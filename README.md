# Magus-0 — Game-Agnostic AI Agent

**Hackathon tracks:** GameCraft Arena · Best Use of InsForge

---

## Thesis

One shared planner (general) + a learning loop to make it expert (skilled).

The agent drives multiple GameWorld browser games from screenshots + a textual board summary alone. It emits ordered action sequences, predicts each outcome, reflects on prediction error to accumulate compact rules, and persists every decision, prediction, outcome, and screenshot to InsForge.

In parallel, a **π0-style distillation loop** uses a classical expert teacher to generate training labels, fine-tunes a small LLM via LoRA, and targets a student that can play at near-teacher level without calling a frontier model at inference time. The honest bound: imitation distillation can *match* the teacher, not beat it — surpassing the teacher requires RL with the game's verifiable reward signal, a future direction.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        run_episode loop                       │
│                                                              │
│  Screenshot + state_text ──► PlannerContext ──► ModelProvider.decide()  │
│                                   │                    │     │
│                           legal_actions         Decision:    │
│                           learned_rules           action     │
│                           recent_outcomes         actions[]  │
│                           board state_text        confidence │
│                                                       │      │
│                    [pause] → actions macro applied → [resume]│
│                                                       │      │
│                              new screenshot + metric delta   │
│                                           │                  │
│                                 OutcomeDetector              │
│                                    (improved?)               │
│                                           │                  │
│                     ┌─────── memory mode ─┴─ baseline mode  │
│                     │                        (skip reflect)  │
│               Reflector                                      │
│                  │                                           │
│               Rulebook (accumulates learned rules)           │
│                                                              │
│  DualWriteStore ─► LocalStore (JSONL + PNG)                  │
│                 └► InsForgeStore (Postgres + S3)             │
└──────────────────────────────────────────────────────────────┘
```

**Key design decisions:**

- **Thin adapters.** Each game provides only legal actions, control mappings, and metric extraction — zero strategy. The planner is entirely game-agnostic.
- **Board-state text alongside screenshots.** `GameWorldClient.state_text()` derives column heights, hole count, current/next/held piece from `window.gameAPI.getState()` and injects it as `PlannerContext.state_text`. Placement becomes arithmetic, not pixel-guessing.
- **Action sequences.** `Decision.actions` carries an ordered macro (e.g. `["left","left","rotate","drop"]`) applied atomically per turn. The loop pauses game gravity during inference and the action sequence, so a piece isn't locked mid-positioning.
- **Pydantic contracts everywhere.** `Decision`, `PlannerContext`, `StepRecord`, `EpisodeResult` — all typed, all validated.
- **Dual-write persistence.** Local store is always required; InsForge write is best-effort. A failed InsForge write logs a warning but never aborts the run.
- **Proven offline.** The entire loop is verified offline: 128 tests, `MockProvider` + `FakeGameWorld`, zero network.

---

## InsForge Integration

| Layer | What Magus-0 writes |
|---|---|
| **Postgres** | One row per step (`StepRecord`) — decision, metrics, rule learned, screenshot reference. One `EpisodeResult` per run. |
| **S3** | Per-step screenshots (`step_XXXX.png`) alongside the JSONL trace. |
| **Model gateway** | `InsForgeGatewayProvider` sends screenshot + board state text + prompt to the InsForge vision endpoint; receives a structured `Decision`. |

`DualWriteStore` wraps `LocalStore` (always) + `InsForgeStore` (best-effort). `FallbackProvider` tries InsForge gateway → Anthropic → Mock in order, logging a `WARNING` on degradation so a misconfigured run is never silent.

---

## Results

### Zero-shot baseline: gpt-4o-mini (no board state, single actions)

Real run, `scripts/experiment.sh`, 12 steps, gpt-4o-mini via the InsForge gateway.

| Game | Mode | Primary metric | Notes |
|---|---|---|---|
| Tetris | baseline | **score 66** | pixel-only; no board summary |
| Tetris | memory | score 44 | memory hurt (see confounds below) |
| Wolf3D | baseline | health 0 (died) | took fire, health dropped to 0 |
| Wolf3D | memory | **health 100** | memory helped — survived the run |

### With board-state text + action sequences: gemini-2.5-flash

Switching the model to `google/gemini-2.5-flash` via the InsForge gateway, adding `state_text` (column heights, holes, current/next piece), and issuing action-sequence macros per turn:

| Game | Result |
|---|---|
| Tetris | **score ~238** vs 66 zero-shot baseline |

The jump from 66 to ~238 comes from three compounding improvements: a stronger reasoning model, arithmetic placement via board state, and atomic multi-move positioning that avoids pieces locking at spawn.

### With candidate-placement lookahead: gemini-2.5-flash

A VLM can't simulate piece interlock from a board summary, so we stopped asking it to: `rank_placements` (in the teacher module) enumerates every legal rotation × column hard-drop in code, scores each resulting board with the full Dellacherie evaluation, and injects the ranked top-6 — with the exact open-loop action macro that reaches each — into the prompt. The model's job collapses to "copy candidate [0] verbatim."

| Game | Result |
|---|---|
| Tetris | **score 466** in 15 steps (vs ~238 free-form, vs 66 zero-shot) |

The macros are rotation-shift-compensated (`_macro_live` replicates the game's own rotation geometry), so a blind open-loop apply lands the piece in the intended column.

### Expert teacher: Dellacherie heuristic

The `ludus/teacher/tetris_heuristic.py` heuristic (Tetris only) enumerates all rotation × column placements, simulates the drop, scores by holes/height/bumpiness/lines, and picks the best. It exists as a label source, not the product.

| | Score |
|---|---|
| Teacher self-play | **~27,000–100,000+**, ~0 holes |

### Distillation student: LoRA fine-tuned Llama-3.2-3B-Instruct — BEATS THE GATEWAY

2,000 simulated teacher self-play examples (`scripts/collect_dataset_sim.py` — no browser, seconds not hours), exported to text SFT format (`scripts/export_nebius.py`), LoRA fine-tuned on Nebius AI Studio (`scripts/finetune_nebius.py`, val_loss 9.3e-06), and served **locally** via ollama (Nebius trains LoRA but cannot serve it — see gotchas).

| Provider | Live Tetris, 15 steps | Offline val (held-out) |
|---|---|---|
| **Student** (Llama-3.2-3B LoRA, local, free) | **1022** | 50/50 exact macro matches |
| gemini-2.5-flash via gateway | 466 | — |
| zero-shot gpt-4o-mini baseline | 66 | — |

The student — a 3B model running on a laptop, no frontier model, no network — copies candidate [0] more faithfully than gemini-flash over the gateway: 100% legal actions, zero omitted rotates, four line clears in 15 steps including a +322 multi-line. This is the distillation thesis demonstrated end-to-end.

**Imitation ceiling:** the student learns to copy the teacher's choice; its per-step decision quality now matches the teacher, so the remaining gap to teacher self-play totals (~27k+) is episode length, not decision quality. Beating the teacher requires RL/self-play with the game's verifiable score signal — a future direction.

### Honest read on the baseline experiment

The memory vs baseline results are mixed, not a clean win:

- **Wolf3D — memory helped.** Baseline health → 0 (died); memory agent finished at 100.
- **Tetris — memory hurt** (44 < 66). Three real confounds:
  1. **Within-episode only.** Rules accumulate within a single run; there is no cross-episode persistence yet.
  2. **Tiny sample.** One 12-step trial per cell; a non-deterministic vision model means results are not averaged over seeds.
  3. **Blunt reflection.** The metric-based `Reflector` emits generic rules ("action X did not improve score"). In Tetris — where most non-drop moves never change `score` — this floods the rulebook and suppresses useful actions.

What the experiment *does* prove: the loop runs end-to-end on real games via real vision, structured output was 100% legal-action compliant in every run, and every step is persisted to InsForge.

---

## Learning loop (π0-style distillation)

The goal is a general AND skilled agent: generality from one shared planner across games; skill from a distillation loop that uses expert demonstrations as training signal.

```
Expert teacher (tetris_heuristic) ──► SIMULATED self-play (tetris_sim, no browser)
        │
        ▼
Dataset collection (scripts/collect_dataset_sim.py) — 2000 examples in seconds
  prompt: rendered by the SAME live state-text function (candidate block included)
  label : candidate [0] (Dellacherie-best) with _macro_live-corrected macro
  + DAgger-style recovery states: epsilon-random placement APPLIED from the
    displayed top-6 menu, label stays the teacher's best (student learns to
    act well from messy boards, not just the teacher's near-perfect line)
        │
        ▼
Export to SFT format (scripts/export_nebius.py --mode text --in examples_sim.jsonl)
  text mode: board reaches student via state_text, no image (Llama-3.2-3B-Instruct)
        │
        ▼
LoRA fine-tune on Nebius AI Studio (scripts/finetune_nebius.py)
  model: meta-llama/Llama-3.2-3B-Instruct — 9 min, val_loss 9.3e-06
        │
        ▼
Serve LOCALLY (Nebius cannot serve LoRA — its support list is empty):
  download adapter -> convert_lora_to_gguf.py -> ollama create (Modelfile ADAPTER)
  NebiusProvider unchanged: NEBIUS_BASE_URL=http://localhost:11434/v1
        │
        ▼
A/B eval: student 1022 vs gemini-2.5-flash 466 (live, 15 steps) ✅
          + 50/50 exact macro matches on held-out validation
```

**Why the dataset was regenerated (v2):** the original ~510 live-collected examples had two fatal flaws — (1) macros from `_macro`, which assumes rotation preserves the leftmost column (the live collector re-read the board after each rotate, but the runtime loop applies macros *blind*, so a perfectly-imitating student would land rotated pieces in the wrong column), and (2) prompts predating the candidate-lookahead format. The sim collector fixes both by construction and adds recovery states.

**Generality note:** the Dellacherie teacher is Tetris-specific. The path to a general-purpose skilled agent (true π0-style) is multi-game distillation using a strong frontier VLM as a universal teacher across many games — the roadmap, not yet built.

---

## Roadmap

1. ~~**Deploy student + A/B**~~ ✅ **Done** — student served locally via ollama, live A/B: student 1022 vs gemini-flash 466 (15 steps), 50/50 offline val.
2. **Cross-episode rule persistence** — persist the rulebook across runs so reflection compounds over time.
3. **Multi-game distillation** — build teachers for Wolf3D and Fireboy & Watergirl; collect multi-game SFT data; fine-tune a shared student.
4. **RL to surpass the teacher** — the cheapest path is improving the *ranking function* itself (tune Dellacherie weights via CMA-ES against the `tetris_sim` simulator, or a small value network on simulated rollouts); the whole pipeline inherits the improvement since the teacher is the label source. Full policy-gradient RL would also train against the simulator, warm-started from the student.
5. **Model-based reflection** — replace the blunt metric-based `Reflector` with a model-authored situational rule, sharper than "action X did not improve score."
6. **Magus-1 (in progress)** — see `docs/specs/2026-07-04-magus-1-induced-world-models-design.md`.
   **M1 auto-onboarding: ✅** `python -m ludus.cli onboard <game>` probes any GameWorld game
   (presses candidate keys, diffs `getState()`, filters ambient drift) and generates a playable
   `GameProfile` — controls, metrics, objective — with zero hand-written config. First live sweep:
   5/5 games produced playable profiles (tetris + breakout with clean movement semantics; snake +
   flappy-bird exposed a known M2 fix: press-window motion attribution in continuous games). Play
   any profile with `--profile profiles/<game>.json`.

---

## How to run

### Setup (first time)

```bash
# 1. Clone this repo, then create one env with the offline core + (optionally) GameWorld:
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

# 2. (Live runs only) Clone GameWorld into ./gameworld and install its deps + Chromium:
git clone https://github.com/gameworld-project/gameworld.git
pip install -r gameworld/requirements.txt && playwright install chromium

# 3. (Live runs only) The 34-game HTML library is nested under benchmark/ in its repo —
#    copy the games into place so they sit at gameworld/games/benchmark/<game>/index.html:
git clone https://github.com/gameworld-dev/gameworld-games.git /tmp/gw-games
cp -R /tmp/gw-games/benchmark/* gameworld/games/benchmark/ && rm -rf /tmp/gw-games
```

GameWorld is found at `./gameworld` by default; set `GAMEWORLD_ROOT=/path/to/gameworld` to override.
Offline (tests + `--fake`) needs only step 1. Live runs need all three + a `.env` (below).

### Offline (no creds, no browser)

```bash
source .venv/bin/activate
pytest -v --ignore=infra                              # 128 tests, all offline
python -m ludus.cli tetris baseline --provider mock --fake --steps 5   # fake run, no browser
```

### Live (browser + creds)

Create a `.env` in the repo root (auto-loaded by the CLI):

```bash
INSFORGE_API_KEY=ik_...
INSFORGE_BASE_URL=https://<id>.us-east.insforge.app
INSFORGE_GATEWAY_URL=https://<id>.us-east.insforge.app/api/ai   # gateway base; chat = {this}/chat/completion
INSFORGE_VISION_MODEL=google/gemini-2.5-flash                   # recommended; any vision-capable gateway model id
INSFORGE_S3_BUCKET=<your-bucket>                                # created in the InsForge dashboard
ANTHROPIC_API_KEY=sk-ant-...                                    # optional fallback provider

# Distillation student (optional). NEBIUS_API_KEY is used for TRAINING only;
# serving is local (Nebius cannot serve LoRA adapters). At run time point the
# provider at the local ollama server instead:
NEBIUS_API_KEY=...                                              # training (finetune_nebius.py)
NEBIUS_BASE_URL=http://localhost:11434/v1                       # local ollama (OpenAI-compatible)
NEBIUS_MODEL=magus-tetris-student                               # ollama model with the LoRA adapter
NEBIUS_MULTIMODAL=0                                             # text-only LoRA student
```

Then create the InsForge tables once, and run (from the activated `.venv`):

```bash
source .venv/bin/activate
python scripts/insforge_setup.py                                  # creates ludus_episodes + ludus_steps
python -m ludus.cli tetris baseline --provider gateway --steps 12
python -m ludus.cli wolf3d memory  --provider gateway --steps 12
python -m ludus.cli tetris baseline --provider nebius --steps 12  # student (after fine-tune)
```

`--provider fallback` chains gateway → Anthropic → mock (warns loudly if it degrades to mock, so a misconfigured run is never silent).

### Experiment (baseline vs memory, 2 games)

```bash
source .venv/bin/activate && bash scripts/experiment.sh          # needs GameWorld + live creds
```

### Distillation pipeline (Tetris)

```bash
source .venv/bin/activate

# 1. Collect simulated teacher self-play — no browser, ~3 seconds for 2000 examples.
#    --epsilon injects DAgger-style recovery states (suboptimal placement applied,
#    teacher's best move kept as the label).
python scripts/collect_dataset_sim.py 2000 --epsilon 0.15 --seed 7

# 2. Export to text SFT format (train/val split):
python scripts/export_nebius.py --mode text --in data/tetris/examples_sim.jsonl

# 3. LoRA fine-tune on Nebius AI Studio (~10 min; fails loudly if no servable id):
python scripts/finetune_nebius.py

# 4. Serve LOCALLY — Nebius trains LoRA but cannot serve it. Download the final
#    checkpoint's files (GET /v1/files/{id}/content, follow redirects) into
#    data/tetris/lora_student/ckpt-NN/, then:
python /path/to/llama.cpp/convert_lora_to_gguf.py \
  --base <dir-with-base-config.json> --outtype f16 \
  --outfile data/tetris/lora_student/tetris-student-lora.gguf \
  data/tetris/lora_student/ckpt-NN
ollama pull llama3.2:3b-instruct-q8_0
ollama create magus-tetris-student -f data/tetris/lora_student/Modelfile
#    (Modelfile: FROM llama3.2:3b-instruct-q8_0 / ADAPTER <abs path to .gguf>
#     / PARAMETER temperature 0 / PARAMETER num_ctx 8192)

# 5. Run the student live (NebiusProvider pointed at the local server):
NEBIUS_BASE_URL=http://localhost:11434/v1 NEBIUS_API_KEY=local-ollama \
NEBIUS_MODEL=magus-tetris-student NEBIUS_MULTIMODAL=0 \
python -m ludus.cli tetris baseline --provider nebius --steps 15
```

Legacy live-browser collection (`scripts/collect_dataset.py`, screenshots included for a
future vision student) and the live teacher demo (`scripts/teacher_play.py`) still work
but need GameWorld + Playwright in the env.

### Dashboard

```bash
.venv/bin/python -m uvicorn server.app:app --port 8137
# open http://localhost:8137 — polls /api/state every 1.2s
```

---

## File map

```
configs/            per-game YAML (objective, legal_actions, control_map, metrics, timing)
ludus/
  schemas.py        Pydantic contracts (Decision, PlannerContext, StepRecord, EpisodeResult)
                    Decision.actions = ordered action macro; PlannerContext.state_text = board summary
  loop.py           game-agnostic run_episode; pause/resume around inference + action sequence
  providers/        ModelProvider protocol + Mock / InsForge / Anthropic / Fallback / Nebius
  adapters/         thin per-game GameAdapter impls (tetris, wolf3d, fireboy_watergirl)
  persistence/      LocalStore, InsForgeStore, DualWriteStore
  outcome.py        OutcomeDetector
  reflection.py     Reflector (metric-based)
  rulebook.py       Rulebook
  teacher/
    tetris_heuristic.py   Dellacherie-style placement heuristic + rank_placements
                          (candidate lookahead: every rotation x column pre-simulated,
                          _macro_live rotation-shift-compensated open-loop macros)
    tetris_sim.py         simulated self-play collector (no browser) — live-format
                          prompts, candidate-[0] labels, DAgger recovery states
  cli.py            entrypoint: python -m ludus.cli <game> <mode> [options]
server/app.py       minimal FastAPI dashboard (read-only from runs/)
scripts/
  experiment.sh           baseline vs memory loop across tetris + wolf3d
  teacher_play.py         drive the live teacher heuristic (calibration + demo)
  collect_dataset.py      LIVE browser collection (legacy; screenshots for vision student)
  collect_dataset_sim.py  SIMULATED collection → data/tetris/examples_sim.jsonl (preferred)
  export_nebius.py        export examples to Nebius/OpenAI SFT format (vision or text mode)
  finetune_nebius.py      launch LoRA fine-tune on Nebius AI Studio; poll to completion;
                          fails loudly if the job yields no servable model id
data/tetris/        distillation dataset (gitignored): examples_sim.jsonl + nebius SFT files
                    + lora_student/ (downloaded adapter, GGUF, ollama Modelfile)
runs/               local persistence output (gitignored)
tests/              128 offline unit tests
```
