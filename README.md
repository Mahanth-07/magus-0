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
- **Proven offline.** The entire loop is verified offline: 92 tests, `MockProvider` + `FakeGameWorld`, zero network.

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

### Expert teacher: Dellacherie heuristic

The `ludus/teacher/tetris_heuristic.py` heuristic (Tetris only) enumerates all rotation × column placements, simulates the drop, scores by holes/height/bumpiness/lines, and picks the best. It exists as a label source, not the product.

| | Score |
|---|---|
| Teacher self-play | **~27,000–100,000+**, ~0 holes |

### Distillation student: LoRA fine-tuned Llama-3.2-3B-Instruct

~510 expert self-play examples collected via `scripts/collect_dataset.py`, exported to text SFT format (`scripts/export_nebius.py`), and LoRA fine-tuned on Nebius AI Studio (`scripts/finetune_nebius.py`). The fine-tune succeeded. **A/B evaluation (student vs gemini-flash vs teacher) is in progress — numbers pending.**

**Imitation ceiling:** the student learns to imitate the teacher; its ceiling is the teacher's performance (~27k+). Beating the teacher requires RL/self-play with the game's verifiable score signal — a future direction.

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
Expert teacher (tetris_heuristic) ──► self-play
        │
        ▼
Dataset collection (scripts/collect_dataset.py)
  pairs: screenshot + state_text + objective + legal_actions ──► teacher Decision (action macro)
        │
        ▼
Export to SFT format (scripts/export_nebius.py)
  text mode: board reaches student via state_text, no image (Llama-3.2-3B-Instruct)
  vision mode: full multimodal (screenshot + text)
        │
        ▼
LoRA fine-tune on Nebius AI Studio (scripts/finetune_nebius.py)
  model: meta-llama/Llama-3.2-3B-Instruct
  dataset: ~510 expert placements
  status: SUCCEEDED
        │
        ▼
Serve the adapter (NebiusProvider, --provider nebius, NEBIUS_MODEL=<ft-model-id>)
  text-only student: NEBIUS_MULTIMODAL=0
        │
        ▼
A/B eval: student vs gemini-2.5-flash vs teacher   ← IN PROGRESS
```

**Generality note:** the Dellacherie teacher is Tetris-specific. The path to a general-purpose skilled agent (true π0-style) is multi-game distillation using a strong frontier VLM as a universal teacher across many games — the roadmap, not yet built.

---

## Roadmap

1. **Deploy student + A/B** — serve the LoRA adapter via `NebiusProvider`, run A/B (student vs gemini-flash vs teacher) on Tetris, report numbers.
2. **Cross-episode rule persistence** — persist the rulebook across runs so reflection compounds over time.
3. **Multi-game distillation** — build teachers for Wolf3D and Fireboy & Watergirl; collect multi-game SFT data; fine-tune a shared student.
4. **RL to surpass the teacher** — use the game's verifiable score signal (PPO or similar) to push past the imitation ceiling.
5. **Model-based reflection** — replace the blunt metric-based `Reflector` with a model-authored situational rule, sharper than "action X did not improve score."

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
pytest -v --ignore=infra                              # 92 tests, all offline
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

# Distillation student (optional):
NEBIUS_API_KEY=...
NEBIUS_BASE_URL=https://api.studio.nebius.com/v1
NEBIUS_MODEL=<fine-tuned-model-id>                              # from finetune_nebius.py output
NEBIUS_MULTIMODAL=0                                             # set to 0 for the text-only LoRA student
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

### Expert teacher (Tetris only)

```bash
# Watch the heuristic play live (calibrates + verifies rotation tables end-to-end):
source .venv/bin/activate && python scripts/teacher_play.py

# Collect a distillation dataset (~510 examples used for the LoRA fine-tune):
python scripts/collect_dataset.py 500

# Export to Nebius SFT format (text-only mode for Llama-3.2-3B-Instruct):
python scripts/export_nebius.py --mode text

# Launch a LoRA fine-tune on Nebius AI Studio:
python scripts/finetune_nebius.py
```

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
    tetris_heuristic.py   Dellacherie-style placement heuristic (label source for distillation)
  cli.py            entrypoint: python -m ludus.cli <game> <mode> [options]
server/app.py       minimal FastAPI dashboard (read-only from runs/)
scripts/
  experiment.sh         baseline vs memory loop across tetris + wolf3d
  teacher_play.py       drive the live teacher heuristic (calibration + demo)
  collect_dataset.py    collect expert self-play examples → data/tetris/examples.jsonl
  export_nebius.py      export examples to Nebius/OpenAI SFT format (vision or text mode)
  finetune_nebius.py    launch LoRA fine-tune on Nebius AI Studio; poll to completion
data/tetris/        distillation dataset (gitignored): images/ + examples.jsonl + nebius SFT files
runs/               local persistence output (gitignored)
tests/              92 offline unit tests
```
