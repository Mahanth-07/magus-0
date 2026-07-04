# CLAUDE.md — Magus-0 Repo Guide

Magus-0 is a game-agnostic AI agent that plays GameWorld browser games via **semantic macro-actions**. One shared planner drives multiple games; it predicts outcomes, reflects on prediction error, and accumulates a compact **rulebook**. Board state is also sent as text alongside screenshots so placement is arithmetic, not pixel-guessing. Persistence is dual-write: local JSONL/PNG fallback + InsForge (Postgres traces + S3 screenshots).

In parallel, a **π0-style distillation loop** uses a classical expert teacher (Tetris-only Dellacherie heuristic) to generate training labels, fine-tunes a small text LLM (Llama-3.2-3B-Instruct) via LoRA on Nebius AI Studio, and serves the student **locally via ollama** (Nebius cannot serve LoRA). **A/B result: student 1022 vs gemini-2.5-flash 466 on live 15-step Tetris; 50/50 exact macro matches on held-out validation.** The Tetris prompt includes a code-computed "Candidate placements" block (every rotation × column pre-simulated and Dellacherie-ranked); the model's job is to copy candidate [0] verbatim.

**Default vision model:** `google/gemini-2.5-flash` via the InsForge gateway (set `INSFORGE_VISION_MODEL`). Anthropic and Mock remain as fallbacks.

---

## Environment

The recommended setup is **one env** (a `.venv`) holding both the offline core and GameWorld's deps:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
# live runs only: install GameWorld deps into the SAME env
pip install -r gameworld/requirements.txt && playwright install chromium
```

- **Offline** (unit tests, `--fake` runs) needs only `requirements.txt` + the `ludus` package — no playwright.
- **Live** (real browser games) additionally needs GameWorld + playwright in the active env.
- GameWorld is located at `./gameworld` (repo-relative) by default; override with `GAMEWORLD_ROOT=/path/to/gameworld`.

---

## Architecture

```
screenshot + state_text ──► PlannerContext ──► ModelProvider.decide() ──► Decision
                                 │                                            │
                          build_prompt()                           action + actions[]
                          (game-agnostic,                                     │
                           state_text included)               [pause] → macro applied → [resume]
                                                                              │
                                                                   screenshot + metric delta
                                                                              │
                                                                   OutcomeDetector (improved?)
                                                                              │
                                                               Reflector ──► Rulebook (memory mode)
                                                                              │
                                                              DualWriteStore (LocalStore + InsForgeStore)
```

### Core modules

- **`ludus/schemas.py`** — Pydantic contracts:
  - `Decision` — includes `actions: list[str]` (ordered action macro) and `confidence` (coerced from 0–100 to 0–1 for models that return a percentage).
  - `PlannerContext` — includes `state_text: str` (board summary from `GameWorldClient.state_text()`).
  - `StepRecord`, `EpisodeResult`.
- **`ludus/loop.py`** — `run_episode(...)`. Game-agnostic. Pauses the game during model inference + action-sequence application; resumes after. `Decision.actions` macro is applied atomically (falls back to the single `action` if `actions` is empty). Do NOT add game logic here.
- **`ludus/providers/`** — `ModelProvider` protocol + impls:
  - `MockProvider` — deterministic offline, fixed `Decision`.
  - `InsForgeGatewayProvider` (name `"gateway"`) — vision via InsForge model gateway; sends screenshot + board text.
  - `AnthropicProvider` (name `"anthropic"`) — Anthropic API fallback.
  - `FallbackProvider` (name `"fallback"`) — tries gateway → Anthropic → Mock; logs `WARNING` on degradation.
  - `NebiusProvider` (name `"nebius"`) — Nebius AI Studio (OpenAI-compatible). Two modes:
    - **Multimodal** (default): text + screenshot data URI (for a VLM student).
    - **Text-only** (`NEBIUS_MULTIMODAL=0`): system + user STRING only, no image; the student sees the board via `state_text`. This is the mode used by the LoRA fine-tuned Llama-3.2-3B-Instruct.
  - **`_coerce_confidence`** (in `insforge.py`): normalizes model-emitted confidence from 0–100 scale to 0–1 so models that return `95` instead of `0.95` don't cause a Pydantic validation error.
- **`ludus/adapters/`** — thin per-game `GameAdapter` protocol impls: `tetris.py`, `wolf3d.py`, `fireboy_watergirl.py`. Supply legal actions, control map, metrics, timing — no strategy.
- **`ludus/persistence/`** — `LocalStore` (JSONL steps + PNG), `InsForgeStore`, `DualWriteStore` (local required, InsForge best-effort).
- **`ludus/outcome.py`**, **`ludus/reflection.py`**, **`ludus/rulebook.py`** — metric-based reflection pipeline.
- **`ludus/teacher/tetris_heuristic.py`** — Dellacherie-style Tetris placement heuristic. Enumerates every (rotation, column), simulates the drop, scores by holes/height/bumpiness/lines/wells/transitions, returns the best action macro. Also exposes `rank_placements` (the candidate-lookahead tool: top-k placements sorted by the SAME Dellacherie score, so candidate [0] == `best_move`'s choice) and `_macro_live` (open-loop macros that compensate for the game's rotation-induced column shift — required because the runtime loop applies macros blind, without re-reading the board). Tetris-specific; used to generate training labels for distillation, not as the product.
- **`ludus/teacher/tetris_sim.py`** — SIMULATED self-play collector (no browser, no I/O). Prompts rendered by the live client's `_tetris_state_text` over a synthetic getState()-shaped dict (exact runtime prompt distribution, candidate block + rolling live-format `recent_outcomes` included); labels = candidate [0]; DAgger-style recovery states via `epsilon` (suboptimal placement from the displayed top-6 APPLIED, teacher's best kept as label).
- **`configs/`** — per-game YAML: objective, legal_actions, control_map, primary_metric, timing.
- **`server/app.py`** — minimal FastAPI dashboard (read-only from `runs/`). Start: `.venv/bin/python -m uvicorn server.app:app --port 8137`.

### Board state text

`GameWorldClient.state_text()` calls `window.gameAPI.getState()` and derives a compact textual summary: per-column stack heights, total holes, current/next/held piece and its column position. This is injected into `PlannerContext.state_text` and included in the prompt so the model can reason arithmetically about where to place a piece. Returns `""` for non-Tetris games or states with no board grid — safe no-op in those cases.

### Action sequences and pause/resume

`Decision.actions` is an ordered list of 1–5 moves applied atomically per turn (e.g. `["rotate", "left", "left", "drop"]`). The loop **pauses** game gravity before the model call and **resumes** after the macro is fully applied, so a Tetris piece doesn't lock at spawn during the slow vision round-trip.

---

## CLI

```bash
python -m ludus.cli <game> <mode> [--steps N] [--provider mock|gateway|anthropic|fallback|nebius] [--fake]
```

- `game`: `tetris`, `wolf3d`, `fireboy_watergirl`.
- `mode`: `baseline` (no rules) | `memory` (rules fed into prompt).
- `--fake`: uses offline `FakeGameWorld` — no browser, no network.
- `--provider nebius`: routes to `NebiusProvider`. Set `NEBIUS_MODEL` + `NEBIUS_API_KEY`. Set `NEBIUS_MULTIMODAL=0` for the text-only LoRA student.
- `--profile profiles/<game>.json`: play via a machine-generated `GameProfile` (GenericAdapter)
  instead of `configs/<game>.yaml` — works for ANY GameWorld catalog id, no per-game code.

### Auto-onboarding (Magus-1 M1)

```bash
python -m ludus.cli onboard <game> [--out profiles] [--headed]
```

- `<game>`: adapter name (`tetris`), raw catalog id (`05_breakout`), or `synthetic:grid` /
  `synthetic:counter` (offline, used by tests).
- Probes candidate keys (`ludus/onboarding/prober.py`), diffs `raw_state()` around each press,
  filters ambient drift + tick counters, names controls heuristically, picks metrics
  (`ludus/onboarding/metrics.py`), and writes `profiles/<game>.json` (`GameProfile`).
- The ambient baseline samples over press-sized windows (M2 fix), so continuous world motion
  (snake, flappy-bird) lands in `ambient_paths` instead of being attributed to keys. Flaky
  `apply()` calls are skipped and counted (`ProbeReport.apply_errors`), never fatal.

### World-model induction (Magus-1 M2)

```bash
python -m ludus.cli explore <game> [--episodes 5 --steps 40 --seed 7]   # -> data/<game>/transitions.jsonl
python -m ludus.cli induce <game> [--iterations 4]                      # -> worldmodels/<game>/model.py + report.json
```

- `explore`: uniform-random self-play over the profile's controls, one `Transition`
  (before/action/after raw states) per press (`ludus/worldmodel/explore.py`). `data/` is
  gitignored.
- `induce`: an LLM (`ludus/worldmodel/llm.py` — anthropic default, gateway fallback, set
  `LUDUS_SYNTH_BACKEND`/`LUDUS_SYNTH_MODEL`) writes `predict(state, action) -> next_state` as
  pure Python; candidates pass an AST allowlist gate and run in a subprocess sandbox
  (`sandbox.py`); a field-weighted validator (`validate.py`, tick/wall-clock paths excluded,
  primary-metric + player paths weigh 3x) gates INDUCED on BOTH train and holdout; train
  counterexamples drive up to `--iterations` repair rounds. Artifacts always saved, FAILED
  verdicts included (`worldmodels/<game>/report.json` has per-field accuracies).
- Live status: `synthetic:grid` INDUCED (1.000 holdout); `29_tetris` FAILED honestly at 0.800
  (piece y-positions are wall-clock-dependent — gravity during the press window). Known M3
  carry-forward: mechanics rarer than (1−threshold) in the data can pass the gate wrong
  (grid's goal mechanic, 4 events in 100 train cases).

---

## Running tests

```bash
.venv/bin/pytest -v --ignore=infra   # 128 tests, all offline
```

Tests cover schemas (including `actions` macro + `state_text`), mock provider, fallback provider, outcome detection, reflection, rulebook, local store, dual store, loop, config loader, and the tetris adapter.

---

## Distillation pipeline

### Scripts

| Script | What it does |
|---|---|
| `scripts/teacher_play.py` | Drive the live Dellacherie teacher heuristic in the real game (calibration + demo). |
| `scripts/collect_dataset_sim.py` | **Preferred.** Simulated teacher self-play (no browser) → `data/tetris/examples_sim.jsonl`. 2000 examples in ~3s. `--epsilon` controls DAgger recovery states. |
| `scripts/collect_dataset.py` | LIVE browser collection (legacy; keeps screenshots for a future vision student) → `data/tetris/examples.jsonl` + `images/`. |
| `scripts/export_nebius.py` | Export examples to Nebius/OpenAI SFT format. `--mode text --in data/tetris/examples_sim.jsonl` produces text-only train/val splits (for Llama-3.2-3B-Instruct). Threads `recent_outcomes` into the prompt. |
| `scripts/finetune_nebius.py` | Upload SFT files, create a LoRA fine-tuning job on Nebius AI Studio, poll to completion. Fails LOUDLY if the succeeded job has no servable model id (Nebius returns `fine_tuned_model: null` + checkpoint files only — never put a `file-...` id in `NEBIUS_MODEL`). |

### Serving the student (local — Nebius cannot serve LoRA)

Nebius trains LoRA but its LoRA-inference support list is empty (`POST /v0/models` rejects all base models). Serve locally instead:

1. Download the final checkpoint's files: `GET {NEBIUS_BASE_URL}/fine_tuning/jobs/{job}/checkpoints` for the checkpoint id, then `GET /files/{id}/content` (**follow redirects — `curl -L`**, plain curl yields 0-byte files) into `data/tetris/lora_student/ckpt-NN/`.
2. Convert PEFT → GGUF: `python llama.cpp/convert_lora_to_gguf.py --base <dir with base config.json> --outtype f16 --outfile tetris-student-lora.gguf ckpt-NN/` (ollama's safetensors `ADAPTER` import rejects Nebius's PEFT layout; GGUF works).
3. `ollama create magus-tetris-student -f data/tetris/lora_student/Modelfile` (FROM `llama3.2:3b-instruct-q8_0`, `ADAPTER` = absolute path to the .gguf, temperature 0, num_ctx 8192).
4. Run: `NEBIUS_BASE_URL=http://localhost:11434/v1 NEBIUS_API_KEY=local-ollama NEBIUS_MODEL=magus-tetris-student NEBIUS_MULTIMODAL=0 python -m ludus.cli tetris baseline --provider nebius --steps 15` (`NebiusProvider` is OpenAI-compatible; no code changes).

### Data

`data/tetris/` (gitignored): `examples_sim.jsonl` (current dataset, 2000 rows), `examples.jsonl` (legacy live), `nebius_text_train.jsonl` / `nebius_text_val.jsonl` (SFT splits), `lora_student/` (adapter + GGUF + Modelfile).

### Results

Student (local 3B LoRA): **1022** on live 15-step Tetris vs gemini-2.5-flash **466**; 50/50 exact macro matches on held-out val; 100% legal actions, zero omitted rotates. Fine-tune: 9 min on Nebius, val_loss 9.3e-06.

### Imitation ceiling

The student copies candidate [0], so its per-step decision quality equals the teacher's; the gap to teacher self-play totals (~27k+) is episode length. Beating the teacher means improving the ranking itself (tune Dellacherie weights against the simulator, or RL with the verifiable game reward) — not yet built.

---

## Games working live

| Game | GameWorld ID | Agent controls | Primary metric |
|---|---|---|---|
| Tetris | `29_tetris` | left/right/rotate=x/drop=Space | `score` |
| Wolf3D | `31_wolf3d` | move=arrows/attack=x | `kills` (+ health/ammo/lives) |
| Fireboy & Watergirl | `12_fireboy-and-watergirl` | Watergirl: a/d/w; Fireboy: arrow keys (human) | `score` (diamonds) |

Fireboy & Watergirl requires `?autostart=1` — handled internally.

---

## How to add a game

1. Add `configs/<game>.yaml` — `objective`, `legal_actions`, `control_map`, `primary_metric`, `step_interval_ms`.
2. Create `ludus/adapters/<game>.py` — implement `GameAdapter` protocol. Supply legal actions, metric extraction, timing. Zero strategy.
3. Add the `<game>` → `<gameworld_id>` mapping to `GAMEWORLD_IDS` in `ludus/cli.py`.
4. Cross-check control keys against the real GameWorld game before committing — metric field names must match what the GameWorld API returns.
5. Add adapter tests to `tests/test_<game>_adapter.py` following the tetris pattern.

---

## Providers / InsForge

- `InsForgeGatewayProvider` — model gateway (primary, vision-capable). Default model: `google/gemini-2.5-flash` (set via `INSFORGE_VISION_MODEL`).
- `AnthropicProvider` — fallback via Anthropic API.
- `MockProvider` — deterministic, offline, returns a fixed `Decision`. Used by all 128 tests.
- `NebiusProvider` — Nebius AI Studio; text-only or multimodal depending on `NEBIUS_MULTIMODAL`.
- `FallbackProvider` — ordered list; degrades on error and logs `WARNING: degrading to ...`.
- `DualWriteStore` — writes local first (required); InsForge write is best-effort. Emits a warning on InsForge failure, never raises.

---

## Persistence layout

```
runs/<episode_id>/
  steps.jsonl          # one StepRecord JSON per line (appended)
  step_XXXX.png        # screenshot per step
  episode.json         # EpisodeResult (written at end of episode)
```

`runs/` is gitignored. The dashboard server reads this layout directly.

---

## Experiment script

```bash
bash scripts/experiment.sh
```

Loops tetris + wolf3d × baseline + memory, 12 steps each, with `--provider fallback`. Run from the active `.venv` (with GameWorld installed) + live creds. Override interpreter via `LUDUS_PYTHON`, step count via `STEPS`.

---

## Gotchas

- `GameWorldClient` is a context manager — always use `with GameWorldClient(...) as gw:` or call `gw.close()` explicitly (no browser leak).
- `GameWorldClient.pause()` / `resume()` freeze game gravity during inference. `FakeGameWorld` doesn't implement these (no-op handled by the loop via `hasattr`).
- `state_text()` returns `""` for non-Tetris games or loading states — the prompt section is omitted when empty.
- **Confidence coercion:** models that return confidence as 0–100 (e.g. `95`) are normalized to 0–1 by `_coerce_confidence` in `insforge.py` before Pydantic validation. Values slightly over 1.0 (e.g. `1.2`) are clamped, not divided.
- Fireboy & Watergirl is co-op: the agent controls Watergirl; Fireboy keys are recorded via a partner-key recorder but not sent by the agent.
- `FallbackProvider` logs a WARNING (not an error) when it degrades to mock — a misconfigured live run is clearly visible in logs.
- Metric field names must exactly match what the GameWorld API returns. A mismatch emits a warning, not an error — silent wrong data if ignored.
- The teacher scripts (`teacher_play.py`, `collect_dataset.py`) require GameWorld + Playwright in the active env. `collect_dataset_sim.py` does NOT (pure Python).
- **Episode ids collide across runs:** `episode_id = f"{game}-{mode}"` (`cli.py`), so re-running appends to `runs/<id>/steps.jsonl` and OVERWRITES `episode.json`. Back up the dir first if the old result matters.
- `_load_dotenv` never overrides already-set env vars — inline `NEBIUS_*=... python -m ludus.cli ...` wins over `.env` (this is how the local-ollama student is selected without editing `.env`).
- **Nebius LoRA:** training works, serving does not (empty support list). Never set `NEBIUS_MODEL` to a `file-...` id. See "Serving the student" above for the local ollama path.
- Nebius `GET /files/{id}/content` redirects to S3 — use `curl -L` or you get 0-byte files.
