# CLAUDE.md — Magus-0 Repo Guide

Magus-0 is a game-agnostic AI agent that plays GameWorld browser games via **semantic macro-actions**. One shared planner drives multiple games; it predicts outcomes, reflects on prediction error, and accumulates a compact **rulebook**. Board state is also sent as text alongside screenshots so placement is arithmetic, not pixel-guessing. Persistence is dual-write: local JSONL/PNG fallback + InsForge (Postgres traces + S3 screenshots).

In parallel, a **π0-style distillation loop** uses a classical expert teacher (Tetris-only Dellacherie heuristic) to generate training labels, fine-tunes a small text LLM (Llama-3.2-3B-Instruct) via LoRA on Nebius AI Studio, and targets a student that plays at near-teacher level without a frontier model at inference time. A/B evaluation is in progress.

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
- **`ludus/teacher/tetris_heuristic.py`** — Dellacherie-style Tetris placement heuristic. Enumerates every (rotation, column), simulates the drop, scores by holes/height/bumpiness/lines/wells/transitions, returns the best action macro. Tetris-specific; used to generate training labels for distillation, not as the product.
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

---

## Running tests

```bash
.venv/bin/pytest -v --ignore=infra   # 92 tests, all offline
```

Tests cover schemas (including `actions` macro + `state_text`), mock provider, fallback provider, outcome detection, reflection, rulebook, local store, dual store, loop, config loader, and the tetris adapter.

---

## Distillation pipeline

### Scripts

| Script | What it does |
|---|---|
| `scripts/teacher_play.py` | Drive the live Dellacherie teacher heuristic in the real game (calibration + demo). |
| `scripts/collect_dataset.py` | Run teacher self-play and log one SFT example per placement → `data/tetris/examples.jsonl` + `data/tetris/images/`. |
| `scripts/export_nebius.py` | Export `examples.jsonl` to Nebius/OpenAI SFT format. `--mode text` produces text-only train/val splits (for Llama-3.2-3B-Instruct). `--mode vision` embeds screenshots as data URIs. |
| `scripts/finetune_nebius.py` | Upload SFT files, create a LoRA fine-tuning job on Nebius AI Studio, poll to completion. Prints the fine-tuned model id on success — set it as `NEBIUS_MODEL`. |

### Data

`data/tetris/` (gitignored): `images/ex_NNNNNN.png`, `examples.jsonl`, `nebius_text_sft.jsonl` (train), `nebius_text_val.jsonl` (val).

### Env vars for the student

```bash
NEBIUS_API_KEY=...
NEBIUS_BASE_URL=https://api.studio.nebius.com/v1   # default
NEBIUS_MODEL=<fine-tuned-model-id>
NEBIUS_MULTIMODAL=0   # 0 = text-only; omit or set 1 for multimodal
```

### Imitation ceiling

The LoRA student is trained by imitation; its performance ceiling is the teacher's (~27k+ Tetris score). Beating the teacher requires RL with a verifiable game reward — not yet built.

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
- `MockProvider` — deterministic, offline, returns a fixed `Decision`. Used by all 92 tests.
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
- The teacher scripts (`teacher_play.py`, `collect_dataset.py`) require GameWorld + Playwright in the active env.
