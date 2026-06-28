# CLAUDE.md — Ludus Repo Guide

Ludus is a game-agnostic AI agent that plays GameWorld browser games via **semantic macro-actions**. One shared planner drives multiple games; it predicts outcomes, reflects on prediction error, and accumulates a compact **rulebook**. Persistence is dual-write: local JSONL/PNG fallback + InsForge (Postgres traces + S3 screenshots).

**Thesis:** memory (learned rules) beats baseline (no rules) on state-verifiable metrics, across >1 game, with one shared brain.

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
- GameWorld is located at `./gameworld` (repo-relative) by default; override with `GAMEWORLD_ROOT=/path/to/gameworld`. `gameworld_client.py` adds it to `sys.path`.

(Historically this machine used a separate base-anaconda env for live runs; the single-`.venv` setup above is the portable default.)

---

## Architecture

```
PlannerContext ──► ModelProvider.decide() ──► Decision
                        │
                 build_prompt() (game-agnostic)
                        │
                 GameWorldClient.apply(action)
                        │
              screenshot + metric delta
                        │
               OutcomeDetector (improved?)
                        │
              Reflector ──► Rulebook (memory mode only)
                        │
             DualWriteStore (LocalStore + InsForgeStore)
```

- **`ludus/schemas.py`** — Pydantic contracts: `Decision`, `ActionArgs`, `PlannerContext`, `StepRecord`, `EpisodeResult`.
- **`ludus/loop.py`** — `run_episode(adapter, provider, store, mode, steps)`. Game-agnostic. Do NOT edit to add game logic.
- **`ludus/providers/`** — `ModelProvider` protocol + `MockProvider` (deterministic), `InsForgeGatewayProvider`, `AnthropicProvider`, `FallbackProvider` (tries in order, logs WARNING on degradation to mock).
- **`ludus/adapters/`** — thin per-game `GameAdapter` protocol impls: `tetris.py`, `wolf3d.py`, `fireboy_watergirl.py`. Supply legal actions, control map, metrics, timing — no strategy.
- **`ludus/persistence/`** — `LocalStore` (JSONL steps + PNG), `InsForgeStore`, `DualWriteStore` (local required, InsForge best-effort).
- **`ludus/outcome.py`**, **`ludus/reflection.py`**, **`ludus/rulebook.py`** — metric-based reflection pipeline.
- **`configs/`** — per-game YAML: objective, legal_actions, control_map, primary_metric, timing.
- **`server/app.py`** — minimal FastAPI dashboard (read-only from `runs/`). Start: `.venv/bin/python -m uvicorn server.app:app --port 8137`.

---

## CLI

```bash
python -m ludus.cli <game> <mode> [--steps N] [--provider mock|gateway|anthropic|fallback] [--fake]   # from the active .venv
```

- `game`: `tetris`, `wolf3d`, `fireboy_watergirl` (mapped to GameWorld IDs internally via `GAMEWORLD_IDS`).
- `mode`: `baseline` (no rules) | `memory` (rules fed into prompt).
- `--fake`: uses offline `FakeGameWorld` — no browser, no network.
- `--provider fallback`: tries InsForge gateway → Anthropic → Mock, logs WARNING on degradation.

---

## Running tests

```bash
.venv/bin/pytest -v          # or: source .venv/bin/activate && pytest -v
```

43 tests, all offline. Tests cover schemas, mock provider, fallback provider, outcome detection, reflection, rulebook, local store, dual store, loop, config loader, and the tetris adapter.

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
4. Cross-check the control keys against the real GameWorld game before committing — metric field names must match what the GameWorld API returns (inspect with `--fake --provider mock` first, then verify live).
5. Add adapter tests to `tests/test_<game>_adapter.py` following the tetris pattern.

---

## Providers / InsForge

- `InsForgeGatewayProvider` — model gateway (primary, vision-capable).
- `AnthropicProvider` — fallback via Anthropic API.
- `MockProvider` — deterministic, offline, returns a fixed `Decision`. Used by all 43 tests.
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

`runs/` is gitignored. The dashboard server (`server/app.py`) reads this layout directly.

---

## Experiment script

```bash
bash scripts/experiment.sh
```

Loops tetris + wolf3d × baseline + memory, 12 steps each, with `--provider fallback`. Run from the active `.venv` (with GameWorld installed) + live creds. Override interpreter via `LUDUS_PYTHON`, step count via `STEPS`.

---

## Gotchas

- `GameWorldClient` is a context manager — always use `with GameWorldClient(...) as gw:` or call `gw.close()` explicitly (no browser leak).
- Fireboy & Watergirl is co-op: the agent controls Watergirl; Fireboy keys are recorded via a partner-key recorder but not sent by the agent.
- `FallbackProvider` logs a WARNING (not an error) when it degrades to mock — a misconfigured live run will be clearly visible in logs.
- Metric field names must exactly match what the GameWorld API returns. A mismatch emits a warning, not an error — silent wrong data if ignored.
- InsForge egress: LiveKit appends `-TR_<trackId>` to egress filenames — parse the participant UUID from `track_<uuid>-TR_<id>.ogg`, not the whole token. (Voice product, not core Ludus.)

---

## Cut order (if time-constrained)

Rocket League 2D → custom Next.js dashboard → model-based reflection (metric-based is the default) → extra analytics.
