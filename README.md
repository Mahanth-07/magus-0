# Ludus — Game-Agnostic AI Agent

**Hackathon tracks:** GameCraft Arena · Best Use of InsForge

---

## Thesis

One shared brain — one planner, one rulebook — beats the no-rules baseline across multiple games. The agent learns compact rules by reflecting on prediction error and reuses them in subsequent steps. The memory advantage is measurable on state-verifiable game metrics (score, kills, diamonds collected).

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        run_episode loop                       │
│                                                              │
│  Screenshot ──► PlannerContext ──► ModelProvider.decide()    │
│                     │                      │                 │
│              legal_actions          Decision (action,        │
│              learned_rules           expected_result,        │
│              recent_outcomes         confidence)             │
│                                           │                  │
│                             GameWorldClient.apply(action)    │
│                                           │                  │
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
- **Pydantic contracts everywhere.** `Decision`, `PlannerContext`, `StepRecord`, `EpisodeResult` — all typed, all validated.
- **Dual-write persistence.** Local store is always required; InsForge write is best-effort. A failed InsForge write logs a warning but never aborts the run.
- **Survival hedge.** The entire loop is proven offline: 43 tests, `MockProvider` + `FakeGameWorld`, zero network.

---

## InsForge Integration

| Layer | What Ludus writes |
|---|---|
| **Postgres** | One row per step (`StepRecord`) — decision, metrics, rule learned, screenshot reference. One `EpisodeResult` per run. |
| **S3** | Per-step screenshots (`step_XXXX.png`) alongside the JSONL trace. |
| **Model gateway** | `InsForgeGatewayProvider` sends screenshot + prompt to the InsForge vision endpoint; receives a structured `Decision`. |

`DualWriteStore` wraps `LocalStore` (always) + `InsForgeStore` (best-effort). `FallbackProvider` tries InsForge gateway → Anthropic → Mock in order, logging a `WARNING` on degradation so a misconfigured run is never silent.

---

## Verified live evidence

### Tetris

Score climbs across steps; pieces respond to left/right/rotate/drop key sequences. Metric: `score` + `lines_remaining`.

### Wolf3D

Ammo drops 8 → 4 on attack actions (confirmed by metric delta). Guard returns fire: `health` 100 → 90. Metrics: `kills`, `health`, `ammo`, `lives`.

### Fireboy & Watergirl (co-op)

Agent controls Watergirl (a/d/w keys); Fireboy arrows captured by a partner-key recorder. Watergirl traverses the level. Metric: `score` (diamonds collected).

---

## Results: baseline vs memory

> ⏳ Filled after the live experiment run (`scripts/experiment.sh` with vision creds).

---

## How to run

### Offline (no creds, no browser)

```bash
# Unit tests (43, all offline):
/Users/mahanth/ludus/.venv/bin/pytest -v

# Fake run (no browser, mock provider):
/opt/anaconda3/bin/python -m ludus.cli tetris baseline --provider mock --fake --steps 5
```

### Live (browser + creds)

```bash
# Set env: INSFORGE_API_KEY, INSFORGE_DB_URL, AWS_* for S3
/opt/anaconda3/bin/python -m ludus.cli tetris baseline --provider fallback --steps 12
/opt/anaconda3/bin/python -m ludus.cli wolf3d memory --provider fallback --steps 12
```

### Experiment (baseline vs memory, 2 games)

```bash
bash scripts/experiment.sh   # requires anaconda python + live creds
```

### Dashboard

```bash
/Users/mahanth/ludus/.venv/bin/python -m uvicorn server.app:app --port 8137
# open http://localhost:8137 — polls /api/state every 1.2s
```

---

## File map

```
configs/            per-game YAML (objective, legal_actions, control_map, metrics, timing)
ludus/
  schemas.py        Pydantic contracts (Decision, PlannerContext, StepRecord, EpisodeResult)
  loop.py           game-agnostic run_episode
  providers/        ModelProvider protocol + Mock / InsForge / Anthropic / Fallback
  adapters/         thin per-game GameAdapter impls (tetris, wolf3d, fireboy_watergirl)
  persistence/      LocalStore, InsForgeStore, DualWriteStore
  outcome.py        OutcomeDetector
  reflection.py     Reflector (metric-based)
  rulebook.py       Rulebook
  cli.py            entrypoint: python -m ludus.cli <game> <mode> [options]
server/app.py       minimal FastAPI dashboard (read-only from runs/)
scripts/
  experiment.sh     baseline vs memory loop across tetris + wolf3d
runs/               local persistence output (gitignored)
tests/              43 offline unit tests
```
