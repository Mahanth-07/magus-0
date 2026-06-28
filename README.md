# Ludus — Game-Agnostic AI Agent

**Hackathon tracks:** GameCraft Arena · Best Use of InsForge

---

## Thesis

One shared brain — one planner, one rulebook — plays multiple games from screenshots alone. The agent emits semantic actions, predicts each outcome, reflects on prediction error to learn compact rules, and reuses them in later steps. Every decision, prediction, outcome, and screenshot is persisted to InsForge.

Whether learned memory *beats* the no-rules baseline is measured directly on state-verifiable metrics (score, kills, health, diamonds) rather than asserted — see **Results** below for the honest, mixed findings and what they reveal about the reflection design.

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

Real run, `scripts/experiment.sh`, 12 steps each, **gpt-4o-mini vision via the InsForge gateway**. All decisions/outcomes/screenshots were persisted to InsForge during these runs.

| Game | Mode | Primary metric | Secondary | Legal-action rate | Rules learned |
|---|---|---|---|---|---|
| Tetris | baseline | **score 66** | 14 cells | 100% | 0 |
| Tetris | memory | score 44 | 10 cells | 100% | 3 |
| Wolf3D | baseline | kills 0 | **health 0 (died)** | 100% | 0 |
| Wolf3D | memory | kills 0 | **health 100 (preserved)** | 100% | 5 |

**Honest read.** The results are mixed, not a clean memory win:

- **Wolf3D — memory helped.** The baseline agent's health dropped to 0 (it took fire); the memory agent finished with full health (100), directly matching the "preserve health" objective. Neither scored a kill at this step budget.
- **Tetris — memory hurt** (44 < 66). Memory learned 3 rules but scored lower.
- **Structured output never failed** — 100% legal-action rate in every run; the planner always returned a parseable `Decision`.

**Why "memory beats baseline" is not cleanly shown here — three real confounds:**

1. **Within-episode only.** Each run starts with a *fresh* rulebook; rules learned in early steps only help later steps of the *same* 12-step run. There is no cross-episode accumulation yet.
2. **Tiny sample.** One 12-step trial per cell, with a non-deterministic vision model. No averaging over seeds/trials.
3. **Blunt reflection.** This build uses the spec's cut-down **metric-based** `Reflector`, which emits generic "action X did not improve <metric>; try an alternative" rules. In Tetris — where most non-drop moves never change `score` — that floods the rulebook with discouraging rules that can suppress useful actions. The spec's **model-based** reflection (deferred) would phrase sharper, situational rules.

**What the experiment *does* prove:** the full learn-and-reuse loop runs end-to-end on real games via real vision, the rulebook genuinely accumulates and feeds back, structured outputs are robust, and every step is persisted to InsForge. The harness can now answer "does memory help?" empirically — the next step to make it a clean win is cross-episode rule persistence + model-based reflection + multi-trial averaging.

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

Create `~/ludus/.env` (auto-loaded by the CLI — no sourcing needed):

```bash
INSFORGE_API_KEY=ik_...
INSFORGE_BASE_URL=https://<id>.us-east.insforge.app
INSFORGE_GATEWAY_URL=https://<id>.us-east.insforge.app/api/ai   # gateway base; chat = {this}/chat/completion
INSFORGE_VISION_MODEL=openai/gpt-4o-mini                        # any vision-capable gateway model id
INSFORGE_S3_BUCKET=<your-bucket>                                # created in the InsForge dashboard
ANTHROPIC_API_KEY=sk-ant-...                                    # optional fallback provider
```

Then create the InsForge tables once, and run:

```bash
/opt/anaconda3/bin/python scripts/insforge_setup.py            # creates ludus_episodes + ludus_steps
/opt/anaconda3/bin/python -m ludus.cli tetris baseline --provider gateway --steps 12
/opt/anaconda3/bin/python -m ludus.cli wolf3d memory  --provider gateway --steps 12
```

`--provider fallback` chains gateway → Anthropic → mock (warns loudly if it degrades to mock, so a misconfigured run is never silent). Live runs need the **base anaconda** python (it has Playwright + GameWorld's deps); the offline tests run in `.venv`.

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
