# Magus-1 — Induced World Models: a general game agent that writes each game's simulator

**Date:** 2026-07-04
**Status:** Approved design, pre-implementation
**Extends:** Magus-0 (this repo). The loop, providers, persistence, rulebook, and
distillation pipeline are reused; everything hand-written-per-game becomes
machine-generated-per-game.

---

## 1. Problem and goal

Magus-0 proved a recipe on Tetris: externalize the world model into code
(`rank_placements` pre-simulates every placement), let the neural net only
perceive and select. Evidence ladder: free-form VLM play 238 → code-simulated
candidate list 466 (gateway) / 1022 (local 3B student) → pure code planning
(teacher) 27k+. The neural net was never the strength; **the simulator was**.

The blocker to generality: the simulator, the teacher, the adapter config, and
the state-text renderer were all hand-written for Tetris. Repeating that per
game does not scale.

**Goal:** an agent that "catches on" to a new game autonomously, where catching
on has a concrete mechanism — the agent *induces the game's rules as executable
Python*, validates them against real transitions, plans against its own
simulator, and self-distills the resulting expert play into one local
multi-game student.

**Success criteria (user-set):**
1. On games the system has never seen, after bounded autonomous interaction it
   **beats a zero-shot frontier VLM** playing the same game with the same step
   budget.
2. Horizon goal: competent, fully autonomous play — point at a game id, no
   human input.
3. Signature demo: **Fireboy & Watergirl co-op** — the agent plays *with a
   human partner* well, using partner-behavior prediction.

**Context:** side project + portfolio piece. No hard deadline; rigor and
writeup-worthiness matter. Budget: API calls + laptop-scale local models; no
GPU-farm assumptions.

**Game universe:** GameWorld's 34-game benchmark first (all expose
`window.gameAPI.getState()`, pause/resume, keyboard controls). Architecture
keeps a pluggable state-extraction seam so pixels-only games can be added
later; only the GameWorld path is built now.

---

## 2. Architecture: five stages, each independently demoable

```
                        ┌─ Stage 1: AUTO-ONBOARD ─────────────────────────┐
 game id ──────────────►│ probe gameAPI: discover controls (press keys,   │
                        │ diff state), find metrics, infer objective       │──► GameProfile (JSON, cached)
                        └──────────────────────────────────────────────────┘
                        ┌─ Stage 2: WORLD-MODEL INDUCTION ────────────────┐
 GameProfile ──────────►│ explore episodes → log (state, action, state')  │
                        │ transitions → LLM writes predict()/reward() as  │──► worldmodels/<game>/model.py
                        │ Python → validate on held-out transitions →     │    + validation report
                        │ repair loop with counterexamples until ≥90%     │    (INDUCED or FAILED)
                        └──────────────────────────────────────────────────┘
                        ┌─ Stage 3: MODEL-BASED PLANNER ──────────────────┐
 induced model ────────►│ depth-limited rollout search over the agent's   │
                        │ OWN simulator → ranked candidate macros →       │──► beats zero-shot VLM
                        │ selector copies best (general rank_placements)  │    (the head-to-head demo)
                        └──────────────────────────────────────────────────┘
                        ┌─ Stage 4: SELF-DISTILLATION ────────────────────┐
 planner episodes ─────►│ (prompt, macro) pairs from ALL induced games →  │──► one local student =
                        │ one multi-game LoRA student → student becomes   │    policy prior for search
                        │ the planner's policy prior (AlphaZero-flavored) │    + acts alone when confident
                        └──────────────────────────────────────────────────┘
                        ┌─ Stage 5: CO-OP (Fireboy & Watergirl) ──────────┐
 partner keys ─────────►│ predict the human's next moves → plan jointly   │──► plays WITH you
                        │ against the induced model                        │
                        └──────────────────────────────────────────────────┘

 Fallback path (always available): induction FAILED → frontier VLM + rulebook,
 still auto-onboarded, still plays — the system degrades to today's Magus-0.
```

### Design commitments

1. **State is a pluggable source.** `StateSource` protocol
   (`raw_state() -> dict`); gameAPI implementation now, a vision-based
   extractor can slot in later without touching anything downstream.
2. **The world model is code, not weights.** The agent's understanding of a
   game is a Python `predict(state, action) -> next_state` function it wrote
   and debugged itself, validated against held-out real transitions.
   Inspectable, versionable, testable.
3. **Every stage gates on measurement.** World models are only planned against
   if validated; the planner only claims victory via same-budget head-to-heads;
   the student only ships if it matches planner decisions on held-out states.

### Why the distillation stage matters (vs plain universal-teacher distillation)

A student distilled from a frontier VLM cannot beat a zero-shot frontier VLM
on unseen games — imitation caps at the teacher, and frontier VLMs are
mediocre at these games. Here, **the planner generates trajectories better
than any frontier model** wherever induction succeeds; distilling those is a
genuine self-improvement loop: better prior → deeper effective search → better
data → better prior.

---

## 3. Components

```
ludus/
  onboarding/
    prober.py       # ControlProber: launch game, press candidate keys, diff raw
                    # state, label each key by observed effect. MetricFinder:
                    # numeric fields that move + LLM naming pass -> objective text.
    profile.py      # GameProfile (pydantic): game_id, controls {semantic: key},
                    # metrics, primary_metric, state_schema, timing. Cached as
                    # profiles/<game>.json — replaces hand-written configs/*.yaml.
  worldmodel/
    transitions.py  # TransitionStore: append/load (state, macro, state', deltas)
                    # JSONL per game; train/holdout split by episode.
    inducer.py      # Synthesis loop: prompt LLM with schema + sampled
                    # transitions -> predict()/reward()/legal() source; on
                    # validation failure, re-prompt with counterexamples. Budgeted.
    sandbox.py      # Runs induced code in a subprocess: timeout, no filesystem/
                    # network, import allowlist (math, copy, itertools).
    validate.py     # Field-weighted prediction accuracy on held-out transitions;
                    # emits the validation report that gates promotion.
  planning/
    planner.py      # Depth-limited rollout search over the induced model:
                    # enumerate macros from discovered controls, simulate d steps,
                    # score by induced reward; top-k out. (rank_placements is the
                    # hand-written special case this generalizes.)
    candidates.py   # Renders top-k into the proven candidate-block prompt format;
                    # selector model copies candidate [0].
  partner/
    model.py        # PartnerModel: predict human's next macro from recent
                    # partner keys + state (Markov baseline, LLM refinement);
                    # planner rolls out WITH predicted partner actions (Stage 5).
```

**Reused as-is:** `loop.py` (episode driver), all providers (selector and
fallback are providers), `persistence/`, `rulebook.py`/`reflection.py`
(residual dynamics the induced code can't express), the distillation scripts
(Stage 4 = today's pipeline pointed at multi-game planner data instead of the
Tetris teacher).

**CLI — one subcommand per stage, so every stage is a demo:**

```
python -m ludus.cli onboard <game>                 # -> profiles/<game>.json
python -m ludus.cli explore <game>                 # -> data/<game>/transitions.jsonl
python -m ludus.cli induce <game>                  # -> worldmodels/<game>/model.py + report
python -m ludus.cli play <game> --provider planner # model-based play
python -m ludus.cli duel <game> --steps 30         # planner vs zero-shot VLM, same budget
```

**Data flow, end to end:** game id → prober → `GameProfile` → explorer plays
ε-random episodes → `transitions.jsonl` → inducer ⇄ sandbox/validator until
pass or budget out → `model.py` → planner plays for real (score, and SFT rows
as exhaust) → rows from all induced games → one LoRA student → student
re-enters the planner as its policy prior.

**v1 simplification:** the planner searches macro sequences of discovered
semantic actions at small depth (2–4 plies), not full game-tree search. Tetris
needed exactly depth-1 over ~40 candidates to reach 466/1022. Depth is a
tunable, not an architecture change.

---

## 4. Error handling — every stage can fail, no stage can lie

- **Sandboxed induced code.** All `model.py` execution in a subprocess with
  timeout, import allowlist, no filesystem/network. A synthesized function that
  hangs or throws marks the candidate failed — never crashes the agent.
- **Validation gates promotion.** Below-threshold world models (target: ≥90%
  field-weighted accuracy on held-out transitions) are stored with reports but
  never planned against. `duel` refuses to run against an unvalidated model.
- **Graceful degradation is a first-class path.** Induction FAILED → the game
  is still playable via frontier VLM + rulebook (today's Magus-0), still
  auto-onboarded. Coverage claims stay honest: "N of 34 induced; the rest
  degrade to reactive play."
- **Budget caps everywhere.** Synthesis: max repair iterations + max API spend
  per game. Exploration: max episodes/steps. Resistant games cost a bounded
  amount, then get flagged.
- **Prober safety.** Some keys pause/reset/quit; the prober snapshots state
  before each probe and relaunches on irrecoverable states; probing is
  idempotent per key.

---

## 5. Testing — offline-first, same discipline as the existing 128 tests

- **Toy games with known ground truth.** A `SyntheticGame` suite (grid-pusher,
  counter-game, simple faller — pure Python, FakeGameWorld-style) where true
  dynamics are known. CI asserts: prober discovers controls, inducer recovers
  dynamics ~100%, planner reaches optimal play. Whole pipeline runs in CI with
  a mocked LLM (canned synthesis responses) — zero network.
- **Tetris as the gold game.** The induced Tetris model must agree with the
  hand-written `drop_piece` on a corpus of real transitions — the one game
  with a perfect reference world model is the calibration standard.
- **The duel harness is the science.** Same game, same step budget, same seeds
  where possible: planner-agent vs zero-shot gemini-flash vs (where it exists)
  the hand-written teacher. Every writeup claim comes from `duel` output,
  persisted like any episode.
- **Student regression.** Held-out (game, state) pairs: student's macro must
  match the planner's choice ≥ threshold per game — catches a student that
  silently forgot a game when new ones are added.

---

## 6. Milestones — each a portfolio artifact

1. **M1:** `onboard` works on 5+ games with zero hand-written config; VLM plays
   them via generated profiles. *(Kills the per-game tedium.)*
2. **M2:** First non-Tetris game fully induced + planner **wins a duel** vs
   zero-shot frontier VLM. *(The headline result.)*
3. **M3:** Sweep all 34 games; publish the induction success table.
   *(The benchmark table.)*
4. **M4:** Multi-game student distilled from planner data; transfer numbers on
   held-out games. *(The "foundation model" claim, honestly scoped.)*
5. **M5:** Fireboy & Watergirl co-op with partner prediction. *(The demo
   people remember.)*

---

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Physics-heavy games (continuous state, e.g. Wolf3D raycasting) resist code induction | Partial models: predict discrete/eventful parts ("attack while facing enemy in range decrements HP"), leave pixel-precise motion to the rulebook; degrade gracefully |
| Exploration data too shallow → induced model overfits early-game states | Curiosity-weighted exploration (prefer actions from states unlike anything logged); grow the transition store across sessions |
| Induced `reward()` mis-specifies the objective → planner optimizes the wrong thing | Reward derived from prober-discovered `primary_metric` deltas (ground truth from the game), not synthesized from scratch |
| Prober breaks game state (quit/reset keys) | Snapshot + relaunch; per-key idempotent probing |
| LLM synthesis cost balloons on hard games | Hard per-game API budget; FAILED is an acceptable, reported outcome |
| Selector model ignores candidate [0] | Already solved in Magus-0: recency-positioned directive + (later) the distilled student, 50/50 exact-copy on validation |

---

## 8. Out of scope for v1

- Pixels-only games (seam exists via `StateSource`; not built).
- Full game-tree search / MCTS with UCB (depth-limited rollout only).
- RL fine-tuning of the student (the planner already exceeds the reactive
  ceiling; RL remains a future direction).
- Multiplayer beyond the single human co-op partner in Fireboy & Watergirl.
