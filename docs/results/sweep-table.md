| game | verdict | induction (overall / primary) | duel (planner vs baseline) |
|---|---|---|---|
| 01_2048 | DUEL_WON | 0.89 / 1.00 | 172 vs 24 |
| 02_another-gentlemans-adventure | DUEL_LOST | 0.91 / 1.00 | 0 vs 2 |
| 03_astray | INDUCTION_FAILED | 0.86 / 0.82 | — |
| 04_boxel-rebound | ONBOARD_FAILED | — | — |
| 05_breakout | DUEL_LOST | 0.96 / 0.96 | 0.0333333 vs 0.0666667 |
| 06_captaincallisto | DUEL_LOST | 0.77 / 1.00 | 0 vs 2 |
| 07_chrome-dino | EXPLORE_FAILED | — | — |
| 08_core-ball | ONBOARD_FAILED | — | — |
| 09_cubefield | INDUCTION_FAILED | 0.80 / 0.06 | — |
| 10_doodle-jump | DUEL_WON | 0.95 / 1.00 | 67 vs 15 |
| 11_edge-surf | DUEL_LOST | 1.00 / 1.00 | 32 vs 33 |
| 13_flappy-bird | EXPLORE_FAILED | — | — |
| 14_geodash | INDUCTION_FAILED | 0.79 / 0.28 | — |
| 15_google-snake | DUEL_LOST | 0.90 / 1.00 | 0 vs 1 |
| 16_hextris | DUEL_TIE | 0.92 / 1.00 | 0 vs 0 |
| 17_mario-game | DUEL_TIE | 0.95 / 1.00 | 0 vs 0 |
| 18_minecraft-clone-glm | DUEL_TIE | 0.94 / 1.00 | 124 vs 124 |
| 19_minesweeper | EXPLORE_FAILED | — | — |
| 20_monkey-mart | EXPLORE_FAILED | — | — |
| 21_ns-shaft | INDUCTION_FAILED | 0.74 / 1.00 | — |
| 22_ovo | ONBOARD_FAILED | — | — |
| 23_pacman | EXPLORE_FAILED | — | — |
| 24_restless-wing-syndrome | INDUCTION_FAILED | 0.89 / 0.72 | — |
| 25_rocket-league-2d | ONBOARD_FAILED | — | — |
| 26_run-3 | ONBOARD_FAILED | — | — |
| 27_stack | DUEL_WON | 0.97 / 1.00 | 4 vs 0 |
| 28_temple-run-2 | INDUCTION_FAILED | 0.81 / 0.00 | — |
| 29_tetris | INDUCTION_FAILED | 0.00 / 0.00 | — |
| 30_vex-3 | ONBOARD_FAILED | — | — |
| 31_wolf3d | DUEL_TIE | 0.96 / 1.00 | 0 vs 0 |
| 32_wordle | EXPLORE_FAILED | — | — |
| 33_worlds-hardest-game | ONBOARD_FAILED | — | — |
| 34_worlds-hardest-game-2 | ONBOARD_FAILED | — | — |

Totals: DUEL_LOST=5, DUEL_TIE=4, DUEL_WON=3, EXPLORE_FAILED=6, INDUCTION_FAILED=7, ONBOARD_FAILED=8
Totals: DUEL_LOST=5, DUEL_TIE=4, DUEL_WON=3, EXPLORE_FAILED=6, INDUCTION_FAILED=7, ONBOARD_FAILED=8

## Student eval (2026-07-12): planner-teacher vs fine-tuned VLM student (15 steps)

Student = Qwen3-VL-8B LoRA on 886 reward-filtered planner trajectories, screenshots only.
| game | planner [raw_state] | student [screenshot] | winner |
|---|---|---|---|
| 01_2048 | 56 | 68 | student |
| 10_doodle-jump | 67 | 67 | tie |
| 27_stack | 4 | 0 | planner |

## Student v2 (RECAP-lite, 2026-07-12): 5038 rows (60 expert/44 novice episodes), expert-conditioned eval

| game | planner | student v2 | (student v1) |
|---|---|---|---|
| 01_2048 | 56 | 68 | 68 |
| 10_doodle-jump | 107 | 30 | 67 |
| 27_stack | 4 | 0 | 0 |
| 16_hextris | 0 | 0 | — |

Honest read: v2 (5.7x data + RECAP conditioning) did NOT beat v1 at 15-step duels;
doodle-jump regressed. Caveats: N=1 per cell (planner itself swung 67->107 across
runs), and 42% novice rows may dilute despite conditioning. Next levers: repeated
duels for variance, expert-only ablation, longer-horizon evals.

## N=5 variance duels (planner vs student v1, 15 steps)

| game | planner mean±std | student v1 mean±std | winner (means) |
|---|---|---|---|
| 01_2048 | 102.4±56.8 [56..172] | 68±0.0 | planner |
| 10_doodle-jump | 83±19.6 | **106.8±0.4** | **student** |
| 27_stack | 4.8±0.4 | 0±0.0 | planner |

Corrections vs N=1 claims: the "student beats planner on 2048" was a planner low-roll
(bimodal 56/172); at N=5 planner wins 2048. But the student WINS doodle-jump decisively
(106.8 vs 83) and is nearly deterministic everywhere (temp-0, std ~0) while the planner
is high-variance. The distillation claim that survives: on 1 of 3 games the pixel-only
student outplays its own raw-state teacher; on all games it is far more consistent.

## Onboarding round 1 re-sweep (2026-07-13, sweep2.jsonl)

Wake-up keys + guardrails GRADUATED 2 of 14 dead games all the way to duels:
13_flappy-bird (INDUCED, DUEL_TIE) and 32_wordle (INDUCED, DUEL_TIE). Coverage
now 14/33 INDUCED. Remaining 9 ONBOARD_FAILED are the deep bucket: client boot
hangs (wait_until_actionable) and mouse-dependent games — Horizon-1 item 2.

## Client-level boot fixes + mouse support (2026-07-13, sweep3.jsonl)

9 target failures re-swept after adding: (1) 15s initial boot wait + keyboard/mouse
wake sequence in `_connect()`; (2) `mouse:center/top/lower/3q` pseudo-keys in
`WAKE_KEYS`; (3) `_is_actionable()` helper accepting "ready" || "playing" in prober
and explore; (4) `ReadyGame`/`ClickMenuGame` synthetic games for TDD. 16 new tests,
369→385 passing.

| game | sweep2 verdict | sweep3 verdict | notes |
|---|---|---|---|
| 04_boxel-rebound | ONBOARD_FAILED (boot hang) | INDUCTION_FAILED (0.81/0.00) | Booted! 10 controls. Induction fails: world state not learnable (primary_accuracy=0.0 after 3 rounds) |
| 08_core-ball | ONBOARD_FAILED (boot hang) | **DUEL_WON** (8 vs 6) | Booted via mouse:center. 1 control. INDUCED 0.993/1.0. Planner wins |
| 19_minesweeper | ONBOARD_FAILED (never playing) | ONBOARD_FAILED | Mouse-only game; no keyboard controls discovered. Needs DEFAULT_PROBE_KEYS mouse extension |
| 20_monkey-mart | ONBOARD_FAILED (no controls) | ONBOARD_FAILED | Mouse-only game; no keyboard controls discovered |
| 22_ovo | ONBOARD_FAILED (boot hang) | DUEL_LOST (0 vs 1) | Booted via mouse:center. 11 controls. INDUCED 0.869/1.0. Planner loses (narrow game) |
| 23_pacman | ONBOARD_FAILED (no controls) | EXPLORE_FAILED | 6 controls found. explore.py "ready" bug fixed after this run; needs re-sweep |
| 26_run-3 | ONBOARD_FAILED (boot hang) | ONBOARD_FAILED | Booted (status→playing), but 0 keyboard controls. Mouse-only |
| 30_vex-3 | ONBOARD_FAILED (boot hang) | ONBOARD_FAILED | Wake keys exhausted; game stays in menu. Canvas-specific click needed |
| 33_worlds-hardest-game | ONBOARD_FAILED (boot hang) | DUEL_TIE (1 vs 1) | Booted via mouse:lower. 11 controls. INDUCED 0.950/1.0. 0-0 tie in 15 steps |
| 34_worlds-hardest-game-2 | ONBOARD_FAILED (boot hang) | EXPLORE_FAILED | Booted (per boot log), but explore produced 0 transitions (likely same "ready" explore bug as pacman) |

Net: 4 of 9 ONBOARD_FAILEDs escaped: 08_core-ball → DUEL_WON; 22_ovo → DUEL_LOST;
33_worlds-hardest-game → DUEL_TIE; 04_boxel-rebound → INDUCTION_FAILED (progress).
Mouse-only games (minesweeper, monkey-mart, run-3) and vex-3 remain blocked —
need DEFAULT_PROBE_KEYS mouse extension (future work). pacman/34_worlds-hardest-game-2
hit the explore "ready" bug fixed mid-sweep; require re-sweep.

INDUCED coverage: 14 → 16 (added core-ball, ovo, worlds-hardest-game; boxel-rebound
INDUCTION_FAILED counts as partial progress).

## Tie-break round (40 steps × 3 repeats, 2026-07-13)

| game | planner | gateway | verdict |
|---|---|---|---|
| 16_hextris | 78±17 | 107±25 | baseline |
| 31_wolf3d | 0±0 | 0.7±0.9 | baseline |
| 18_minecraft | 124±0 | 124±0 | structural tie (metric accrues identically) |
| 33_whg | 1±0 | 1±0 | structural tie |
| 17_mario, 13_flappy, 32_wordle | 0±0 | 0±0 | structural tie (metric never moves) |

Finding: longer horizons resolved 2/7 ties (both planner losses). The other 5 are
STRUCTURAL: the primary metric never moves (or moves identically) under either agent —
wrong metric choice or missing action vocabulary (wordle needs typing; mario/flappy die
instantly). This is an onboarding-quality issue, not a planning one.

## v3-expert N=5 duels (2026-07-14, corrected eval — model verified v3)

```
==== DUEL 01_2048 (15 steps x5, metric=primary_score) ====
  planner   (planner ): primary_score=102.4±56.83  scores=[172.0, 172.0, 56.0, 56.0, 56.0]  legal_rate=1.00  [raw_state]
  baseline  (together): primary_score=8±0.00  scores=[8.0, 8.0, 8.0, 8.0, 8.0]  legal_rate=1.00  [screenshot]
  winner: planner
==== DUEL 10_doodle-jump (15 steps x5, metric=primary_score) ====
  planner   (planner ): primary_score=30±0.00  scores=[30.0, 30.0, 30.0, 30.0, 30.0]  legal_rate=1.00  [raw_state]
  baseline  (together): primary_score=68.2±34.44  scores=[107.0, 67.0, 30.0, 107.0, 30.0]  legal_rate=1.00  [screenshot]
  winner: baseline
==== DUEL 27_stack (15 steps x5, metric=primary_score) ====
  planner   (planner ): primary_score=4.4±0.49  scores=[4.0, 4.0, 5.0, 4.0, 5.0]  legal_rate=1.00  [raw_state]
  baseline  (together): primary_score=0±0.00  scores=[0.0, 0.0, 0.0, 0.0, 0.0]  legal_rate=1.00  [screenshot]
  winner: planner
```
Compare: v1 doodle 106.8±0.4 (beat planner); v2 unconditioned doodle 60.2±28.7.

**Ablation verdict:** v3-expert REGRESSED (2048 collapsed 68→8; doodle 68±34 vs v1's
106.8±0.4). Novice-row dilution was NOT v2's problem — both retrains underperform v1.
**v1 (886 reward-filtered rows) remains the best student.** Clean finding: at this
scale, the small focused first recipe beats 3-6x more data under either labeling scheme.

## v4 (16-game corpus, core-ball HELD OUT) — N=5, 2026-07-14

```
==== DUEL 01_2048 (15 steps x5, metric=primary_score) ====
  planner   (planner ): primary_score=56±0.00  scores=[56.0, 56.0, 56.0, 56.0, 56.0]  legal_rate=1.00  [raw_state]
  baseline  (together): primary_score=44±0.00  scores=[44.0, 44.0, 44.0, 44.0, 44.0]  legal_rate=1.00  [screenshot]
  winner: planner
==== DUEL 10_doodle-jump (15 steps x5, metric=primary_score) ====
  planner   (planner ): primary_score=99±16.00  scores=[107.0, 107.0, 67.0, 107.0, 107.0]  legal_rate=1.00  [raw_state]
  baseline  (together): primary_score=37.4±14.80  scores=[30.0, 30.0, 30.0, 30.0, 67.0]  legal_rate=1.00  [screenshot]
  winner: planner
==== DUEL 08_core-ball (15 steps x5, metric=primary_score) ====
  planner   (planner ): primary_score=3±0.00  scores=[3.0, 3.0, 3.0, 3.0, 3.0]  legal_rate=1.00  [raw_state]
  baseline  (together): primary_score=6.6±2.80  scores=[8.0, 8.0, 8.0, 8.0, 1.0]  legal_rate=1.00  [screenshot]
  winner: baseline
```
08_core-ball is the TRANSFER TEST: the student never trained on it.

**v4 verdict — FIRST TRANSFER RESULT:** on held-out core-ball (never in training),
v4 scored 6.6±2.8 vs the planner's 3±0 — the student BEAT the planner on a game it
never saw, while regressing on trained games (2048 44 vs v1's 68; doodle 37 vs 106.8).
Multi-game training traded per-game peak for cross-game generalization. First evidence
for the Horizon-4 claim.

## v5 transfer replication: 10_doodle-jump held out — N=5, 2026-07-22

**INVALID — serving failure, not a measurement:** the student's legal_rate was 0.00 (every decide() errored; adapter registration silently failed on a missing project id). Retained for honesty; superseded by the rerun below.
```
==== DUEL 10_doodle-jump (15 steps x5, metric=primary_score) ====
  planner   (planner ): primary_score=107±0.00  scores=[107.0, 107.0, 107.0, 107.0, 107.0]  legal_rate=1.00  [raw_state]
  baseline  (together): primary_score=0±0.00  scores=[0.0, 0.0, 0.0, 0.0, 0.0]  legal_rate=0.00  [screenshot]
  winner: planner
```

## CMA-ES teacher tuning (2026-07-23): sim tuning + real N=5 validation

### 01_2048
```
  "baseline_sim_fitness": 8.5,
  "iterations": 1
}
-- real duel (tuned planner vs gateway, N=5):
==== DUEL 01_2048 (15 steps x5, metric=primary_score) ====
  planner   (planner ): primary_score=56±0.00  scores=[56.0, 56.0, 56.0, 56.0, 56.0]  legal_rate=1.00  [raw_state]
  baseline  (gateway ): primary_score=40±15.39  scores=[24.0, 56.0, 60.0, 24.0, 36.0]  legal_rate=1.00  [screenshot]
  winner: planner
```

### 10_doodle-jump
```
  "baseline_sim_fitness": 0.0,
  "iterations": 1
}
-- real duel (tuned planner vs gateway, N=5):
==== DUEL 10_doodle-jump (15 steps x5, metric=primary_score) ====
  planner   (planner ): primary_score=75.6±28.98  scores=[30.0, 67.0, 107.0, 67.0, 107.0]  legal_rate=1.00  [raw_state]
  baseline  (gateway ): primary_score=29.6±19.11  scores=[14.0, 14.0, 53.0, 53.0, 14.0]  legal_rate=1.00  [screenshot]
  winner: planner
```

### 08_core-ball
```
  "baseline_sim_fitness": 2.625,
  "iterations": 1
}
-- real duel (tuned planner vs gateway, N=5):
==== DUEL 08_core-ball (15 steps x5, metric=primary_score) ====
  planner   (planner ): primary_score=4±2.00  scores=[3.0, 3.0, 3.0, 8.0, 3.0]  legal_rate=1.00  [raw_state]
  baseline  (gateway ): primary_score=8±0.00  scores=[8.0, 8.0, 8.0, 8.0, 8.0]  legal_rate=1.00  [screenshot]
  winner: baseline
```

**CMA-ES round-1 verdict — NULL, with a precise diagnosis:** the optimizer returned
default weights on all 3 games (sim_fitness == baseline exactly); the duels above
therefore measure unchanged-planner variance, not tuning. Root causes: (a) doodle's
induced model yields ZERO score under rollout — it cannot express scoring dynamics
forward, so no scoring function can help; (b) 2048/core-ball landscapes flat at this
budget (240 evals, 8 rollouts × 25 steps). Inert weight files removed. Round-2 fixes
queued: fitness against models with primary_accuracy 1.0 only, larger budgets, longer
rollouts, and feature set that can matter at depth-3 argmax.

## Planning-grade audit (2026-07-22)

Validation-grade (dual gate: predicts observed transitions) vs planning-grade
(forward planner rollout inside the model can generate positive sign-adjusted
primary-metric delta; `planning_grade` in ludus/planning/tuning.py, default
weights, 6 rollouts x 15 steps, depth 2, beam 8, inducer holdout split).
Backfilled into every `worldmodels/*/report.json` by
`scripts/backfill_planning_grade.py`.

| game | validation overall/primary | planning-grade | mean rollout |
| --- | --- | --- | --- |
| 01_2048 | 0.891 / 1.00 (INDUCED) | yes | 6.00 |
| 02_another-gentlemans-adventure | 0.910 / 1.00 (INDUCED) | no | 0.00 |
| 03_astray | 0.859 / 0.82 (FAILED) | no | 0.00 |
| 04_boxel-rebound | 0.807 / 0.00 (FAILED) | yes | 0.22 |
| 05_breakout | 0.960 / 0.96 (INDUCED) | no | 0.00 |
| 06_captaincallisto | 0.766 / 1.00 (INDUCED) | no | 0.00 |
| 08_core-ball | 0.993 / 1.00 (INDUCED) | yes | 2.83 |
| 09_cubefield | 0.801 / 0.06 (FAILED) | yes | 6000.00 |
| 10_doodle-jump | 0.946 / 1.00 (INDUCED) | no | 0.00 |
| 11_edge-surf | 1.000 / 1.00 (INDUCED) | no | 0.00 |
| 13_flappy-bird | 0.776 / 1.00 (INDUCED) | no | 0.00 |
| 14_geodash | 0.790 / 0.28 (FAILED) | yes | 15.00 |
| 15_google-snake | 0.896 / 1.00 (INDUCED) | no | 0.00 |
| 16_hextris | 0.917 / 1.00 (INDUCED) | no | 0.00 |
| 17_mario-game | 0.947 / 1.00 (INDUCED) | no | 0.00 |
| 18_minecraft-clone-glm | 0.939 / 1.00 (INDUCED) | no | 0.00 |
| 21_ns-shaft | 0.736 / 1.00 (FAILED) | no | 0.00 |
| 22_ovo | 0.869 / 1.00 (INDUCED) | no | 0.00 |
| 24_restless-wing-syndrome | 0.890 / 0.72 (FAILED) | no | 0.00 |
| 27_stack | 0.973 / 1.00 (INDUCED) | yes | 1.83 |
| 28_temple-run-2 | 0.810 / 0.00 (FAILED) | yes | 255.00 |
| 29_tetris | 0.000 / 0.00 (FAILED) | no | 0.00 |
| 31_wolf3d | 0.960 / 1.00 (INDUCED) | no | 0.00 |
| 32_wordle | 0.952 / 1.00 (INDUCED) | no | 0.00 |
| 33_worlds-hardest-game | 0.950 / 1.00 (INDUCED) | no | 0.00 |
| synthetic:grid | 0.990 / 1.00 (INDUCED) | yes | 13.33 |

8/26 planning-grade. Read the two quadrants carefully: doodle-jump is the
canonical validation-yes/planning-no case (the CMA-ES round-1 diagnosis, now
recorded), and it has 14 INDUCED siblings with the same failure mode. The four
FAILED-but-planning-yes models (boxel, cubefield, geodash, temple-run-2) score
in rollout precisely because their primary-metric predictions are WRONG
(primary 0.00-0.28) — hallucinated reward, not usable dynamics. Only models
with BOTH gates (2048, core-ball, stack, synthetic:grid) are candidates for
in-model tuning (round-2 lever).
