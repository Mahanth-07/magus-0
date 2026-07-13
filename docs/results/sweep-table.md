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
