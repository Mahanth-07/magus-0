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
