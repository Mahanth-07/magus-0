# CMA-ES teacher tuning (Horizon 2, item 1) — approved design

Parameterize planner beam scoring as weights over generic features of predicted
next-states: [primary_delta, state_change_frac, terminal_penalty,
secondary_metric_delta_sum, macro_len]. Default weights [1,0,0,0,0] == current
behavior (backward compat).

Tuning: per INDUCED game, CMA-ES (pip `cma`) over the 5-vector; fitness = mean
final simulated primary metric over K-step planner rollouts INSIDE the induced
world model (sandbox), start states sampled from data/<game>/transitions.jsonl.
Output: worldmodels/<game>/planner_weights.json {weights, sim_fitness, baseline_sim_fitness}.
PlannerProvider loads weights when file present.

Honest gate: sim gains mean nothing until a real N=5 duel beats the recorded
real-game baseline (2048: 102.4±56.8; doodle: 83-107; core-ball: 3±0).
Fully local; zero cloud dependencies.
