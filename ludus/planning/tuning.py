# ludus/planning/tuning.py
"""CMA-ES teacher tuning (Horizon 2): tune the planner's beam-scoring weights
per game by simulated self-play INSIDE the induced world model.

Fitness of a weight vector = mean final-minus-initial primary-metric delta
(sign-adjusted) over K-step planner rollouts from start states sampled out of
data/<game>/transitions.jsonl. Fully local: the "environment" is the induced
model.py running in the prediction sandbox — zero cloud dependencies.

Honest gate (spec): sim gains mean nothing until a real N=5 duel beats the
recorded real-game baseline. This module only produces the candidate weights
(worldmodels/<game>/planner_weights.json); PlannerProvider picks them up.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from ludus.onboarding.profile import GameProfile
from ludus.planning.planner import DEFAULT_WEIGHTS, _metric, rank_macros
from ludus.worldmodel.sandbox import run_predictions
from ludus.worldmodel.transitions import TransitionStore

log = logging.getLogger("ludus.planning.tuning")

WEIGHT_KEYS = ("primary", "change", "terminal", "secondary", "length")


def _to_vector(weights: dict) -> list[float]:
    return [float(weights[k]) for k in WEIGHT_KEYS]


def _to_weights(vector) -> dict:
    return {k: float(v) for k, v in zip(WEIGHT_KEYS, vector)}


def simulate_rollout(
    model_source: str,
    start_state: dict,
    actions: list[str],
    *,
    weights: dict,
    primary_metric: str,
    higher_is_better: bool,
    steps: int = 25,
    depth: int = 3,
    beam: int = 8,
) -> float:
    """One planner self-play episode inside the induced model.

    Repeatedly rank_macros -> take the best macro -> advance the state through
    run_predictions one action at a time (predictions are sequential). Stops
    on terminal status, a failed prediction, an empty ranking, or the step
    budget. Returns the sign-adjusted final-minus-initial primary metric.
    """
    sign = 1.0 if higher_is_better else -1.0
    initial = _metric(start_state, primary_metric)
    state = start_state
    applied = 0
    while applied < steps and str(state.get("status", "playing")) == "playing":
        ranked = rank_macros(
            model_source, state, actions,
            primary_metric=primary_metric, higher_is_better=higher_is_better,
            depth=depth, beam=beam, top_k=1, weights=weights)
        if not ranked:
            break
        stopped = False
        for action in ranked[0]["actions"]:
            if applied >= steps:
                break
            pred = run_predictions(model_source,
                                   [{"state": state, "action": action}])[0]
            applied += 1
            if not isinstance(pred, dict):
                stopped = True
                break
            state = pred
            if str(state.get("status", "playing")) != "playing":
                break
        if stopped:
            break
    return sign * (_metric(state, primary_metric) - initial)


def fitness(
    model_source: str,
    start_states: list[dict],
    actions: list[str],
    *,
    weights: dict,
    primary_metric: str,
    higher_is_better: bool,
    steps: int = 25,
    depth: int = 3,
    beam: int = 8,
) -> float:
    """Mean simulated-rollout score over the start states."""
    if not start_states:
        return 0.0
    total = sum(
        simulate_rollout(model_source, s, actions, weights=weights,
                         primary_metric=primary_metric,
                         higher_is_better=higher_is_better,
                         steps=steps, depth=depth, beam=beam)
        for s in start_states)
    return total / len(start_states)


def tune_game(
    game: str,
    *,
    iterations: int = 40,
    rollouts: int = 8,
    steps: int = 25,
    depth: int = 3,
    beam: int = 8,
    popsize: int = 6,
    sigma0: float = 0.5,
    seed: int = 7,
    max_evals: int | None = None,
    profile_path: Path | str | None = None,
    data_dir: Path | str = "data",
    worldmodels_dir: Path | str = "worldmodels",
) -> dict:
    """CMA-ES over the 5 planner weights, maximizing simulated fitness.

    Requires an INDUCED model (same honesty gate as PlannerProvider). Writes
    worldmodels/<game>/planner_weights.json and returns its contents. The
    shipped weights are the best EVALUATED candidate including the default —
    tuning never ships weights worse (in sim) than the baseline.
    """
    import cma  # local import: only the tune path needs the dependency

    profile = GameProfile.load(profile_path or Path("profiles") / f"{game}.json")
    model_dir = Path(worldmodels_dir) / game
    report_path = model_dir / "report.json"
    if not report_path.exists():
        raise SystemExit(f"no world model for {game}; run: "
                         f"python -m ludus.cli induce {game}")
    report = json.loads(report_path.read_text())
    if report.get("status") != "INDUCED":
        raise SystemExit(
            f"world model for {game} is not INDUCED "
            f"(status={report.get('status')!r}); refusing to tune against an "
            f"unvalidated model")
    model_source = (model_dir / "model.py").read_text()

    transitions = TransitionStore(
        Path(data_dir) / game / "transitions.jsonl").load()
    if not transitions:
        raise SystemExit(f"no transitions for {game}; run: "
                         f"python -m ludus.cli explore {game}")
    rng = random.Random(seed)
    sampled = (rng.sample(transitions, rollouts)
               if len(transitions) > rollouts else list(transitions))
    start_states = [t.before for t in sampled]
    actions = sorted(profile.controls)

    def evaluate(weights: dict) -> float:
        return fitness(model_source, start_states, actions, weights=weights,
                       primary_metric=profile.primary_metric,
                       higher_is_better=profile.higher_is_better,
                       steps=steps, depth=depth, beam=beam)

    baseline = evaluate(dict(DEFAULT_WEIGHTS))
    best_weights, best_fitness = dict(DEFAULT_WEIGHTS), baseline
    log.info("tune %s: baseline sim fitness %.3f over %d start states",
             game, baseline, len(start_states))

    es = cma.CMAEvolutionStrategy(
        _to_vector(DEFAULT_WEIGHTS), sigma0,
        {"popsize": popsize, "seed": seed, "verbose": -9})
    evals = 0
    generations = 0
    while generations < iterations and not es.stop():
        if max_evals is not None and evals >= max_evals:
            break
        solutions = es.ask()
        values = []
        for x in solutions:
            f = evaluate(_to_weights(x))
            evals += 1
            values.append(-f)  # cma minimizes
            if f > best_fitness:
                best_fitness, best_weights = f, _to_weights(x)
        es.tell(solutions, values)
        generations += 1
        log.info("tune %s: gen %d best sim fitness %.3f (%d evals)",
                 game, generations, best_fitness, evals)

    out = {"weights": best_weights,
           "sim_fitness": best_fitness,
           "baseline_sim_fitness": baseline,
           "iterations": generations}
    dest = model_dir / "planner_weights.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    log.info("tune %s: wrote %s (sim %.3f vs baseline %.3f)",
             game, dest, best_fitness, baseline)
    return out
