# tests/test_tuning.py
"""CMA-ES teacher tuning: simulated rollouts inside the induced world model.

All offline — synthetic grid model, no network, no browser."""

import json

import pytest

from ludus.onboarding.profile import GameProfile
from ludus.planning.planner import DEFAULT_WEIGHTS
from ludus.planning.tuning import (fitness, planning_grade, simulate_rollout,
                                   tune_game)
from ludus.worldmodel.transitions import Transition, TransitionStore

GRID_MODEL = '''
import copy

MOVES = {"move_left": (-1, 0), "move_right": (1, 0),
         "move_up": (0, -1), "move_down": (0, 1)}
GOAL_CYCLE = [(3, 2), (0, 0), (4, 4), (0, 4), (4, 0)]

def predict(state, action):
    out = copy.deepcopy(state)
    p = out["game_state"]["player"]
    out["metrics"]["steps"] += 1
    dx, dy = MOVES.get(action, (0, 0))
    p["x"] = min(4, max(0, p["x"] + dx))
    p["y"] = min(4, max(0, p["y"] + dy))
    goal = out["game_state"]["environment"]["goal"]
    if (p["x"], p["y"]) == (goal["x"], goal["y"]):
        out["metrics"]["score"] += 10
        try:
            i = GOAL_CYCLE.index((goal["x"], goal["y"]))
            nx, ny = GOAL_CYCLE[(i + 1) % len(GOAL_CYCLE)]
            goal["x"], goal["y"] = nx, ny
        except ValueError:
            pass
    return out
'''

ACTIONS = ["move_left", "move_right", "move_up", "move_down"]

TERMINAL_MODEL = '''
import copy

def predict(state, action):
    out = copy.deepcopy(state)
    out["metrics"]["score"] += 5
    out["status"] = "over"
    return out
'''


def _state(px, py, gx, gy, score=0):
    return {"status": "playing",
            "metrics": {"score": score, "steps": 0},
            "game_state": {"player": {"x": px, "y": py},
                           "environment": {"goal": {"x": gx, "y": gy}}}}


def test_simulate_rollout_reaches_positive_score_from_spawn():
    delta = simulate_rollout(
        GRID_MODEL, _state(2, 2, 3, 2), ACTIONS,
        weights=dict(DEFAULT_WEIGHTS), primary_metric="score",
        higher_is_better=True, steps=6, depth=3, beam=8)
    assert delta >= 10.0


def test_simulate_rollout_returns_delta_not_absolute():
    # starting score 90 must not inflate the fitness
    delta = simulate_rollout(
        GRID_MODEL, _state(2, 2, 3, 2, score=90), ACTIONS,
        weights=dict(DEFAULT_WEIGHTS), primary_metric="score",
        higher_is_better=True, steps=1, depth=2, beam=4)
    assert delta == 10.0


def test_simulate_rollout_stops_on_terminal():
    delta = simulate_rollout(
        TERMINAL_MODEL, _state(0, 0, 4, 4), ACTIONS,
        weights=dict(DEFAULT_WEIGHTS), primary_metric="score",
        higher_is_better=True, steps=10, depth=2, beam=4)
    assert delta == 5.0  # one prediction, then status != playing -> stop


def test_simulate_rollout_broken_model_scores_zero():
    delta = simulate_rollout(
        "def predict(s, a):\n    raise RuntimeError()\n",
        _state(2, 2, 3, 2), ACTIONS,
        weights=dict(DEFAULT_WEIGHTS), primary_metric="score",
        higher_is_better=True, steps=5, depth=2, beam=4)
    assert delta == 0.0


def test_fitness_is_mean_over_start_states():
    starts = [_state(0, 0, 4, 4), _state(2, 2, 3, 2)]
    f = fitness(TERMINAL_MODEL, starts, ACTIONS,
                weights=dict(DEFAULT_WEIGHTS), primary_metric="score",
                higher_is_better=True, steps=3, depth=2, beam=4)
    assert f == 5.0  # every rollout terminates after one +5 step


COPY_MODEL = '''
import copy

def predict(state, action):
    return copy.deepcopy(state)
'''


def test_planning_grade_true_for_model_that_scores_forward():
    pg = planning_grade(
        GRID_MODEL, [_state(2, 2, 3, 2), _state(0, 0, 0, 4)], ACTIONS,
        primary_metric="score", higher_is_better=True,
        rollouts=2, steps=6, depth=2, beam=8)
    assert pg["planning_grade"] is True
    assert pg["scoring_rollouts_frac"] > 0
    assert pg["mean_rollout_score"] > 0


def test_planning_grade_false_for_copy_state_model():
    # predicts "nothing changes" perfectly — validation-grade could pass,
    # but no rollout can ever generate score: planning-grade must be False
    pg = planning_grade(
        COPY_MODEL, [_state(2, 2, 3, 2)], ACTIONS,
        primary_metric="score", higher_is_better=True,
        rollouts=1, steps=3, depth=2, beam=4)
    assert pg == {"mean_rollout_score": 0.0, "scoring_rollouts_frac": 0.0,
                  "planning_grade": False}


def test_planning_grade_empty_start_states_is_zeros():
    pg = planning_grade(
        GRID_MODEL, [], ACTIONS,
        primary_metric="score", higher_is_better=True)
    assert pg == {"mean_rollout_score": 0.0, "scoring_rollouts_frac": 0.0,
                  "planning_grade": False}


def test_planning_grade_broken_source_guards_to_zeros():
    pg = planning_grade(
        "this is not python (", [_state(2, 2, 3, 2)], ACTIONS,
        primary_metric="score", higher_is_better=True,
        rollouts=1, steps=2, depth=2, beam=4)
    assert pg == {"mean_rollout_score": 0.0, "scoring_rollouts_frac": 0.0,
                  "planning_grade": False}


def test_planning_grade_caps_rollouts_at_budget():
    starts = [_state(2, 2, 3, 2)] * 5
    pg = planning_grade(
        GRID_MODEL, starts, ACTIONS,
        primary_metric="score", higher_is_better=True,
        rollouts=2, steps=2, depth=2, beam=4)
    # only 2 of the 5 starts run; every run scores identically
    assert pg["scoring_rollouts_frac"] in (0.0, 1.0)


def _tuning_dirs(tmp_path, status="INDUCED", n_transitions=6):
    game = "synthetic:grid"
    wm = tmp_path / "worldmodels" / game
    wm.mkdir(parents=True)
    (wm / "model.py").write_text(GRID_MODEL)
    (wm / "report.json").write_text(json.dumps({"status": status,
                                                "overall": 1.0}))
    profile = GameProfile(
        game_id=game,
        controls={"move_left": "ArrowLeft", "move_right": "ArrowRight",
                  "move_up": "ArrowUp", "move_down": "ArrowDown"},
        control_effects={}, metrics=["score", "steps"],
        primary_metric="score", higher_is_better=True, timing_ms=10,
        objective="test", state_schema={})
    profile_path = tmp_path / "profiles" / f"{game}.json"
    profile.save(profile_path)
    store = TransitionStore(tmp_path / "data" / game / "transitions.jsonl")
    for i in range(n_transitions):
        before = _state(2, 2, 3, 2)
        after = _state(3, 2, 0, 0, score=10)
        store.append(Transition(game_id=game, episode_id=f"e{i}", step=0,
                                action="move_right", key="ArrowRight",
                                before=before, after=after))
    return tmp_path, profile_path


def test_tune_game_smoke_writes_weights_file(tmp_path):
    root, profile_path = _tuning_dirs(tmp_path)
    result = tune_game(
        "synthetic:grid", iterations=2, rollouts=2, steps=3,
        depth=2, beam=4, max_evals=10, seed=7,
        profile_path=profile_path,
        data_dir=root / "data", worldmodels_dir=root / "worldmodels")
    out_path = root / "worldmodels" / "synthetic:grid" / "planner_weights.json"
    assert out_path.exists()
    saved = json.loads(out_path.read_text())
    assert saved == result
    assert set(result["weights"]) == {"primary", "change", "terminal",
                                      "secondary", "length"}
    assert result["baseline_sim_fitness"] > 0
    # loose stochastic gate: tuned must not collapse below half the baseline
    assert result["sim_fitness"] >= result["baseline_sim_fitness"] * 0.5
    assert result["iterations"] >= 1


def test_cmd_tune_prints_baseline_vs_tuned(tmp_path, capsys):
    from ludus.cli import cmd_tune
    root, profile_path = _tuning_dirs(tmp_path)
    result = cmd_tune("synthetic:grid", iterations=1, rollouts=2, steps=3,
                      max_evals=6, profile_path=profile_path,
                      data_dir=root / "data",
                      worldmodels_dir=root / "worldmodels")
    out = capsys.readouterr().out
    assert "baseline" in out and "tuned" in out
    assert "planner_weights.json" in out
    assert result["iterations"] >= 1


def test_tune_game_refuses_non_induced_model(tmp_path):
    root, profile_path = _tuning_dirs(tmp_path, status="FAILED")
    with pytest.raises(SystemExit, match="not INDUCED"):
        tune_game("synthetic:grid", iterations=1, rollouts=1, steps=2,
                  max_evals=2, profile_path=profile_path,
                  data_dir=root / "data",
                  worldmodels_dir=root / "worldmodels")


def test_tune_game_refuses_missing_transitions(tmp_path):
    root, profile_path = _tuning_dirs(tmp_path, n_transitions=0)
    with pytest.raises(SystemExit, match="no transitions"):
        tune_game("synthetic:grid", iterations=1, rollouts=1, steps=2,
                  max_evals=2, profile_path=profile_path,
                  data_dir=root / "data",
                  worldmodels_dir=root / "worldmodels")
