# tests/test_planner_provider.py
"""PlannerProvider — the induced model driving the real loop."""

import json

import pytest

from ludus.onboarding.profile import GameProfile
from ludus.planning.provider import PlannerProvider
from ludus.schemas import PlannerContext

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
        i = GOAL_CYCLE.index((goal["x"], goal["y"]))
        nx, ny = GOAL_CYCLE[(i + 1) % len(GOAL_CYCLE)]
        goal["x"], goal["y"] = nx, ny
    return out
'''


def _grid_profile() -> GameProfile:
    return GameProfile(
        game_id="synthetic:grid",
        controls={"move_left": "ArrowLeft", "move_right": "ArrowRight",
                  "move_up": "ArrowUp", "move_down": "ArrowDown"},
        control_effects={}, metrics=["score", "steps"], primary_metric="score",
        higher_is_better=True, timing_ms=10, objective="test", state_schema={},
    )


def _model_dir(tmp_path, status="INDUCED"):
    d = tmp_path / "synthetic:grid"
    d.mkdir(parents=True)
    (d / "model.py").write_text(GRID_MODEL)
    (d / "report.json").write_text(json.dumps({"status": status, "overall": 1.0}))
    return tmp_path


def _ctx(raw_state):
    return PlannerContext(objective="o", legal_actions=["move_right"],
                          screenshot_png=b"x", raw_state=raw_state)


def test_provider_refuses_non_induced_model(tmp_path):
    root = _model_dir(tmp_path, status="FAILED")
    with pytest.raises(SystemExit, match="not INDUCED"):
        PlannerProvider(_grid_profile(), worldmodels_dir=root)


def test_provider_plans_toward_goal(tmp_path):
    root = _model_dir(tmp_path)
    provider = PlannerProvider(_grid_profile(), worldmodels_dir=root, depth=3)
    state = {"status": "playing", "metrics": {"score": 0, "steps": 0},
             "game_state": {"player": {"x": 2, "y": 2},
                            "environment": {"goal": {"x": 3, "y": 2}}}}
    decision = provider.decide(_ctx(state))
    assert decision.actions == ["move_right"]
    assert decision.action == "move_right"
    assert "predicted score" in decision.reason


def test_provider_without_raw_state_falls_back_to_first_action(tmp_path):
    root = _model_dir(tmp_path)
    provider = PlannerProvider(_grid_profile(), worldmodels_dir=root)
    decision = provider.decide(_ctx(None))
    assert decision.actions == [sorted(_grid_profile().controls)[0]]


def test_provider_end_to_end_scores_in_real_episode(tmp_path):
    """The M3 offline acceptance: planner over an induced model beats random.
    From spawn (2,2) with goal (3,2), 5 steps must collect >= 10 score."""
    from ludus.adapters.generic import GenericAdapter
    from ludus.loop import run_episode
    from ludus.persistence.local import LocalStore
    from ludus.rulebook import Rulebook
    from ludus.synthetic import GridWorldGame

    root = _model_dir(tmp_path)
    profile = _grid_profile()
    provider = PlannerProvider(profile, worldmodels_dir=root, depth=3)
    result = run_episode(
        adapter=GenericAdapter(profile.to_game_config()), provider=provider,
        gameworld=GridWorldGame(), store=LocalStore(root=tmp_path / "runs"),
        rulebook=Rulebook(), mode="baseline", max_steps=5,
        episode_id="planner-e2e")
    assert result.final_metrics["score"] >= 10.0
