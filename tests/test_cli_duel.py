# tests/test_cli_duel.py
"""Duel harness — planner vs a baseline provider, same game, same budget."""

import json

from ludus.cli import cmd_duel
from ludus.providers.mock import MockProvider

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


def _setup(tmp_path):
    from ludus.cli import cmd_onboard
    cmd_onboard(game="synthetic:grid", out_dir=tmp_path / "profiles")
    d = tmp_path / "wm" / "synthetic:grid"
    d.mkdir(parents=True)
    (d / "model.py").write_text(GRID_MODEL)
    (d / "report.json").write_text(json.dumps({"status": "INDUCED", "overall": 1.0}))
    return tmp_path


def test_duel_planner_beats_mock_on_grid(tmp_path, capsys):
    root = _setup(tmp_path)
    result = cmd_duel(
        game="synthetic:grid",
        profile_path=root / "profiles" / "synthetic:grid.json",
        worldmodels_dir=root / "wm", steps=6,
        baseline_provider=MockProvider(),
        runs_dir=root / "runs",
    )
    assert result["planner"]["score"] >= 10.0
    assert result["baseline"]["score"] <= result["planner"]["score"]
    assert result["winner"] == "planner"
    out = capsys.readouterr().out
    assert "DUEL" in out and "planner" in out and "winner" in out


def test_duel_result_is_persisted(tmp_path):
    root = _setup(tmp_path)
    result = cmd_duel(
        game="synthetic:grid",
        profile_path=root / "profiles" / "synthetic:grid.json",
        worldmodels_dir=root / "wm", steps=4,
        baseline_provider=MockProvider(),
        runs_dir=root / "runs",
    )
    saved = json.loads((root / "runs" / "duel-synthetic:grid.json").read_text())
    assert saved == result
