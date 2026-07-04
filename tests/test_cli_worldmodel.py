# tests/test_cli_worldmodel.py
"""CLI explore/induce paths — offline via synthetic games + scripted synthesizer."""

import json

from ludus.cli import cmd_explore, cmd_induce
from ludus.worldmodel.transitions import TransitionStore

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


def _onboard_grid(tmp_path):
    from ludus.cli import cmd_onboard
    cmd_onboard(game="synthetic:grid", out_dir=tmp_path / "profiles")
    return tmp_path / "profiles" / "synthetic:grid.json"


def test_cmd_explore_writes_transitions(tmp_path, capsys):
    profile_path = _onboard_grid(tmp_path)
    cmd_explore(game="synthetic:grid", profile_path=profile_path,
                data_dir=tmp_path / "data", episodes=2, steps=8, seed=4)
    ts = TransitionStore(tmp_path / "data" / "synthetic:grid" / "transitions.jsonl").load()
    assert len(ts) == 16
    assert "transitions" in capsys.readouterr().out


def test_cmd_induce_produces_artifacts(tmp_path, capsys):
    profile_path = _onboard_grid(tmp_path)
    cmd_explore(game="synthetic:grid", profile_path=profile_path,
                data_dir=tmp_path / "data", episodes=4, steps=15, seed=4)
    cmd_induce(game="synthetic:grid", profile_path=profile_path,
               data_dir=tmp_path / "data", out_dir=tmp_path / "wm",
               synthesizer=lambda s, u: f"```python\n{GRID_MODEL}\n```",
               max_iterations=2)
    report = json.loads((tmp_path / "wm" / "synthetic:grid" / "report.json").read_text())
    assert report["status"] == "INDUCED"
    out = capsys.readouterr().out
    assert "INDUCED" in out
