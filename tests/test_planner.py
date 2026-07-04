"""Beam search over a sandboxed induced model."""

from ludus.planning.planner import rank_macros

# Correct gridworld world model (same dynamics as ludus/synthetic.py).
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
            pass  # goal not in standard cycle (test-only positions); keep it
    return out
'''

ACTIONS = ["move_left", "move_right", "move_up", "move_down"]


def _state(px, py, gx, gy, score=0):
    return {"status": "playing",
            "metrics": {"score": score, "steps": 0},
            "game_state": {"player": {"x": px, "y": py},
                           "environment": {"goal": {"x": gx, "y": gy}}}}


def test_finds_adjacent_goal_in_one_move():
    ranked = rank_macros(GRID_MODEL, _state(2, 2, 3, 2), ACTIONS,
                         primary_metric="score", higher_is_better=True,
                         depth=3, beam=8, top_k=4)
    best = ranked[0]
    assert best["actions"] == ["move_right"]
    assert best["predicted_reward"] == 10.0


def test_finds_two_move_goal_and_prefers_shortest_path():
    # goal at (4,3) from (2,2): reachable in 3 (right,right,down in some order)
    ranked = rank_macros(GRID_MODEL, _state(2, 2, 4, 3), ACTIONS,
                         primary_metric="score", higher_is_better=True,
                         depth=3, beam=16, top_k=4)
    best = ranked[0]
    assert best["predicted_reward"] == 10.0
    assert len(best["actions"]) == 3
    assert sorted(best["actions"]) == ["move_down", "move_right", "move_right"]


def test_no_reward_in_horizon_still_returns_ranked_macros():
    # goal unreachable within depth 2 from (0,0) at (4,4): all rewards 0
    ranked = rank_macros(GRID_MODEL, _state(0, 0, 4, 4), ACTIONS,
                         primary_metric="score", higher_is_better=True,
                         depth=2, beam=8, top_k=4)
    assert len(ranked) == 4
    assert all(r["predicted_reward"] == 0.0 for r in ranked)
    assert all(1 <= len(r["actions"]) <= 2 for r in ranked)


def test_broken_model_returns_empty():
    ranked = rank_macros("def predict(s, a):\n    raise RuntimeError()\n",
                         _state(2, 2, 3, 2), ACTIONS,
                         primary_metric="score", higher_is_better=True,
                         depth=2, beam=4, top_k=4)
    assert ranked == []


def test_terminal_branches_are_not_expanded():
    dying_model = '''
import copy

def predict(state, action):
    out = copy.deepcopy(state)
    if action == "move_right":
        out["metrics"]["score"] += 5
        out["status"] = "terminal"
    return out
'''
    ranked = rank_macros(dying_model, _state(2, 2, 3, 2), ["move_right", "move_left"],
                         primary_metric="score", higher_is_better=True,
                         depth=3, beam=8, top_k=10)
    # move_right yields +5 then terminal: macros extending past it must not exist
    for r in ranked:
        if r["actions"][0] == "move_right":
            assert len(r["actions"]) == 1
