"""Beam search over a sandboxed induced model."""

from ludus.planning.planner import DEFAULT_WEIGHTS, rank_macros

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


def test_zero_reward_ties_prefer_state_changing_macros():
    # jammed-board scenario: "noop" leaves the state identical, "shift"
    # moves an entity but scores nothing. The planner must prefer shift —
    # frozen no-ops caused the live 2048 deadlock (score 4 vs VLM 72).
    model = '''
import copy

def predict(state, action):
    out = copy.deepcopy(state)
    if action == "shift":
        out["game_state"]["player"]["x"] += 1
    return out
'''
    state = {"status": "playing", "metrics": {"score": 0, "steps": 0},
             "game_state": {"player": {"x": 0, "y": 0}}}
    ranked = rank_macros(model, state, ["noop", "shift"],
                         primary_metric="score", higher_is_better=True,
                         depth=2, beam=4, top_k=4)
    assert ranked[0]["actions"][0] == "shift"
    assert ranked[0]["changes_state"] is True


DEADLOCK_MODEL = '''
import copy

def predict(state, action):
    out = copy.deepcopy(state)
    if action == "shift":
        out["game_state"]["player"]["x"] += 1
    return out
'''

DEADLOCK_STATE = {"status": "playing", "metrics": {"score": 0, "steps": 0},
                  "game_state": {"player": {"x": 0, "y": 0}}}


def test_default_weights_match_no_weights_exactly():
    """weights=None and the explicit default dict must produce IDENTICAL
    rankings — the regression gate for the CMA-ES parameterization."""
    kwargs = dict(primary_metric="score", higher_is_better=True,
                  depth=3, beam=16, top_k=6)
    baseline = rank_macros(GRID_MODEL, _state(2, 2, 4, 3), ACTIONS, **kwargs)
    weighted = rank_macros(GRID_MODEL, _state(2, 2, 4, 3), ACTIONS,
                           weights=dict(DEFAULT_WEIGHTS), **kwargs)
    assert weighted == baseline


def test_default_weights_constant_is_current_behavior():
    assert DEFAULT_WEIGHTS == {"primary": 1.0, "change": 0.0, "terminal": 0.0,
                               "secondary": 0.0, "length": 0.0}


def test_change_weight_steers_zero_reward_ties():
    # deadlock scenario: all primary rewards 0. A positive change weight must
    # SCORE state-changing macros above no-ops (not just tie-break them).
    ranked = rank_macros(DEADLOCK_MODEL, DEADLOCK_STATE, ["noop", "shift"],
                         primary_metric="score", higher_is_better=True,
                         depth=2, beam=4, top_k=6,
                         weights={"primary": 1.0, "change": 1.0})
    assert ranked[0]["actions"] == ["shift"]
    assert ranked[0]["changes_state"] is True
    # a pure no-op macro must rank below every shift-containing macro
    noop_ranks = [i for i, r in enumerate(ranked)
                  if all(a == "noop" for a in r["actions"])]
    shift_ranks = [i for i, r in enumerate(ranked)
                   if "shift" in r["actions"]]
    assert max(shift_ranks) < min(noop_ranks)


def test_terminal_weight_penalizes_dying():
    dying_model = '''
import copy

def predict(state, action):
    out = copy.deepcopy(state)
    if action == "move_right":
        out["metrics"]["score"] += 5
        out["status"] = "terminal"
    return out
'''
    # unweighted, move_right (+5 then dead) wins; terminal=-10 flips it
    ranked = rank_macros(dying_model, _state(2, 2, 3, 2),
                         ["move_right", "move_left"],
                         primary_metric="score", higher_is_better=True,
                         depth=2, beam=4, top_k=4,
                         weights={"primary": 1.0, "terminal": -10.0})
    assert ranked[0]["actions"][0] == "move_left"


def test_secondary_weight_scores_non_primary_metrics():
    coin_model = '''
import copy

def predict(state, action):
    out = copy.deepcopy(state)
    if action == "collect":
        out["metrics"]["coins"] += 1
    return out
'''
    state = {"status": "playing", "metrics": {"score": 0, "coins": 0},
             "game_state": {"player": {"x": 0}}}
    ranked = rank_macros(coin_model, state, ["collect", "noop"],
                         primary_metric="score", higher_is_better=True,
                         depth=2, beam=4, top_k=4,
                         weights={"primary": 1.0, "secondary": 1.0})
    assert ranked[0]["actions"] == ["collect", "collect"]


def test_length_weight_prefers_longer_macros():
    # zero-reward everywhere: the default tie-break prefers SHORT macros; a
    # positive length weight must flip that.
    ranked = rank_macros(GRID_MODEL, _state(0, 0, 4, 4), ACTIONS,
                         primary_metric="score", higher_is_better=True,
                         depth=2, beam=8, top_k=4,
                         weights={"primary": 1.0, "length": 1.0})
    assert all(len(r["actions"]) == 2 for r in ranked)


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
