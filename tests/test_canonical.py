# tests/test_canonical.py
"""Canonical ordering: set-valued entity lists must compare order-free."""

from ludus.worldmodel.canonical import canonicalize
from ludus.worldmodel.transitions import Transition
from ludus.worldmodel.validate import validate_model


def test_lists_of_dicts_sort_by_content():
    a = {"entities": [{"x": 3, "y": 1}, {"x": 0, "y": 0}]}
    b = {"entities": [{"x": 0, "y": 0}, {"x": 3, "y": 1}]}
    assert canonicalize(a) == canonicalize(b)


def test_scalar_lists_and_nesting_untouched():
    s = {"board": [[0, 2], [4, 0]], "queue": ["i", "o"]}
    assert canonicalize(s) == s


def test_perfect_prediction_in_shuffled_order_scores_one():
    # model returns the right tiles, reversed order — must be 1.0 not ~0
    src = '''
import copy

def predict(state, action):
    out = copy.deepcopy(state)
    out["game_state"]["entities"] = list(reversed(out["game_state"]["entities"]))
    return out
'''
    ts = [Transition(
        game_id="g", episode_id=f"e{i}", step=0, action="noop", key="n",
        before={"metrics": {"score": 0},
                "game_state": {"entities": [{"x": 0, "y": 0}, {"x": 3, "y": 1}]}},
        after={"metrics": {"score": 0},
               "game_state": {"entities": [{"x": 0, "y": 0}, {"x": 3, "y": 1}]}},
    ) for i in range(3)]
    report = validate_model(src, ts, important_paths=["metrics.score"],
                            threshold=0.9)
    assert report.overall == 1.0
    assert report.passed
