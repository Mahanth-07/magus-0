# tests/test_canonical.py
"""Canonical ordering: set-valued entity lists must compare order-free."""

from ludus.worldmodel.canonical import canonicalize
from ludus.worldmodel.transitions import Transition
from ludus.worldmodel.validate import validate_model


def test_split_entities_extracts_lists_of_dicts():
    from ludus.worldmodel.canonical import split_entities
    state = {"metrics": {"score": 4},
             "game_state": {"entities": [{"x": 1, "y": 0}, {"x": 0, "y": 0}]}}
    scalar, ents = split_entities(state)
    assert scalar["game_state"]["entities"] == 2          # count stand-in
    assert scalar["metrics"]["score"] == 4
    assert "game_state.entities" in ents
    assert len(ents["game_state.entities"]) == 2


def test_spawned_entity_costs_one_unit_not_index_catastrophe():
    # Perfect persistent prediction + inability to guess the 1 random spawn:
    # must score N/(N+1) on the entity list, NOT ~0 from index shifting.
    from ludus.worldmodel.transitions import Transition
    from ludus.worldmodel.validate import validate_model

    src = '''
import copy

def predict(state, action):
    out = copy.deepcopy(state)   # keeps all persistent entities, no spawn guess
    return out
'''
    before_ents = [{"type": "tile", "x": i, "y": 0, "props": {"value": 2}}
                   for i in range(5)]
    after_ents = before_ents + [{"type": "tile", "x": 2, "y": 3,
                                 "props": {"value": 4}}]   # the random spawn
    ts = [Transition(
        game_id="g", episode_id=f"e{i}", step=0, action="left", key="L",
        before={"metrics": {"score": 0}, "game_state": {"entities": before_ents}},
        after={"metrics": {"score": 0}, "game_state": {"entities": after_ents}},
    ) for i in range(3)]
    report = validate_model(src, ts, important_paths=["metrics.score"],
                            threshold=0.9)
    ent_acc = report.per_field["game_state.entities"]
    assert abs(ent_acc - 5 / 6) < 1e-9
    # counterexample names the missing entity
    missing = [c for c in report.counterexamples
               if c["field"] == "game_state.entities"]
    assert missing and "value" in str(missing[0]["actual"])


def test_entity_count_standin_is_scored_as_scalar():
    # model predicting wrong entity COUNT gets that scalar path wrong too
    from ludus.worldmodel.transitions import Transition
    from ludus.worldmodel.validate import validate_model
    src = '''
import copy

def predict(state, action):
    out = copy.deepcopy(state)
    out["game_state"]["entities"] = out["game_state"]["entities"][:1]
    return out
'''
    ts = [Transition(
        game_id="g", episode_id="e0", step=0, action="a", key="k",
        before={"metrics": {"score": 0},
                "game_state": {"entities": [{"x": 0}, {"x": 1}]}},
        after={"metrics": {"score": 0},
               "game_state": {"entities": [{"x": 0}, {"x": 1}]}},
    )]
    report = validate_model(src, ts, important_paths=["metrics.score"],
                            threshold=0.99)
    assert report.per_field["game_state.entities"] == 0.5
    assert report.passed is False


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
