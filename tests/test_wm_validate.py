"""Field-weighted validation of induced models against real transitions."""

from ludus.worldmodel.transitions import Transition
from ludus.worldmodel.validate import validate_model

PERFECT = '''
import copy

def predict(state, action):
    out = copy.deepcopy(state)
    if action == "inc":
        out["metrics"]["score"] += 1
    return out
'''

WRONG_SCORE = '''
import copy

def predict(state, action):
    return copy.deepcopy(state)   # never updates score
'''


def _transitions(n=4):
    return [Transition(
        game_id="g", episode_id=f"ep{i}", step=0, action="inc", key="x",
        before={"status": "playing", "metrics": {"score": i},
                "game_state": {"player": {"x": 1}}},
        after={"status": "playing", "metrics": {"score": i + 1},
               "game_state": {"player": {"x": 1}}},
    ) for i in range(n)]


def test_perfect_model_scores_one_and_passes():
    report = validate_model(PERFECT, _transitions(),
                            important_paths=["metrics.score"], threshold=0.9)
    assert report.overall == 1.0
    assert report.passed is True
    assert report.per_field["metrics.score"] == 1.0
    assert report.n_cases == 4
    assert report.counterexamples == []


def test_wrong_model_fails_with_counterexamples():
    report = validate_model(WRONG_SCORE, _transitions(),
                            important_paths=["metrics.score"], threshold=0.9)
    assert report.passed is False
    assert report.per_field["metrics.score"] == 0.0
    assert report.per_field["game_state.player.x"] == 1.0   # untouched field fine
    ce = report.counterexamples[0]
    assert ce["field"] == "metrics.score"
    assert ce["predicted"] != ce["actual"]


def test_important_paths_weigh_more():
    # score wrong (weight 3), player.x + status right (weight 1 each):
    # weighted accuracy = 2/5, NOT the unweighted 2/3.
    report = validate_model(WRONG_SCORE, _transitions(),
                            important_paths=["metrics.score"], threshold=0.9)
    assert abs(report.overall - 2 / 5) < 1e-9


def test_float_tolerance():
    src = '''
import copy

def predict(state, action):
    out = copy.deepcopy(state)
    out["metrics"]["score"] = state["metrics"]["score"] + 1 + 1e-9
    return out
'''
    report = validate_model(src, _transitions(),
                            important_paths=["metrics.score"], threshold=0.9)
    assert report.per_field["metrics.score"] == 1.0


def test_crashed_prediction_counts_all_fields_wrong():
    src = 'def predict(state, action):\n    raise RuntimeError("boom")\n'
    report = validate_model(src, _transitions(),
                            important_paths=["metrics.score"], threshold=0.9)
    assert report.overall == 0.0
    assert report.passed is False


def test_wall_clock_fields_are_not_graded():
    src = '''
import copy

def predict(state, action):
    out = copy.deepcopy(state)
    if action == "inc":
        out["metrics"]["score"] += 1
    return out
'''
    ts = [Transition(
        game_id="g", episode_id=f"e{i}", step=0, action="inc", key="x",
        before={"metrics": {"score": i}, "timestampMs": 1000 + i},
        after={"metrics": {"score": i + 1}, "timestampMs": 2000 + i},
    ) for i in range(3)]
    report = validate_model(src, ts, important_paths=["metrics.score"],
                            threshold=0.9)
    assert "timestampMs" not in report.per_field
    assert report.overall == 1.0
    assert report.passed


def test_dual_gate_passes_on_reward_fidelity_despite_stochastic_noise():
    # perfect on score, wrong on a noisy field: primary >= threshold AND
    # overall >= floor -> passed (the measured-2048 situation)
    src = '''
import copy

def predict(state, action):
    out = copy.deepcopy(state)
    if action == "inc":
        out["metrics"]["score"] += 1
    return out    # never predicts the noisy field
'''
    ts = [Transition(
        game_id="g", episode_id=f"e{i}", step=0, action="inc", key="x",
        before={"metrics": {"score": i}, "game_state": {"noise": i * 7 % 3}},
        after={"metrics": {"score": i + 1}, "game_state": {"noise": (i + 1) * 5 % 3}},
    ) for i in range(4)]
    report = validate_model(src, ts, important_paths=["metrics.score"],
                            threshold=0.9, primary_metric="score")
    assert report.primary_accuracy == 1.0
    assert report.overall < 1.0
    assert report.passed is True


def test_dual_gate_fails_when_reward_wrong_even_if_overall_high():
    src = '''
import copy

def predict(state, action):
    return copy.deepcopy(state)   # everything right EXCEPT the score delta
'''
    ts = [Transition(
        game_id="g", episode_id=f"e{i}", step=0, action="inc", key="x",
        before={"metrics": {"score": i}, "game_state": {"a": 1, "b": 2, "c": 3}},
        after={"metrics": {"score": i + 1}, "game_state": {"a": 1, "b": 2, "c": 3}},
    ) for i in range(4)]
    report = validate_model(src, ts, important_paths=["metrics.score"],
                            threshold=0.9, primary_metric="score")
    assert report.primary_accuracy == 0.0
    assert report.passed is False


def test_dual_gate_fails_below_overall_floor():
    src = '''
def predict(state, action):
    return {"metrics": {"score": state["metrics"]["score"] + 1}}   # score only
'''
    ts = [Transition(
        game_id="g", episode_id=f"e{i}", step=0, action="inc", key="x",
        before={"metrics": {"score": i},
                "game_state": {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6,
                               "g": 7, "h": 8, "i": 9, "j": 10}},
        after={"metrics": {"score": i + 1},
               "game_state": {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6,
                              "g": 7, "h": 8, "i": 9, "j": 10}},
    ) for i in range(4)]
    report = validate_model(src, ts, important_paths=["metrics.score"],
                            threshold=0.9, primary_metric="score")
    assert report.primary_accuracy == 1.0
    assert report.overall < 0.75
    assert report.passed is False


def test_legacy_single_threshold_without_primary_metric():
    # no primary_metric kwarg -> old semantics (existing callers/tests)
    src = 'import copy\n\ndef predict(state, action):\n    return copy.deepcopy(state)\n'
    ts = [Transition(
        game_id="g", episode_id="e0", step=0, action="a", key="k",
        before={"metrics": {"v": 1}}, after={"metrics": {"v": 1}},
    )]
    report = validate_model(src, ts, important_paths=[], threshold=0.9)
    assert report.passed is True
