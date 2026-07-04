"""Sandbox — induced code runs contained: AST allowlist + subprocess + timeout."""

from ludus.worldmodel.sandbox import check_source, run_predictions

GOOD = '''
import copy

def predict(state, action):
    out = copy.deepcopy(state)
    if action == "inc":
        out["metrics"]["score"] += 1
    return out
'''


def test_check_source_accepts_allowlisted_imports():
    assert check_source(GOOD) == []


def test_check_source_rejects_forbidden_import():
    bad = "import os\n\ndef predict(state, action):\n    return state\n"
    violations = check_source(bad)
    assert violations and "os" in violations[0]


def test_check_source_rejects_dangerous_names():
    bad = "def predict(state, action):\n    return eval('state')\n"
    assert check_source(bad)


def test_check_source_rejects_syntax_errors():
    assert check_source("def predict(state action):")


def test_run_predictions_happy_path():
    cases = [
        {"state": {"metrics": {"score": 0}}, "action": "inc"},
        {"state": {"metrics": {"score": 5}}, "action": "noop"},
    ]
    preds = run_predictions(GOOD, cases)
    assert preds[0]["metrics"]["score"] == 1
    assert preds[1]["metrics"]["score"] == 5


def test_run_predictions_per_case_error_is_none():
    src = '''
def predict(state, action):
    if action == "boom":
        raise ValueError("nope")
    return state
'''
    preds = run_predictions(src, [
        {"state": {"a": 1}, "action": "boom"},
        {"state": {"a": 1}, "action": "ok"},
    ])
    assert preds[0] is None
    assert preds[1] == {"a": 1}


def test_run_predictions_timeout_returns_all_none():
    src = '''
def predict(state, action):
    while True:
        pass
'''
    preds = run_predictions(src, [{"state": {}, "action": "x"}], timeout_s=2.0)
    assert preds == [None]


def test_run_predictions_missing_predict_returns_all_none():
    preds = run_predictions("x = 1\n", [{"state": {}, "action": "a"}])
    assert preds == [None]
