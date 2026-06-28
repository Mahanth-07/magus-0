# tests/test_outcome.py
from ludus.outcome import OutcomeDetector, Outcome


def test_higher_is_better_improvement():
    o = OutcomeDetector().detect(
        pre={"lines": 0.0}, post={"lines": 1.0},
        primary_metric="lines", higher_is_better=True,
    )
    assert o.primary_delta == 1.0
    assert o.improved is True


def test_lower_is_better_improvement():
    o = OutcomeDetector().detect(
        pre={"holes": 3.0}, post={"holes": 1.0},
        primary_metric="holes", higher_is_better=False,
    )
    assert o.primary_delta == -2.0
    assert o.improved is True


def test_no_improvement_when_flat():
    o = OutcomeDetector().detect(
        pre={"lines": 2.0}, post={"lines": 2.0},
        primary_metric="lines", higher_is_better=True,
    )
    assert o.improved is False


def test_delta_covers_all_keys():
    o = OutcomeDetector().detect(
        pre={"a": 1.0}, post={"b": 5.0},
        primary_metric="a", higher_is_better=True,
    )
    assert o.delta == {"a": -1.0, "b": 5.0}
