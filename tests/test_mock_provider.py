from ludus.schemas import PlannerContext, Decision
from ludus.providers.mock import MockProvider


def _ctx():
    return PlannerContext(objective="o", legal_actions=["left", "right", "drop"])


def test_mock_is_available():
    assert MockProvider().available() is True


def test_mock_returns_valid_decision_with_legal_action():
    d = MockProvider().decide(_ctx())
    assert isinstance(d, Decision)
    assert d.action in ["left", "right", "drop"]


def test_mock_follows_script_in_order():
    p = MockProvider(script=["right", "drop"])
    assert p.decide(_ctx()).action == "right"
    assert p.decide(_ctx()).action == "drop"
    assert p.decide(_ctx()).action == "right"  # wraps
