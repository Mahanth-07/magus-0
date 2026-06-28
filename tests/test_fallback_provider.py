import pytest
from ludus.schemas import PlannerContext
from ludus.providers.mock import MockProvider
from ludus.providers.fallback import FallbackProvider


class _Unavailable:
    name = "down"
    def available(self): return False
    def decide(self, ctx): raise AssertionError("must not be called")


class _Raises:
    name = "boom"
    def available(self): return True
    def decide(self, ctx): raise RuntimeError("gateway 500")


def _ctx():
    return PlannerContext(objective="o", legal_actions=["a"])


def test_falls_through_unavailable_to_mock():
    p = FallbackProvider([_Unavailable(), MockProvider()])
    assert p.decide(_ctx()).action == "a"


def test_falls_through_raising_to_mock():
    p = FallbackProvider([_Raises(), MockProvider()])
    assert p.decide(_ctx()).action == "a"


def test_raises_when_all_fail():
    p = FallbackProvider([_Unavailable(), _Raises()])
    with pytest.raises(RuntimeError):
        p.decide(_ctx())


def test_warns_when_falling_back_to_mock(caplog):
    import logging
    p = FallbackProvider([_Unavailable(), MockProvider()])
    with caplog.at_level(logging.WARNING, logger="ludus.providers"):
        p.decide(_ctx())
    assert any("degraded to MockProvider" in r.message for r in caplog.records)
