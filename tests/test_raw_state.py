# tests/test_raw_state.py
"""StateSource seam: every playable client exposes raw_state() -> dict."""

from ludus.gameworld_client import FakeGameWorld, GameWorldClient
from ludus.state_source import StateSource
from ludus.synthetic import CounterGame, GridWorldGame


def test_synthetic_and_fake_satisfy_statesource_protocol():
    for obj in (GridWorldGame(), CounterGame(), FakeGameWorld([{"score": 1.0}])):
        assert isinstance(obj, StateSource)


def test_fake_gameworld_raw_state_reflects_current_metrics():
    fake = FakeGameWorld([{"score": 3.0}, {"score": 7.0}])
    st = fake.raw_state()
    assert st["status"] == "playing"
    assert st["metrics"]["score"] == 3.0
    fake.apply({"key": "x", "duration_ms": 10})
    assert fake.raw_state()["metrics"]["score"] == 7.0


def test_gameworld_client_raw_state_delegates_to_manager(monkeypatch):
    # No browser: patch the async bridge and assert delegation + None-guard.
    import asyncio

    client = GameWorldClient.__new__(GameWorldClient)  # skip __init__ (no threads)
    monkeypatch.setattr(client, "_ensure_started", lambda: None, raising=False)

    class _Mgr:
        async def get_game_state(self):
            return {"status": "playing", "metrics": {"score": 42}}

    client._mgr = _Mgr()
    monkeypatch.setattr(client, "_submit", lambda coro: asyncio.run(coro), raising=False)
    assert client.raw_state()["metrics"]["score"] == 42
