import json as _json

from ludus.providers import build_provider
from ludus.providers.fireworks import FireworksProvider
from ludus.providers.insforge import build_user_text, parse_decision, _SYSTEM
from ludus.schemas import PlannerContext


_DECISION_JSON = (
    '{"scene_summary":"s","controlled_entity":"the active tetromino",'
    '"current_subgoal":"g","action":"left","actions":["left","drop"],'
    '"action_args":{"duration_ms":120},"expected_result":"e","reason":"r",'
    '"confidence":1.0}'
)


class _Resp:
    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"content": _DECISION_JSON}}]}


def _capture_post(monkeypatch):
    captured = {}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return _Resp()

    monkeypatch.setattr("ludus.providers.fireworks.httpx.post", _fake_post)
    return captured


def test_build_provider_constructs_fireworks():
    p = build_provider("fireworks")
    assert isinstance(p, FireworksProvider)
    assert p.name == "fireworks"


def test_available_requires_key_and_model(monkeypatch):
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    monkeypatch.delenv("FIREWORKS_MODEL", raising=False)
    assert FireworksProvider().available() is False

    monkeypatch.setenv("FIREWORKS_API_KEY", "k")
    assert FireworksProvider().available() is False  # model still missing

    monkeypatch.setenv("FIREWORKS_MODEL", "accounts/x/models/ludus-tetris")
    assert FireworksProvider().available() is True


def test_default_base_url(monkeypatch):
    monkeypatch.delenv("FIREWORKS_BASE_URL", raising=False)
    assert FireworksProvider()._base == "https://api.fireworks.ai/inference/v1"


def test_base_url_override(monkeypatch):
    monkeypatch.setenv("FIREWORKS_BASE_URL", "https://example.test/v1")
    assert FireworksProvider()._base == "https://example.test/v1"


def test_fireworks_decide_text_only_request_and_parse_roundtrip(monkeypatch):
    """No live call: stub httpx.post and assert the request is TEXT-only (system +
    user STRING messages, NO image_url anywhere), carries the right auth/model/temp,
    and the OpenAI-style response body parses into a Decision via parse_decision."""
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setenv("FIREWORKS_MODEL", "accounts/x/models/ludus-tetris")
    captured = _capture_post(monkeypatch)

    ctx = PlannerContext(
        objective="win", legal_actions=["left", "right", "rotate", "drop"],
        state_text="col heights: 0 0 0", screenshot_png=b"\x89PNG fake",
    )
    d = FireworksProvider().decide(ctx)

    # Parse round-trip: identical contract to every other provider.
    assert d.action == "left"
    assert d.actions == ["left", "drop"]
    assert d.confidence == 1.0
    assert d == parse_decision(_DECISION_JSON)

    # Request shape: OpenAI chat — auth header, model, temp 0, correct endpoint.
    assert captured["url"] == "https://api.fireworks.ai/inference/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["json"]["model"] == "accounts/x/models/ludus-tetris"
    assert captured["json"]["temperature"] == 0

    # Text-only: system + user, both STRING content, no content arrays, no image.
    msgs = captured["json"]["messages"]
    assert [m["role"] for m in msgs] == ["system", "user"]
    for m in msgs:
        assert isinstance(m["content"], str)
    assert msgs[0]["content"] == _SYSTEM
    assert msgs[1]["content"] == build_user_text(
        objective=ctx.objective,
        legal_actions=ctx.legal_actions,
        state_text=ctx.state_text,
        recent_outcomes=ctx.recent_outcomes,
        partner_recent_actions=ctx.partner_recent_actions,
        learned_rules=ctx.learned_rules,
    )
    assert "image_url" not in _json.dumps(captured["json"])
