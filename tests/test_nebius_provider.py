import os

from ludus.providers import build_provider
from ludus.providers.nebius import NebiusProvider
from ludus.providers.insforge import build_prompt, parse_decision
from ludus.schemas import PlannerContext


def test_build_provider_constructs_nebius():
    p = build_provider("nebius")
    assert isinstance(p, NebiusProvider)
    assert p.name == "nebius"


def test_available_requires_key_and_model(monkeypatch):
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    monkeypatch.delenv("NEBIUS_MODEL", raising=False)
    assert NebiusProvider().available() is False

    monkeypatch.setenv("NEBIUS_API_KEY", "k")
    assert NebiusProvider().available() is False  # model still missing

    monkeypatch.setenv("NEBIUS_MODEL", "some/model")
    assert NebiusProvider().available() is True


def test_default_base_url(monkeypatch):
    monkeypatch.delenv("NEBIUS_BASE_URL", raising=False)
    assert NebiusProvider()._base == "https://api.studio.nebius.com/v1"


def test_nebius_decide_parses_openai_choices_shape(monkeypatch):
    """No live call: stub httpx.post and assert the OpenAI-style response body is
    routed through parse_decision into a Decision (same contract as runtime)."""
    decision_json = (
        '{"scene_summary":"s","controlled_entity":"the active tetromino",'
        '"current_subgoal":"g","action":"left","actions":["left","drop"],'
        '"action_args":{"duration_ms":120},"expected_result":"e","reason":"r",'
        '"confidence":1.0}'
    )

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": decision_json}}]}

    captured = {}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return _Resp()

    monkeypatch.setenv("NEBIUS_API_KEY", "test-key")
    monkeypatch.setenv("NEBIUS_MODEL", "test/model")
    monkeypatch.setattr("ludus.providers.nebius.httpx.post", _fake_post)

    ctx = PlannerContext(
        objective="win", legal_actions=["left", "right", "rotate", "drop"],
        state_text="col heights: 0 0 0", screenshot_png=b"\x89PNG fake",
    )
    d = NebiusProvider().decide(ctx)

    assert d.action == "left"
    assert d.actions == ["left", "drop"]
    assert d.confidence == 1.0

    # Request shape: OpenAI vision chat — auth header, model, temp 0, text + image_url.
    assert captured["url"].endswith("/chat/completions")
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["json"]["model"] == "test/model"
    assert captured["json"]["temperature"] == 0
    content = captured["json"]["messages"][0]["content"]
    assert content[0]["type"] == "text" and build_prompt(ctx)[:20] in content[0]["text"]
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
