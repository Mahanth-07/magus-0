import os

from ludus.providers import build_provider
from ludus.providers.nebius import NebiusProvider
from ludus.providers.insforge import build_prompt, build_user_text, parse_decision, _SYSTEM
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


def _capture_post(monkeypatch):
    captured = {}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return _Resp()

    monkeypatch.setattr("ludus.providers.nebius.httpx.post", _fake_post)
    return captured


def test_text_only_mode_sends_no_image_url(monkeypatch):
    """NEBIUS_MULTIMODAL falsey -> text-only request: system + user STRING messages,
    NO image_url anywhere, parsed through the same parse_decision contract."""
    monkeypatch.setenv("NEBIUS_API_KEY", "test-key")
    monkeypatch.setenv("NEBIUS_MODEL", "ft:llama-3.2-3b")
    monkeypatch.setenv("NEBIUS_MULTIMODAL", "0")
    captured = _capture_post(monkeypatch)

    ctx = PlannerContext(
        objective="win", legal_actions=["left", "right", "rotate", "drop"],
        state_text="col heights: 0 0 0", screenshot_png=b"\x89PNG fake",
    )
    d = NebiusProvider().decide(ctx)
    assert d.action == "left" and d.actions == ["left", "drop"]

    msgs = captured["json"]["messages"]
    # system + user, both STRING content, no content arrays, no image_url at all.
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
    assert "image_url" not in json_dumps(captured["json"])
    assert captured["json"]["temperature"] == 0


def test_multimodal_default_when_env_unset_keeps_image(monkeypatch):
    """No NEBIUS_MULTIMODAL set -> default multimodal: image_url present."""
    monkeypatch.setenv("NEBIUS_API_KEY", "test-key")
    monkeypatch.setenv("NEBIUS_MODEL", "some/vlm")
    monkeypatch.delenv("NEBIUS_MULTIMODAL", raising=False)
    captured = _capture_post(monkeypatch)

    ctx = PlannerContext(
        objective="win", legal_actions=["left", "drop"],
        state_text="x", screenshot_png=b"\x89PNG fake",
    )
    NebiusProvider().decide(ctx)
    assert "image_url" in json_dumps(captured["json"])


def test_multimodal_truthy_string_keeps_image(monkeypatch):
    """NEBIUS_MULTIMODAL=1 -> multimodal path with image_url."""
    monkeypatch.setenv("NEBIUS_API_KEY", "test-key")
    monkeypatch.setenv("NEBIUS_MODEL", "some/vlm")
    monkeypatch.setenv("NEBIUS_MULTIMODAL", "1")
    captured = _capture_post(monkeypatch)
    ctx = PlannerContext(objective="w", legal_actions=["drop"], screenshot_png=b"png")
    NebiusProvider().decide(ctx)
    assert "image_url" in json_dumps(captured["json"])


def json_dumps(obj) -> str:
    import json as _json
    return _json.dumps(obj)
