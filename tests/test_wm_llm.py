"""chat() seam — payload construction + response parsing, no network."""

import httpx
import pytest

from ludus.worldmodel import llm


def test_anthropic_payload_and_parse(monkeypatch):
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured.update(url=url, headers=headers, json=json)
        return httpx.Response(200, json={"content": [{"text": "SOURCE"}]},
                              request=httpx.Request("POST", url))

    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setattr(llm.httpx, "post", fake_post)
    out = llm.chat("sys", "user text", backend="anthropic")
    assert out == "SOURCE"
    assert captured["url"].endswith("/v1/messages")
    assert captured["json"]["system"] == "sys"
    assert captured["json"]["messages"][0]["content"] == "user text"


def test_gateway_payload_and_parse(monkeypatch):
    def fake_post(url, headers=None, json=None, timeout=None):
        assert url.endswith("/chat/completion")
        assert json["messages"][0]["role"] == "system"
        return httpx.Response(200, json={"text": "GW"},
                              request=httpx.Request("POST", url))

    monkeypatch.setenv("INSFORGE_GATEWAY_URL", "https://x/api/ai")
    monkeypatch.setenv("INSFORGE_API_KEY", "k")
    monkeypatch.setenv("INSFORGE_VISION_MODEL", "m")
    monkeypatch.setattr(llm.httpx, "post", fake_post)
    assert llm.chat("sys", "u", backend="gateway") == "GW"


def test_backend_default_prefers_anthropic_when_key_present(monkeypatch):
    monkeypatch.delenv("LUDUS_SYNTH_BACKEND", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    assert llm.default_backend() == "anthropic"
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    monkeypatch.setenv("INSFORGE_GATEWAY_URL", "https://x")
    assert llm.default_backend() == "gateway"


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        llm.chat("s", "u", backend="nope")
