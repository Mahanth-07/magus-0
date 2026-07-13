"""TDD tests for TogetherProvider — written BEFORE the implementation.

Tests verify:
  1. build_provider("together") constructs a TogetherProvider
  2. available() gates on TOGETHER_API_KEY + TOGETHER_MODEL env vars
  3. Payload shape: system=plain string, user=content array [text, image_url(JPEG)]
  4. Image downscaling uses _encode_image from scripts/export_together exactly
  5. parse_decision wiring: OpenAI-shape response -> Decision
"""
import io
import json as _json

import pytest
from PIL import Image

from ludus.providers import build_provider
from ludus.providers.together import TogetherProvider
from ludus.providers.insforge import build_user_text, _SYSTEM, parse_decision
from ludus.schemas import PlannerContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DECISION_JSON = (
    '{"scene_summary":"s","controlled_entity":"the active tetromino",'
    '"current_subgoal":"g","action":"left","actions":["left","drop"],'
    '"action_args":{"duration_ms":120},"expected_result":"e","reason":"r",'
    '"confidence":1.0}'
)


def _make_png(width=800, height=600) -> bytes:
    """Create a real PNG so _encode_image (which calls PIL.Image.open) succeeds."""
    img = Image.new("RGB", (width, height), color=(128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


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

    monkeypatch.setattr("ludus.providers.together.httpx.post", _fake_post)
    return captured


# ---------------------------------------------------------------------------
# Test 1: build_provider wiring
# ---------------------------------------------------------------------------

def test_build_provider_constructs_together():
    p = build_provider("together")
    assert isinstance(p, TogetherProvider)
    assert p.name == "together"


# ---------------------------------------------------------------------------
# Test 2: available() gates on env vars
# ---------------------------------------------------------------------------

def test_available_requires_key_and_model(monkeypatch):
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    monkeypatch.delenv("TOGETHER_MODEL", raising=False)
    assert TogetherProvider().available() is False

    monkeypatch.setenv("TOGETHER_API_KEY", "k")
    assert TogetherProvider().available() is False  # model still missing

    monkeypatch.setenv("TOGETHER_MODEL", "mahanth1112_3532/Qwen3-VL-8B-Instruct-magus-student-v1-4d2347c5")
    assert TogetherProvider().available() is True


# ---------------------------------------------------------------------------
# Test 3: payload shape — system as plain string, user as content array w/ JPEG
# ---------------------------------------------------------------------------

def test_together_payload_shape_system_string_user_array(monkeypatch):
    """Together chat API: system is a plain string, user is a content array
    containing text + image_url (JPEG data-URI, matching training format)."""
    monkeypatch.setenv("TOGETHER_API_KEY", "test-key")
    monkeypatch.setenv("TOGETHER_MODEL", "test/model")
    captured = _capture_post(monkeypatch)

    ctx = PlannerContext(
        objective="win", legal_actions=["left", "right", "rotate", "drop"],
        state_text="col heights: 0 0 0", screenshot_png=_make_png(),
    )
    d = TogetherProvider().decide(ctx)

    # Response round-trips through parse_decision
    assert d.action == "left"
    assert d.actions == ["left", "drop"]
    assert d.confidence == 1.0

    # Auth / endpoint
    assert captured["url"] == "https://api.together.xyz/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["json"]["model"] == "test/model"
    assert captured["json"]["temperature"] == 0

    msgs = captured["json"]["messages"]
    roles = [m["role"] for m in msgs]
    assert roles == ["system", "user"]

    # System: plain string (not a content array)
    sys_msg = msgs[0]
    assert isinstance(sys_msg["content"], str), (
        f"system content must be a plain string, got {type(sys_msg['content'])}"
    )
    assert sys_msg["content"] == _SYSTEM

    # User: content array [text, image_url]
    user_msg = msgs[1]
    assert isinstance(user_msg["content"], list), (
        "user content must be a list (content array)"
    )
    content = user_msg["content"]
    assert content[0]["type"] == "text"
    expected_text = build_user_text(
        objective=ctx.objective,
        legal_actions=ctx.legal_actions,
        state_text=ctx.state_text,
        recent_outcomes=ctx.recent_outcomes,
        partner_recent_actions=ctx.partner_recent_actions,
        learned_rules=ctx.learned_rules,
    )
    assert content[0]["text"] == expected_text

    assert content[1]["type"] == "image_url"
    url = content[1]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,"), (
        f"image must be JPEG data-URI (matching training), got prefix: {url[:30]}"
    )


# ---------------------------------------------------------------------------
# Test 4: image downscaling — _encode_image is invoked and produces JPEG <=512px
# ---------------------------------------------------------------------------

def test_image_is_downscaled_to_jpeg_via_encode_image(monkeypatch):
    """Provider must call _encode_image (or equivalent) that downscales to
    max 512px longest side and JPEG-encodes at q=80, matching training."""
    monkeypatch.setenv("TOGETHER_API_KEY", "test-key")
    monkeypatch.setenv("TOGETHER_MODEL", "test/model")

    encode_calls = []
    import scripts.export_together as et_module

    original = et_module._encode_image

    def _spy(image_bytes: bytes) -> str:
        encode_calls.append(image_bytes)
        return original(image_bytes)

    monkeypatch.setattr("ludus.providers.together._encode_image", _spy)
    _capture_post(monkeypatch)

    png_800x600 = _make_png(800, 600)
    ctx = PlannerContext(
        objective="win", legal_actions=["drop"],
        state_text="", screenshot_png=png_800x600,
    )
    TogetherProvider().decide(ctx)

    assert len(encode_calls) == 1, "_encode_image should be called exactly once per decide()"
    # Verify the data-URI produced has JPEG content (JPEG magic bytes: FF D8)
    result_uri = et_module._encode_image(encode_calls[0])
    import base64
    jpeg_bytes = base64.b64decode(result_uri.split(",", 1)[1])
    assert jpeg_bytes[:2] == b"\xff\xd8", "Output must be JPEG (FF D8 magic)"

    # Verify downscaling: 800x600 -> longest side <=512
    decoded_img = Image.open(io.BytesIO(jpeg_bytes))
    assert max(decoded_img.size) <= 512, (
        f"Image not downscaled: got {decoded_img.size}, expected max dim <=512"
    )


# ---------------------------------------------------------------------------
# Test 5: base URL override via TOGETHER_BASE_URL
# ---------------------------------------------------------------------------

def test_base_url_defaults_and_override(monkeypatch):
    monkeypatch.delenv("TOGETHER_BASE_URL", raising=False)
    assert TogetherProvider()._base == "https://api.together.xyz/v1"

    monkeypatch.setenv("TOGETHER_BASE_URL", "https://custom.test/v1")
    assert TogetherProvider()._base == "https://custom.test/v1"
