"""Tests for scripts/export_together.py — Together VLM fine-tuning export.

Pure-unit tests: all image I/O is replaced with tiny synthetic PNG/JPEG bytes;
no network calls; no disk access beyond tmp_path fixtures.
"""

from __future__ import annotations

import base64
import io
import json
import random
from pathlib import Path

import pytest
from PIL import Image

from scripts.export_together import _encode_image, _split_paths, row_to_together, export, MAX_PX


# ---------------------------------------------------------------------------
# Helpers: generate minimal real PNG bytes for PIL to read
# ---------------------------------------------------------------------------

def _make_png_bytes(width: int = 64, height: int = 64, color: tuple = (128, 64, 200)) -> bytes:
    """Return raw PNG bytes for a solid-colour image of given size."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_sft_row(system: str = "sys", user: str = "usr", assistant: str = '{"action":"up"}') -> dict:
    """Minimal sft_v1 row dict (no image path — tests supply bytes directly)."""
    return {"system": system, "user": user, "assistant": assistant, "meta": {"game": "test"}}


# ---------------------------------------------------------------------------
# _encode_image tests
# ---------------------------------------------------------------------------

def test_encode_image_returns_jpeg_data_uri():
    png = _make_png_bytes(64, 64)
    uri = _encode_image(png)
    assert uri.startswith("data:image/jpeg;base64,")


def test_encode_image_downscales_large_image():
    """An image wider than MAX_PX must be shrunk so longest side <= MAX_PX."""
    big_png = _make_png_bytes(1024, 768)
    uri = _encode_image(big_png)
    # Decode and check dimensions
    b64_data = uri.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_data)
    img = Image.open(io.BytesIO(img_bytes))
    assert max(img.size) <= MAX_PX


def test_encode_image_small_image_stays_small():
    """An image that's already small should not be upscaled."""
    tiny_png = _make_png_bytes(32, 32)
    uri = _encode_image(tiny_png)
    b64_data = uri.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_data)
    img = Image.open(io.BytesIO(img_bytes))
    # Must remain at most 32x32
    assert max(img.size) <= 32


def test_encode_image_is_valid_base64():
    png = _make_png_bytes(64, 64)
    uri = _encode_image(png)
    b64_part = uri.split(",", 1)[1]
    # Should not raise
    decoded = base64.b64decode(b64_part)
    assert len(decoded) > 0


# ---------------------------------------------------------------------------
# row_to_together tests: message structure
# ---------------------------------------------------------------------------

def test_row_to_together_message_roles():
    row = _make_sft_row()
    png = _make_png_bytes()
    out = row_to_together(row, png)
    roles = [m["role"] for m in out["messages"]]
    assert roles == ["system", "user", "assistant"]


def test_row_to_together_system_content_is_text_block():
    row = _make_sft_row(system="system instruction")
    out = row_to_together(row, _make_png_bytes())
    sys_msg = out["messages"][0]
    assert sys_msg["role"] == "system"
    content = sys_msg["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[0]["text"] == "system instruction"


def test_row_to_together_user_has_text_then_image():
    row = _make_sft_row(user="what should I do?")
    out = row_to_together(row, _make_png_bytes())
    user_msg = out["messages"][1]
    assert user_msg["role"] == "user"
    content = user_msg["content"]
    # First block: text
    assert content[0]["type"] == "text"
    assert content[0]["text"] == "what should I do?"
    # Second block: image_url with base64 data-URI
    assert content[1]["type"] == "image_url"
    url = content[1]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")


def test_row_to_together_image_embedding_present_and_non_empty():
    row = _make_sft_row()
    out = row_to_together(row, _make_png_bytes(128, 128))
    url = out["messages"][1]["content"][1]["image_url"]["url"]
    b64 = url.split(",", 1)[1]
    decoded = base64.b64decode(b64)
    assert len(decoded) > 100  # actual JPEG content, not empty


def test_row_to_together_assistant_content_is_text_block():
    row = _make_sft_row(assistant='{"action":"left"}')
    out = row_to_together(row, _make_png_bytes())
    asst_msg = out["messages"][2]
    assert asst_msg["role"] == "assistant"
    content = asst_msg["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[0]["text"] == '{"action":"left"}'


def test_row_to_together_pure_function_no_io(tmp_path):
    """row_to_together must not perform any file I/O — give it bytes, get dict."""
    row = _make_sft_row()
    png = _make_png_bytes()
    # If this raises FileNotFoundError it means the function tried to open a file
    result = row_to_together(row, png)
    assert "messages" in result


# ---------------------------------------------------------------------------
# _split_paths tests
# ---------------------------------------------------------------------------

def test_split_paths_derive_correct_names():
    out = Path("/tmp/data/together_sft.jsonl")
    train, val = _split_paths(out)
    assert train.name == "together_train.jsonl"
    assert val.name == "together_val.jsonl"
    assert train.parent == out.parent
    assert val.parent == out.parent


# ---------------------------------------------------------------------------
# export() integration tests (uses tmp_path for I/O, synthetic PNG files)
# ---------------------------------------------------------------------------

def _write_fake_dataset(tmp_path: Path, n: int = 10, seed: int = 0) -> tuple[Path, Path]:
    """Write a fake sft_v1.jsonl with n rows and synthetic PNG images.
    Returns (jsonl_path, images_dir).
    """
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    jsonl = tmp_path / "sft_v1.jsonl"
    rng = random.Random(seed)
    with jsonl.open("w") as fh:
        for i in range(n):
            img_path = img_dir / f"step_{i:04d}.png"
            img_path.write_bytes(_make_png_bytes(rng.randint(32, 256), rng.randint(32, 256)))
            row = {
                "image": str(img_path),
                "system": "sys",
                "user": f"user text {i}",
                "assistant": json.dumps({"action": "up", "actions": ["up"]}),
                "meta": {"game": "test", "step_index": i},
            }
            fh.write(json.dumps(row) + "\n")
    return jsonl, img_dir


def test_export_writes_expected_files(tmp_path):
    in_path, _ = _write_fake_dataset(tmp_path, n=10)
    out_path = tmp_path / "out" / "together_sft.jsonl"
    export(in_path, out_path, val_frac=0.2, seed=13)
    assert out_path.exists()
    assert (tmp_path / "out" / "together_train.jsonl").exists()
    assert (tmp_path / "out" / "together_val.jsonl").exists()


def test_export_total_row_count(tmp_path):
    in_path, _ = _write_fake_dataset(tmp_path, n=10)
    out_path = tmp_path / "together_sft.jsonl"
    info = export(in_path, out_path, val_frac=0.2, seed=13)
    assert info["total"] == 10
    assert info["train"] + info["val"] == 10


def test_export_split_determinism(tmp_path):
    """Same seed -> same train/val split across two calls."""
    in_path, _ = _write_fake_dataset(tmp_path, n=20)

    out1 = tmp_path / "run1" / "together_sft.jsonl"
    out2 = tmp_path / "run2" / "together_sft.jsonl"
    info1 = export(in_path, out1, val_frac=0.1, seed=13)
    info2 = export(in_path, out2, val_frac=0.1, seed=13)

    assert info1["train"] == info2["train"]
    assert info1["val"] == info2["val"]

    # The actual JSONL lines must be identical
    train1 = (tmp_path / "run1" / "together_train.jsonl").read_text().splitlines()
    train2 = (tmp_path / "run2" / "together_train.jsonl").read_text().splitlines()
    assert train1 == train2


def test_export_different_seeds_give_different_splits(tmp_path):
    in_path, _ = _write_fake_dataset(tmp_path, n=50)
    out1 = tmp_path / "run1" / "together_sft.jsonl"
    out2 = tmp_path / "run2" / "together_sft.jsonl"
    export(in_path, out1, val_frac=0.2, seed=7)
    export(in_path, out2, val_frac=0.2, seed=42)
    train1 = (tmp_path / "run1" / "together_train.jsonl").read_text()
    train2 = (tmp_path / "run2" / "together_train.jsonl").read_text()
    # Different seeds should give different orderings (with high probability for n=50)
    assert train1 != train2


def test_export_output_rows_are_valid_together_format(tmp_path):
    in_path, _ = _write_fake_dataset(tmp_path, n=5)
    out_path = tmp_path / "together_sft.jsonl"
    export(in_path, out_path, val_frac=0.2, seed=13)

    for line in out_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        assert "messages" in row
        msgs = row["messages"]
        assert len(msgs) == 3
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"
        # User must have both text and image blocks
        user_content = msgs[1]["content"]
        types = [c["type"] for c in user_content]
        assert "text" in types
        assert "image_url" in types
        # Image must be a base64 data-URI
        img_block = next(c for c in user_content if c["type"] == "image_url")
        assert img_block["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_export_images_downscaled(tmp_path):
    """Images larger than MAX_PX must be downscaled in the output."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    big_png = img_dir / "big.png"
    big_png.write_bytes(_make_png_bytes(1024, 768))

    jsonl = tmp_path / "sft_v1.jsonl"
    jsonl.write_text(json.dumps({
        "image": str(big_png),
        "system": "s",
        "user": "u",
        "assistant": '{"action":"up"}',
        "meta": {},
    }) + "\n")

    out_path = tmp_path / "together_sft.jsonl"
    export(jsonl, out_path, val_frac=0.5, seed=0)

    # Parse any output row and check the embedded image dimensions
    out_lines = [l for l in out_path.read_text().splitlines() if l.strip()]
    row = json.loads(out_lines[0])
    user_content = row["messages"][1]["content"]
    img_block = next(c for c in user_content if c["type"] == "image_url")
    b64 = img_block["image_url"]["url"].split(",", 1)[1]
    decoded = base64.b64decode(b64)
    img = Image.open(io.BytesIO(decoded))
    assert max(img.size) <= MAX_PX
