"""Tests for scripts/finetune_together.py — Together VLM fine-tune launcher.

Tests cover the model-id extraction logic (the critical safety gate that prevents
a file-id being reported as a servable model id).
"""

from __future__ import annotations

from scripts.finetune_together import _model_id_from, TERMINAL


# ---------------------------------------------------------------------------
# _model_id_from tests
# ---------------------------------------------------------------------------

def test_model_id_prefers_output_name():
    job = {"output_name": "Qwen/Qwen3-VL-8B-Instruct:magus-student-v1:abc123",
           "fine_tuned_model": None}
    assert _model_id_from(job) == "Qwen/Qwen3-VL-8B-Instruct:magus-student-v1:abc123"


def test_model_id_falls_back_to_fine_tuned_model():
    job = {"output_name": None,
           "fine_tuned_model": "Qwen/Qwen3-VL-8B-Instruct:magus-student-v1:xyz"}
    assert _model_id_from(job) == "Qwen/Qwen3-VL-8B-Instruct:magus-student-v1:xyz"


def test_model_id_returns_none_when_both_absent():
    job = {"status": "succeeded"}
    assert _model_id_from(job) is None


def test_model_id_returns_none_for_null_fields():
    job = {"output_name": None, "fine_tuned_model": None}
    assert _model_id_from(job) is None


def test_model_id_never_returns_file_id_in_output_name():
    job = {"output_name": "file-deadbeef1234", "fine_tuned_model": None}
    assert _model_id_from(job) is None


def test_model_id_never_returns_file_id_in_fine_tuned_model():
    job = {"output_name": None, "fine_tuned_model": "file-abc123"}
    assert _model_id_from(job) is None


def test_model_id_file_id_skipped_falls_back_to_second_field():
    """If output_name is a file-id but fine_tuned_model is valid, use fine_tuned_model."""
    job = {
        "output_name": "file-badid",
        "fine_tuned_model": "Qwen/Qwen3-VL-8B-Instruct:good-model",
    }
    assert _model_id_from(job) == "Qwen/Qwen3-VL-8B-Instruct:good-model"


# ---------------------------------------------------------------------------
# TERMINAL set tests
# ---------------------------------------------------------------------------

def test_terminal_set_contains_expected_statuses():
    assert "succeeded" in TERMINAL
    assert "failed" in TERMINAL
    assert "cancelled" in TERMINAL
    assert "error" in TERMINAL


def test_terminal_set_does_not_contain_running_statuses():
    for status in ("running", "queued", "pending", "compiling", "uploading"):
        assert status not in TERMINAL
