"""Tests for the fine-tune job result extraction.

Regression: the June run's job returned fine_tuned_model=null, the extractor
fell back to result_files[0] (a "file-..." checkpoint FILE id), and that id
was pasted into .env as NEBIUS_MODEL — a model id that can never serve chat
completions. A file id must never be reported as a servable model id.
"""

from scripts.finetune_nebius import _model_id_from


def test_model_id_prefers_fine_tuned_model():
    job = {"fine_tuned_model": "meta-llama/Llama-3.2-3B-Instruct-LoRa:tetris-abc"}
    assert _model_id_from(job) == "meta-llama/Llama-3.2-3B-Instruct-LoRa:tetris-abc"


def test_model_id_falls_back_to_checkpoint_field():
    job = {"fine_tuned_model": None, "fine_tuned_model_checkpoint": "ft-checkpoint-xyz"}
    assert _model_id_from(job) == "ft-checkpoint-xyz"


def test_model_id_never_returns_a_file_id():
    job = {
        "status": "succeeded",
        "fine_tuned_model": None,
        "result_files": ["file-6b88516daa27555cb27832adfc34557d"],
    }
    assert _model_id_from(job) is None


def test_model_id_rejects_file_id_in_named_fields():
    # even if the API stuffs a file id into a model field, refuse it
    job = {"fine_tuned_model": "file-deadbeef"}
    assert _model_id_from(job) is None
