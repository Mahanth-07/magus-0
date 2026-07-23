# tests/test_backfill_planning_grade.py
"""Backfill smoke: patch planning-grade fields into existing report.json."""

import json

from scripts.backfill_planning_grade import backfill
from tests.test_tuning import _tuning_dirs


def test_backfill_patches_reports_and_returns_rows(tmp_path):
    root, _ = _tuning_dirs(tmp_path)
    rows = backfill(worldmodels_dir=root / "worldmodels",
                    data_dir=root / "data",
                    profiles_dir=root / "profiles",
                    rollouts=2, steps=4)
    assert len(rows) == 1
    row = rows[0]
    assert row["game"] == "synthetic:grid"
    assert row["planning_grade"] is True
    assert row["mean_rollout_score"] > 0
    patched = json.loads(
        (root / "worldmodels" / "synthetic:grid" / "report.json").read_text())
    assert patched["planning_grade"] is True
    assert patched["scoring_rollouts_frac"] > 0
    assert patched["mean_rollout_score"] > 0
    assert patched["status"] == "INDUCED"  # pre-existing fields preserved


def test_backfill_zeros_when_model_missing(tmp_path):
    root, _ = _tuning_dirs(tmp_path)
    (root / "worldmodels" / "synthetic:grid" / "model.py").unlink()
    rows = backfill(worldmodels_dir=root / "worldmodels",
                    data_dir=root / "data",
                    profiles_dir=root / "profiles",
                    rollouts=1, steps=2)
    assert rows[0]["planning_grade"] is False
    assert rows[0]["mean_rollout_score"] == 0.0
