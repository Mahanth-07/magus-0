# tests/test_cli_onboard.py
"""CLI onboarding path — offline via synthetic games."""

from ludus.cli import cmd_onboard
from ludus.onboarding.profile import GameProfile


def test_cmd_onboard_synthetic_writes_profile(tmp_path, capsys):
    cmd_onboard(game="synthetic:grid", out_dir=tmp_path)
    profile = GameProfile.load(tmp_path / "synthetic:grid.json")
    assert profile.controls["move_right"] == "ArrowRight"
    assert profile.primary_metric == "score"
    out = capsys.readouterr().out
    assert "move_right" in out          # human-readable summary printed
    assert "score" in out


def test_cmd_onboard_counter_synthetic(tmp_path):
    cmd_onboard(game="synthetic:counter", out_dir=tmp_path)
    profile = GameProfile.load(tmp_path / "synthetic:counter.json")
    assert profile.controls == {"use_x": "x", "use_z": "z"}
