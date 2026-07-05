# tests/test_collect_trajectories_smoke.py
"""Import smoke test for scripts/collect_trajectories.py.

Verifies the script can be imported and its helper functions are callable
without any live game infrastructure (no browser, no provider credentials).
"""

from __future__ import annotations

from pathlib import Path

import pytest


def test_script_imports_cleanly():
    """The script must be importable without runtime side-effects."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "collect_trajectories",
        Path(__file__).resolve().parent.parent / "scripts" / "collect_trajectories.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert callable(mod.main)
    assert callable(mod._induced_games)
    assert callable(mod._collect_episode)


def test_induced_games_returns_empty_for_missing_dir(tmp_path):
    """_induced_games returns [] when the worldmodels dir does not exist."""
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location(
        "collect_trajectories",
        Path(__file__).resolve().parent.parent / "scripts" / "collect_trajectories.py",
    )
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)

    result = mod._induced_games(tmp_path / "nonexistent")
    assert result == []


def test_induced_games_finds_induced_status(tmp_path):
    """_induced_games returns only games with status == 'INDUCED'."""
    import json
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location(
        "collect_trajectories",
        Path(__file__).resolve().parent.parent / "scripts" / "collect_trajectories.py",
    )
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)

    wm = tmp_path / "worldmodels"
    (wm / "game_a").mkdir(parents=True)
    (wm / "game_b").mkdir(parents=True)
    (wm / "game_c").mkdir(parents=True)
    (wm / "game_a" / "report.json").write_text(json.dumps({"status": "INDUCED"}))
    (wm / "game_b" / "report.json").write_text(json.dumps({"status": "FAILED"}))
    (wm / "game_c" / "report.json").write_text(json.dumps({"status": "INDUCED"}))

    result = mod._induced_games(wm)
    assert sorted(result) == ["game_a", "game_c"]
