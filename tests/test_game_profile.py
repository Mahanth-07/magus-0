# tests/test_game_profile.py
"""GameProfile — machine-generated per-game config, cached as JSON."""

import json

from ludus.config import GameConfig
from ludus.onboarding.profile import GameProfile


def _profile() -> GameProfile:
    return GameProfile(
        game_id="gridworld",
        controls={"move_left": "ArrowLeft", "move_right": "ArrowRight"},
        control_effects={"move_left": ["game_state.player.x"],
                         "move_right": ["game_state.player.x"]},
        metrics=["score", "steps"],
        primary_metric="score",
        higher_is_better=True,
        timing_ms=120,
        objective="Maximize score. Controls: move_left (ArrowLeft), move_right (ArrowRight).",
        state_schema={"metrics.score": "int", "game_state.player.x": "int"},
    )


def test_roundtrips_through_json_file(tmp_path):
    p = _profile()
    path = tmp_path / "gridworld.json"
    p.save(path)
    loaded = GameProfile.load(path)
    assert loaded == p
    # file is plain JSON (inspectable artifact, spec commitment #2)
    assert json.loads(path.read_text())["game_id"] == "gridworld"


def test_to_game_config_bridges_to_existing_loop():
    cfg = _profile().to_game_config()
    assert isinstance(cfg, GameConfig)
    assert cfg.name == "gridworld"
    assert cfg.legal_actions == ["move_left", "move_right"]
    assert cfg.control_map == {"move_left": "ArrowLeft", "move_right": "ArrowRight"}
    assert cfg.primary_metric == "score"
    assert cfg.relevant_metrics == ["score", "steps"]
    assert cfg.timing_ms == 120
    assert cfg.higher_is_better is True
    assert cfg.objective == "Maximize score. Controls: move_left (ArrowLeft), move_right (ArrowRight)."
