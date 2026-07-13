# tests/test_onboard_orchestrator.py
"""End-to-end offline onboarding: synthetic game -> GameProfile -> playable."""

import pytest

from ludus.adapters.generic import GenericAdapter
from ludus.onboarding.onboard import onboard_game
from ludus.onboarding.profile import GameProfile
from ludus.synthetic import CounterGame, GridWorldGame


def test_onboard_gridworld_produces_complete_profile(tmp_path):
    profile = onboard_game(
        "gridworld", GridWorldGame,
        probe_keys=["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "z"],
        out_dir=tmp_path,
    )
    assert profile.game_id == "gridworld"
    assert profile.controls == {
        "move_left": "ArrowLeft", "move_right": "ArrowRight",
        "move_up": "ArrowUp", "move_down": "ArrowDown",
    }
    assert profile.primary_metric == "score"
    assert profile.higher_is_better is True
    assert "score" in profile.objective
    assert "move_left (ArrowLeft)" in profile.objective
    assert "maximizing" in profile.objective
    assert "maximizeing" not in profile.objective
    # cached to disk
    assert GameProfile.load(tmp_path / "gridworld.json") == profile


def test_onboard_empty_controls_saves_profile_then_raises(tmp_path):
    """When probing discovers no controls, profile is saved (honest artifact)
    but onboard_game raises SystemExit with a clear message."""
    # ArrowLeft has no effect in CounterGame -> empty controls
    with pytest.raises(SystemExit, match="no controls discovered"):
        onboard_game(
            "counter-empty", CounterGame,
            probe_keys=["ArrowLeft"],   # no-op key only
            out_dir=tmp_path,
        )
    # Profile must exist on disk (honest artifact)
    saved = GameProfile.load(tmp_path / "counter-empty.json")
    assert saved.controls == {}


def test_onboarded_profile_drives_a_real_episode_offline(tmp_path):
    """The M1 acceptance test: onboard with zero hand-written config, then
    run the EXISTING episode loop against the same game via GenericAdapter."""
    from ludus.loop import run_episode
    from ludus.persistence.local import LocalStore
    from ludus.providers.mock import MockProvider
    from ludus.rulebook import Rulebook

    profile = onboard_game(
        "gridworld", GridWorldGame,
        probe_keys=["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"],
        out_dir=tmp_path,
    )
    adapter = GenericAdapter(profile.to_game_config())
    game = GridWorldGame()
    result = run_episode(
        adapter=adapter, provider=MockProvider(), gameworld=game,
        store=LocalStore(root=tmp_path / "runs"), rulebook=Rulebook(),
        mode="baseline", max_steps=3, episode_id="gridworld-onboard-test",
    )
    assert result.steps == 3
