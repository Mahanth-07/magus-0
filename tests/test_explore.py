"""Explorer — uniform-random self-play that logs induction transitions."""

import pytest

from ludus.onboarding.profile import GameProfile
from ludus.synthetic import GridWorldGame, MenuGame, ReadyGame
from ludus.worldmodel.explore import explore_game
from ludus.worldmodel.transitions import TransitionStore


def _grid_profile() -> GameProfile:
    return GameProfile(
        game_id="synthetic:grid",
        controls={"move_left": "ArrowLeft", "move_right": "ArrowRight",
                  "move_up": "ArrowUp", "move_down": "ArrowDown"},
        control_effects={}, metrics=["score", "steps"], primary_metric="score",
        higher_is_better=True, timing_ms=50, objective="test",
        state_schema={},
    )


def test_explore_logs_requested_transitions(tmp_path):
    store = TransitionStore(tmp_path / "t.jsonl")
    summary = explore_game(_grid_profile(), GridWorldGame, store=store,
                           episodes=3, steps_per_episode=10, seed=5)
    ts = store.load()
    assert len(ts) == 30
    assert summary["transitions"] == 30 and summary["episodes"] == 3
    t0 = ts[0]
    assert t0.action in _grid_profile().controls
    assert t0.key == _grid_profile().controls[t0.action]
    assert t0.before["status"] == "playing"
    assert t0.game_id == "synthetic:grid"


def test_explore_episode_ids_distinct_and_stable(tmp_path):
    store = TransitionStore(tmp_path / "t.jsonl")
    explore_game(_grid_profile(), GridWorldGame, store=store,
                 episodes=2, steps_per_episode=4, seed=1)
    eps = [t.episode_id for t in store.load()]
    assert len(set(eps)) == 2
    assert eps[0] != eps[-1]


def test_explore_deterministic_for_seed(tmp_path):
    s1 = TransitionStore(tmp_path / "a.jsonl")
    s2 = TransitionStore(tmp_path / "b.jsonl")
    explore_game(_grid_profile(), GridWorldGame, store=s1,
                 episodes=2, steps_per_episode=6, seed=9)
    explore_game(_grid_profile(), GridWorldGame, store=s2,
                 episodes=2, steps_per_episode=6, seed=9)
    a = [(t.action, t.after) for t in s1.load()]
    b = [(t.action, t.after) for t in s2.load()]
    assert a == b


def test_explore_duration_override(tmp_path):
    # turn-based games need TAP presses: key-repeat during a long hold makes
    # one press execute multiple moves, corrupting induction transitions
    class DurationRecorder(GridWorldGame):
        durations = []

        def apply(self, command):
            DurationRecorder.durations.append(command["duration_ms"])
            super().apply(command)

    DurationRecorder.durations = []
    store = TransitionStore(tmp_path / "t.jsonl")
    explore_game(_grid_profile(), DurationRecorder, store=store,
                 episodes=1, steps_per_episode=3, seed=1, duration_ms=30)
    assert DurationRecorder.durations == [30, 30, 30]

    DurationRecorder.durations = []
    store2 = TransitionStore(tmp_path / "t2.jsonl")
    explore_game(_grid_profile(), DurationRecorder, store=store2,
                 episodes=1, steps_per_episode=2, seed=1)
    assert DurationRecorder.durations == [50, 50]   # profile.timing_ms default


def test_explore_ends_episode_on_terminal(tmp_path):
    # add the quit key as a "control": pressing it ends the episode early;
    # the explorer must record terminal_after=True and start a fresh episode.
    profile = _grid_profile()
    profile.controls = {"use_q": "q"}
    store = TransitionStore(tmp_path / "t.jsonl")
    explore_game(profile, GridWorldGame, store=store,
                 episodes=2, steps_per_episode=5, seed=2)
    ts = store.load()
    assert len(ts) == 2                      # each episode ends on press 1
    assert all(t.terminal_after for t in ts)
    assert len({t.episode_id for t in ts}) == 2


# --- wake-up and empty-controls guardrails -----------------------------------

def _menu_profile() -> GameProfile:
    return GameProfile(
        game_id="synthetic:menu",
        controls={"use_x": "x"},
        control_effects={}, metrics=["score"], primary_metric="score",
        higher_is_better=True, timing_ms=50, objective="test",
        state_schema={},
    )


def test_explore_wakes_menu_game_before_episode(tmp_path):
    """explore_game must press Enter to leave the menu before each episode."""
    store = TransitionStore(tmp_path / "t.jsonl")
    summary = explore_game(_menu_profile(), MenuGame, store=store,
                           episodes=2, steps_per_episode=3, seed=1)
    ts = store.load()
    # All before-states must be "playing" (menu was exited)
    assert all(t.before["status"] == "playing" for t in ts)
    assert summary["transitions"] == 6


def test_explore_handles_ready_status_as_actionable(tmp_path):
    """Games that start with status='ready' must be explored normally.

    Regression for pacman/minesweeper: before the fix, the per-step
    `if not _is_actionable(before): break` check caused an immediate break
    because 'ready' was treated as non-actionable, yielding 0 transitions.
    """
    profile = GameProfile(
        game_id="synthetic:ready",
        controls={"use_x": "x"},
        control_effects={}, metrics=["score"], primary_metric="score",
        higher_is_better=True, timing_ms=50, objective="test",
        state_schema={},
    )
    store = TransitionStore(tmp_path / "t.jsonl")
    summary = explore_game(profile, ReadyGame, store=store,
                           episodes=2, steps_per_episode=3, seed=1)
    ts = store.load()
    # All before-states must have an actionable status (ready or playing)
    assert all(t.before["status"] in ("ready", "playing") for t in ts)
    assert summary["transitions"] == 6


def test_explore_raises_value_error_on_empty_controls(tmp_path):
    empty_profile = GameProfile(
        game_id="bad", controls={}, control_effects={},
        metrics=[], primary_metric="score", higher_is_better=True,
        timing_ms=50, objective="test", state_schema={},
    )
    store = TransitionStore(tmp_path / "t.jsonl")
    with pytest.raises(ValueError, match="no controls"):
        explore_game(empty_profile, MenuGame, store=store,
                     episodes=1, steps_per_episode=1, seed=0)
