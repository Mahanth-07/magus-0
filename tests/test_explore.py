"""Explorer — uniform-random self-play that logs induction transitions."""

from ludus.onboarding.profile import GameProfile
from ludus.synthetic import GridWorldGame
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
