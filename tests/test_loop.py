# tests/test_loop.py
from ludus.adapters.base import GameAdapter
from ludus.gameworld_client import FakeGameWorld
from ludus.providers.mock import MockProvider
from ludus.persistence.local import LocalStore
from ludus.rulebook import Rulebook
from ludus.loop import run_episode


class _TetrisLikeAdapter:
    name = "tetris"
    objective = "place pieces, minimize holes"
    legal_actions = ["left", "right", "drop"]
    primary_metric = "holes"
    higher_is_better = False
    timing_ms = 50

    def semantic_to_gameworld(self, action: str, args) -> dict:
        return {"key": {"left": "ArrowLeft", "right": "ArrowRight", "drop": "Space"}[action]}


def test_loop_runs_baseline_no_rules(tmp_path):
    gw = FakeGameWorld(metric_script=[{"holes": 2.0}, {"holes": 2.0}, {"holes": 2.0}])
    rb = Rulebook()
    result = run_episode(
        adapter=_TetrisLikeAdapter(), provider=MockProvider(script=["left", "right"]),
        gameworld=gw, store=LocalStore(root=tmp_path), rulebook=rb,
        mode="baseline", max_steps=2, episode_id="ep-base",
    )
    assert result.steps == 2
    assert result.rules == []                       # baseline never learns
    assert result.legal_action_rate == 1.0


def test_loop_memory_mode_learns_rule(tmp_path):
    # holes never decrease → every confident step is a miss → rule added
    gw = FakeGameWorld(metric_script=[{"holes": 3.0}, {"holes": 3.0}, {"holes": 3.0}])
    rb = Rulebook()
    result = run_episode(
        adapter=_TetrisLikeAdapter(), provider=MockProvider(script=["left", "right"]),
        gameworld=gw, store=LocalStore(root=tmp_path), rulebook=rb,
        mode="memory", max_steps=2, episode_id="ep-mem",
    )
    assert result.steps == 2
    assert len(result.rules) >= 1                    # memory mode learns


def test_loop_memory_no_rule_from_illegal_action(tmp_path):
    gw = FakeGameWorld(metric_script=[{"holes": 3.0}, {"holes": 3.0}])
    rb = Rulebook()
    result = run_episode(
        adapter=_TetrisLikeAdapter(), provider=MockProvider(script=["jump"]),  # illegal
        gameworld=gw, store=LocalStore(root=tmp_path), rulebook=rb,
        mode="memory", max_steps=1, episode_id="ep-illegal-mem",
    )
    assert result.legal_action_rate == 0.0
    assert result.rules == []   # illegal action must not produce a learned rule


def test_loop_flags_illegal_action(tmp_path):
    gw = FakeGameWorld(metric_script=[{"holes": 1.0}, {"holes": 1.0}])
    result = run_episode(
        adapter=_TetrisLikeAdapter(), provider=MockProvider(script=["jump"]),  # illegal
        gameworld=gw, store=LocalStore(root=tmp_path), rulebook=Rulebook(),
        mode="baseline", max_steps=1, episode_id="ep-illegal",
    )
    assert result.legal_action_rate == 0.0
