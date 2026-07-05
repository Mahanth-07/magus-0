"""The loop must hand providers the raw state dict (planner seam)."""

from ludus.loop import run_episode
from ludus.persistence.local import LocalStore
from ludus.rulebook import Rulebook
from ludus.schemas import Decision, PlannerContext
from ludus.synthetic import GridWorldGame
from ludus.adapters.generic import GenericAdapter
from ludus.config import GameConfig


def _cfg() -> GameConfig:
    return GameConfig(
        name="gridworld", objective="test", legal_actions=["move_right"],
        primary_metric="score", higher_is_better=True, timing_ms=10,
        control_map={"move_right": "ArrowRight"}, relevant_metrics=["score"],
    )


class _CapturingProvider:
    name = "capture"

    def __init__(self):
        self.contexts = []

    def decide(self, ctx: PlannerContext) -> Decision:
        self.contexts.append(ctx)
        return Decision(
            scene_summary="s", controlled_entity="e", current_subgoal="g",
            action="move_right", actions=["move_right"],
            expected_result="r", reason="t", confidence=1.0,
        )


def test_planner_context_accepts_raw_state_default_none():
    ctx = PlannerContext(objective="o", legal_actions=["a"],
                         screenshot_png=b"x")
    assert ctx.raw_state is None


def test_loop_threads_raw_state_to_provider(tmp_path):
    provider = _CapturingProvider()
    run_episode(adapter=GenericAdapter(_cfg()), provider=provider,
                gameworld=GridWorldGame(), store=LocalStore(root=tmp_path),
                rulebook=Rulebook(), mode="baseline", max_steps=2,
                episode_id="rawstate-test")
    assert len(provider.contexts) == 2
    rs = provider.contexts[0].raw_state
    assert rs is not None
    assert rs["game_state"]["player"]["x"] == 2  # GridWorld spawn


def test_loop_survives_gameworld_without_raw_state(tmp_path):
    class NoRawState:
        def screenshot(self):
            return b"png"

        def metrics(self, keys):
            return {k: 0.0 for k in keys}

        def apply(self, command):
            pass

    provider = _CapturingProvider()
    run_episode(adapter=GenericAdapter(_cfg()), provider=provider,
                gameworld=NoRawState(), store=LocalStore(root=tmp_path),
                rulebook=Rulebook(), mode="baseline", max_steps=1,
                episode_id="norawstate-test")
    assert provider.contexts[0].raw_state is None
