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


class _SeqProvider:
    """Returns a Decision carrying an `actions` macro on every call."""

    def __init__(self, actions, primary):
        self._actions = actions
        self._primary = primary

    def decide(self, ctx):
        from ludus.schemas import Decision
        return Decision(
            scene_summary="s", controlled_entity="piece", current_subgoal="g",
            action=self._primary, actions=self._actions,
            expected_result="e", reason="r", confidence=0.9,
        )


def test_loop_applies_action_sequence(tmp_path):
    # A Decision with actions=["left","left","drop"] must apply 3 commands.
    gw = FakeGameWorld(metric_script=[{"holes": 2.0}, {"holes": 1.0}])
    result = run_episode(
        adapter=_TetrisLikeAdapter(),
        provider=_SeqProvider(actions=["left", "left", "drop"], primary="drop"),
        gameworld=gw, store=LocalStore(root=tmp_path), rulebook=Rulebook(),
        mode="baseline", max_steps=1, episode_id="ep-seq",
    )
    assert len(gw.applied) == 3
    assert [c["key"] for c in gw.applied] == ["ArrowLeft", "ArrowLeft", "Space"]
    assert result.legal_action_rate == 1.0   # primary action "drop" is legal


def test_loop_sequence_skips_illegal_moves_in_macro(tmp_path):
    # Illegal entries inside the macro are skipped; legal ones still apply.
    gw = FakeGameWorld(metric_script=[{"holes": 2.0}, {"holes": 2.0}])
    run_episode(
        adapter=_TetrisLikeAdapter(),
        provider=_SeqProvider(actions=["left", "jump", "drop"], primary="left"),
        gameworld=gw, store=LocalStore(root=tmp_path), rulebook=Rulebook(),
        mode="baseline", max_steps=1, episode_id="ep-seq-illegal",
    )
    assert [c["key"] for c in gw.applied] == ["ArrowLeft", "Space"]  # "jump" skipped


def test_loop_empty_actions_uses_single_action(tmp_path):
    # actions=[] -> single-action path (backward compatible).
    gw = FakeGameWorld(metric_script=[{"holes": 2.0}, {"holes": 2.0}])
    run_episode(
        adapter=_TetrisLikeAdapter(),
        provider=_SeqProvider(actions=[], primary="right"),
        gameworld=gw, store=LocalStore(root=tmp_path), rulebook=Rulebook(),
        mode="baseline", max_steps=1, episode_id="ep-seq-empty",
    )
    assert [c["key"] for c in gw.applied] == ["ArrowRight"]


def test_loop_flags_illegal_action(tmp_path):
    gw = FakeGameWorld(metric_script=[{"holes": 1.0}, {"holes": 1.0}])
    result = run_episode(
        adapter=_TetrisLikeAdapter(), provider=MockProvider(script=["jump"]),  # illegal
        gameworld=gw, store=LocalStore(root=tmp_path), rulebook=Rulebook(),
        mode="baseline", max_steps=1, episode_id="ep-illegal",
    )
    assert result.legal_action_rate == 0.0


class _FlakyProvider:
    """Raises on the given (0-based) call indices, else returns a legal Decision.

    Simulates a transient model-call failure mid-episode.
    """

    def __init__(self, fail_on, action="left"):
        self._fail_on = set(fail_on)
        self._action = action
        self._i = -1

    def decide(self, ctx):
        self._i += 1
        if self._i in self._fail_on:
            raise RuntimeError("transient gateway 500")
        from ludus.schemas import Decision
        return Decision(
            scene_summary="s", controlled_entity="c", current_subgoal="g",
            action=self._action, expected_result="e", reason="r", confidence=0.9,
        )


class _StateTextFakeGameWorld:
    """FakeGameWorld variant that returns a non-empty state_text for testing."""

    def __init__(self, metric_script, state_text_value="board: col heights 1 2 3"):
        self._metrics = metric_script
        self._i = 0
        self.applied: list[dict] = []
        self._state_text_value = state_text_value

    def screenshot(self) -> bytes:
        return b"\x89PNG fake-frame"

    def metrics(self, keys):
        m = self._metrics[min(self._i, len(self._metrics) - 1)]
        return {k: m.get(k, 0.0) for k in keys}

    def apply(self, command: dict) -> None:
        self.applied.append(command)
        self._i = min(self._i + 1, len(self._metrics) - 1)

    def state_text(self) -> str:
        return self._state_text_value


def test_loop_persists_state_text_in_step_records(tmp_path):
    """run_episode must capture ctx.state_text in the persisted StepRecord."""
    import json

    gw = _StateTextFakeGameWorld(
        metric_script=[{"holes": 2.0}, {"holes": 2.0}, {"holes": 2.0}],
        state_text_value="col heights: 1 2 0\nholes=0",
    )
    run_episode(
        adapter=_TetrisLikeAdapter(),
        provider=MockProvider(script=["left"]),
        gameworld=gw,
        store=LocalStore(root=tmp_path),
        rulebook=Rulebook(),
        mode="baseline",
        max_steps=1,
        episode_id="ep-state-text",
    )
    lines = (tmp_path / "ep-state-text" / "steps.jsonl").read_text().splitlines()
    step = json.loads(lines[0])
    assert step["state_text"] == "col heights: 1 2 0\nholes=0"


def test_loop_persists_empty_state_text_when_gameworld_returns_empty(tmp_path):
    """When state_text() returns '', StepRecord.state_text should be ''."""
    import json

    gw = FakeGameWorld(metric_script=[{"holes": 2.0}, {"holes": 2.0}])
    run_episode(
        adapter=_TetrisLikeAdapter(),
        provider=MockProvider(script=["left"]),
        gameworld=gw,
        store=LocalStore(root=tmp_path),
        rulebook=Rulebook(),
        mode="baseline",
        max_steps=1,
        episode_id="ep-no-state-text",
    )
    lines = (tmp_path / "ep-no-state-text" / "steps.jsonl").read_text().splitlines()
    step = json.loads(lines[0])
    assert step["state_text"] == ""


def test_loop_survives_transient_decide_failure(tmp_path):
    # One failed decide() must NOT crash the episode: it's skipped, counted, and
    # excluded from the legal-action denominator.
    gw = FakeGameWorld(metric_script=[{"holes": 2.0}] * 4)
    result = run_episode(
        adapter=_TetrisLikeAdapter(),
        provider=_FlakyProvider(fail_on=[1], action="left"),
        gameworld=gw, store=LocalStore(root=tmp_path), rulebook=Rulebook(),
        mode="baseline", max_steps=3, episode_id="ep-flaky",
    )
    assert result.steps == 3
    assert result.errored_steps == 1
    assert result.legal_action_rate == 1.0          # 2 attempted, both legal
    assert len(gw.applied) == 2                      # only successful steps applied
