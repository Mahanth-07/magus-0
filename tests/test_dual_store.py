# tests/test_dual_store.py
from ludus.schemas import Decision, ActionArgs, StepRecord
from ludus.persistence.dual import DualWriteStore


class _Recording:
    def __init__(self): self.steps = []; self.shots = []; self.episodes = []
    def save_screenshot(self, eid, i, png): self.shots.append((eid, i)); return f"{eid}/{i}.png"
    def save_step(self, step): self.steps.append(step)
    def save_episode(self, result): self.episodes.append(result)


class _Broken:
    def save_screenshot(self, eid, i, png): raise RuntimeError("s3 down")
    def save_step(self, step): raise RuntimeError("pg down")
    def save_episode(self, result): raise RuntimeError("pg down")


def _step():
    return StepRecord(episode_id="ep1", step_index=0, mode="memory", game="tetris",
        decision=Decision(scene_summary="s", controlled_entity="c", current_subgoal="g",
            action="left", action_args=ActionArgs(), expected_result="e", reason="r", confidence=0.5),
        primary_metric="holes", primary_delta=-1.0, improved=True, metric_delta={"holes": -1.0})


def test_writes_to_both():
    a, b = _Recording(), _Recording()
    store = DualWriteStore(primary=a, secondary=b)
    store.save_step(_step())
    assert len(a.steps) == 1 and len(b.steps) == 1


def test_secondary_failure_does_not_break_primary():
    a = _Recording()
    store = DualWriteStore(primary=a, secondary=_Broken())
    store.save_step(_step())                       # must not raise
    ref = store.save_screenshot("ep1", 0, b"x")    # must not raise
    assert len(a.steps) == 1
    assert ref == "ep1/0.png"                       # primary ref is returned
