# tests/test_local_store.py
import json
from ludus.schemas import Decision, ActionArgs, StepRecord, EpisodeResult
from ludus.persistence.local import LocalStore


def _step(episode_id="ep1", i=0):
    return StepRecord(
        episode_id=episode_id, step_index=i, mode="memory", game="tetris",
        decision=Decision(scene_summary="s", controlled_entity="c", current_subgoal="g",
                          action="left", action_args=ActionArgs(), expected_result="e",
                          reason="r", confidence=0.5),
        primary_metric="holes", primary_delta=-1.0, improved=True,
        metric_delta={"holes": -1.0}, rule_added=None,
    )


def test_save_screenshot_returns_ref_and_writes_file(tmp_path):
    store = LocalStore(root=tmp_path)
    ref = store.save_screenshot("ep1", 0, b"\x89PNG data")
    assert (tmp_path / ref).read_bytes() == b"\x89PNG data"


def test_save_step_appends_jsonl(tmp_path):
    store = LocalStore(root=tmp_path)
    store.save_step(_step(i=0))
    store.save_step(_step(i=1))
    lines = (tmp_path / "ep1" / "steps.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["step_index"] == 1


def test_save_episode_writes_summary(tmp_path):
    store = LocalStore(root=tmp_path)
    result = EpisodeResult(episode_id="ep1", game="tetris", mode="memory",
                           steps=2, legal_action_rate=1.0, final_metrics={"holes": 1.0},
                           rules=["avoid holes"])
    store.save_episode(result)
    data = json.loads((tmp_path / "ep1" / "episode.json").read_text())
    assert data["steps"] == 2 and data["rules"] == ["avoid holes"]
