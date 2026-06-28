from ludus.schemas import Decision, ActionArgs, StepRecord
from ludus.persistence.insforge import step_to_row


def test_step_to_row_flattens_decision():
    step = StepRecord(episode_id="ep1", step_index=2, mode="memory", game="tetris",
        decision=Decision(scene_summary="s", controlled_entity="c", current_subgoal="g",
            action="drop", action_args=ActionArgs(), expected_result="clear line",
            reason="r", confidence=0.9),
        primary_metric="holes", primary_delta=-1.0, improved=True,
        metric_delta={"holes": -1.0}, rule_added=None, screenshot_ref="ep1/step_0002.png")
    row = step_to_row(step)
    assert row["action"] == "drop" and row["confidence"] == 0.9
    assert row["episode_id"] == "ep1" and row["screenshot_ref"] == "ep1/step_0002.png"
