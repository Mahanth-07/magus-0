import pytest
from pydantic import ValidationError
from ludus.schemas import Decision, ActionArgs, PlannerContext


def test_decision_parses_full_payload():
    d = Decision(
        scene_summary="An enemy is visible to the right.",
        controlled_entity="First-person player",
        current_subgoal="Center the enemy",
        action="turn_right",
        action_args=ActionArgs(duration_ms=300),
        expected_result="enemy moves toward center",
        reason="firing now would miss",
        confidence=0.82,
    )
    assert d.action == "turn_right"
    assert d.action_args.duration_ms == 300


def test_decision_defaults_action_args():
    d = Decision(
        scene_summary="s", controlled_entity="c", current_subgoal="g",
        action="wait", expected_result="e", reason="r", confidence=0.5,
    )
    assert d.action_args.duration_ms == 200


def test_decision_defaults_empty_actions():
    d = Decision(
        scene_summary="s", controlled_entity="c", current_subgoal="g",
        action="wait", expected_result="e", reason="r", confidence=0.5,
    )
    assert d.actions == []


def test_decision_parses_actions_sequence():
    d = Decision(
        scene_summary="s", controlled_entity="c", current_subgoal="g",
        action="drop", actions=["left", "left", "rotate", "drop"],
        expected_result="e", reason="r", confidence=0.7,
    )
    assert d.actions == ["left", "left", "rotate", "drop"]
    assert d.action == "drop"


def test_decision_rejects_out_of_range_confidence():
    with pytest.raises(ValidationError):
        Decision(
            scene_summary="s", controlled_entity="c", current_subgoal="g",
            action="wait", expected_result="e", reason="r", confidence=1.5,
        )


def test_decision_rejects_unknown_field():
    with pytest.raises(ValidationError):
        Decision(
            scene_summary="s", controlled_entity="c", current_subgoal="g",
            action="wait", expected_result="e", reason="r", confidence=0.5,
            bogus="x",
        )


def test_planner_context_round_trips_screenshot_bytes():
    ctx = PlannerContext(
        objective="place pieces",
        legal_actions=["left", "right", "drop"],
        recent_outcomes=["left -> improved"],
        learned_rules="- avoid holes",
        screenshot_png=b"\x89PNG fake",
    )
    assert ctx.legal_actions == ["left", "right", "drop"]
    assert ctx.screenshot_png.startswith(b"\x89PNG")
