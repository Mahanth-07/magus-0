# tests/test_reflection.py
from ludus.schemas import Decision, ActionArgs
from ludus.outcome import Outcome
from ludus.reflection import Reflector


def _decision(action="left", confidence=0.8, subgoal="reduce holes"):
    return Decision(
        scene_summary="s", controlled_entity="c", current_subgoal=subgoal,
        action=action, action_args=ActionArgs(), expected_result="fewer holes",
        reason="r", confidence=confidence,
    )


def _outcome(improved):
    return Outcome(delta={"holes": 0.0}, primary_metric="holes",
                   primary_delta=0.0, improved=improved, summary="holes +0.00 (no improvement)")


def test_no_rule_when_improved():
    assert Reflector().reflect(_decision(), _outcome(True), "tetris") is None


def test_rule_when_confident_miss():
    rule = Reflector().reflect(_decision(confidence=0.8), _outcome(False), "tetris")
    assert rule is not None
    assert "tetris" in rule and "left" in rule and "holes" in rule


def test_no_rule_when_low_confidence_miss():
    assert Reflector().reflect(_decision(confidence=0.2), _outcome(False), "tetris") is None
