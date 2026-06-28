import base64
import json

from scripts.export_nebius import example_to_sft
from ludus.providers.insforge import _SYSTEM, build_user_text, parse_decision


def _example() -> dict:
    """A minimal example record shaped exactly like collect_dataset.py writes."""
    target = {
        "scene_summary": "Board 10x20",
        "controlled_entity": "the active tetromino",
        "current_subgoal": "place the i to minimize holes and clear lines",
        "action": "left",
        "actions": ["left", "left", "drop"],
        "action_args": {"duration_ms": 120},
        "expected_result": "piece settles at column 1; holes stay low",
        "reason": "expert placement (holes=0, height=4)",
        "confidence": 1.0,
    }
    return {
        "image": "images/ex_000001.png",
        "objective": "Tetris. Place pieces to clear lines.",
        "legal_actions": ["left", "right", "rotate", "drop"],
        "state_text": "current=i at_cols=3-4-5-6\nBoard 10x20 (col heights):\n0 0 0 0 0 0 0 0 0 0",
        "target": target,
        "reward": {"score_after": 100, "holes_after": 0},
    }


IMG = b"\x89PNG\r\n\x1a\n fake-image-bytes"


def test_message_roles_and_order():
    sft = example_to_sft(_example(), IMG)
    msgs = sft["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]


def test_system_message_is_the_runtime_system_string():
    sft = example_to_sft(_example(), IMG)
    assert sft["messages"][0]["content"] == _SYSTEM


def test_user_content_has_text_block_then_image_data_uri():
    ex = _example()
    sft = example_to_sft(ex, IMG)
    user = sft["messages"][1]
    content = user["content"]
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"
    # image is the base64 data URI of the supplied bytes
    expected_uri = "data:image/png;base64," + base64.b64encode(IMG).decode()
    assert content[1]["image_url"]["url"] == expected_uri


def test_user_text_matches_runtime_prompt_helper_and_excludes_image_and_system():
    ex = _example()
    sft = example_to_sft(ex, IMG)
    text = sft["messages"][1]["content"][0]["text"]
    # Exactly the shared runtime user-text helper (objective + legal actions + state_text).
    assert text == build_user_text(
        objective=ex["objective"],
        legal_actions=ex["legal_actions"],
        state_text=ex["state_text"],
    )
    assert ex["objective"] in text
    assert "left, right, rotate, drop" in text
    assert "Board 10x20" in text
    # The user turn must NOT inline the system preamble or the base64 image.
    assert _SYSTEM not in text
    assert "base64" not in text


def test_assistant_content_parses_back_into_a_decision_with_matching_actions():
    ex = _example()
    sft = example_to_sft(ex, IMG)
    assistant = sft["messages"][2]
    # Assistant content is the JSON STRING the student should produce.
    assert isinstance(assistant["content"], str)
    decision = parse_decision(assistant["content"])
    assert decision.actions == ex["target"]["actions"]
    assert decision.action == ex["target"]["action"]
    assert decision.confidence == 1.0
    # round-trips as valid JSON
    assert json.loads(assistant["content"])["controlled_entity"] == "the active tetromino"
