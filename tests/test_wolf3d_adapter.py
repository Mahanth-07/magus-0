# tests/test_wolf3d_adapter.py
from ludus.config import GameConfig
from ludus.adapters.wolf3d import Wolf3DAdapter
from ludus.schemas import ActionArgs


def _cfg():
    return GameConfig(name="wolf3d", objective="kill one enemy, preserve health",
        legal_actions=["move_forward", "turn_left", "turn_right", "attack", "wait"],
        primary_metric="enemies_killed", higher_is_better=True, timing_ms=300,
        control_map={"move_forward": "ArrowUp", "turn_left": "ArrowLeft",
                     "turn_right": "ArrowRight", "attack": "Control", "wait": ""},
        relevant_metrics=["enemies_killed", "health"])


def test_wolf3d_fields_and_mapping():
    a = Wolf3DAdapter(_cfg())
    assert a.primary_metric == "enemies_killed" and a.higher_is_better is True
    assert a.semantic_to_gameworld("attack", ActionArgs())["key"] == "Control"
