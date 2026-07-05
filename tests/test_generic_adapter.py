# tests/test_generic_adapter.py
"""GenericAdapter — GameAdapter protocol from any GameConfig, no game class."""

from ludus.adapters.generic import GenericAdapter
from ludus.config import GameConfig
from ludus.schemas import ActionArgs


def _cfg() -> GameConfig:
    return GameConfig(
        name="gridworld", objective="Maximize score.",
        legal_actions=["move_left", "move_right"],
        primary_metric="score", higher_is_better=True, timing_ms=100,
        control_map={"move_left": "ArrowLeft", "move_right": "ArrowRight"},
        relevant_metrics=["score", "steps"],
    )


def test_exposes_adapter_protocol_fields():
    a = GenericAdapter(_cfg())
    assert a.name == "gridworld"
    assert a.legal_actions == ["move_left", "move_right"]
    assert a.primary_metric == "score"
    assert a.higher_is_better is True
    assert a.relevant_metrics == ["score", "steps"]
    assert a.timing_ms == 100


def test_semantic_to_gameworld_maps_key_and_duration():
    a = GenericAdapter(_cfg())
    cmd = a.semantic_to_gameworld("move_left", ActionArgs(duration_ms=100))
    assert cmd == {"key": "ArrowLeft", "duration_ms": 100}
