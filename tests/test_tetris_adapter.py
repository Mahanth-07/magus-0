from ludus.config import GameConfig
from ludus.adapters.tetris import TetrisAdapter
from ludus.schemas import ActionArgs


def _cfg():
    return GameConfig(name="tetris", objective="o", legal_actions=["left", "drop"],
        primary_metric="holes", higher_is_better=False, timing_ms=250,
        control_map={"left": "ArrowLeft", "drop": "Space"},
        relevant_metrics=["holes", "lines"])


def test_adapter_exposes_config_fields():
    a = TetrisAdapter(_cfg())
    assert a.name == "tetris"
    assert a.legal_actions == ["left", "drop"]
    assert a.primary_metric == "holes"
    assert a.higher_is_better is False


def test_semantic_to_gameworld_maps_keys():
    a = TetrisAdapter(_cfg())
    cmd = a.semantic_to_gameworld("drop", ActionArgs(duration_ms=100))
    assert cmd["key"] == "Space"
    assert cmd["duration_ms"] == 100
