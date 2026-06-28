from ludus.config import GameConfig
from ludus.adapters.fireboy_watergirl import FireboyWatergirlAdapter
from ludus.schemas import ActionArgs


def _cfg():
    return GameConfig(name="fireboy_watergirl", objective="o",
        legal_actions=["move_left", "move_right", "jump", "wait_for_partner"],
        primary_metric="score", higher_is_better=True, timing_ms=250,
        control_map={"move_left": "a", "move_right": "d", "jump": "w"},
        relevant_metrics=["score", "completion_progress"])


def test_adapter_exposes_config_fields():
    a = FireboyWatergirlAdapter(_cfg())
    assert a.name == "fireboy_watergirl"
    assert "wait_for_partner" in a.legal_actions
    assert a.primary_metric == "score"
    assert a.higher_is_better is True


def test_semantic_to_gameworld_maps_watergirl_keys():
    a = FireboyWatergirlAdapter(_cfg())
    assert a.semantic_to_gameworld("move_right", ActionArgs(duration_ms=200))["key"] == "d"
    assert a.semantic_to_gameworld("jump", ActionArgs())["key"] == "w"


def test_wait_for_partner_is_noop_empty_key():
    a = FireboyWatergirlAdapter(_cfg())
    cmd = a.semantic_to_gameworld("wait_for_partner", ActionArgs(duration_ms=300))
    assert cmd["key"] == ""
    assert cmd["duration_ms"] == 300
