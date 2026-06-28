from ludus.config import load_game_config


def test_loads_tetris_config(tmp_path):
    p = tmp_path / "tetris.yaml"
    p.write_text(
        "name: tetris\n"
        "objective: place pieces, minimize holes\n"
        "legal_actions: [left, right, rotate, drop]\n"
        "primary_metric: holes\n"
        "higher_is_better: false\n"
        "timing_ms: 250\n"
        "control_map: {left: ArrowLeft, right: ArrowRight, rotate: ArrowUp, drop: Space}\n"
        "relevant_metrics: [holes, lines, height]\n"
    )
    cfg = load_game_config(p)
    assert cfg.name == "tetris"
    assert cfg.legal_actions == ["left", "right", "rotate", "drop"]
    assert cfg.higher_is_better is False
    assert cfg.control_map["drop"] == "Space"
