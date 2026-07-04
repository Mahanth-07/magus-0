"""Candidate-block rendering — the generalized M0 'copy candidate [0]' prompt."""

from ludus.planning.candidates import render_candidate_block


def test_renders_m0_compatible_block():
    ranked = [
        {"actions": ["move_right"], "predicted_reward": 10.0},
        {"actions": ["move_left", "move_up"], "predicted_reward": 0.0},
    ]
    block = render_candidate_block(ranked, primary_metric="score")
    lines = block.splitlines()
    assert lines[0].startswith("Candidate placements")
    assert "[0]" in lines[1] and "actions=['move_right']" in lines[1]
    assert "predicted score +10.0" in lines[1]
    assert "[1]" in lines[2] and "+0.0" in lines[2]


def test_empty_candidates_render_empty_string():
    assert render_candidate_block([], primary_metric="score") == ""
