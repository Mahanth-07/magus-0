# tests/test_onboarding_diff.py
"""flatten_state/diff_states — the change-attribution primitives for onboarding."""

from ludus.onboarding.diff import diff_states, flatten_state


def test_flatten_nested_dicts_to_dot_paths():
    state = {"metrics": {"score": 5}, "game_state": {"player": {"x": 2, "y": 3}}}
    flat = flatten_state(state)
    assert flat["metrics.score"] == 5
    assert flat["game_state.player.x"] == 2
    assert flat["game_state.player.y"] == 3


def test_flatten_indexes_small_lists():
    state = {"blocks": [{"x": 1}, {"x": 4}]}
    flat = flatten_state(state)
    assert flat["blocks[0].x"] == 1
    assert flat["blocks[1].x"] == 4


def test_flatten_summarizes_long_lists_as_length_only():
    # board grids (e.g. 20 rows) would create noise; record length, not elements
    state = {"board": [[0] * 10 for _ in range(20)]}
    flat = flatten_state(state)
    assert flat["board.__len__"] == 20
    assert not any(k.startswith("board[") for k in flat)


def test_flatten_keeps_scalars_and_none():
    flat = flatten_state({"status": "playing", "held": None})
    assert flat["status"] == "playing"
    assert flat["held"] is None


def test_diff_reports_changed_added_removed():
    before = {"metrics": {"score": 0, "lives": 3}, "phase": "a"}
    after = {"metrics": {"score": 10, "lives": 3}, "spawn": True}
    d = diff_states(flatten_state(before), flatten_state(after))
    assert d["metrics.score"] == (0, 10)
    assert d["phase"] == ("a", None)      # removed
    assert d["spawn"] == (None, True)     # added
    assert "metrics.lives" not in d       # unchanged


def test_diff_identical_states_is_empty():
    s = flatten_state({"a": {"b": 1}})
    assert diff_states(s, s) == {}


def test_diff_detects_removal_of_none_valued_field():
    # a stored None is a real value (e.g. Tetris held=null); losing the key
    # entirely must still be reported, not confused with "no change"
    before = flatten_state({"held": None})
    after = flatten_state({})
    assert "held" in diff_states(before, after)


def test_flatten_list_boundary_at_max_indexed():
    from ludus.onboarding.diff import MAX_INDEXED_LIST
    exactly = {"xs": list(range(MAX_INDEXED_LIST))}
    over = {"xs": list(range(MAX_INDEXED_LIST + 1))}
    assert f"xs[{MAX_INDEXED_LIST - 1}]" in flatten_state(exactly)
    assert flatten_state(over) == {"xs.__len__": MAX_INDEXED_LIST + 1}
