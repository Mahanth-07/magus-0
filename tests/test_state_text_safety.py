"""state_text must be a safe no-op for ANY game's state shape."""

from ludus.gameworld_client import _tetris_state_text


def test_list_valued_environment_is_safe_noop():
    # 2048 exposes game_state.environment as a LIST (the 4x4 grid); the
    # Tetris renderer must return "" instead of crashing (live duel finding)
    state = {"status": "playing", "metrics": {"primary_score": 4},
             "game_state": {"environment": [[0, 2], [2, 0]],
                            "entities": [{"x": 0, "y": 0}]}}
    assert _tetris_state_text(state) == ""


def test_non_dict_player_is_safe_noop():
    state = {"game_state": {"environment": {"board": [[0]]}, "player": "??"}}
    # board exists but player is junk — must not crash (returns str)
    assert isinstance(_tetris_state_text(state), str)
