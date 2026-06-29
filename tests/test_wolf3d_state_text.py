# tests/test_wolf3d_state_text.py
"""Unit tests for the Wolf3D state_text helpers in ludus.gameworld_client.

The angle/coordinate convention was calibrated LIVE against 31_wolf3d:
  forward = (cos(angle_deg), sin(angle_deg)), +y DOWN; angle_to_enemy = atan2(dy, dx).
  bearing > 0 -> turn LEFT, bearing < 0 -> turn RIGHT.
These tests pin that convention so a future refactor can't silently flip the sign.
"""
import math

from ludus.gameworld_client import (
    _relative_bearing,
    _bearing_phrase,
    _wolf3d_state_text,
)


def test_bearing_enemy_directly_ahead():
    # Facing angle 0 = +x. Enemy straight ahead in +x.
    assert abs(_relative_bearing(0, 0, 0, 5, 0)) < 1e-6
    assert "AHEAD" in _bearing_phrase(0)


def test_bearing_enemy_at_plus90_turns_left():
    # Facing +x (angle 0); enemy in +y (which is "screen down"). atan2(1,0)=+90.
    b = _relative_bearing(0, 0, 0, 0, 5)
    assert abs(b - 90) < 1e-6
    assert "turn LEFT" in _bearing_phrase(b)


def test_bearing_enemy_at_minus90_turns_right():
    # Enemy in -y. atan2(-1,0) = -90 -> turn RIGHT.
    b = _relative_bearing(0, 0, 0, 0, -5)
    assert abs(b + 90) < 1e-6
    assert "turn RIGHT" in _bearing_phrase(b)


def test_bearing_behind():
    b = _relative_bearing(0, 0, 0, -5, 0)
    assert abs(abs(b) - 180) < 1e-6
    assert "BEHIND" in _bearing_phrase(b)


def test_live_calibration_case_turns_right_63deg():
    # The exact INITIAL frame from the live probe: player (34,3) facing 270,
    # enemy guard at (30,1). Verified live that turning RIGHT centers it.
    b = _relative_bearing(34, 3, 270, 30, 1)
    assert b < 0  # negative -> RIGHT
    assert 60 < abs(b) < 67  # ~63 deg
    assert "turn RIGHT" in _bearing_phrase(b)


def _wolf_state(enemy=True):
    state = {
        "game_state": {
            "player": {
                "tile_x": 34, "tile_y": 3, "angle_deg": 270,
                "health": 100, "ammo": 8, "weapon": "pistol",
            },
            "environment": {},
        },
        "metrics": {"monsters_remaining": 6, "total_monsters": 6},
    }
    if enemy:
        state["game_state"]["environment"]["nearest_enemy"] = {
            "type": "guard", "state": "idle",
            "distance_tiles": 4.472, "tile_x": 30, "tile_y": 1,
        }
    return state


def test_state_text_with_enemy():
    txt = _wolf3d_state_text(_wolf_state(enemy=True))
    assert "Player tile=(34,3) facing=270deg" in txt
    assert "health=100 ammo=8 weapon=pistol" in txt
    assert "guard" in txt and "4.5 tiles away" in txt
    assert "turn RIGHT" in txt
    assert "Monsters remaining: 6 of 6" in txt
    assert "Tactics:" in txt


def test_state_text_no_enemy_graceful():
    txt = _wolf3d_state_text(_wolf_state(enemy=False))
    assert "NONE listed" in txt
    assert "scan" in txt
    assert "Tactics:" in txt


def test_state_text_returns_empty_for_non_wolf3d():
    # No player.angle_deg -> not a Wolf3D state.
    assert _wolf3d_state_text({"game_state": {"player": {"tile_x": 1}}}) == ""
    assert _wolf3d_state_text({}) == ""
