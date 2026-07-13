# tests/test_prober.py
"""ControlProber against synthetic ground truth."""

import pytest

from ludus.onboarding.prober import (
    ACTIONABLE_STATUSES,
    DEFAULT_PROBE_KEYS,
    WAKE_KEYS,
    ControlProber,
    ProbeReport,
    _is_actionable,
)
from ludus.synthetic import ClickMenuGame, CounterGame, GridWorldGame, MenuGame, ReadyGame


def test_discovers_gridworld_movement_controls_with_semantic_names():
    prober = ControlProber(GridWorldGame, probe_keys=DEFAULT_PROBE_KEYS, ambient_window_s=0.0)
    report = prober.probe()
    controls = report.controls  # semantic -> key
    assert controls["move_left"] == "ArrowLeft"
    assert controls["move_right"] == "ArrowRight"
    assert controls["move_up"] == "ArrowUp"
    assert controls["move_down"] == "ArrowDown"


def test_noop_keys_are_excluded():
    report = ControlProber(GridWorldGame, probe_keys=["ArrowLeft", "z"], ambient_window_s=0.0).probe()
    assert "ArrowLeft" in report.controls.values()
    assert "z" not in report.controls.values()


def test_ambient_step_counter_not_attributed_to_keys():
    # metrics.steps ticks on EVERY apply -> it must not appear as a key effect.
    report = ControlProber(GridWorldGame, probe_keys=["ArrowRight"], ambient_window_s=0.0).probe()
    effects = report.control_effects["move_right"]
    assert "game_state.player.x" in effects
    assert "metrics.steps" not in effects


def test_counter_game_action_key_gets_use_name():
    report = ControlProber(CounterGame, probe_keys=["x", "ArrowLeft"], ambient_window_s=0.0).probe()
    assert report.controls == {"use_x": "x"}  # arrows do nothing in CounterGame


def test_prober_survives_quit_key_via_relaunch():
    # 'q' makes GridWorld terminal; the prober must relaunch and keep probing.
    report = ControlProber(GridWorldGame, probe_keys=["q", "ArrowRight"], ambient_window_s=0.0).probe()
    assert report.controls.get("move_right") == "ArrowRight"
    assert report.relaunches >= 1
    # 'q' itself only flips status -> excluded from controls
    assert "q" not in report.controls.values()


def test_state_schema_captured():
    report = ControlProber(GridWorldGame, probe_keys=["ArrowRight"], ambient_window_s=0.0).probe()
    assert report.state_schema["metrics.score"] == "int"
    assert report.state_schema["game_state.player.x"] == "int"


# --- wake-up phase -----------------------------------------------------------

def test_wake_keys_constant_exists_and_includes_enter_and_space():
    assert "Enter" in WAKE_KEYS
    assert "Space" in WAKE_KEYS


def test_probe_report_has_wake_key_field_default_none():
    r = ProbeReport()
    assert r.wake_key is None


def test_prober_sets_wake_key_none_for_already_playing_game():
    report = ControlProber(GridWorldGame, probe_keys=["ArrowRight"],
                           ambient_window_s=0.0).probe()
    assert report.wake_key is None


def test_prober_discovers_controls_on_menu_game_with_enter_wake():
    report = ControlProber(MenuGame, probe_keys=["x"], ambient_window_s=0.0).probe()
    assert report.controls == {"use_x": "x"}
    assert report.wake_key == "Enter"


def test_prober_raises_runtime_error_when_no_wake_key_works():
    class AlwaysMenuGame(MenuGame):
        """Never transitions out of menu status regardless of input."""
        def apply(self, command: dict) -> None:
            pass  # always stays "menu"

    with pytest.raises(RuntimeError, match="wake keys"):
        ControlProber(AlwaysMenuGame, probe_keys=["x"],
                      ambient_window_s=0.0).probe()


# --- new boot-fix behaviors ---------------------------------------------------

def test_actionable_statuses_includes_playing_and_ready():
    assert "playing" in ACTIONABLE_STATUSES
    assert "ready" in ACTIONABLE_STATUSES


def test_is_actionable_returns_true_for_playing():
    assert _is_actionable({"status": "playing"}) is True


def test_is_actionable_returns_true_for_ready():
    assert _is_actionable({"status": "ready"}) is True


def test_is_actionable_returns_false_for_menu():
    assert _is_actionable({"status": "menu"}) is False


def test_wake_keys_includes_mouse_center():
    assert "mouse:center" in WAKE_KEYS


def test_prober_accepts_ready_status_as_actionable():
    """ReadyGame starts with status='ready'; prober must treat it as actionable
    (no wake-up attempt) and discover controls normally."""
    report = ControlProber(ReadyGame, probe_keys=["x"], ambient_window_s=0.0).probe()
    assert report.controls == {"use_x": "x"}
    assert report.wake_key is None  # already actionable, no wake needed


def test_prober_wakes_click_menu_game_via_mouse_pseudo_key():
    """ClickMenuGame only wakes on mouse:* keys. Prober must discover controls
    despite the keyboard-only WAKE_KEYS being no-ops."""
    report = ControlProber(ClickMenuGame, probe_keys=["x"], ambient_window_s=0.0).probe()
    assert report.controls == {"use_x": "x"}
    assert report.wake_key is not None
    assert report.wake_key.startswith("mouse:")
