# tests/test_prober_hardening.py
"""M1 live findings: press-window ambient sampling + apply() resilience."""

import pytest

from ludus.onboarding.prober import ControlProber
from ludus.synthetic import AutoRunnerGame, GridWorldGame


def test_autorunner_drifts_with_wall_clock():
    g = AutoRunnerGame(drift_hz=50)
    x0 = g.raw_state()["game_state"]["player"]["x"]
    import time
    time.sleep(0.1)
    assert g.raw_state()["game_state"]["player"]["x"] > x0


def test_press_window_ambient_filters_continuous_motion():
    # AutoRunner's player.x advances with TIME, not input. With press-window
    # ambient sampling the drift lands in ambient_paths and 'z' (a real no-op)
    # is not credited with movement. This is the snake/flappy scramble fix.
    prober = ControlProber(lambda: AutoRunnerGame(drift_hz=50),
                           probe_keys=["z"], duration_ms=80)
    report = prober.probe()
    assert any(p.endswith("player.x") for p in report.ambient_paths)
    assert report.controls == {}


def test_press_window_does_not_break_static_games():
    # GridWorld has no wall-clock dynamics; discovery must be unchanged.
    report = ControlProber(GridWorldGame, probe_keys=["ArrowRight"]).probe()
    assert report.controls == {"move_right": "ArrowRight"}


class _FlakyGrid(GridWorldGame):
    """apply() raises once (simulated Playwright timeout), then recovers."""

    def __init__(self):
        super().__init__()
        self.raised = False

    def apply(self, command):
        if not self.raised and command.get("key") == "ArrowRight":
            self.raised = True
            raise TimeoutError("simulated Playwright hiccup")
        super().apply(command)


def test_apply_exception_skips_press_not_probe():
    report = ControlProber(_FlakyGrid, probe_keys=["ArrowRight", "ArrowLeft"]).probe()
    # one press lost, but 2/3 remaining still make the majority -> discovered
    assert report.controls.get("move_right") == "ArrowRight"
    assert report.controls.get("move_left") == "ArrowLeft"
    assert report.apply_errors == 1


def test_second_chance_recovers_situational_noop_keys():
    from ludus.synthetic import WallGame
    # probe order matters: ArrowRight FIRST (no-op at the wall), ArrowLeft
    # second (works) -> first pass drops ArrowRight; second chance scrambles
    # with ArrowLeft and recovers it.
    report = ControlProber(WallGame, probe_keys=["ArrowRight", "ArrowLeft"],
                           ambient_window_s=0.0).probe()
    assert report.controls.get("move_left") == "ArrowLeft"
    assert report.controls.get("move_right") == "ArrowRight"
    assert report.second_chance == ["ArrowRight"]


def test_second_chance_skipped_when_nothing_effective():
    # CounterGame arrows are ALWAYS no-ops; with no effective keys there is
    # no scrambler, so no second chance and no infinite retry.
    from ludus.synthetic import CounterGame
    report = ControlProber(CounterGame, probe_keys=["ArrowLeft", "ArrowRight"],
                           ambient_window_s=0.0).probe()
    assert report.controls == {}
    assert report.second_chance == []
