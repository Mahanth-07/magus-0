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


def test_reset_like_keys_are_not_used_for_scrambling():
    from ludus.synthetic import WallGame

    class ResetWallGame(WallGame):
        """'r' resets the whole game (x→4 + increments many mirror fields);
        scrambling with it re-walls the player and sabotages second-chance
        probing — the 2048 Space finding."""

        def __init__(self):
            super().__init__()       # self.x = 4 (right wall)
            self._reset_count = 0

        def apply(self, command):
            if command.get("key") == "r":
                self.steps += 1
                self.x = 4           # reset to right wall
                self._reset_count += 1
                return
            super().apply(command)

        def raw_state(self):
            st = super().raw_state()
            rc = self._reset_count
            # mirror depends on reset_count (not x), so 'r' changes
            # reset_count + mirror.a/b/c/d = 5 paths every single press.
            # Schema total: status + metrics.score + metrics.steps +
            # player.x + player.y + reset_count + mirror.a/b/c/d = 10.
            # threshold = 0.5 * 10 = 5.0 → 'r': 5 < 5.0 = False → excluded.
            # ArrowLeft changes only player.x (1 path): 1 < 5.0 → included.
            st["game_state"]["reset_count"] = rc
            st["game_state"]["mirror"] = {
                "a": rc, "b": rc * 2, "c": rc * 3, "d": rc * 4,
            }
            return st

    # Probe order: 'r' first (effective, large effect set), 'ArrowRight'
    # second (no-op at the right wall x=4), 'ArrowLeft' third (effective,
    # moves player left).  Scramble for ArrowRight must use ArrowLeft but
    # NOT 'r' — otherwise reset re-walls the player and ArrowRight stays
    # at most 1/3 effective (not majority) → undiscovered.
    report = ControlProber(ResetWallGame,
                           probe_keys=["r", "ArrowRight", "ArrowLeft"],
                           ambient_window_s=0.0).probe()
    assert report.controls.get("move_right") == "ArrowRight"
    assert "ArrowRight" in report.second_chance
