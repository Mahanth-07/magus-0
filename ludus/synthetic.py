# ludus/synthetic.py
"""Synthetic games — pure-Python stand-ins that speak the GameWorld client
interface (raw_state/apply/metrics/screenshot). Deterministic, no browser.

These are the CI ground truth for the onboarding pipeline (M1) and, later,
for world-model induction (M2): we KNOW the true controls/metrics/dynamics,
so tests can assert the pipeline recovers them exactly.

State dicts mimic GameWorld's getState() shape (docs/DISCOVERY.md):
top-level `status` in {"playing","terminal"}, `metrics`, nested `game_state`.
"""

from __future__ import annotations


class _SyntheticBase:
    """Shared client-interface plumbing (see GameWorldClient for the contract)."""

    def screenshot(self) -> bytes:
        return b"\x89PNG synthetic-frame"

    def metrics(self, keys: list[str]) -> dict[str, float]:
        m = self.raw_state().get("metrics", {})
        return {k: float(m.get(k, 0.0)) for k in keys}

    def read_partner_actions(self) -> list[str]:
        return []

    def state_text(self) -> str:
        return ""

    def raw_state(self) -> dict:  # pragma: no cover - overridden
        raise NotImplementedError

    def apply(self, command: dict) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def close(self) -> None:
        pass


class GridWorldGame(_SyntheticBase):
    """5x5 grid. Arrows move the player (clamped at walls). Reaching the goal
    scores +10 and the goal respawns along a fixed deterministic cycle.
    'q' ends the game (status -> terminal). Every apply() ticks `steps`."""

    SIZE = 5
    GOAL_CYCLE = [(3, 2), (0, 0), (4, 4), (0, 4), (4, 0)]
    MOVES = {"ArrowLeft": (-1, 0), "ArrowRight": (1, 0),
             "ArrowUp": (0, -1), "ArrowDown": (0, 1)}

    def __init__(self) -> None:
        self.x, self.y = 2, 2
        self._goal_i = 0
        self.score = 0
        self.steps = 0
        self.terminal = False

    @property
    def _goal(self) -> tuple[int, int]:
        return self.GOAL_CYCLE[self._goal_i % len(self.GOAL_CYCLE)]

    def apply(self, command: dict) -> None:
        if self.terminal:
            return
        self.steps += 1
        key = command.get("key")
        if key == "q":
            self.terminal = True
            return
        if key in self.MOVES:
            dx, dy = self.MOVES[key]
            self.x = min(self.SIZE - 1, max(0, self.x + dx))
            self.y = min(self.SIZE - 1, max(0, self.y + dy))
            if (self.x, self.y) == self._goal:
                self.score += 10
                self._goal_i += 1

    def raw_state(self) -> dict:
        gx, gy = self._goal
        return {
            "status": "terminal" if self.terminal else "playing",
            "metrics": {"score": self.score, "steps": self.steps},
            "game_state": {
                "player": {"x": self.x, "y": self.y},
                "environment": {"goal": {"x": gx, "y": gy}},
            },
        }


class AutoRunnerGame(_SyntheticBase):
    """Continuous-motion regression game: player.x advances with WALL-CLOCK
    time (like snake/flappy), independent of input. 'x' bumps score; all other
    keys are no-ops. Used to pin the press-window ambient-sampling fix — an
    instant-snapshot ambient baseline misses the drift and misattributes it
    to whatever key was pressed."""

    def __init__(self, drift_hz: int = 50) -> None:
        import time as _time
        self._time = _time
        self._t0 = _time.monotonic()
        self._drift_hz = drift_hz
        self.score = 0
        self.steps = 0

    def apply(self, command: dict) -> None:
        self.steps += 1
        if command.get("key") == "x":
            self.score += 1

    def raw_state(self) -> dict:
        x = int((self._time.monotonic() - self._t0) * self._drift_hz)
        return {
            "status": "playing",
            "metrics": {"score": self.score, "steps": self.steps},
            "game_state": {"player": {"x": x, "y": 0}},
        }


class WallGame(_SyntheticBase):
    """Second-chance probing repro: the player spawns AT the right wall
    (x=4), so ArrowRight is a no-op from the spawn state — exactly like
    2048's situational no-op arrows. After any ArrowLeft, ArrowRight works."""

    def __init__(self) -> None:
        self.x = 4
        self.steps = 0

    def apply(self, command: dict) -> None:
        self.steps += 1
        key = command.get("key")
        if key == "ArrowLeft":
            self.x = max(0, self.x - 1)
        elif key == "ArrowRight":
            self.x = min(4, self.x + 1)

    def raw_state(self) -> dict:
        return {"status": "playing",
                "metrics": {"score": 0, "steps": self.steps},
                "game_state": {"player": {"x": self.x, "y": 0}}}


class CounterGame(_SyntheticBase):
    """'x' increments score. 'z' costs 10 health (floor 0 = terminal).
    Arrows do nothing. Every apply() ticks `steps`."""

    def __init__(self) -> None:
        self.score = 0
        self.health = 100
        self.steps = 0

    @property
    def terminal(self) -> bool:
        return self.health <= 0

    def apply(self, command: dict) -> None:
        if self.health <= 0:
            return
        self.steps += 1
        key = command.get("key")
        if key == "x":
            self.score += 1
        elif key == "z":
            self.health = max(0, self.health - 10)

    def raw_state(self) -> dict:
        return {
            "status": "terminal" if self.health <= 0 else "playing",
            "metrics": {"score": self.score, "health": self.health,
                        "steps": self.steps},
            "game_state": {"player": {"health": self.health}},
        }
