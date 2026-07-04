# ludus/onboarding/prober.py
"""ControlProber — discover a game's controls by pressing keys and diffing state.

Algorithm per probe() run:
  1. Ambient baseline: take consecutive raw_state() snapshots with NO input;
     any path that changes is "ambient" (gravity, timers, animations) and is
     never attributed to a key. metrics.steps-style tick counters are also
     excluded per-press (they change on every apply).
  2. Per key, `repeats` times: snapshot -> press -> snapshot -> diff. A path
     counts as a key effect if it changes in a majority of repeats and is not
     ambient. If the game leaves "playing" status (quit/reset keys, game
     over), relaunch via the client factory and continue.
  3. Heuristic semantic naming: player-position effects become move_left/
     right/up/down (screen convention: y grows downward); any other consistent
     effect becomes use_<key>. Keys with no consistent effect are dropped.

The prober needs a client FACTORY (not an instance) precisely because probing
can kill a game session. Works on anything with raw_state() + apply() —
GameWorldClient and the synthetic games alike.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ludus.onboarding.diff import diff_states, flatten_state

# GameWorld games draw from a small key vocabulary (docs/DISCOVERY.md + the
# three hand-written configs). Order matters: earlier keys claim semantic
# names first on collision.
DEFAULT_PROBE_KEYS = [
    "ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown",
    "Space", "x", "z", "a", "d", "w", "s",
]

# Tick counters change on every apply(); never attribute them to a key.
_TICK_HINTS = ("steps", "frame", "tick", "time")


@dataclass
class ProbeReport:
    controls: dict[str, str] = field(default_factory=dict)         # semantic -> key
    control_effects: dict[str, list[str]] = field(default_factory=dict)
    state_schema: dict[str, str] = field(default_factory=dict)     # path -> type name
    ambient_paths: list[str] = field(default_factory=list)
    relaunches: int = 0


def _status(state: dict) -> str:
    return str(state.get("status", "playing"))


def _is_tick(path: str) -> bool:
    leaf = path.rsplit(".", 1)[-1].lower()
    return any(h in leaf for h in _TICK_HINTS)


class ControlProber:
    def __init__(
        self,
        client_factory,
        probe_keys: list[str] | None = None,
        repeats: int = 3,
        duration_ms: int = 80,
    ) -> None:
        self._factory = client_factory
        self._keys = list(probe_keys or DEFAULT_PROBE_KEYS)
        self._repeats = repeats
        self._duration_ms = duration_ms

    def probe(self) -> ProbeReport:
        report = ProbeReport()
        client = self._factory()
        try:
            client, ambient = self._ambient_baseline(client, report)
            report.ambient_paths = sorted(ambient)
            report.state_schema = {
                p: type(v).__name__
                for p, v in flatten_state(client.raw_state()).items()
            }

            key_effects: dict[str, dict[str, list]] = {}
            for key in self._keys:
                client, effects = self._probe_key(client, key, ambient, report)
                if effects:
                    key_effects[key] = effects

            self._name_controls(key_effects, report)
            return report
        finally:
            if hasattr(client, "close"):
                client.close()

    # -- phases ---------------------------------------------------------------

    def _relaunch(self, client, report: ProbeReport):
        if hasattr(client, "close"):
            client.close()
        report.relaunches += 1
        return self._factory()

    def _ambient_baseline(self, client, report: ProbeReport):
        """Paths that change with NO input across `repeats` snapshot pairs."""
        ambient: set[str] = set()
        for _ in range(self._repeats):
            if _status(client.raw_state()) != "playing":
                client = self._relaunch(client, report)
            before = flatten_state(client.raw_state())
            after = flatten_state(client.raw_state())
            ambient |= set(diff_states(before, after))
        return client, ambient

    def _probe_key(self, client, key: str, ambient: set, report: ProbeReport):
        """Press `key` `repeats` times; return {path: [deltas]} for paths that
        changed in a majority of presses (ambient/tick paths excluded)."""
        changed_counts: dict[str, int] = {}
        deltas: dict[str, list] = {}
        for _ in range(self._repeats):
            if _status(client.raw_state()) != "playing":
                client = self._relaunch(client, report)
            before = flatten_state(client.raw_state())
            client.apply({"key": key, "duration_ms": self._duration_ms})
            after = flatten_state(client.raw_state())
            for path, (b, a) in diff_states(before, after).items():
                if path in ambient or _is_tick(path) or path == "status":
                    continue
                changed_counts[path] = changed_counts.get(path, 0) + 1
                if isinstance(b, (int, float)) and isinstance(a, (int, float)):
                    deltas.setdefault(path, []).append(a - b)
                else:
                    deltas.setdefault(path, []).append(None)
        majority = math.ceil(self._repeats / 2)
        return client, {
            p: deltas[p] for p, n in changed_counts.items() if n >= majority
        }

    # -- naming -----------------------------------------------------------------

    @staticmethod
    def _movement_name(effects: dict[str, list]) -> str | None:
        """move_left/right/up/down if the dominant effect is a player-position
        axis with a consistent sign. Screen convention: y grows DOWNWARD."""
        for axis, neg, pos in (("x", "move_left", "move_right"),
                               ("y", "move_up", "move_down")):
            paths = sorted(p for p in effects
                           if "player" in p and p.endswith(f".{axis}"))
            for p in paths:
                nums = [d for d in effects[p] if isinstance(d, (int, float))]
                if nums and all(d < 0 for d in nums):
                    return neg
                if nums and all(d > 0 for d in nums):
                    return pos
        return None

    def _name_controls(self, key_effects: dict, report: ProbeReport) -> None:
        for key in self._keys:  # deterministic claim order
            effects = key_effects.get(key)
            if not effects:
                continue
            name = self._movement_name(effects) or f"use_{key.lower()}"
            if name in report.controls:  # collision: later key gets suffixed
                name = f"{name}_{key.lower()}"
            report.controls[name] = key
            report.control_effects[name] = sorted(effects)
