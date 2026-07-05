# ludus/onboarding/diff.py
"""State flatten/diff primitives for onboarding.

flatten_state turns a nested getState()-shaped dict into {dot.path: scalar}.
Lists up to MAX_INDEXED_LIST elements are indexed ([i]); longer lists (board
grids) are summarized as `path.__len__` only — element-level noise from a
20x10 grid would swamp change attribution.
"""

from __future__ import annotations

MAX_INDEXED_LIST = 16

# Tick/wall-clock fields change on every frame and are unknowable to a
# transition function — they are not game rules.  Shared by prober.py
# (control attribution) and validate.py (grading parity).
TICK_HINTS = ("steps", "frame", "tick", "time")


def is_tick_path(path: str) -> bool:
    """Return True if *path* looks like a tick counter or wall-clock field."""
    leaf = path.rsplit(".", 1)[-1].lower()
    return any(h in leaf for h in TICK_HINTS)

_SCALARS = (str, int, float, bool)
_MISSING = object()


def flatten_state(value, prefix: str = "") -> dict:
    """Recursively flatten `value` to {dot.path: scalar}."""
    flat: dict = {}
    if isinstance(value, dict):
        for k, v in value.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            flat.update(flatten_state(v, key))
    elif isinstance(value, list):
        if len(value) > MAX_INDEXED_LIST:
            flat[f"{prefix}.__len__"] = len(value)
        else:
            for i, v in enumerate(value):
                flat.update(flatten_state(v, f"{prefix}[{i}]"))
    elif value is None or isinstance(value, _SCALARS):
        flat[prefix] = value
    else:  # unknown object; record its repr so changes are still visible
        flat[prefix] = repr(value)
    return flat


def diff_states(before: dict, after: dict) -> dict:
    """{path: (before_value, after_value)} for changed/added/removed paths.
    Missing-side values are None."""
    out: dict = {}
    for path in before.keys() | after.keys():
        b = before.get(path, _MISSING)
        a = after.get(path, _MISSING)
        if b is _MISSING or a is _MISSING or b != a:
            out[path] = (None if b is _MISSING else b,
                         None if a is _MISSING else a)
    return out
