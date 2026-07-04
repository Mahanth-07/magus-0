# ludus/onboarding/diff.py
"""State flatten/diff primitives for onboarding.

flatten_state turns a nested getState()-shaped dict into {dot.path: scalar}.
Lists up to MAX_INDEXED_LIST elements are indexed ([i]); longer lists (board
grids) are summarized as `path.__len__` only — element-level noise from a
20x10 grid would swamp change attribution.
"""

from __future__ import annotations

MAX_INDEXED_LIST = 16

_SCALARS = (str, int, float, bool)


def flatten_state(value, prefix: str = "") -> dict:
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
        b, a = before.get(path), after.get(path)
        if b != a:
            out[path] = (b, a)
    return out
