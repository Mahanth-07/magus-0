# ludus/worldmodel/canonical.py
"""Canonical state ordering for induction and validation.

Games expose set-valued state as LISTS (2048 tiles, entity arrays) whose
order shuffles as members move/merge/spawn. Index-aligned comparison then
scores a perfect prediction as wrong whenever order differs. Canonicalizing
— recursively sorting lists-of-dicts by their content serialization — makes
prompt diffs and validation order-insensitive without touching the games or
the induced models (which may emit entities in any order)."""

from __future__ import annotations

import json


def canonicalize(value):
    """Recursively sort lists of dicts by canonical JSON of their (already
    canonicalized) content. Scalars/other lists pass through unchanged."""
    if isinstance(value, dict):
        return {k: canonicalize(v) for k, v in value.items()}
    if isinstance(value, list):
        canon = [canonicalize(v) for v in value]
        if canon and all(isinstance(v, dict) for v in canon):
            return sorted(canon, key=lambda d: json.dumps(d, sort_keys=True,
                                                          default=str))
        return canon
    return value


def split_entities(state, prefix: str = "") -> tuple[dict, dict[str, list[str]]]:
    """Split a state into (scalar_state, entity_map).

    scalar_state: the state with every list-of-dicts REPLACED by its length
    (kept as an int at the list's key so flatten yields one countable path).
    entity_map: {dot.path: [entity_key]} where
    entity_key = json.dumps(canonicalize(entity), sort_keys=True, default=str).
    Multiset semantics: duplicate keys appear twice.
    """
    entities: dict[str, list[str]] = {}

    def walk(value, path):
        if isinstance(value, dict):
            return {k: walk(v, f"{path}.{k}" if path else str(k))
                    for k, v in value.items()}
        if isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
            entities[path] = sorted(
                json.dumps(canonicalize(v), sort_keys=True, default=str)
                for v in value)
            return len(value)   # scalar stand-in: entity count
        return value

    scalar_state = walk(state, prefix)
    return scalar_state, entities
