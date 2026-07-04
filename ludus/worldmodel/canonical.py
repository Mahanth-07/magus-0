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
