"""Transition dataset for world-model induction.

One Transition per key press: the full raw state before and after (the
induction target is `predict(before, action) ~= after`). JSONL on disk so
exploration can append across sessions. Train/holdout split is by EPISODE —
within-episode transitions are heavily correlated, so a within-episode split
would leak and inflate validation accuracy.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from pydantic import BaseModel


class Transition(BaseModel):
    game_id: str
    episode_id: str
    step: int
    action: str          # semantic action name (from GameProfile.controls)
    key: str             # the raw key actually pressed
    before: dict
    after: dict
    terminal_after: bool = False


class TransitionStore:
    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)

    def append(self, t: Transition) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a") as f:
            f.write(t.model_dump_json() + "\n")

    def load(self) -> list[Transition]:
        if not self._path.exists():
            return []
        out = []
        with self._path.open() as f:
            for line in f:
                if line.strip():
                    out.append(Transition.model_validate_json(line))
        return out

    def split(self, holdout_frac: float, seed: int) -> tuple[list[Transition], list[Transition]]:
        """Episode-keyed split: whole episodes go to one side or the other."""
        transitions = self.load()
        episodes = sorted({t.episode_id for t in transitions})
        rng = random.Random(seed)
        rng.shuffle(episodes)
        n_hold = max(1, round(len(episodes) * holdout_frac)) if episodes else 0
        holdout_eps = set(episodes[:n_hold])
        train = [t for t in transitions if t.episode_id not in holdout_eps]
        holdout = [t for t in transitions if t.episode_id in holdout_eps]
        return train, holdout
