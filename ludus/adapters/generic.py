# ludus/adapters/generic.py
"""GenericAdapter — GameAdapter for machine-generated GameProfiles.

Identical shape to the hand-written adapters (see tetris.py): mapping +
metadata only, zero strategy. The difference: it needs no per-game class,
so onboarded games are playable without writing code."""

from ludus.config import GameConfig
from ludus.schemas import ActionArgs


class GenericAdapter:
    def __init__(self, cfg: GameConfig) -> None:
        self._cfg = cfg
        self.name = cfg.name
        self.objective = cfg.objective
        self.legal_actions = cfg.legal_actions
        self.primary_metric = cfg.primary_metric
        self.higher_is_better = cfg.higher_is_better
        self.relevant_metrics = cfg.relevant_metrics
        self.timing_ms = cfg.timing_ms

    def semantic_to_gameworld(self, action: str, args: ActionArgs) -> dict:
        return {"key": self._cfg.control_map[action], "duration_ms": args.duration_ms}
