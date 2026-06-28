from ludus.config import GameConfig
from ludus.schemas import ActionArgs


class FireboyWatergirlAdapter:
    """Thin adapter for the co-op game — mapping + metadata only, NO strategy.

    The agent controls WATERGIRL (keys w/a/d). Fireboy's arrow keys are the human
    partner's and are NOT in our control_map. `wait_for_partner` is a co-op no-op:
    it maps to an empty key so the loop spends a step observing (and draining the
    partner buffer) without injecting any Watergirl input.
    """

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
        if action == "wait_for_partner":
            return {"key": "", "duration_ms": args.duration_ms}
        return {"key": self._cfg.control_map[action], "duration_ms": args.duration_ms}
