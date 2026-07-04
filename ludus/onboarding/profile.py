# ludus/onboarding/profile.py
"""GameProfile — the machine-generated replacement for configs/*.yaml.

Produced by the onboarding pipeline (prober + metric finder), cached as plain
JSON under profiles/<game>.json, and bridged into the existing episode loop
via to_game_config(). The profile is an inspectable artifact: everything the
system believes about a game's interface, with evidence (control_effects).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from ludus.config import GameConfig


class GameProfile(BaseModel):
    game_id: str
    controls: dict[str, str]                 # semantic action -> playwright key
    control_effects: dict[str, list[str]]    # semantic action -> changed state paths (evidence)
    metrics: list[str]                       # numeric metric names (from metrics.*)
    primary_metric: str
    higher_is_better: bool = True
    timing_ms: int = 120
    objective: str
    state_schema: dict[str, str]             # flattened path -> python type name

    def to_game_config(self) -> GameConfig:
        return GameConfig(
            name=self.game_id,
            objective=self.objective,
            legal_actions=list(self.controls),
            primary_metric=self.primary_metric,
            higher_is_better=self.higher_is_better,
            timing_ms=self.timing_ms,
            control_map=dict(self.controls),
            relevant_metrics=list(self.metrics),
        )

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.model_dump_json(indent=2) + "\n")

    @classmethod
    def load(cls, path: Path | str) -> "GameProfile":
        return cls.model_validate_json(Path(path).read_text())
