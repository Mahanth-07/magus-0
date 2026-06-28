from pathlib import Path
import yaml
from pydantic import BaseModel


class GameConfig(BaseModel):
    name: str
    objective: str
    legal_actions: list[str]
    primary_metric: str
    higher_is_better: bool
    timing_ms: int
    control_map: dict[str, str]
    relevant_metrics: list[str]


def load_game_config(path: Path | str) -> GameConfig:
    data = yaml.safe_load(Path(path).read_text())
    return GameConfig(**data)
