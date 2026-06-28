from typing import Protocol, runtime_checkable
from ludus.schemas import ActionArgs


@runtime_checkable
class GameAdapter(Protocol):
    name: str
    objective: str
    legal_actions: list[str]
    primary_metric: str
    higher_is_better: bool
    timing_ms: int
    def semantic_to_gameworld(self, action: str, args: ActionArgs) -> dict: ...
