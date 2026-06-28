from typing import Protocol, runtime_checkable
from ludus.schemas import Decision, PlannerContext


@runtime_checkable
class ModelProvider(Protocol):
    name: str
    def available(self) -> bool: ...
    def decide(self, context: PlannerContext) -> Decision: ...
