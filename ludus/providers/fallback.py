import logging
from ludus.schemas import Decision, PlannerContext
from ludus.providers.base import ModelProvider

log = logging.getLogger("ludus.providers")


class FallbackProvider:
    name = "fallback"

    def __init__(self, providers: list[ModelProvider]) -> None:
        if not providers:
            raise ValueError("FallbackProvider needs at least one provider")
        self._providers = providers

    def available(self) -> bool:
        return any(p.available() for p in self._providers)

    def decide(self, context: PlannerContext) -> Decision:
        last_exc: Exception | None = None
        for p in self._providers:
            if not p.available():
                continue
            try:
                return p.decide(context)
            except Exception as exc:  # provider-level failure → try next
                log.warning("provider %s failed: %s", p.name, exc)
                last_exc = exc
        raise RuntimeError(f"all providers failed (last: {last_exc})")
