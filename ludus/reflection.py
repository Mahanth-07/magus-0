from ludus.schemas import Decision
from ludus.outcome import Outcome


class Reflector:
    """Metric-based reflection: turn a confident prediction-error into a compact rule.

    Model-based reflection (an extra LLM call to phrase the rule) is an optional
    enhancement; the spec's cut order falls back to exactly this metric-based path.
    """

    def __init__(self, min_confidence: float = 0.4) -> None:
        self._min_confidence = min_confidence

    def reflect(self, decision: Decision, outcome: Outcome, game: str) -> str | None:
        if outcome.improved:
            return None
        if decision.confidence < self._min_confidence:
            return None
        return (
            f"[{game}] In subgoal '{decision.current_subgoal}', action "
            f"'{decision.action}' did not improve {outcome.primary_metric}; "
            f"try an alternative."
        )
