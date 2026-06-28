from ludus.schemas import ActionArgs, Decision, PlannerContext


class MockProvider:
    name = "mock"

    def __init__(self, script: list[str] | None = None) -> None:
        self._script = script or []
        self._i = 0

    def available(self) -> bool:
        return True

    def decide(self, context: PlannerContext) -> Decision:
        if self._script:
            action = self._script[self._i % len(self._script)]
            self._i += 1
        else:
            action = context.legal_actions[0]
        return Decision(
            scene_summary="mock scene",
            controlled_entity="mock entity",
            current_subgoal="mock subgoal",
            action=action,
            action_args=ActionArgs(duration_ms=100),
            expected_result="mock expected result",
            reason="mock reason",
            confidence=0.5,
        )
