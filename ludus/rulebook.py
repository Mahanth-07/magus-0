class Rulebook:
    def __init__(self, max_rules: int = 20) -> None:
        self._rules: list[str] = []
        self._max = max_rules

    def add(self, rule: str | None) -> bool:
        if not rule or rule in self._rules:
            return False
        self._rules.append(rule)
        if len(self._rules) > self._max:
            self._rules.pop(0)
        return True

    def rules(self) -> list[str]:
        return list(self._rules)

    def render(self) -> str:
        return "\n".join(f"- {r}" for r in self._rules)
