"""GameWorld driving client.

FakeGameWorld is the offline test double (no browser). GameWorldClient (Phase 5)
wraps the REAL GameWorld API discovered in docs/DISCOVERY.md and exposes the same
three methods: screenshot() -> bytes, metrics(keys) -> dict, apply(command) -> None.
"""


class GameWorldClient:
    """Real GameWorld driving client. Methods to be filled from docs/DISCOVERY.md after Phase 0.
    Same interface as FakeGameWorld: screenshot()->bytes, metrics(keys)->dict, apply(command)->None."""

    def __init__(self, game_id: str, timing_ms: int, headless: bool = True) -> None:
        self.game_id = game_id
        self.timing_ms = timing_ms
        self.headless = headless

    def _not_ready(self):
        raise NotImplementedError(
            "GameWorldClient not yet implemented — fill from docs/DISCOVERY.md (Phase 0). Use --fake for now."
        )

    def screenshot(self) -> bytes:
        self._not_ready()

    def metrics(self, keys: list[str]) -> dict[str, float]:
        self._not_ready()

    def apply(self, command: dict) -> None:
        self._not_ready()


class FakeGameWorld:
    def __init__(self, metric_script: list[dict[str, float]]) -> None:
        self._metrics = metric_script
        self._i = 0
        self.applied: list[dict] = []

    def screenshot(self) -> bytes:
        return b"\x89PNG fake-frame"

    def metrics(self, keys: list[str]) -> dict[str, float]:
        m = self._metrics[min(self._i, len(self._metrics) - 1)]
        return {k: m.get(k, 0.0) for k in keys}

    def apply(self, command: dict) -> None:
        self.applied.append(command)
        self._i = min(self._i + 1, len(self._metrics) - 1)
