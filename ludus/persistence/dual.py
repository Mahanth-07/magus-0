import logging
from ludus.schemas import StepRecord, EpisodeResult
from ludus.persistence.base import Store

log = logging.getLogger("ludus.persistence")


class DualWriteStore:
    """Writes to primary (local — must succeed) and secondary (InsForge — best-effort)."""

    name = "dual"

    def __init__(self, primary: Store, secondary: Store) -> None:
        self._primary = primary
        self._secondary = secondary

    def _try_secondary(self, fn) -> None:
        try:
            fn(self._secondary)
        except Exception as exc:
            log.warning("secondary store failed: %s", exc)

    def save_screenshot(self, episode_id: str, step_index: int, png: bytes) -> str:
        ref = self._primary.save_screenshot(episode_id, step_index, png)
        self._try_secondary(lambda s: s.save_screenshot(episode_id, step_index, png))
        return ref

    def save_step(self, step: StepRecord) -> None:
        self._primary.save_step(step)
        self._try_secondary(lambda s: s.save_step(step))

    def save_episode(self, result: EpisodeResult) -> None:
        self._primary.save_episode(result)
        self._try_secondary(lambda s: s.save_episode(result))
