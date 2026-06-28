from pathlib import Path
from ludus.schemas import StepRecord, EpisodeResult


class LocalStore:
    name = "local"

    def __init__(self, root: Path | str = "runs") -> None:
        self._root = Path(root)

    def _dir(self, episode_id: str) -> Path:
        d = self._root / episode_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_screenshot(self, episode_id: str, step_index: int, png: bytes) -> str:
        rel = Path(episode_id) / f"step_{step_index:04d}.png"
        path = self._root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(png)
        return str(rel)

    def save_step(self, step: StepRecord) -> None:
        path = self._dir(step.episode_id) / "steps.jsonl"
        with path.open("a") as f:
            f.write(step.model_dump_json() + "\n")

    def save_episode(self, result: EpisodeResult) -> None:
        path = self._dir(result.episode_id) / "episode.json"
        path.write_text(result.model_dump_json(indent=2))
