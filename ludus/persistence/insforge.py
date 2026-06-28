import logging
import os
import httpx
from ludus.schemas import StepRecord, EpisodeResult

log = logging.getLogger("ludus.persistence")


def step_to_row(step: StepRecord) -> dict:
    return {
        "episode_id": step.episode_id, "step_index": step.step_index,
        "mode": step.mode, "game": step.game, "action": step.decision.action,
        "expected_result": step.decision.expected_result,
        "primary_metric": step.primary_metric, "primary_delta": step.primary_delta,
        "improved": step.improved, "metric_delta": step.metric_delta,
        "rule_added": step.rule_added, "screenshot_ref": step.screenshot_ref,
        "confidence": step.decision.confidence,
    }


class InsForgeStore:
    """Episode traces in InsForge Postgres + screenshots in InsForge Storage.

    Real API (discovered live; see docs/DISCOVERY.md InsForge section):
      - insert rows: POST {base}/api/database/records/{table}  (body MUST be a JSON array)
      - upload file: PUT  {base}/api/storage/buckets/{bucket}/objects/{key}  (multipart `file`)
    Used as the best-effort secondary in DualWriteStore, so failures never block a run.
    """

    name = "insforge"

    def __init__(self) -> None:
        self._base = os.environ.get("INSFORGE_BASE_URL", "").rstrip("/")
        self._key = os.environ.get("INSFORGE_API_KEY", "")
        self._bucket = os.environ.get("INSFORGE_S3_BUCKET", "")
        self._json_h = {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}
        self._key_h = {"x-api-key": self._key}

    def _insert(self, table: str, rows: list[dict]) -> None:
        r = httpx.post(f"{self._base}/api/database/records/{table}",
                       headers=self._json_h, json=rows, timeout=20)
        r.raise_for_status()

    def save_screenshot(self, episode_id: str, step_index: int, png: bytes) -> str:
        key = f"{episode_id}/step_{step_index:04d}.png"
        r = httpx.put(
            f"{self._base}/api/storage/buckets/{self._bucket}/objects/{key}",
            headers=self._key_h,
            files={"file": (f"step_{step_index:04d}.png", png, "image/png")},
            timeout=30,
        )
        r.raise_for_status()
        return key

    def save_step(self, step: StepRecord) -> None:
        self._insert("ludus_steps", [step_to_row(step)])

    def save_episode(self, result: EpisodeResult) -> None:
        self._insert("ludus_episodes", [result.model_dump()])
