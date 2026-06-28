import os
import httpx
from ludus.schemas import StepRecord, EpisodeResult


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
    """Postgres rows + S3 screenshots. Endpoints/auth from docs/DISCOVERY.md."""

    name = "insforge"

    def __init__(self) -> None:
        self._base = os.environ.get("INSFORGE_BASE_URL", "")
        self._key = os.environ.get("INSFORGE_API_KEY", "")
        self._bucket = os.environ.get("INSFORGE_S3_BUCKET", "")
        self._h = {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}

    def save_screenshot(self, episode_id: str, step_index: int, png: bytes) -> str:
        key = f"{episode_id}/step_{step_index:04d}.png"
        # CONFIRM against DISCOVERY.md (InsForge section): presigned PUT endpoint path, auth header, bucket param.
        # httpx.put(<presigned_url>, content=png, timeout=30).raise_for_status()
        return key

    def save_step(self, step: StepRecord) -> None:
        # CONFIRM against DISCOVERY.md (InsForge section): PostgREST path (/rest/v1/ vs SDK), auth header shape.
        httpx.post(f"{self._base}/rest/v1/ludus_steps", headers=self._h,
                   json=step_to_row(step), timeout=20).raise_for_status()

    def save_episode(self, result: EpisodeResult) -> None:
        # CONFIRM against DISCOVERY.md (InsForge section): PostgREST path (/rest/v1/ vs SDK), auth header shape.
        httpx.post(f"{self._base}/rest/v1/ludus_episodes", headers=self._h,
                   json=result.model_dump(), timeout=20).raise_for_status()
