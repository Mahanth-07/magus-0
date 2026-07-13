import os
import sys
from pathlib import Path

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from ludus.schemas import Decision, PlannerContext
from ludus.providers.insforge import _SYSTEM, build_user_text, parse_decision

# Reuse the EXACT image preprocessing used during training (max 512px, JPEG q80).
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.export_together import _encode_image  # noqa: E402


class TogetherProvider:
    """Together AI runtime (OpenAI-compatible /chat/completions) — VLM multimodal.

    Serves the LoRA fine-tuned Qwen3-VL-8B student trained on the multi-game
    distillation dataset (sft_v1.jsonl). Training format (from export_together.py):
      - system: plain string (Together chat API normalises arrays to strings anyway)
      - user: content array [text, image_url(JPEG data-URI)]
    Images are downscaled to max 512px longest-side at JPEG q=80 via the EXACT
    same _encode_image from scripts/export_together.py — byte-level parity with
    the training distribution.

    Env vars:
      TOGETHER_API_KEY   — required
      TOGETHER_MODEL     — required (e.g. mahanth1112_3532/Qwen3-VL-8B-...)
      TOGETHER_BASE_URL  — optional override (default https://api.together.xyz/v1)
    """

    name = "together"

    def __init__(self, model: str | None = None) -> None:
        self._base = os.environ.get("TOGETHER_BASE_URL", "https://api.together.xyz/v1")
        self._key = os.environ.get("TOGETHER_API_KEY", "")
        self._model = model or os.environ.get("TOGETHER_MODEL", "")

    def available(self) -> bool:
        return bool(self._key and self._model)

    def _messages(self, ctx: PlannerContext) -> list[dict]:
        import os
        from ludus.student.recap import condition_system
        # Gate: only append expert conditioning when TOGETHER_RECAP=1.
        # Default off — v1 model was trained without conditioning; gate it.
        recap_on = os.environ.get("TOGETHER_RECAP", "0").strip() == "1"
        system_text = _SYSTEM + condition_system("expert") if recap_on else _SYSTEM

        # System as plain string (matching OpenAI chat API standard format;
        # Together normalises content arrays to strings internally).
        user_text = build_user_text(
            objective=ctx.objective,
            legal_actions=ctx.legal_actions,
            state_text=ctx.state_text,
            recent_outcomes=ctx.recent_outcomes,
            partner_recent_actions=ctx.partner_recent_actions,
            learned_rules=ctx.learned_rules,
        )
        # Image: downscaled to max 512px JPEG — EXACT same preprocessing as training.
        data_uri = _encode_image(ctx.screenshot_png)
        return [
            {"role": "system", "content": system_text},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            },
        ]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, max=4))
    def decide(self, ctx: PlannerContext) -> Decision:
        payload = {
            "model": self._model,
            "messages": self._messages(ctx),
            "temperature": 0,
        }
        r = httpx.post(
            f"{self._base}/chat/completions",
            headers={"Authorization": f"Bearer {self._key}"},
            json=payload, timeout=90,
        )
        r.raise_for_status()
        return parse_decision(r.json()["choices"][0]["message"]["content"])
