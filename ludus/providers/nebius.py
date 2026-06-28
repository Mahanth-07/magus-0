import base64
import os
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
from ludus.schemas import Decision, PlannerContext
from ludus.providers.insforge import build_prompt, parse_decision


class NebiusProvider:
    """Vision via Nebius AI Studio (OpenAI-compatible /chat/completions).

    This is the runtime seam for the model fine-tuned by the distillation pipeline:
    once a small VLM is LoRA-fine-tuned on data/tetris/nebius_sft.jsonl and served
    on Nebius, point NEBIUS_MODEL at it and the agent loop drives the student with
    the SAME build_prompt/parse_decision contract every other provider uses.
    """

    name = "nebius"

    def __init__(self, model: str | None = None) -> None:
        self._base = os.environ.get("NEBIUS_BASE_URL", "https://api.studio.nebius.com/v1")
        self._key = os.environ.get("NEBIUS_API_KEY", "")
        self._model = model or os.environ.get("NEBIUS_MODEL", "")

    def available(self) -> bool:
        return bool(self._key and self._model)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, max=4))
    def decide(self, ctx: PlannerContext) -> Decision:
        b64 = base64.b64encode(ctx.screenshot_png).decode()
        payload = {
            "model": self._model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": build_prompt(ctx)},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }],
            "temperature": 0,
        }
        r = httpx.post(
            f"{self._base}/chat/completions",
            headers={"Authorization": f"Bearer {self._key}"},
            json=payload, timeout=90,
        )
        r.raise_for_status()
        return parse_decision(r.json()["choices"][0]["message"]["content"])
