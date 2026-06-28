import base64
import os
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
from ludus.schemas import Decision, PlannerContext
from ludus.providers.insforge import build_prompt, parse_decision


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str = "claude-opus-4-8") -> None:
        self._key = os.environ.get("ANTHROPIC_API_KEY", "")
        self._model = model

    def available(self) -> bool:
        return bool(self._key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, max=4))
    def decide(self, ctx: PlannerContext) -> Decision:
        b64 = base64.b64encode(ctx.screenshot_png).decode()
        payload = {
            "model": self._model,
            "max_tokens": 1024,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                    {"type": "text", "text": build_prompt(ctx)},
                ],
            }],
        }
        r = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": self._key, "anthropic-version": "2023-06-01"},
            json=payload, timeout=60,
        )
        r.raise_for_status()
        return parse_decision(r.json()["content"][0]["text"])
