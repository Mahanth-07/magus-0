import base64
import json
import os
import re
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
from ludus.schemas import Decision, PlannerContext

_SYSTEM = (
    "You control a game via SEMANTIC actions. Look at the screenshot and return ONE action. "
    "You MUST choose action from the legal list. Predict the result before acting. "
    "Respond with ONLY a JSON object with keys: scene_summary, controlled_entity, "
    "current_subgoal, action, action_args, expected_result, reason, confidence."
)


def build_prompt(ctx: PlannerContext) -> str:
    return (
        f"{_SYSTEM}\n\nObjective: {ctx.objective}\n"
        f"Legal actions: {', '.join(ctx.legal_actions)}\n"
        f"Recent outcomes:\n" + ("\n".join(ctx.recent_outcomes) or "(none)") + "\n"
        f"Partner recent actions:\n" + ("\n".join(ctx.partner_recent_actions) or "(none)") + "\n"
        f"Learned rules:\n{ctx.learned_rules or '(none)'}\n"
    )


def parse_decision(raw: str) -> Decision:
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        raise ValueError(f"no JSON object in model output: {raw[:200]}")
    return Decision.model_validate(json.loads(m.group(0)))


class InsForgeGatewayProvider:
    """Vision via InsForge OpenAI-compatible model gateway. Endpoint/shape from DISCOVERY.md."""

    name = "gateway"

    def __init__(self, model: str | None = None) -> None:
        self._base = os.environ.get("INSFORGE_GATEWAY_URL", "")
        self._key = os.environ.get("INSFORGE_API_KEY", "")
        self._model = model or os.environ.get("INSFORGE_VISION_MODEL", "")

    def available(self) -> bool:
        return bool(self._base and self._key and self._model)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, max=4))
    def decide(self, ctx: PlannerContext) -> Decision:
        b64 = base64.b64encode(ctx.screenshot_png).decode()
        # InsForge model gateway: POST {INSFORGE_GATEWAY_URL}/chat/completion (singular).
        # Vision via OpenAI-style content array (text + image_url data URI).
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
            f"{self._base}/chat/completion",
            headers={"Authorization": f"Bearer {self._key}"},
            json=payload, timeout=90,
        )
        r.raise_for_status()
        body = r.json()
        # InsForge gateway returns {"text": ...}; fall back to OpenAI {"choices":[...]} shape.
        content = body.get("text")
        if content is None:
            content = body["choices"][0]["message"]["content"]
        return parse_decision(content)
