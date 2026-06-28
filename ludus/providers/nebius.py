import base64
import os
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
from ludus.schemas import Decision, PlannerContext
from ludus.providers.insforge import _SYSTEM, build_prompt, build_user_text, parse_decision


def _is_multimodal() -> bool:
    """Whether to send the screenshot. Default multimodal unless NEBIUS_MULTIMODAL
    is explicitly falsey ("0"/"false"/"no"/"off"/""), so a non-multimodal student
    (e.g. the LoRA-fine-tuned Llama-3.2-3B-Instruct) gets a text-only prompt."""
    val = os.environ.get("NEBIUS_MULTIMODAL")
    if val is None:
        return True
    return val.strip().lower() not in {"0", "false", "no", "off", ""}


class NebiusProvider:
    """Nebius AI Studio runtime (OpenAI-compatible /chat/completions).

    Runtime seam for the model fine-tuned by the distillation pipeline. Two modes,
    selected by NEBIUS_MULTIMODAL:
      - multimodal (default): VLM path — text prompt + screenshot data URI (vision
        SFT, data/tetris/nebius_sft.jsonl).
      - text-only (NEBIUS_MULTIMODAL falsey): a non-multimodal student — system +
        user STRING messages, NO image. The board reaches the model via state_text
        in build_user_text (text SFT, data/tetris/nebius_text_*.jsonl).
    Either way the loop drives the student with the SAME parse_decision contract
    every other provider uses; point NEBIUS_MODEL at the fine-tuned model id.
    """

    name = "nebius"

    def __init__(self, model: str | None = None) -> None:
        self._base = os.environ.get("NEBIUS_BASE_URL", "https://api.studio.nebius.com/v1")
        self._key = os.environ.get("NEBIUS_API_KEY", "")
        self._model = model or os.environ.get("NEBIUS_MODEL", "")

    def available(self) -> bool:
        return bool(self._key and self._model)

    def _messages(self, ctx: PlannerContext) -> list[dict]:
        if _is_multimodal():
            b64 = base64.b64encode(ctx.screenshot_png).decode()
            return [{
                "role": "user",
                "content": [
                    {"type": "text", "text": build_prompt(ctx)},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }]
        # Text-only: separate system + user STRING messages, no image. The student
        # trains on exactly this prompt (export_text -> nebius_text_*.jsonl).
        user_text = build_user_text(
            objective=ctx.objective,
            legal_actions=ctx.legal_actions,
            state_text=ctx.state_text,
            recent_outcomes=ctx.recent_outcomes,
            partner_recent_actions=ctx.partner_recent_actions,
            learned_rules=ctx.learned_rules,
        )
        return [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user_text},
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
