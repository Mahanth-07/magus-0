import os

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from ludus.schemas import Decision, PlannerContext
from ludus.providers.insforge import _SYSTEM, build_user_text, parse_decision


class FireworksProvider:
    """Fireworks AI runtime (OpenAI-compatible /chat/completions), TEXT-only.

    Runtime seam for the LoRA-fine-tuned Tetris distillation student served
    serverless on Fireworks (base llama-v3.2-3b-instruct; see
    scripts/fireworks_finetune.py). The student is a small TEXT model — there is
    no vision path here, so this provider ALWAYS sends system + user STRING
    messages and NO image. The board reaches the model via state_text in
    build_user_text, exactly the prompt the student trained on
    (data/tetris/nebius_text_*.jsonl). Point FIREWORKS_MODEL at the served model
    id (e.g. accounts/mahanthk7/models/ludus-tetris). The loop drives the student
    with the SAME parse_decision contract every other provider uses.
    """

    name = "fireworks"

    def __init__(self, model: str | None = None) -> None:
        self._base = os.environ.get(
            "FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1"
        )
        self._key = os.environ.get("FIREWORKS_API_KEY", "")
        self._model = model or os.environ.get("FIREWORKS_MODEL", "")

    def available(self) -> bool:
        return bool(self._key and self._model)

    def _messages(self, ctx: PlannerContext) -> list[dict]:
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
