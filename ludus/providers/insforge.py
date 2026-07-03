import base64
import json
import os
import re
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
from ludus.schemas import Decision, PlannerContext

_SYSTEM = (
    "You control a game via SEMANTIC actions. Look at the screenshot and decide what to do. "
    "Every action you use MUST come from the legal list. Predict the result before acting.\n"
    "You may return an ordered \"actions\" array of 1-5 moves that are applied in sequence in "
    "one turn. Use it to POSITION the controlled entity (e.g. move/rotate it into place) BEFORE "
    "committing with a terminal/irreversible move. Prefer positioning over an immediate terminal "
    "action: do not commit until the entity is where you want it. Put any terminal move LAST. "
    "Set \"action\" to the single most important move of the turn (the primary move).\n"
    "Respond with ONLY a JSON object with keys: scene_summary, controlled_entity, "
    "current_subgoal, action, actions, action_args, expected_result, reason, confidence. "
    "\"action\" is a single legal action string; \"actions\" is an ordered list of legal action "
    "strings (may be empty if a single action suffices)."
)


def build_user_text(
    objective: str,
    legal_actions: list[str],
    state_text: str = "",
    recent_outcomes: list[str] | None = None,
    partner_recent_actions: list[str] | None = None,
    learned_rules: str = "",
) -> str:
    """The user-turn text (everything except the _SYSTEM preamble and the image).

    Factored out so the offline SFT export (scripts/export_nebius.py) can build the
    EXACT same user text the runtime sends to the model — the student must train on
    the same prompt it will see at inference, minus the live screenshot (which the
    export embeds as a data URI instead).
    """
    recent_outcomes = recent_outcomes or []
    partner_recent_actions = partner_recent_actions or []
    state_section = f"Game state:\n{state_text}\n" if state_text else ""
    # When the state provides pre-simulated Candidate placements (Tetris lookahead),
    # add a final, unambiguous directive nearest the model's output: copy candidate [0]'s
    # actions verbatim. This is where the model has the strongest recency bias, so it
    # reliably overrides any urge to improvise its own moves.
    candidate_directive = ""
    if "Candidate placements" in (state_text or ""):
        candidate_directive = (
            "IMPORTANT: The candidates above are pre-computed and sorted best-first. "
            "Set \"actions\" to candidate [0]'s actions array EXACTLY as written "
            "(copy every element, including each \"rotate\"), and set \"action\" to its "
            "last element. Do not improvise or omit rotates.\n"
        )
    return (
        f"Objective: {objective}\n"
        f"Legal actions: {', '.join(legal_actions)}\n"
        + state_section
        + f"Recent outcomes:\n" + ("\n".join(recent_outcomes) or "(none)") + "\n"
        f"Partner recent actions:\n" + ("\n".join(partner_recent_actions) or "(none)") + "\n"
        f"Learned rules:\n{learned_rules or '(none)'}\n"
        + candidate_directive
    )


def build_prompt(ctx: PlannerContext) -> str:
    return f"{_SYSTEM}\n\n" + build_user_text(
        objective=ctx.objective,
        legal_actions=ctx.legal_actions,
        state_text=ctx.state_text,
        recent_outcomes=ctx.recent_outcomes,
        partner_recent_actions=ctx.partner_recent_actions,
        learned_rules=ctx.learned_rules,
    )


def _coerce_confidence(data: dict) -> dict:
    """Normalize a model-emitted confidence into [0.0, 1.0].

    Stronger models sometimes return 0-100 scale (e.g. 95) or slightly-over-1
    values. If confidence is a number > 1 and <= 100, treat it as a percentage
    (divide by 100); then clamp into [0.0, 1.0]. Non-numeric/missing values are
    left untouched so Pydantic surfaces the original error.
    """
    conf = data.get("confidence")
    if isinstance(conf, (int, float)) and not isinstance(conf, bool):
        # A value clearly on a 0-100 scale (>= 10) is treated as a percentage and
        # divided by 100 (e.g. 95 -> 0.95). Values only slightly over 1 (e.g. 3.0)
        # are a model overshoot, not a percentage, so they just clamp to 1.0.
        if 10 <= conf <= 100:
            conf = conf / 100.0
        data["confidence"] = max(0.0, min(1.0, float(conf)))
    return data


def parse_decision(raw: str) -> Decision:
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        raise ValueError(f"no JSON object in model output: {raw[:200]}")
    return Decision.model_validate(_coerce_confidence(json.loads(m.group(0))))


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
