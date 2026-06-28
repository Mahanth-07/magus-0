# tests/test_partner_context.py
from ludus.schemas import PlannerContext
from ludus.providers.insforge import build_prompt


def test_prompt_includes_partner_actions():
    ctx = PlannerContext(objective="co-op", legal_actions=["move_right", "wait_for_partner"],
        partner_recent_actions=["fireboy: move_right", "fireboy: jump"])
    text = build_prompt(ctx)
    assert "fireboy: jump" in text
