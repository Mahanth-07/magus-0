from ludus.schemas import PlannerContext
from ludus.providers.insforge import build_prompt, parse_decision


def test_build_prompt_includes_context():
    ctx = PlannerContext(objective="win", legal_actions=["a", "b"],
        recent_outcomes=["a -> ok"], learned_rules="- do x")
    text = build_prompt(ctx)
    assert "win" in text and "a" in text and "do x" in text


def test_parse_decision_from_fenced_json():
    raw = '```json\n{"scene_summary":"s","controlled_entity":"c","current_subgoal":"g",' \
          '"action":"a","action_args":{"duration_ms":100},"expected_result":"e",' \
          '"reason":"r","confidence":0.7}\n```'
    d = parse_decision(raw)
    assert d.action == "a" and d.confidence == 0.7
