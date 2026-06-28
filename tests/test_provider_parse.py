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


def _decision_json(confidence) -> str:
    return ('{"scene_summary":"s","controlled_entity":"c","current_subgoal":"g",'
            '"action":"a","action_args":{"duration_ms":100},"expected_result":"e",'
            f'"reason":"r","confidence":{confidence}}}')


def test_parse_decision_coerces_percentage_confidence():
    # Models that emit 0-100 confidence (e.g. 95) get scaled to 0.95.
    d = parse_decision(_decision_json(95))
    assert d.confidence == 0.95


def test_parse_decision_clamps_slightly_over_one_confidence():
    # A value just over 1 (3.0) clamps to the [0,1] ceiling.
    d = parse_decision(_decision_json(3.0))
    assert d.confidence == 1.0


def test_build_prompt_includes_state_text_when_present():
    ctx = PlannerContext(objective="win", legal_actions=["a"],
        state_text="col heights: 4 0 2\nholes=1")
    text = build_prompt(ctx)
    assert "Game state:" in text and "col heights: 4 0 2" in text and "holes=1" in text


def test_build_prompt_omits_state_section_when_empty():
    ctx = PlannerContext(objective="win", legal_actions=["a"])
    assert "Game state:" not in build_prompt(ctx)
