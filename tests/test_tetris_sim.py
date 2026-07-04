"""Tests for the simulation-based distillation collector (no browser).

The sim collector must produce examples whose prompt is rendered by the SAME
function the live client uses (_tetris_state_text) and whose label is exactly
candidate [0] of the prompt's own candidate list — the invariant the student
is trained on ("copy candidate [0] verbatim").
"""

import ast
import json
import re

from ludus.gameworld_client import _tetris_state_text
from ludus.teacher.tetris_sim import simulate_examples, synthetic_state
from ludus.teacher.tetris_heuristic import COLS, ROWS

OBJECTIVE = "Tetris. Copy candidate [0]."
LEGAL = ["left", "right", "rotate", "drop"]


def _stack_with_heights(heights):
    b = [[0] * COLS for _ in range(ROWS)]
    for c, h in enumerate(heights):
        for r in range(ROWS - h, ROWS):
            b[r][c] = 1
    return b


# --------------------------------------------------------------------------
# synthetic_state — a getState()-shaped dict the live renderer accepts
# --------------------------------------------------------------------------

def test_synthetic_state_renders_via_live_state_text():
    stack = _stack_with_heights([0, 0, 2, 3, 0, 0, 1, 0, 0, 0])
    state = synthetic_state(stack, "t", preview=["i", "o", "s"], score=1200, level=2)
    text = _tetris_state_text(state)
    assert "Candidate placements" in text
    assert f"Board {COLS}x{ROWS}" in text
    assert "current=t" in text
    assert "next=[i,o,s]" in text
    assert "score=1200" in text


def test_synthetic_state_board_includes_active_piece_but_stack_analysis_excludes_it():
    # heights line must describe the SETTLED stack only (spawned piece stripped)
    stack = _stack_with_heights([0] * COLS)
    state = synthetic_state(stack, "o", preview=["i"], score=0, level=1)
    text = _tetris_state_text(state)
    # empty stack -> all-zero heights even though the active o-piece is on the grid
    heights_line = text.splitlines()[2]
    assert heights_line.split() == ["0"] * COLS


# --------------------------------------------------------------------------
# simulate_examples — the collector loop
# --------------------------------------------------------------------------

def _candidate0_actions_from_prompt(state_text: str) -> list[str]:
    m = re.search(r"\[0\][^\n]*actions=(\[[^\]]*\])", state_text)
    assert m, f"no candidate [0] line in:\n{state_text}"
    return ast.literal_eval(m.group(1))


def test_simulate_examples_label_is_candidate0_of_its_own_prompt():
    records = list(simulate_examples(
        30, epsilon=0.3, seed=7, objective=OBJECTIVE,
        legal_actions=LEGAL, duration_ms=120,
    ))
    assert len(records) == 30
    for rec in records:
        assert rec["target"]["actions"] == _candidate0_actions_from_prompt(rec["state_text"])
        assert rec["target"]["actions"][-1] == "drop"
        assert rec["target"]["action"] == rec["target"]["actions"][-1]


def test_simulate_examples_off_policy_moves_come_from_the_displayed_menu():
    # Recovery states must be MILDLY suboptimal: every off-policy application
    # must be one of the top candidates shown in the prompt, not a uniformly
    # random (often catastrophic) placement that floods the data with wrecked
    # boards the student would never actually reach.
    records = list(simulate_examples(
        80, epsilon=0.5, seed=5, objective=OBJECTIVE,
        legal_actions=LEGAL, duration_ms=120,
    ))
    for rec in records:
        if rec["applied"]["teacher"]:
            continue
        shown = set(re.findall(r"\[\d\] rotate x(\d+), col (\d+)", rec["state_text"]))
        applied = (str(rec["applied"]["rotation"]), str(rec["applied"]["target_col"]))
        assert applied in shown, (
            f"off-policy move {applied} not among displayed candidates {shown}"
        )


def test_simulate_examples_epsilon_produces_recovery_states():
    # With epsilon > 0 some applied placements differ from the taught label,
    # so later prompts include messy boards — but every label stays the teacher's.
    records = list(simulate_examples(
        60, epsilon=0.5, seed=3, objective=OBJECTIVE,
        legal_actions=LEGAL, duration_ms=120,
    ))
    off_policy = [r for r in records if not r["applied"]["teacher"]]
    assert off_policy, "epsilon=0.5 over 60 steps must produce off-policy applications"
    holes_seen = {r["reward"]["holes_after"] for r in records}
    assert any(h > 0 for h in holes_seen), "recovery data must include boards with holes"


def test_simulate_examples_deterministic_for_seed():
    a = list(simulate_examples(15, epsilon=0.2, seed=11, objective=OBJECTIVE,
                               legal_actions=LEGAL, duration_ms=120))
    b = list(simulate_examples(15, epsilon=0.2, seed=11, objective=OBJECTIVE,
                               legal_actions=LEGAL, duration_ms=120))
    assert json.dumps(a) == json.dumps(b)


def test_simulate_examples_carry_live_format_recent_outcomes():
    # The live loop feeds the last 5 outcome lines ("<action> -> score +N.NN
    # (improved)") into every prompt after step 1; training rows must match
    # that distribution instead of always showing "(none)".
    records = list(simulate_examples(10, epsilon=0.0, seed=2, objective=OBJECTIVE,
                                     legal_actions=LEGAL, duration_ms=120))
    assert records[0]["recent_outcomes"] == []
    assert records[1]["recent_outcomes"], "step 2 must remember step 1's outcome"
    line_re = re.compile(
        r"^(left|right|rotate|drop) -> score \+?-?\d+\.\d{2} "
        r"\((improved|no improvement)\)$"
    )
    for rec in records:
        assert len(rec["recent_outcomes"]) <= 5
        for line in rec["recent_outcomes"]:
            assert line_re.match(line), f"bad outcome line: {line!r}"


def test_text_export_threads_recent_outcomes_into_the_prompt():
    from scripts.export_nebius import example_to_text_sft

    records = list(simulate_examples(6, epsilon=0.0, seed=2, objective=OBJECTIVE,
                                     legal_actions=LEGAL, duration_ms=120))
    rec = records[-1]
    assert rec["recent_outcomes"]
    row = example_to_text_sft(rec)
    user = row["messages"][1]["content"]
    for line in rec["recent_outcomes"]:
        assert line in user


def test_simulate_examples_records_are_text_export_compatible():
    from scripts.export_nebius import example_to_text_sft

    rec = next(iter(simulate_examples(1, epsilon=0.0, seed=1, objective=OBJECTIVE,
                                      legal_actions=LEGAL, duration_ms=120)))
    row = example_to_text_sft(rec)
    roles = [m["role"] for m in row["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert "Candidate placements" in row["messages"][1]["content"]
    assert json.loads(row["messages"][2]["content"]) == rec["target"]
