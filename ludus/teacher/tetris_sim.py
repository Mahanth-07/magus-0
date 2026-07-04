"""Simulation-based distillation collector — no browser, no I/O.

Generates training examples for the Tetris student entirely in Python using the
teacher's own placement engine (drop_piece). Each example's prompt is rendered
by the SAME function the live client uses (gameworld_client._tetris_state_text
over a synthetic getState()-shaped dict), so the student trains on the exact
prompt distribution it sees at inference — including the candidate-placement
block. The label is candidate [0] of that very list (the Dellacherie-best
placement, with a _macro_live-corrected open-loop macro that survives the
runtime's blind apply).

Recovery states (DAgger-style): with probability `epsilon` the APPLIED
placement is a random non-best candidate, so subsequent boards are messy — but
the LABEL is always the teacher's best move. The student thus learns to act
well from bad positions, not only along the teacher's near-perfect trajectory.
"""

from __future__ import annotations

import random
from typing import Iterator

from ludus.teacher.tetris_heuristic import (
    COLS,
    ROTATIONS,
    ROWS,
    SPAWN_LEFT_DEFAULT,
    count_holes,
    drop_piece,
    rank_placements,
)

_SHAPES = "iojlszt"

# Guideline per-line clear scores; informational (reward metadata), the label
# does not depend on them.
_LINE_SCORES = {0: 0, 1: 100, 2: 300, 3: 500, 4: 800}


def _spawn_cells(shape: str) -> list[tuple[int, int]]:
    """Absolute (row, col) board cells of a freshly-spawned `shape` (rot 0)."""
    return [(r, SPAWN_LEFT_DEFAULT + c) for (c, r) in ROTATIONS[shape.lower()][0]]


def synthetic_state(
    stack: list[list[int]],
    shape: str,
    *,
    preview: list[str],
    score: int,
    level: int,
) -> dict:
    """Build a getState()-shaped dict for _tetris_state_text from a sim board.

    The board grid includes the active piece at its spawn pose (as the live
    game's grid does); player.blocks carries the active cells so the renderer
    can strip them for the settled-stack analysis.
    """
    board = [row[:] for row in stack]
    blocks = []
    for (r, c) in _spawn_cells(shape):
        board[r][c] = 1
        blocks.append({"x": c, "y": r})
    return {
        "game_state": {
            "environment": {
                "board": board,
                "preview_queue": list(preview),
                "held_piece": None,
            },
            "player": {"shape": shape.lower(), "blocks": blocks},
        },
        "metrics": {"score": score, "level": level},
    }


def _build_target(shape: str, cand: dict, duration_ms: int) -> dict:
    """Decision-shaped label for candidate `cand` (mirrors the live collector's
    _build_target, but action = the macro's LAST element per the objective's
    'copy candidate [0], set action to its last element' directive)."""
    actions = list(cand["actions"])
    return {
        "scene_summary": f"Tetris: active {shape} piece falling over the stack",
        "controlled_entity": "the active tetromino",
        "current_subgoal": f"place the {shape} to minimize holes and clear lines",
        "action": actions[-1],
        "actions": actions,
        "action_args": {"duration_ms": duration_ms},
        "expected_result": (
            f"piece settles at column {cand['target_col']}; holes stay low"
        ),
        "reason": (
            f"expert placement (holes={cand['holes']}, max_height={cand['max_height']})"
        ),
        "confidence": 1.0,
    }


def simulate_examples(
    n: int,
    *,
    epsilon: float,
    seed: int,
    objective: str,
    legal_actions: list[str],
    duration_ms: int,
    explore_top: int = 6,
) -> Iterator[dict]:
    """Yield up to `n` examples.jsonl-shaped records from simulated self-play.

    Record shape matches scripts/collect_dataset.py output (image=None: the
    text-only student never sees pixels), plus an "applied" block recording
    which placement was actually taken (teacher or epsilon-random).
    """
    from ludus.gameworld_client import _tetris_state_text

    rng = random.Random(seed)
    stack = [[0] * COLS for _ in range(ROWS)]
    bag: list[str] = []
    score = 0
    level = 1
    # Rolling outcome lines in the loop's exact live format
    # ("<action> -> score +N.NN (improved)"); last 5 are fed into each prompt.
    outcomes: list[str] = []

    def _refill() -> None:
        fresh = list(_SHAPES)
        rng.shuffle(fresh)
        bag.extend(fresh)

    def _reset() -> None:
        nonlocal stack, score
        stack = [[0] * COLS for _ in range(ROWS)]
        score = 0
        outcomes.clear()

    produced = 0
    while produced < n:
        if len(bag) < 4:  # current piece + a 3-piece preview
            _refill()
        shape = bag.pop(0)

        # top-out: the fresh piece can't spawn -> new game
        if any(stack[r][c] for (r, c) in _spawn_cells(shape)):
            _reset()
            continue

        # explore_top matches the prompt's displayed top-k: off-policy moves are
        # then always placements the model could have picked from its own menu
        # (mildly suboptimal), never uniformly-random catastrophic ones.
        pool = rank_placements(
            stack, shape,
            current_left=SPAWN_LEFT_DEFAULT, current_rotation=0,
            top_k=explore_top,
        )
        if not pool:
            _reset()
            continue
        best = pool[0]

        state = synthetic_state(
            stack, shape, preview=bag[:3], score=score, level=level
        )
        state_text = _tetris_state_text(state)
        target = _build_target(shape, best, duration_ms)

        # DAgger-style: sometimes APPLY a suboptimal placement (label unchanged)
        chosen, teacher = best, True
        if len(pool) > 1 and rng.random() < epsilon:
            chosen, teacher = rng.choice(pool[1:]), False

        res = drop_piece(stack, ROTATIONS[shape][chosen["rotation"]], chosen["target_col"])
        if res is None:
            _reset()
            continue
        stack, _, lines = res
        delta = _LINE_SCORES.get(lines, 0)
        score += delta
        recent = outcomes[-5:]
        outcomes.append(
            f"{target['action']} -> score +{float(delta):.2f} "
            f"({'improved' if delta > 0 else 'no improvement'})"
        )

        yield {
            "image": None,
            "objective": objective,
            "legal_actions": list(legal_actions),
            "state_text": state_text,
            "recent_outcomes": recent,
            "target": target,
            "reward": {"score_after": score, "holes_after": count_holes(stack)},
            "applied": {
                "rotation": chosen["rotation"],
                "target_col": chosen["target_col"],
                "teacher": teacher,
            },
            "seed": seed,
            "sim": True,
        }
        produced += 1
