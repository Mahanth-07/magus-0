#!/usr/bin/env python3
"""Collect a supervised distillation dataset from the EXPERT TEACHER's self-play.

Drives the live 29_tetris game with ludus.teacher.tetris_heuristic.best_move and
logs ONE training example per placement. Each example pairs what a VLM would see at
inference (screenshot + state_text + objective + legal actions) with the TEACHER's
move as a Decision-shaped target. A small VLM can later be LoRA-fine-tuned on this
to imitate the expert (pi0-style policy distillation).

The driving logic (re-read-and-correct rotate/shift/drop, no-pause-while-positioning)
is REUSED from scripts.teacher_play — see that module's docstring for the live
calibration. Here we additionally snapshot the inputs/target BEFORE applying the
macro, then read the post-placement score/holes as a reward signal.

Run with the anaconda python (Playwright lives there), sandbox disabled:
    /opt/anaconda3/bin/python scripts/collect_dataset.py [N] [--seed S] [--headed]

Outputs (gitignored):
    data/tetris/images/ex_000001.png   one PNG per example
    data/tetris/examples.jsonl         one JSON record per example
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# repo root on path so `import ludus...` / `import scripts...` resolve as a script
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ludus.config import load_game_config
from ludus.gameworld_client import GameWorldClient
from ludus.teacher.tetris_heuristic import (
    ROTATIONS,
    best_move,
    column_heights,
    count_holes,
    strip_active_piece,
)

# Reuse the live-calibrated driving primitives from the teacher harness.
from scripts.teacher_play import (
    COLS,
    GAME_ID,
    active_info,
    board_of,
    piece_signature,
    raw_state,
    settle_shape_cols,
    status_of,
    tap,
)

DATA_DIR = REPO_ROOT / "data" / "tetris"
IMAGES_DIR = DATA_DIR / "images"
EXAMPLES_PATH = DATA_DIR / "examples.jsonl"


def _build_target(shape: str, move: dict, duration_ms: int) -> dict:
    """A Decision-shaped dict (schemas.Decision) capturing the teacher's placement.

    actions = the teacher macro; action = the first move (or "drop" if the macro is
    just a drop). Text fields are templated so the student learns to narrate its
    placement in the same schema the runtime parses.
    """
    actions = list(move["actions"])
    action = actions[0] if actions else "drop"
    feats = move.get("features", {}) or {}
    holes = feats.get("holes", 0)
    agg_height = feats.get("aggregate_height", feats.get("max_height", 0))
    target_col = move["target_col"]
    return {
        "scene_summary": f"Tetris: active {shape} piece falling over the stack",
        "controlled_entity": "the active tetromino",
        "current_subgoal": f"place the {shape} to minimize holes and clear lines",
        "action": action,
        "actions": actions,
        "action_args": {"duration_ms": duration_ms},
        "expected_result": f"piece settles at column {target_col}; holes stay low",
        "reason": f"expert placement (holes={holes}, height={agg_height})",
        "confidence": 1.0,
    }


def _apply_macro(client: GameWorldClient, shape: str, move: dict) -> None:
    """Apply the teacher macro with the SAME re-read-and-correct loop as teacher_play.

    Rotate to the target orientation (re-reading after each press so base-point
    rotation shifts / wall-kicks can't desync us), shift horizontally using the
    MEASURED leftmost column, then hard-drop. Gravity is NOT paused while
    positioning (the game's input loop only polls keys while it ticks).
    """
    target_rot = move["target_rotation"]
    target_col = move["target_col"]

    # --- rotate to target orientation ---
    want_set = frozenset(ROTATIONS[shape][target_rot])
    for _ in range(4):
        cur = raw_state(client)
        _, cur_blocks, _, _ = active_info(cur)
        cur_set = frozenset(settle_shape_cols(cur_blocks) or ())
        if cur_set == want_set:
            break
        tap(client, "rotate")

    # --- shift horizontally using the MEASURED leftmost col (self-correcting) ---
    guard = 0
    while guard < COLS + 2:
        cur = raw_state(client)
        _, _, cur_left, _ = active_info(cur)
        if cur_left is None:
            break
        delta = target_col - cur_left
        if delta == 0:
            break
        action = "right" if delta > 0 else "left"
        before_left = cur_left
        tap(client, action)
        after = raw_state(client)
        _, _, after_left, _ = active_info(after)
        if after_left == before_left:
            break  # blocked by wall/stack
        guard += 1

    # --- hard drop ---
    tap(client, "drop")
    time.sleep(0.12)  # let the lock + next spawn happen


def _settled_holes(state: dict) -> int:
    """Holes in the settled stack (active piece excluded), from getState()."""
    board = board_of(state)
    if not board:
        return 0
    blocks = active_info(state)[1]
    return count_holes(strip_active_piece(board, blocks))


def collect(
    client: GameWorldClient,
    target_n: int,
    objective: str,
    legal_actions: list[str],
    duration_ms: int,
    start_index: int,
    fout,
    seed: int | None = None,
) -> tuple[int, list[int], list[int]]:
    """Collect up to target_n examples this game. Returns (count, scores, holes_list)."""
    written = 0
    scores: list[int] = []
    holes_list: list[int] = []
    last_sig = None

    # wait for a real active piece
    for _ in range(50):
        st = raw_state(client)
        if status_of(st) == "playing":
            shape, blocks, leftmost, _ = active_info(st)
            if shape and blocks:
                break
        time.sleep(0.1)

    while written < target_n:
        st = raw_state(client)
        if status_of(st) == "terminal":
            print(f"[game over after {written} examples this game]")
            break
        if status_of(st) != "playing":
            time.sleep(0.05)
            continue

        shape, blocks, leftmost, _ = active_info(st)
        board = board_of(st)
        if not (shape and blocks and board and leftmost is not None):
            time.sleep(0.05)
            continue

        sig = piece_signature(blocks)
        if sig == last_sig:
            time.sleep(0.03)
            continue

        # --- decide (the teacher's move for THIS piece) ---
        stack = strip_active_piece(board, blocks)
        move = best_move(stack, shape, current_left=leftmost, current_rotation=0)

        # --- snapshot the INPUTS the student would see, BEFORE applying the macro ---
        png = client.screenshot()
        state_text = client.state_text()
        target = _build_target(shape, move, duration_ms)

        idx = start_index + written
        img_rel = f"images/ex_{idx:06d}.png"
        (IMAGES_DIR / f"ex_{idx:06d}.png").write_bytes(png)

        # --- apply the macro, then read the reward (post-placement) ---
        _apply_macro(client, shape, move)
        post = raw_state(client)
        post_metrics = post.get("metrics") or {}
        score_after = int(post_metrics.get("score", 0) or 0)
        holes_after = _settled_holes(post)

        record = {
            "image": img_rel,
            "objective": objective,
            "legal_actions": legal_actions,
            "state_text": state_text,
            "target": target,
            "reward": {"score_after": score_after, "holes_after": holes_after},
        }
        if seed is not None:
            record["seed"] = seed
        fout.write(json.dumps(record) + "\n")
        fout.flush()

        scores.append(score_after)
        holes_list.append(holes_after)
        written += 1
        last_sig = sig

        if written <= 5 or written % 10 == 0:
            heights = column_heights(strip_active_piece(board_of(post) or board, active_info(post)[1]))
            print(
                f"[{idx:4d}] '{shape}' -> rot{move['target_rotation']} col{move['target_col']} "
                f"| score={score_after} holes={holes_after} maxh={max(heights or [0])}"
            )

    return written, scores, holes_list


def _existing_count(path: Path) -> int:
    """Number of non-blank example records already in `path` (0 if absent).

    Used by --append to keep numbering monotonic across runs so new image files
    never clobber earlier ones and the jsonl stays one-record-per-line.
    """
    if not path.exists():
        return 0
    n = 0
    with path.open() as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("n", nargs="?", type=int, default=150, help="number of examples to collect")
    ap.add_argument("--seed", type=int, default=None, help="RNG seed marker for the run (recorded per example)")
    ap.add_argument(
        "--append",
        action="store_true",
        help="append to (rather than overwrite) examples.jsonl, continuing the image numbering",
    )
    ap.add_argument("--headed", action="store_true", help="show the browser window")
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "tetris.yaml")
    args = ap.parse_args()

    cfg = load_game_config(args.config)
    objective = cfg.objective
    legal_actions = cfg.legal_actions
    duration_ms = cfg.timing_ms

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # In --append mode, continue numbering after the records already on disk so we
    # never overwrite earlier image files; otherwise start fresh (truncate).
    already = _existing_count(EXAMPLES_PATH) if args.append else 0
    open_mode = "a" if args.append else "w"

    print(
        f"Collecting {args.n} teacher examples from {GAME_ID} "
        f"(headless={not args.headed}, append={args.append}, seed={args.seed}) -> {EXAMPLES_PATH}"
    )
    if args.append:
        print(f"[append] {already} examples already on disk; new ones start at index {already + 1}")

    total = 0
    all_scores: list[int] = []
    all_holes: list[int] = []
    games = 0
    # The teacher rarely tops out, so one game usually yields many examples; loop
    # across fresh games until we hit the target (each game is a new browser/seed).
    with EXAMPLES_PATH.open(open_mode) as fout:
        while total < args.n and games < 12:
            games += 1
            remaining = args.n - total
            print(f"\n=== game {games} (need {remaining} more) ===")
            client = GameWorldClient(
                GAME_ID, timing_ms=duration_ms, headless=not args.headed, call_timeout=90.0
            )
            try:
                n, scores, holes = collect(
                    client, remaining, objective, legal_actions, duration_ms,
                    start_index=already + total + 1, fout=fout, seed=args.seed,
                )
            finally:
                client.close()
            total += n
            all_scores += scores
            all_holes += holes
            if n == 0:
                print("[no progress this game; stopping]")
                break

    # --- summary ---
    print("\n==== COLLECTION SUMMARY ====")
    print(f"examples written : {total}")
    print(f"games played     : {games}")
    if all_scores:
        print(f"score range      : {min(all_scores)} .. {max(all_scores)}")
    if all_holes:
        print(f"holes avg/max    : {sum(all_holes)/len(all_holes):.2f} / {max(all_holes)}")
    print(f"jsonl            : {EXAMPLES_PATH}")
    print(f"images dir       : {IMAGES_DIR}")


if __name__ == "__main__":
    main()
