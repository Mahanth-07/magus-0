#!/usr/bin/env python3
"""Live teacher-driven play harness for the real 29_tetris game.

Drives the REAL browser game with the pure heuristic (ludus.teacher.tetris_heuristic),
verifying end-to-end that the search actually CONTROLS the game well. This is the
live calibration the offline tests can't do: it checks that pressing `rotate` /
`left` / `right` moves the piece the way the heuristic's tables assume, and uses a
re-read-and-correct loop so the teacher's intended placement matches what lands.

Run with the anaconda python (Playwright lives there), sandbox disabled:
    /opt/anaconda3/bin/python scripts/teacher_play.py [N_PIECES]

Key design — why this is robust to coordinate/rotation quirks:
  * After each `rotate` we RE-READ the live state and confirm the rotation index
    actually advanced (and which way).
  * We compute the target leftmost column offline, but the LEFT/RIGHT moves are
    driven by the *measured* current leftmost column after rotation (so base-point
    rotation shifts and wall-kicks can't desync us), correcting one step at a time.
  * We DON'T pause gravity while positioning: the game's input loop only polls
    keys while its update loop runs, so pausing freezes left/right/rotate
    (verified live). Gravity at low levels is slow enough to position with
    discrete taps before the piece locks.

Live calibration result (verified, headless, seed 42):
  ArrowRight -> leftmost col +1, ArrowLeft -> leftmost col -1, `x` -> clockwise
  rotation matching the derived ROTATIONS tables EXACTLY (every probe reported
  "moved/expected OK"). No table or direction fixes were needed once pausing was
  removed. A 28-piece run scored 2510 with 0 holes and max height 2 (the VLM
  baseline was ~238) — see the module's __main__ output.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# repo root on path so `import ludus...` works when run as a script
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ludus.gameworld_client import GameWorldClient
from ludus.teacher.tetris_heuristic import (
    ROTATIONS,
    best_move,
    column_heights,
    count_holes,
    strip_active_piece,
)

GAME_ID = "29_tetris"
ROWS, COLS = 20, 10

# Per-action key hold; the game's input loop polls held keys. Short, discrete taps.
TAP_MS = 60


def raw_state(client: GameWorldClient) -> dict:
    """Reach the raw getState() dict through the client's async bridge (reuse)."""
    client._ensure_started()
    return client._submit(client._mgr.get_game_state()) or {}


def active_info(state: dict):
    """Return (shape, blocks, leftmost_col, bottomed) for the active piece, or Nones."""
    gs = state.get("game_state") or {}
    player = gs.get("player") or {}
    shape = player.get("shape")
    blocks = player.get("blocks") or []
    xs = [b["x"] for b in blocks if isinstance(b, dict) and isinstance(b.get("x"), int)]
    leftmost = min(xs) if xs else None
    bottomed = player.get("bottomed")
    return shape, blocks, leftmost, bottomed


def board_of(state: dict):
    gs = state.get("game_state") or {}
    env = gs.get("environment") or {}
    return env.get("board")


def status_of(state: dict) -> str:
    return state.get("status") or "unknown"


def piece_signature(blocks) -> frozenset:
    """A hashable signature of the active piece's exact cells (to detect changes)."""
    return frozenset(
        (b["x"], b["y"]) for b in blocks
        if isinstance(b, dict) and isinstance(b.get("x"), int) and isinstance(b.get("y"), int)
    )


def tap(client: GameWorldClient, action: str) -> None:
    key = {"left": "ArrowLeft", "right": "ArrowRight", "rotate": "x", "drop": "Space"}[action]
    client.apply({"key": key, "duration_ms": TAP_MS})


def settle_shape_cols(blocks):
    """Set of (col, row) of the active piece, normalized to its own top-left —
    so we can compare the live orientation against ROTATIONS entries."""
    cells = [(b["x"], b["y"]) for b in blocks if isinstance(b, dict)]
    if not cells:
        return None
    minx = min(c for c, _ in cells)
    miny = min(r for _, r in cells)
    return tuple(sorted((c - minx, r - miny) for c, r in cells))


def play(client: GameWorldClient, n_pieces: int, calibrate: bool = True):
    placed = 0
    last_sig = None
    calib_report = []

    # wait for a real active piece
    for _ in range(50):
        st = raw_state(client)
        if status_of(st) == "playing":
            shape, blocks, leftmost, _ = active_info(st)
            if shape and blocks:
                break
        time.sleep(0.1)

    while placed < n_pieces:
        st = raw_state(client)
        if status_of(st) == "terminal":
            print(f"[game over after {placed} pieces]")
            break
        if status_of(st) != "playing":
            time.sleep(0.05)
            continue

        shape, blocks, leftmost, bottomed = active_info(st)
        board = board_of(st)
        if not (shape and blocks and board and leftmost is not None):
            time.sleep(0.05)
            continue

        sig = piece_signature(blocks)
        if sig == last_sig:
            # same piece still in play (hasn't spawned a new one yet); wait briefly
            time.sleep(0.03)
            continue

        # --- decide ---
        stack = strip_active_piece(board, blocks)
        move = best_move(stack, shape, current_left=leftmost, current_rotation=0)
        target_rot = move["target_rotation"]
        target_col = move["target_col"]

        do_calib = calibrate and placed < 3

        # NOTE: we deliberately do NOT pause() while positioning. The game's input
        # loop only polls keys while the update loop runs, so pausing freezes
        # left/right/rotate (verified live). Gravity at low levels is slow enough
        # that discrete taps + re-reads position the piece before it locks; if a
        # piece ever bottoms mid-macro we just drop from wherever it is.

        # --- rotate to target orientation, re-reading after each press ---
        want_set = frozenset(ROTATIONS[shape][target_rot])
        rotations_done = 0
        for _ in range(4):
            cur = raw_state(client)
            _, cur_blocks, _, _ = active_info(cur)
            cur_set = frozenset(settle_shape_cols(cur_blocks) or ())
            if cur_set == want_set:
                break
            before = settle_shape_cols(cur_blocks)
            tap(client, "rotate")
            after_state = raw_state(client)
            _, after_blocks, _, _ = active_info(after_state)
            after = settle_shape_cols(after_blocks)
            rotations_done += 1
            if do_calib:
                changed = (after != before)
                calib_report.append(
                    f"  rotate #{rotations_done} on '{shape}': "
                    f"{'CHANGED' if changed else 'no-change'} "
                    f"{before} -> {after}"
                )

        # --- shift horizontally using the MEASURED leftmost col (self-correcting) ---
        guard = 0
        while guard < COLS + 2:
            cur = raw_state(client)
            _, cur_blocks, cur_left, _ = active_info(cur)
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
            if do_calib:
                moved = (after_left - before_left) if after_left is not None else 0
                expect = 1 if action == "right" else -1
                calib_report.append(
                    f"  {action} on '{shape}': leftcol {before_left} -> {after_left} "
                    f"(moved {moved}, expected {expect}) "
                    f"{'OK' if moved == expect else ('WALL/BLOCKED' if moved == 0 else 'MISMATCH')}"
                )
            if after_left == before_left:
                # blocked by a wall/stack on this side — can't reach target; stop.
                break
            guard += 1

        # --- hard drop ---
        tap(client, "drop")
        time.sleep(0.12)  # let the lock + next spawn happen

        placed += 1
        last_sig = sig

        # progress read
        post = raw_state(client)
        pb = board_of(post) or board
        m = post.get("metrics") or {}
        holes_now = count_holes(strip_active_piece(pb, active_info(post)[1]))
        if placed <= 6 or placed % 5 == 0:
            print(
                f"[{placed:2d}] placed '{shape}' rot{target_rot}->col{target_col} "
                f"| score={m.get('score')} occ={m.get('occupied_cells')} "
                f"holes={holes_now} maxh={max(column_heights(strip_active_piece(pb, active_info(post)[1])) or [0])}"
            )

        if do_calib and placed == 3 and calib_report:
            print("\n--- CALIBRATION (first 3 pieces) ---")
            for line in calib_report:
                print(line)
            print("--- end calibration ---\n")

    # final report
    final = raw_state(client)
    fb = board_of(final) or []
    fblocks = active_info(final)[1]
    fstack = strip_active_piece(fb, fblocks) if fb else []
    fm = final.get("metrics") or {}
    heights = column_heights(fstack) if fstack else [0]
    print("\n==== FINAL ====")
    print(f"pieces placed     : {placed}")
    print(f"score             : {fm.get('score')}")
    print(f"level             : {fm.get('level')}")
    print(f"lines_remaining   : {fm.get('lines_remaining')}")
    print(f"occupied_cells    : {fm.get('occupied_cells')}")
    print(f"holes (settled)   : {count_holes(fstack) if fstack else 'n/a'}")
    print(f"max col height    : {max(heights)}")
    print(f"col heights L->R  : {heights}")
    print(f"aggregate height  : {sum(heights)}")
    print(f"bumpiness         : {sum(abs(heights[i]-heights[i+1]) for i in range(len(heights)-1))}")
    return {
        "pieces": placed,
        "score": fm.get("score"),
        "occupied_cells": fm.get("occupied_cells"),
        "holes": count_holes(fstack) if fstack else None,
        "heights": heights,
    }


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    headless = "--headed" not in sys.argv
    print(f"Launching {GAME_ID} (headless={headless}) for {n} teacher-driven pieces...")
    client = GameWorldClient(GAME_ID, timing_ms=TAP_MS, headless=headless, call_timeout=90.0)
    try:
        result = play(client, n_pieces=n)
        return result
    finally:
        client.close()


if __name__ == "__main__":
    main()
