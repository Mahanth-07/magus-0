"""Pure Tetris placement heuristic — the EXPERT TEACHER's brain.

No browser, no I/O. Given the settled board + the active piece's shape, enumerate
every (rotation, target column), settle the piece, score the resulting board with
a Dellacherie-style feature set, and return the best placement as an action macro
the game can execute.

Board convention (matches the real 29_tetris window.gameAPI.getState()):
  board[row][col], ROWS x COLS, row 0 = TOP, value 0/1. The grid may include the
  active falling piece; pass it through strip_active_piece() first so the search
  reasons about the settled STACK, not the in-flight piece.

Cell convention for ROTATIONS:
  each orientation is a tuple of (col, row) offsets, normalized so min col == 0
  and min row == 0. Row grows DOWNWARD (matches the board). `left` in drop_piece
  is the absolute board column the orientation's col-0 maps to.

The ROTATION tables were DERIVED by replicating the game's own
ControlGroup.turn(cw) math (29_tetris/ControlGroup.js + Shapes.js) for rotation
states 0..3, then normalizing — so they match what the `x` (rotate-cw) key
actually produces in the live game. See scripts/teacher_play.py for the live
calibration that verifies this end-to-end.
"""

from __future__ import annotations

import copy

ROWS = 20
COLS = 10

# Distinct rotation orientations per shape, indexed by the number of cw `rotate`
# presses (mod the cycle). Each entry is a tuple of (col, row) offsets normalized
# to the top-left. Derived from the game's turn-cw transform (see module docstring).
#
#  i (2):  rot0 = ####            rot1 = vertical bar
#  o (1):  the 2x2 square (rotation is a no-op)
#  j (4),  l (4),  t (4):  all four orientations distinct
#  s (2),  z (2):  two orientations
ROTATIONS: dict[str, list[tuple[tuple[int, int], ...]]] = {
    "i": [
        ((0, 0), (1, 0), (2, 0), (3, 0)),          # rot0: horizontal
        ((0, 0), (0, 1), (0, 2), (0, 3)),          # rot1: vertical
    ],
    "o": [
        ((0, 0), (1, 0), (0, 1), (1, 1)),          # rot0: square
    ],
    "j": [
        ((0, 0), (0, 1), (1, 1), (2, 1)),          # rot0:  #.. / ###
        ((0, 0), (1, 0), (0, 1), (0, 2)),          # rot1:  ## / #. / #.
        ((0, 0), (1, 0), (2, 0), (2, 1)),          # rot2:  ### / ..#
        ((1, 0), (1, 1), (0, 2), (1, 2)),          # rot3:  .# / .# / ##
    ],
    "l": [
        ((2, 0), (0, 1), (1, 1), (2, 1)),          # rot0:  ..# / ###
        ((0, 0), (0, 1), (0, 2), (1, 2)),          # rot1:  #. / #. / ##
        ((0, 0), (1, 0), (2, 0), (0, 1)),          # rot2:  ### / #..
        ((0, 0), (1, 0), (1, 1), (1, 2)),          # rot3:  ## / .# / .#
    ],
    "s": [
        ((1, 0), (2, 0), (0, 1), (1, 1)),          # rot0:  .## / ##.
        ((0, 0), (0, 1), (1, 1), (1, 2)),          # rot1:  #. / ## / .#
    ],
    "z": [
        ((0, 0), (1, 0), (1, 1), (2, 1)),          # rot0:  ##. / .##
        ((1, 0), (0, 1), (1, 1), (0, 2)),          # rot1:  .# / ## / #.
    ],
    "t": [
        ((1, 0), (0, 1), (1, 1), (2, 1)),          # rot0:  .#. / ###
        ((0, 0), (0, 1), (1, 1), (0, 2)),          # rot1:  #. / ## / #.
        ((0, 0), (1, 0), (2, 0), (1, 1)),          # rot2:  ### / .#.
        ((1, 0), (0, 1), (1, 1), (1, 2)),          # rot3:  .# / ## / .#
    ],
}

# The spawn column (leftmost active cell) the heuristic assumes when translating
# the chosen target into a left/right macro. The live harness re-reads the true
# leftmost column after rotating, so this default only matters for the offline
# macro; callers pass the real `current_left` from getState().
SPAWN_LEFT_DEFAULT = 3

# --------------------------------------------------------------------------
# Dellacherie-style feature weights. Standard, well-known good values:
# heavily penalize holes, penalize aggregate height / bumpiness / landing
# height / wells, reward line clears. Higher score = better placement.
# --------------------------------------------------------------------------
WEIGHTS = {
    "aggregate_height": -0.510066,
    "lines_cleared": 0.760666,
    "holes": -0.85,          # holes are the worst sin; bump above the classic 0.36
    "bumpiness": -0.184483,
    "landing_height": -0.30,
    "wells": -0.18,          # deep single-wide wells (besides one tucked column)
    "row_transitions": -0.18,
    "col_transitions": -0.30,
}


# --------------------------------------------------------------------------
# Board helpers
# --------------------------------------------------------------------------

def strip_active_piece(board: list[list[int]], active_blocks) -> list[list[int]]:
    """Return a COPY of `board` with the active piece's cells cleared.

    `active_blocks` is the getState() player.blocks list of {"x","y"} (y may be
    negative / above the field). Mirrors gameworld_client._tetris_state_text so
    the heuristic reasons about the settled stack, not the falling piece.
    Subtracting a cell that isn't set is a harmless no-op.
    """
    b = [row[:] for row in board]
    for cell in active_blocks or []:
        if isinstance(cell, dict):
            x, y = cell.get("x"), cell.get("y")
        else:  # tolerate (x, y) tuples
            x, y = cell
        if isinstance(x, int) and isinstance(y, int) and 0 <= y < len(b) and 0 <= x < len(b[0]):
            b[y][x] = 0
    return b


def column_heights(board: list[list[int]]) -> list[int]:
    """Per-column height measured from the floor (0 = empty column)."""
    rows = len(board)
    cols = len(board[0]) if rows else 0
    heights = [0] * cols
    for c in range(cols):
        for r in range(rows):
            if board[r][c]:
                heights[c] = rows - r
                break
    return heights


def count_holes(board: list[list[int]]) -> int:
    """Empty cell with at least one filled cell ABOVE it in the same column."""
    rows = len(board)
    cols = len(board[0]) if rows else 0
    holes = 0
    for c in range(cols):
        seen = False
        for r in range(rows):
            if board[r][c]:
                seen = True
            elif seen:
                holes += 1
    return holes


def _bumpiness(heights: list[int]) -> int:
    return sum(abs(heights[i] - heights[i + 1]) for i in range(len(heights) - 1))


def _row_transitions(board: list[list[int]]) -> int:
    """Filled<->empty transitions scanning each row (walls count as filled)."""
    rows = len(board)
    cols = len(board[0]) if rows else 0
    transitions = 0
    for r in range(rows):
        prev = 1  # left wall
        for c in range(cols):
            cell = board[r][c]
            if cell != prev:
                transitions += 1
            prev = cell
        if prev != 1:  # right wall
            transitions += 1
    return transitions


def _col_transitions(board: list[list[int]]) -> int:
    """Filled<->empty transitions scanning each column (floor counts as filled)."""
    rows = len(board)
    cols = len(board[0]) if rows else 0
    transitions = 0
    for c in range(cols):
        prev = 0  # above the stack is empty
        for r in range(rows):
            cell = board[r][c]
            if cell != prev:
                transitions += 1
            prev = cell
        if prev != 1:  # floor below
            transitions += 1
    return transitions


def _wells(heights: list[int]) -> int:
    """Sum of well depths: a column lower than BOTH neighbors forms a well.

    We count the cumulative depth (1+2+...+d for a depth-d well) which is the
    classic Dellacherie 'cumulative wells' contribution. Edge columns use a
    virtual wall (very tall) as the missing neighbor.
    """
    cols = len(heights)
    total = 0
    for c in range(cols):
        left = heights[c - 1] if c > 0 else ROWS
        right = heights[c + 1] if c < cols - 1 else ROWS
        depth = min(left, right) - heights[c]
        if depth > 0:
            total += depth * (depth + 1) // 2
    return total


# --------------------------------------------------------------------------
# Placement simulation
# --------------------------------------------------------------------------

def drop_piece(board: list[list[int]], cells, left: int):
    """Hard-drop a piece orientation into a copy of `board` at column `left`.

    `cells` = orientation offsets (col, row) normalized to top-left.
    `left`  = absolute board column the orientation's col-0 maps to.

    Returns (new_board, landing_top_row, lines_cleared), or None if the
    placement is horizontally out of bounds or cannot fit at all.
    landing_top_row = the smallest board row index any cell occupies after the
    drop (used for landing-height; lower row index = higher up = worse).
    """
    rows = len(board)
    cols = len(board[0]) if rows else 0

    width = max(c for c, _ in cells) + 1
    if left < 0 or left + width > cols:
        return None

    # absolute columns/rows of each cell at drop offset 0
    abs_cells = [(left + c, r) for (c, r) in cells]

    # Find the maximum downward shift `d` such that every cell stays in-bounds
    # and lands on an empty cell. Start d at the highest legal value and walk up.
    max_row_off = max(r for _, r in cells)

    def fits(d: int) -> bool:
        for (x, r) in abs_cells:
            ry = r + d
            if ry < 0:
                continue  # above the field is fine while falling
            if ry >= rows:
                return False
            if board[ry][x]:
                return False
        return True

    # Largest d that fits; if even d=0 doesn't fit (stack reaches the top), invalid.
    d = -1
    # the piece can fall at most until its lowest cell hits the floor
    max_d = rows  # generous upper bound
    for cand in range(max_d + 1):
        if fits(cand):
            d = cand
        else:
            break
    if d < 0:
        return None

    new_board = [row[:] for row in board]
    landing_top_row = rows
    for (x, r) in abs_cells:
        ry = r + d
        if ry < 0:
            # piece settled partially above the visible field -> topped out;
            # treat as invalid so the search never picks a self-burying move.
            return None
        new_board[ry][x] = 1
        landing_top_row = min(landing_top_row, ry)

    # clear completed lines
    kept = [row for row in new_board if not all(row)]
    lines_cleared = rows - len(kept)
    if lines_cleared:
        kept = [[0] * cols for _ in range(lines_cleared)] + kept
    new_board = kept

    return new_board, landing_top_row, lines_cleared


# --------------------------------------------------------------------------
# Scoring + search
# --------------------------------------------------------------------------

def _score_board(board: list[list[int]], landing_top_row: int, lines_cleared: int) -> tuple[float, dict]:
    heights = column_heights(board)
    agg = sum(heights)
    holes = count_holes(board)
    bump = _bumpiness(heights)
    wells = _wells(heights)
    row_tr = _row_transitions(board)
    col_tr = _col_transitions(board)
    # landing height = height (from floor) at which the piece came to rest.
    landing_height = ROWS - landing_top_row

    feats = {
        "aggregate_height": agg,
        "max_height": max(heights) if heights else 0,
        "holes": holes,
        "bumpiness": bump,
        "wells": wells,
        "row_transitions": row_tr,
        "col_transitions": col_tr,
        "landing_height": landing_height,
        "lines_cleared": lines_cleared,
    }

    score = (
        WEIGHTS["aggregate_height"] * agg
        + WEIGHTS["lines_cleared"] * lines_cleared
        + WEIGHTS["holes"] * holes
        + WEIGHTS["bumpiness"] * bump
        + WEIGHTS["landing_height"] * landing_height
        + WEIGHTS["wells"] * wells
        + WEIGHTS["row_transitions"] * row_tr
        + WEIGHTS["col_transitions"] * col_tr
    )
    return score, feats


def _macro(target_rotation: int, target_col: int, current_left: int) -> list[str]:
    """Translate (rotation, target leftmost col) into an action macro from spawn.

    Order: rotate (cw) N times, then shift left/right to move the leftmost cell
    from `current_left` to `target_col`, then `drop` last.

    NOTE: this offline macro assumes rotation does not shift the leftmost column.
    Rotation around the piece's base point CAN shift it (and wall-kicks too), so
    the LIVE harness re-reads the actual leftmost column after rotating and
    recomputes the left/right delta. This macro is exact for offline tests where
    the displacement equals target_col - current_left.
    """
    actions: list[str] = []
    actions.extend(["rotate"] * target_rotation)
    delta = target_col - current_left
    if delta > 0:
        actions.extend(["right"] * delta)
    elif delta < 0:
        actions.extend(["left"] * (-delta))
    actions.append("drop")
    return actions


def best_move(
    board: list[list[int]],
    shape: str,
    *,
    current_left: int = SPAWN_LEFT_DEFAULT,
    current_rotation: int = 0,
) -> dict:
    """Enumerate every (rotation, column), settle, score; return the best move.

    Args:
      board: settled stack (strip the active piece first via strip_active_piece).
      shape: one of i,o,j,l,s,t,z (case-insensitive).
      current_left: leftmost column of the active piece NOW (for the macro).
      current_rotation: current rotation index of the active piece NOW.

    Returns dict:
      actions: list[str] in {left,right,rotate,drop}, ending in "drop".
      target_col: chosen leftmost column.
      target_rotation: chosen rotation index (number of cw rotates from rot 0).
      features: feature dict of the resulting board.
    """
    shape = shape.lower()
    if shape not in ROTATIONS:
        raise ValueError(f"unknown shape {shape!r}")

    rows = len(board)
    cols = len(board[0]) if rows else 0

    best = None  # (score, rotation, left, feats)
    for rot_idx, cells in enumerate(ROTATIONS[shape]):
        width = max(c for c, _ in cells) + 1
        for left in range(0, cols - width + 1):
            res = drop_piece(board, cells, left)
            if res is None:
                continue
            new_board, landing_top_row, lines = res
            score, feats = _score_board(new_board, landing_top_row, lines)
            cand = (score, rot_idx, left, feats)
            if best is None or score > best[0]:
                best = cand

    if best is None:
        # No legal placement anywhere (board topped out). Emit a bare drop so the
        # caller still has a valid macro; the game will end.
        return {
            "actions": ["drop"],
            "target_col": current_left,
            "target_rotation": current_rotation,
            "features": {"holes": count_holes(board), "max_height": max(column_heights(board) or [0])},
        }

    _, rot_idx, left, feats = best
    # number of cw rotate presses needed to reach rot_idx from current_rotation,
    # within this shape's distinct-orientation cycle.
    cycle = len(ROTATIONS[shape])
    rotate_presses = (rot_idx - current_rotation) % cycle
    actions = _macro(rotate_presses, left, current_left)

    return {
        "actions": actions,
        "target_col": left,
        "target_rotation": rot_idx,
        "features": feats,
    }
