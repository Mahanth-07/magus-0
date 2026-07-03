"""Tests for the pure Tetris placement heuristic (ludus/teacher/tetris_heuristic.py).

All synthetic — no browser, no I/O. We prove the SEARCH logic here; the live
harness (scripts/teacher_play.py) proves it controls the real game.

Board convention (matches the real 29_tetris getState):
  board[row][col], 20 rows x 10 cols, row 0 = TOP, value 0/1.
"""

from ludus.teacher.tetris_heuristic import (
    ROTATIONS,
    best_move,
    column_heights,
    count_holes,
    drop_piece,
    rank_placements,
    strip_active_piece,
)

ROWS, COLS = 20, 10


def empty_board(rows=ROWS, cols=COLS):
    return [[0] * cols for _ in range(rows)]


def render(board):
    return "\n".join("".join("#" if c else "." for c in row) for row in board)


# --------------------------------------------------------------------------
# Rotation tables (derived from the game's own ControlGroup.turn math)
# --------------------------------------------------------------------------

def test_rotation_tables_present_for_all_7_pieces():
    for shape in ["i", "o", "j", "l", "s", "t", "z"]:
        assert shape in ROTATIONS
        assert len(ROTATIONS[shape]) >= 1
        for cells in ROTATIONS[shape]:
            assert len(cells) == 4  # every tetromino is 4 cells


def test_rotation_distinct_counts():
    # i,s,z have 2 distinct orientations; o has 1; j,l,t have 4.
    assert len(ROTATIONS["o"]) == 1
    assert len(ROTATIONS["i"]) == 2
    assert len(ROTATIONS["s"]) == 2
    assert len(ROTATIONS["z"]) == 2
    assert len(ROTATIONS["j"]) == 4
    assert len(ROTATIONS["l"]) == 4
    assert len(ROTATIONS["t"]) == 4


def test_rotations_are_normalized_to_origin():
    # Each orientation's cells should be normalized: min col == 0, min row == 0.
    for shape, rots in ROTATIONS.items():
        for cells in rots:
            assert min(c for c, r in cells) == 0
            assert min(r for c, r in cells) == 0


def test_o_piece_is_a_2x2_square():
    cells = set(ROTATIONS["o"][0])
    assert cells == {(0, 0), (1, 0), (0, 1), (1, 1)}


def test_i_piece_flat_is_horizontal_bar():
    cells = set(ROTATIONS["i"][0])
    assert cells == {(0, 0), (1, 0), (2, 0), (3, 0)}


# --------------------------------------------------------------------------
# Feature helpers
# --------------------------------------------------------------------------

def test_column_heights_empty_board_all_zero():
    assert column_heights(empty_board()) == [0] * COLS


def test_column_heights_basic():
    b = empty_board()
    b[ROWS - 1][0] = 1  # one block on the floor, col 0
    b[ROWS - 2][0] = 1  # stacked
    b[ROWS - 1][3] = 1
    h = column_heights(b)
    assert h[0] == 2
    assert h[3] == 1
    assert h[1] == 0


def test_count_holes_none_on_solid_stack():
    b = empty_board()
    for r in range(ROWS - 3, ROWS):
        b[r][0] = 1
    assert count_holes(b) == 0


def test_count_holes_detects_covered_empty():
    b = empty_board()
    b[ROWS - 3][0] = 1   # filled, with empties below it -> 2 holes
    assert count_holes(b) == 2


def test_strip_active_piece_removes_cells():
    b = empty_board()
    b[0][4] = 1
    b[0][5] = 1
    b[ROWS - 1][0] = 1  # settled block stays
    active = [{"x": 4, "y": 0}, {"x": 5, "y": 0}]
    stripped = strip_active_piece(b, active)
    assert stripped[0][4] == 0
    assert stripped[0][5] == 0
    assert stripped[ROWS - 1][0] == 1
    # original not mutated
    assert b[0][4] == 1


# --------------------------------------------------------------------------
# drop_piece
# --------------------------------------------------------------------------

def test_drop_i_flat_on_empty_lands_on_floor():
    b = empty_board()
    cells = ROTATIONS["i"][0]  # horizontal bar
    res = b_after = drop_piece(b, cells, left=0)
    assert res is not None
    new_board, landing_top_row, lines = res
    # all four cells should be on the bottom row
    bottom = ROWS - 1
    for c in range(4):
        assert new_board[bottom][c] == 1
    assert lines == 0
    # height now 1 across cols 0-3
    h = column_heights(new_board)
    assert h[0] == 1 and h[3] == 1


def test_drop_into_deep_notch_no_holes():
    # Build a wall with a 1-wide, 4-deep notch at col 0; vertical I fills it flush.
    b = empty_board()
    for r in range(ROWS - 4, ROWS):
        for c in range(1, COLS):
            b[r][c] = 1
    # col 0 is an open 4-deep well; vertical I (rot 1) dropped at col 0 fills it.
    cells = ROTATIONS["i"][1]  # vertical
    res = drop_piece(b, cells, left=0)
    assert res is not None
    new_board, _, lines = res
    # 4 full rows cleared
    assert lines == 4
    assert count_holes(new_board) == 0


def test_drop_clears_a_line():
    b = empty_board()
    bottom = ROWS - 1
    # fill bottom row except cols 0-3
    for c in range(4, COLS):
        b[bottom][c] = 1
    cells = ROTATIONS["i"][0]  # flat I fills cols 0-3 -> completes the row
    res = drop_piece(b, cells, left=0)
    new_board, _, lines = res
    assert lines == 1


def test_drop_returns_none_when_out_of_bounds():
    b = empty_board()
    cells = ROTATIONS["i"][0]  # width 4
    # placing flat-I with left=7 would need cols 7,8,9,10 -> 10 is OOB
    assert drop_piece(b, cells, left=7) is None
    # left=6 -> cols 6,7,8,9 ok
    assert drop_piece(b, cells, left=6) is not None


# --------------------------------------------------------------------------
# best_move — the integrated search
# --------------------------------------------------------------------------

def _assert_valid_actions(move):
    acts = move["actions"]
    assert isinstance(acts, list)
    assert len(acts) >= 1
    assert acts[-1] == "drop"
    assert all(a in {"left", "right", "rotate", "drop"} for a in acts)


def test_best_move_empty_board_i_places_flat_low():
    b = empty_board()
    move = best_move(b, "i", current_left=3, current_rotation=0)
    _assert_valid_actions(move)
    feats = move["features"]
    # flat placement on empty board => 0 holes, max height 1
    assert feats["holes"] == 0
    assert feats["max_height"] == 1


def test_best_move_actions_nonempty_ends_in_drop_all_shapes():
    b = empty_board()
    for shape in ["i", "o", "j", "l", "s", "t", "z"]:
        move = best_move(b, shape, current_left=3, current_rotation=0)
        _assert_valid_actions(move)
        assert 0 <= move["target_col"] <= COLS - 1
        assert 0 <= move["target_rotation"] <= 3


def test_best_move_fills_deep_well_with_i():
    # A 4-deep well at col 9; the heuristic should drop a vertical I there and
    # clear lines, ending with 0 holes — strictly better than naive spawn drop.
    b = empty_board()
    for r in range(ROWS - 4, ROWS):
        for c in range(0, COLS - 1):
            b[r][c] = 1
    move = best_move(b, "i", current_left=3, current_rotation=0)
    _assert_valid_actions(move)
    assert move["features"]["holes"] == 0
    # should have cleared all four rows
    assert move["features"]["lines_cleared"] == 4


def test_best_move_avoids_stacking_on_tall_column():
    # One very tall column at col 0; an O piece should NOT be stacked there.
    b = empty_board()
    for r in range(ROWS - 12, ROWS):
        b[r][0] = 1
    move = best_move(b, "o", current_left=3, current_rotation=0)
    _assert_valid_actions(move)
    # O is 2 wide; target_col (leftmost) should avoid col 0 entirely.
    assert move["target_col"] != 0


def test_best_move_no_worse_than_naive_spawn_drop_holes():
    # Invariant from the spec: chosen placement's holes <= naive "drop at spawn
    # column, no rotation" holes. Use a jagged board + an S piece (tricky).
    b = empty_board()
    # jagged surface
    heights = [5, 2, 6, 1, 4, 3, 7, 2, 5, 1]
    for c, h in enumerate(heights):
        for r in range(ROWS - h, ROWS):
            b[r][c] = 1

    for shape in ["i", "o", "j", "l", "s", "t", "z"]:
        move = best_move(b, shape, current_left=3, current_rotation=0)
        chosen_holes = move["features"]["holes"]

        # naive: rotation 0, dropped so leftmost cell == spawn col 3
        cells = ROTATIONS[shape][0]
        naive = drop_piece(b, cells, left=3)
        if naive is None:
            continue
        naive_board, _, _ = naive
        naive_holes = count_holes(naive_board)
        assert chosen_holes <= naive_holes, (
            f"{shape}: chosen holes {chosen_holes} > naive {naive_holes}"
        )


def test_best_move_target_rotation_matches_chosen_cells():
    # Deep narrow well forces a vertical I (rotation 1, not 0).
    b = empty_board()
    for r in range(ROWS - 6, ROWS):
        for c in range(0, COLS - 1):
            b[r][c] = 1
    move = best_move(b, "i", current_left=3, current_rotation=0)
    assert move["target_rotation"] == 1  # vertical orientation


def test_best_move_macro_translation_left_right_counts():
    # On an empty board, ask for a move and check the macro's left/right count
    # equals |target_col - current_left| and the rotate count == target_rotation.
    b = empty_board()
    move = best_move(b, "t", current_left=3, current_rotation=0)
    acts = move["actions"]
    n_rot = acts.count("rotate")
    n_left = acts.count("left")
    n_right = acts.count("right")
    assert n_rot == move["target_rotation"]
    # net horizontal displacement matches target - spawn
    assert (n_right - n_left) == (move["target_col"] - 3)


def test_best_move_does_not_mutate_input_board():
    b = empty_board()
    b[ROWS - 1][5] = 1
    snapshot = [row[:] for row in b]
    best_move(b, "l", current_left=3, current_rotation=0)
    assert b == snapshot


# --------------------------------------------------------------------------
# rank_placements — the lookahead tool the VLM picks from
# --------------------------------------------------------------------------

def test_rank_placements_returns_candidates_with_expected_keys():
    b = empty_board()
    cands = rank_placements(b, "i", current_left=3, current_rotation=0, top_k=6)
    assert isinstance(cands, list)
    assert 1 <= len(cands) <= 6
    for c in cands:
        for key in ("rotation", "target_col", "holes", "max_height",
                    "lines_cleared", "actions"):
            assert key in c
        assert isinstance(c["actions"], list)


def test_rank_placements_flat_i_candidate_exists_with_no_holes():
    # On an empty board there must be a flat-laying I placement with holes=0.
    b = empty_board()
    cands = rank_placements(b, "i", current_left=3, current_rotation=0, top_k=10)
    flat = [c for c in cands if c["rotation"] == 0 and c["holes"] == 0]
    assert flat, "expected at least one flat-I candidate with 0 holes"


def test_rank_placements_sorted_best_first_min_holes():
    # The chosen best (index 0) must have the minimum holes among all candidates.
    b = empty_board()
    heights = [5, 2, 6, 1, 4, 3, 7, 2, 5, 1]
    for c, h in enumerate(heights):
        for r in range(ROWS - h, ROWS):
            b[r][c] = 1
    cands = rank_placements(b, "s", current_left=3, current_rotation=0, top_k=20)
    assert cands
    best = cands[0]
    assert best["holes"] == min(c["holes"] for c in cands)
    # confirm the sort ordering is actually best-first by the spec key
    keys = [(-c["lines_cleared"], c["holes"], c["max_height"]) for c in cands]
    assert keys == sorted(keys)


def test_rank_placements_every_candidate_action_ends_in_drop():
    b = empty_board()
    for shape in ["i", "o", "j", "l", "s", "t", "z"]:
        cands = rank_placements(b, shape, current_left=3, current_rotation=0)
        assert cands
        for c in cands:
            assert c["actions"], "actions must be non-empty"
            assert c["actions"][-1] == "drop"
            assert all(a in {"left", "right", "rotate", "drop"} for a in c["actions"])


def test_rank_placements_top_k_respected():
    b = empty_board()
    cands = rank_placements(b, "t", current_left=3, current_rotation=0, top_k=3)
    assert len(cands) <= 3


def test_rotation_left_shift_matches_live_j_piece():
    # Live-observed (29_tetris, headless): the J piece's leftmost column shifts
    # +1 after a single clockwise rotate from spawn (rot0 -> rot1).
    from ludus.teacher.tetris_heuristic import _rotation_left_shift
    assert _rotation_left_shift("j", 0) == 0
    assert _rotation_left_shift("j", 1) == 1
    # full cycle returns to the spawn position
    assert _rotation_left_shift("j", 4) == 0
    # o-piece (2x2 square) never shifts under rotation
    assert _rotation_left_shift("o", 1) == 0
    assert _rotation_left_shift("o", 2) == 0


def test_rank_placements_prefers_line_clear_first():
    # Deep well at col 9 -> a vertical I clears 4 lines; that must rank #1.
    b = empty_board()
    for r in range(ROWS - 4, ROWS):
        for c in range(0, COLS - 1):
            b[r][c] = 1
    cands = rank_placements(b, "i", current_left=3, current_rotation=0, top_k=6)
    assert cands[0]["lines_cleared"] == 4
    assert cands[0]["rotation"] == 1  # vertical


def test_rank_placements_macro_matches_pose():
    # The macro is computed from the current pose AND compensates for the live
    # rotation-induced column shift: net horizontal == target - (current_left + shift),
    # so a blind open-loop apply lands the piece in target_col.
    from ludus.teacher.tetris_heuristic import _rotation_left_shift
    b = empty_board()
    cands = rank_placements(b, "j", current_left=5, current_rotation=0, top_k=10)
    for c in cands:
        acts = c["actions"]
        n_right = acts.count("right")
        n_left = acts.count("left")
        rot = acts.count("rotate")
        shift = _rotation_left_shift("j", rot)
        assert (n_right - n_left) == (c["target_col"] - (5 + shift))
