import copy
import math

def predict(state, action):
    s = copy.deepcopy(state)
    gs = s["game_state"]
    metrics = s["metrics"]
    player = gs["player"]
    env = gs["environment"]
    cols = env["cols"]
    rows = env["rows"]
    board = env["board"]
    blocks = player["blocks"]

    def get_occupied():
        count = 0
        for r in range(rows):
            for c in range(cols):
                if board[r][c] != 0:
                    count += 1
        return count

    def can_place_list(blist):
        for b in blist:
            if b["x"] < 0 or b["x"] >= cols:
                return False
            if b["y"] >= rows:
                return False
            if b["y"] >= 0 and board[b["y"]][b["x"]] != 0:
                return False
        return True

    def place_on_board(blist):
        for b in blist:
            if 0 <= b["y"] < rows and 0 <= b["x"] < cols:
                board[b["y"]][b["x"]] = 1

    def remove_from_board(blist):
        for b in blist:
            if 0 <= b["y"] < rows and 0 <= b["x"] < cols:
                board[b["y"]][b["x"]] = 0

    def rotate_cw(blist):
        xs = [b["x"] for b in blist]
        ys = [b["y"] for b in blist]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        new_blocks = []
        for b in blist:
            dx = b["x"] - cx
            dy = b["y"] - cy
            nx = cx + dy
            ny = cy - dx
            new_blocks.append({"x": round(nx), "y": round(ny)})
        return new_blocks

    def rotate_ccw(blist):
        xs = [b["x"] for b in blist]
        ys = [b["y"] for b in blist]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        new_blocks = []
        for b in blist:
            dx = b["x"] - cx
            dy = b["y"] - cy
            nx = cx - dy
            ny = cy + dx
            new_blocks.append({"x": round(nx), "y": round(ny)})
        return new_blocks

    def get_spawn_blocks(shape):
        if shape == "j":
            return [{"x": 3, "y": -2}, {"x": 3, "y": -1}, {"x": 4, "y": -1}, {"x": 5, "y": -1}]
        elif shape == "o":
            return [{"x": 4, "y": -2}, {"x": 5, "y": -2}, {"x": 4, "y": -1}, {"x": 5, "y": -1}]
        elif shape == "z":
            return [{"x": 3, "y": -1}, {"x": 4, "y": -1}, {"x": 4, "y": 0}, {"x": 5, "y": 0}]
        elif shape == "s":
            return [{"x": 4, "y": -1}, {"x": 5, "y": -1}, {"x": 3, "y": 0}, {"x": 4, "y": 0}]
        elif shape == "l":
            return [{"x": 5, "y": -2}, {"x": 3, "y": -1}, {"x": 4, "y": -1}, {"x": 5, "y": -1}]
        elif shape == "t":
            return [{"x": 4, "y": -2}, {"x": 3, "y": -1}, {"x": 4, "y": -1}, {"x": 5, "y": -1}]
        elif shape == "i":
            return [{"x": 3, "y": -1}, {"x": 4, "y": -1}, {"x": 5, "y": -1}, {"x": 6, "y": -1}]
        else:
            return [{"x": 4, "y": -2}, {"x": 5, "y": -2}, {"x": 4, "y": -1}, {"x": 5, "y": -1}]

    # Remove current piece from board before processing
    remove_from_board(blocks)

    if action == "move_down":
        drop = 3
        moved = [dict(b) for b in blocks]
        actual_drop = 0
        for _ in range(drop):
            next_pos = [{"x": b["x"], "y": b["y"] + 1} for b in moved]
            if not can_place_list(next_pos):
                break
            moved = next_pos
            actual_drop += 1

        player["blocks"] = moved
        blocks = moved

        # Check if bottomed
        next_pos = [{"x": b["x"], "y": b["y"] + 1} for b in moved]
        if not can_place_list(next_pos):
            player["bottomed"] = True

        score_gain = actual_drop
        gs["score"] = gs.get("score", 0) + score_gain
        metrics["score"] = metrics.get("score", 0) + score_gain
        metrics["primary_score"] = metrics.get("primary_score", 0) + score_gain

        place_on_board(blocks)
        metrics["occupied_cells"] = get_occupied()

    elif action in ("move_down_a", "move_down_w", "move_down_x"):
        if action == "move_down_x":
            rotated = rotate_cw(blocks)
        elif action == "move_down_a":
            rotated = rotate_ccw(blocks)
        else:  # move_down_w
            rotated = rotate_cw(blocks)

        if can_place_list(rotated):
            blocks = rotated
        player["blocks"] = blocks
        place_on_board(blocks)
        metrics["occupied_cells"] = get_occupied()

    elif action == "move_left":
        new_blocks = [{"x": b["x"] - 1, "y": b["y"]} for b in blocks]
        if can_place_list(new_blocks):
            blocks = new_blocks
        player["blocks"] = blocks
        place_on_board(blocks)
        metrics["occupied_cells"] = get_occupied()

    elif action == "move_right":
        new_blocks = [{"x": b["x"] + 1, "y": b["y"]} for b in blocks]
        if can_place_list(new_blocks):
            blocks = new_blocks
        player["blocks"] = blocks
        place_on_board(blocks)
        metrics["occupied_cells"] = get_occupied()

    elif action == "use_arrowup":
        rotated = rotate_cw(blocks)
        if can_place_list(rotated):
            blocks = rotated
        player["blocks"] = blocks
        place_on_board(blocks)
        metrics["occupied_cells"] = get_occupied()

    elif action == "use_z":
        rotated = rotate_ccw(blocks)
        if can_place_list(rotated):
            blocks = rotated
        player["blocks"] = blocks
        place_on_board(blocks)
        metrics["occupied_cells"] = get_occupied()

    elif action == "use_space":
        # Hard drop: drop piece to bottom
        current = [dict(b) for b in blocks]
        drop_dist = 0
        while True:
            moved = [{"x": b["x"], "y": b["y"] + 1} for b in current]
            if not can_place_list(moved):
                break
            current = moved
            drop_dist += 1

        place_on_board(current)

        # Check for full lines
        lines_to_clear = []
        for r in range(rows):
            if all(board[r][c] != 0 for c in range(cols)):
                lines_to_clear.append(r)

        lines_cleared = len(lines_to_clear)
        score_gain = drop_dist * 2

        if lines_cleared > 0:
            level = gs.get("level", 1)
            line_scores = {1: 100, 2: 300, 3: 500, 4: 800}
            score_gain += line_scores.get(lines_cleared, lines_cleared * 100) * level
            for r in sorted(lines_to_clear, reverse=True):
                board.pop(r)
                board.insert(0, [0] * cols)
            metrics["lines_remaining"] = max(0, metrics.get("lines_remaining", 5) - lines_cleared)

        gs["score"] = gs.get("score", 0) + score_gain
        metrics["score"] = metrics.get("score", 0) + score_gain
        metrics["primary_score"] = metrics.get("primary_score", 0) + score_gain

        # Spawn next piece from queue
        queue = env["preview_queue"]
        next_shape = queue[0]
        new_queue = list(queue[1:])
        # The last element is unpredictable (new random piece), keep it unchanged
        new_queue.append(queue[-1])
        env["preview_queue"] = new_queue

        player["shape"] = next_shape
        player["bottomed"] = False
        spawn_b = get_spawn_blocks(next_shape)
        player["blocks"] = spawn_b
        blocks = spawn_b

        place_on_board(blocks)
        metrics["occupied_cells"] = get_occupied()

    else:
        place_on_board(blocks)
        metrics["occupied_cells"] = get_occupied()

    return s
