import copy
import math


def _slide_line(vals):
    # vals: list from the "start" end toward far end; merge toward start (index 0)
    nonzero = [v for v in vals if v != 0]
    result = []
    gained = 0
    i = 0
    while i < len(nonzero):
        if i + 1 < len(nonzero) and nonzero[i] == nonzero[i + 1]:
            merged = nonzero[i] * 2
            result.append(merged)
            gained += merged
            i += 2
        else:
            result.append(nonzero[i])
            i += 1
    while len(result) < len(vals):
        result.append(0)
    return result, gained


def predict(state, action):
    ns = copy.deepcopy(state)
    gs = ns["game_state"]

    N = 4
    # build grid[x][y] from environment (environment indexed [x][y])
    env = gs.get("environment")
    grid = [[0] * N for _ in range(N)]
    for x in range(N):
        for y in range(N):
            try:
                grid[x][y] = env[x][y]
            except Exception:
                grid[x][y] = 0

    total_gained = 0
    moved = False

    if action in ("use_arrowleft", "use_arrowright"):
        for y in range(N):
            line = [grid[x][y] for x in range(N)]
            if action == "use_arrowright":
                line = line[::-1]
            newline, g = _slide_line(line)
            total_gained += g
            if action == "use_arrowright":
                newline = newline[::-1]
            for x in range(N):
                if grid[x][y] != newline[x]:
                    moved = True
                grid[x][y] = newline[x]
    elif action in ("use_arrowup", "use_arrowdown"):
        for x in range(N):
            line = [grid[x][y] for y in range(N)]
            if action == "use_arrowdown":
                line = line[::-1]
            newline, g = _slide_line(line)
            total_gained += g
            if action == "use_arrowdown":
                newline = newline[::-1]
            for y in range(N):
                if grid[x][y] != newline[y]:
                    moved = True
                grid[x][y] = newline[y]
    elif action == "use_space":
        # reset - unpredictable random spawns; reset scalar fields only
        gs["score"] = 0
        gs["level"] = 2
        gs["completion_progress"] = 1.0 / 11.0
        m = ns.get("metrics", {})
        m["primary_score"] = 0
        m["max_tile"] = 2
        return ns
    else:
        return ns

    # rebuild environment
    new_env = [[0] * N for _ in range(N)]
    for x in range(N):
        for y in range(N):
            new_env[x][y] = grid[x][y]
    gs["environment"] = new_env

    # rebuild entities (omit random spawn)
    entities = []
    max_tile = 0
    filled = 0
    for x in range(N):
        for y in range(N):
            v = grid[x][y]
            if v != 0:
                entities.append({"type": "tile", "x": x, "y": y, "props": {"value": v}})
                filled += 1
                if v > max_tile:
                    max_tile = v
    gs["entities"] = entities

    if max_tile < 2:
        max_tile = 2

    # score
    gs["score"] = gs.get("score", 0) + total_gained
    gs["level"] = max_tile
    gs["completion_progress"] = math.log2(max_tile) / 11.0

    m = ns.get("metrics", {})
    m["primary_score"] = gs["score"]
    m["max_tile"] = max_tile
    m["filled_cells"] = filled
    m["board_fill_ratio"] = filled / 16.0

    return ns
