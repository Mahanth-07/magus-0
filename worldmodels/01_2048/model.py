import copy
import math

def predict(state, action):
    ns = copy.deepcopy(state)
    gs = ns["game_state"]
    env = gs["environment"]
    N = len(env)
    m = ns["metrics"]

    if action == "use_space":
        # reset: deterministic scalar fields, random spawns unpredictable
        gs["score"] = 0
        gs["level"] = 2
        gs["completion_progress"] = math.log2(2) / 11
        m["primary_score"] = 0
        m["max_tile"] = 2
        m["filled_cells"] = 2
        m["board_fill_ratio"] = 2 / (N * N)
        return ns

    if action not in ("use_arrowleft", "use_arrowright", "use_arrowup", "use_arrowdown"):
        return ns

    # environment is indexed env[x][y]
    orig = [[env[x][y] for y in range(N)] for x in range(N)]
    grid = [[env[x][y] for y in range(N)] for x in range(N)]

    score_gain = 0

    def slide_line(line):
        nonlocal score_gain
        vals = [v for v in line if v != 0]
        res = []
        i = 0
        while i < len(vals):
            if i + 1 < len(vals) and vals[i] == vals[i + 1]:
                merged = vals[i] * 2
                res.append(merged)
                score_gain += merged
                i += 2
            else:
                res.append(vals[i])
                i += 1
        while len(res) < len(line):
            res.append(0)
        return res

    if action == "use_arrowleft":  # slide toward x=0
        for y in range(N):
            line = [grid[x][y] for x in range(N)]
            line = slide_line(line)
            for x in range(N):
                grid[x][y] = line[x]
    elif action == "use_arrowright":  # slide toward x=N-1
        for y in range(N):
            line = [grid[x][y] for x in range(N)]
            line = list(reversed(slide_line(list(reversed(line)))))
            for x in range(N):
                grid[x][y] = line[x]
    elif action == "use_arrowup":  # slide toward y=0
        for x in range(N):
            line = grid[x][:]
            line = slide_line(line)
            grid[x] = line
    elif action == "use_arrowdown":  # slide toward y=N-1
        for x in range(N):
            line = grid[x][:]
            line = list(reversed(slide_line(list(reversed(line)))))
            grid[x] = line

    # write back
    for x in range(N):
        for y in range(N):
            env[x][y] = grid[x][y]

    gs["score"] = gs.get("score", 0) + score_gain

    # rebuild deterministic tile entities (omit random spawn)
    entities = []
    for x in range(N):
        for y in range(N):
            if grid[x][y] != 0:
                entities.append({
                    "type": "tile",
                    "x": x,
                    "y": y,
                    "props": {"value": grid[x][y]}
                })
    gs["entities"] = entities

    filled = sum(1 for x in range(N) for y in range(N) if grid[x][y] != 0)
    max_tile = max((grid[x][y] for x in range(N) for y in range(N)), default=0)
    total = N * N

    m["primary_score"] = gs["score"]
    m["max_tile"] = max_tile
    m["filled_cells"] = filled
    m["total_cells"] = total
    m["board_fill_ratio"] = filled / total

    gs["level"] = max_tile
    gs["completion_progress"] = math.log2(max_tile) / 11 if max_tile > 0 else 0.0

    return ns
