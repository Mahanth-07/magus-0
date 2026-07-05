import copy

DIRS = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
}

REVERSE = {
    "UP": "DOWN",
    "DOWN": "UP",
    "LEFT": "RIGHT",
    "RIGHT": "LEFT",
}

ACTION_DIR = {
    "move_up": "UP",
    "move_up_w": "UP",
    "move_up_space": "UP",
    "move_up_s": "DOWN",
    "move_up_arrowdown": "DOWN",
    "move_up_x": None,          # unknown / no direction change observed cleanly
    "use_a": "LEFT",
    "use_arrowleft": "LEFT",
    "use_arrowright": "RIGHT",
    "use_d": "RIGHT",
}


def predict(state, action):
    s = copy.deepcopy(state)

    gs = s.get("game_state", {})
    player = gs.get("player", {})
    cur_dir = player.get("direction", "RIGHT")

    # movement uses CURRENT direction, then direction is updated afterwards
    dx, dy = DIRS.get(cur_dir, (0, 0))

    entities = gs.get("entities", [])

    # find head
    head = None
    for e in entities:
        if e.get("type") == "snake_head":
            head = e
            break

    if head is not None:
        old_head = (head["x"], head["y"])
        new_head = (old_head[0] + dx, old_head[1] + dy)

        # bodies in order (list order = tail..head as observed)
        bodies = [e for e in entities if e.get("type") == "snake_body"]

        # shift: each body takes the position of the next entity ahead of it
        # order in list: body0(tail) .. body_{n-1} then head
        positions = [(b["x"], b["y"]) for b in bodies] + [old_head]
        # new positions: drop tail, append new head
        new_positions = positions[1:] + [new_head]

        for i, b in enumerate(bodies):
            b["x"], b["y"] = new_positions[i]
        head["x"], head["y"] = new_head

        player["x"], player["y"] = new_head

    # update direction (reversal blocked)
    req = ACTION_DIR.get(action)
    if req is not None and req != REVERSE.get(cur_dir):
        new_dir = req
    else:
        new_dir = cur_dir

    player["direction"] = new_dir
    if "debug" in s:
        s["debug"]["direction"] = new_dir
        if "ticks" in s["debug"]:
            s["debug"]["ticks"] = s["debug"]["ticks"] + 1

    if "gameTimeMs" in s:
        s["gameTimeMs"] = s["gameTimeMs"] + 270

    return s
