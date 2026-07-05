import copy

def predict(state, action):
    s = copy.deepcopy(state)
    gs = s["game_state"]
    p = gs["player"]
    goal = gs["environment"]["goal"]

    s["metrics"]["steps"] += 1

    x, y = p["x"], p["y"]
    if action == "move_right":
        x = min(4, x + 1)
    elif action == "move_left":
        x = max(0, x - 1)
    elif action == "move_up":
        y = max(0, y - 1)
    elif action == "move_down":
        y = min(4, y + 1)

    p["x"], p["y"] = x, y

    if x == goal["x"] and y == goal["y"]:
        s["metrics"]["score"] += 10
        # goal respawns randomly; leave unchanged

    return s
