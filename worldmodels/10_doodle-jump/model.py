import copy

def predict(state, action):
    s = copy.deepcopy(state)
    gs = s.get("game_state", {})
    p = gs.get("player", {})
    vy = p.get("vy", 0)
    x = p.get("x", 0)

    dvy = 1.6
    new_vy = vy + dvy
    dy = (vy + 0.8) * 9.0
    p["y"] = p.get("y", 0) + dy
    p["vy"] = new_vy

    if action == "move_right":
        p["x"] = x + 0.45
        p["dir"] = "right"
    elif action == "move_left":
        p["x"] = x - 0.45
        p["dir"] = "left"

    dt = 143
    if "gameTimeMs" in s:
        s["gameTimeMs"] = s["gameTimeMs"] + dt
    if "timestampMs" in s:
        s["timestampMs"] = s["timestampMs"] + dt

    return s
