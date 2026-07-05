import copy

SCROLL = 13.88
MOVE = 6.96

def predict(state, action):
    s = copy.deepcopy(state)

    gs = s.get("game_state", {})
    player = gs.get("player", {})
    env = gs.get("environment", {})
    plats = env.get("nearby_platforms", [])

    if isinstance(s.get("gameTimeMs"), (int, float)):
        s["gameTimeMs"] = s["gameTimeMs"] + 140
    if isinstance(s.get("timestampMs"), (int, float)):
        s["timestampMs"] = s["timestampMs"] + 140

    on_floor = player.get("on_floor", False)

    if isinstance(player.get("y"), (int, float)):
        player["y"] = player["y"] - SCROLL

    player["vy"] = -0.1
    player["on_floor"] = True

    if action == "move_right":
        if isinstance(player.get("x"), (int, float)):
            player["x"] = player["x"] + MOVE
    elif action == "move_left":
        if isinstance(player.get("x"), (int, float)):
            player["x"] = player["x"] - MOVE

    for p in plats:
        if isinstance(p.get("y"), (int, float)):
            p["y"] = p["y"] - SCROLL

    return s
