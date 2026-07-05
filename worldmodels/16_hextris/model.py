import copy

def predict(state, action):
    s = copy.deepcopy(state)
    gs = s.get("game_state", {})
    player = gs.get("player", {})

    # advance time approximately
    dt = 139
    if "gameTimeMs" in s and isinstance(s["gameTimeMs"], (int, float)):
        s["gameTimeMs"] = s["gameTimeMs"] + dt
    if "timestampMs" in s and isinstance(s["timestampMs"], (int, float)):
        s["timestampMs"] = s["timestampMs"] + dt

    # player x advances by 8
    if "x" in player and isinstance(player["x"], (int, float)):
        player["x"] = player["x"] + 8

    # distance mirrors x
    if "metrics" in s and isinstance(s["metrics"], dict):
        if "distance" in s["metrics"] and isinstance(s["metrics"]["distance"], (int, float)):
            s["metrics"]["distance"] = s["metrics"]["distance"] + 8

    # rotation actions
    left_actions = {"use_arrowleft", "use_a", "use_space"}
    right_actions = {"use_arrowright", "use_d"}

    lane = player.get("lane_rotation")
    angle = player.get("angle")

    if action in left_actions:
        if isinstance(lane, (int, float)):
            player["lane_rotation"] = (lane + 1) % 6
        if isinstance(angle, (int, float)):
            player["angle"] = angle - 60
    elif action in right_actions:
        if isinstance(lane, (int, float)):
            player["lane_rotation"] = (lane - 1) % 6
        if isinstance(angle, (int, float)):
            player["angle"] = angle + 60

    return s
