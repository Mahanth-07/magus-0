import copy

def predict(state, action):
    s = copy.deepcopy(state)
    gs = s.get("game_state", {})
    player = gs.get("player", {})
    metrics = s.get("metrics", {})

    if s.get("gameTimeMs") is not None:
        s["gameTimeMs"] = s["gameTimeMs"] + 138
    if s.get("timestampMs") is not None:
        s["timestampMs"] = s["timestampMs"] + 137

    x = player.get("x")
    if action == "move_left" and x is not None:
        player["x"] = max(0, x - 12)
    elif action == "move_right" and x is not None:
        player["x"] = x + 12

    # flap_meter decrements on "flap" actions
    flap_actions = {"use_z", "move_down", "move_down_s", "move_down_space", "move_right"}
    if action in flap_actions:
        fm = player.get("flap_meter")
        if fm is not None and fm > 0:
            player["flap_meter"] = fm - 1

    # score resets to 0 on leftward/downward reset moves
    if action in ("move_left", "move_down_s"):
        if "score" in gs:
            gs["score"] = 0
        if "primary_score" in metrics:
            metrics["primary_score"] = 0

    return s
