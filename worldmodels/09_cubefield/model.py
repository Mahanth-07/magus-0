import copy

def predict(state, action):
    ns = copy.deepcopy(state)

    gs = ns.get("game_state", {})
    metrics = ns.get("metrics", {})

    inc = 400
    if "score" in gs:
        gs["score"] = gs["score"] + inc
    if "primary_score" in metrics:
        metrics["primary_score"] = metrics["primary_score"] + inc
    if "distance" in metrics:
        metrics["distance"] = metrics["distance"] + inc

    # speed_side, roll_angle, cubes_alive, and entity positions are
    # governed by variable dt / random spawns -> copy through unchanged.

    return ns
