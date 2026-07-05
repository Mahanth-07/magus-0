import copy

def predict(state, action):
    s = copy.deepcopy(state)
    gs = s.get("game_state", {})
    player = gs.get("player", {})
    pos = player.get("position", {})
    env = gs.get("environment", {})
    metrics = s.get("metrics", {})

    dt_time = 138
    if s.get("gameTimeMs") is not None:
        s["gameTimeMs"] += dt_time
    if s.get("timestampMs") is not None:
        s["timestampMs"] += dt_time

    old_dist = gs.get("distance", 0.0)
    old_z = pos.get("z", 0.0)

    z_delta = 1.734
    dist_delta = 1.734

    new_dist = old_dist + dist_delta
    new_z = old_z - z_delta

    gs["distance"] = new_dist
    metrics["distance"] = new_dist
    pos["z"] = new_z

    score_delta = int(round(dist_delta * 10))
    old_score = gs.get("score", 0)
    new_score = old_score + score_delta
    gs["score"] = new_score
    metrics["score"] = new_score
    metrics["primary_score"] = new_score

    pp = env.get("piece_progress", 0)
    if pp is None:
        pp = 0
    pp_new = pp + dist_delta * 0.171
    if pp_new >= 1.0:
        pp_new -= 1.0
    env["piece_progress"] = pp_new

    y = pos.get("y", 0.0)
    pos["y"] = y - 1.0e-7

    x = pos.get("x", 0.0)
    pos["x"] = x - 1.1e-7

    return s
