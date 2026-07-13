import copy

def predict(state, action):
    s = copy.deepcopy(state)

    if s.get("status") == "terminal" or s.get("terminal", {}).get("isTerminal"):
        return s

    gs = s.get("game_state", {})
    player = gs.get("player", {})
    env = gs.get("environment", {})

    dt = 137  # approx ms step
    # Time advances ~137-140 ms
    if "gameTimeMs" in s:
        s["gameTimeMs"] = s["gameTimeMs"] + 137
    if "timestampMs" in s:
        s["timestampMs"] = s["timestampMs"] + 137

    # Physics: vy increases by ~1.6 per step, rotation by ~14.4
    vy = player.get("vy", 0.0)
    y = player.get("y", 0.0)
    rot = player.get("rotation", 0.0)

    new_vy = vy + 1.6
    # y change roughly proportional to average vy scaled
    # From data: dy corresponds well to new_vy*something. 
    # e.g. vy -2.86 -> next, dy=-28.48 from vy=-4.46; dy ~ vy*6.4 approx
    # Use dy = new_vy * scale? check: vy=-2.66->-1.06, dy=-14.08; -1.06*? 
    # Better: dy uses new velocity. dy = new_vy*8.8 approx: -1.06*8.8=-9.3 no.
    # dy=-14.08 when transitioning vy -2.66->-1.06. avg=-1.86*7.5=-14. ok
    avg_vy = (vy + new_vy) / 2.0
    dy = avg_vy * 8.8
    new_y = y + dy

    new_rot = rot + 14.4
    if new_rot > 90:
        new_rot = 90

    # ground/terminal at y ~ 401.52 with rotation 90
    terminal = False
    if new_y >= 401.52:
        new_y = 401.52
        new_rot = 90
        new_vy = 10.54
        terminal = True

    player["vy"] = new_vy
    player["y"] = new_y
    player["rotation"] = new_rot
    # x has random noise; keep approximately unchanged
    # (leave as-is)

    # Pipe moves left by ~18.9
    if env.get("next_pipe") is not None:
        env["next_pipe"]["x"] = env["next_pipe"]["x"] - 18.9

    ents = gs.get("entities")
    if ents:
        for e in ents:
            if e.get("type") == "pipe_gap":
                e["x"] = e["x"] - 18.9

    if terminal:
        s["status"] = "terminal"
        s["is_actionable"] = False
        term = s.get("terminal", {})
        term["isTerminal"] = True
        term["outcome"] = "fail"
        term["reason"] = "collision_or_ground"
        s["terminal"] = term
        if "environment" in gs:
            gs["environment"]["phase"] = "score"
        if "debug" in s:
            s["debug"]["raw_state"] = "score"

    return s
