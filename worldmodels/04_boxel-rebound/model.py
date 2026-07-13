import copy

def predict(state, action):
    s = copy.deepcopy(state)
    gs = s.get("game_state", {})
    player = gs.get("player", {})
    env = gs.get("environment", {})

    dt = 136
    if "gameTimeMs" in s:
        s["gameTimeMs"] = s["gameTimeMs"] + dt
    if "timestampMs" in s:
        s["timestampMs"] = s["timestampMs"] + dt

    x = player.get("x", 0)
    vx = player.get("vx", 0.3)

    # horizontal advance: dx ~ vx * 138.5 (empirically ~41.5 at vx=0.3)
    dx = vx * 138.5
    new_x = x + dx
    player["x"] = new_x

    jump_actions = {"move_up", "use_space"}
    if action in jump_actions and player.get("jump_ready", True):
        player["jump_ready"] = False

    level_width = env.get("level_width", 2820)
    prog = new_x / level_width
    gs["completion_progress"] = prog
    if "metrics" in s:
        s["metrics"]["primary_score"] = prog

    if "score" in gs and "gameTimeMs" in s:
        gs["score"] = s["gameTimeMs"]

    return s
