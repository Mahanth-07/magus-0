import copy, math

def predict(state, action):
    ns = copy.deepcopy(state)
    gs = ns["game_state"]
    p = gs["player"]

    decay = 0.6634
    dt = 0.1477

    # vertical impulse based on arrow keys
    imp_y = 0.0
    if "arrowup" in action:
        imp_y = 0.5518
    elif "arrowdown" in action:
        imp_y = -0.5518 * decay

    vy_old = p.get("vy", 0.0)
    vy_new = vy_old * decay + imp_y

    # horizontal impulse
    imp_x = 0.0
    if action.startswith("move_left"):
        imp_x = 0.0
    if action.startswith("move_right"):
        imp_x = 0.0

    vx_old = p.get("vx", 0.0)
    vx_new = vx_old * decay + imp_x

    p["vy"] = vy_new
    p["vx"] = vx_new
    p["y"] = p["y"] + vy_new * dt
    p["x"] = p["x"] + vx_new * dt

    return ns
