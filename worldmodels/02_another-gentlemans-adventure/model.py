import copy

def predict(state, action):
    s = copy.deepcopy(state)
    gs = s.get("game_state", {})
    p = gs.get("player", {})
    props = p.get("props", {})

    # Time progression: ~140ms per step
    dt_game = 140
    if "gameTimeMs" in s and isinstance(s["gameTimeMs"], (int, float)):
        s["gameTimeMs"] = s["gameTimeMs"] + dt_game
    if "timestampMs" in s and isinstance(s["timestampMs"], (int, float)):
        s["timestampMs"] = s["timestampMs"] + dt_game

    dt = 0.140  # seconds
    max_speed = props.get("max_speed", 50)
    ground_y = 39.96330622223366

    x = p.get("x", 0)
    y = p.get("y", 0)
    vy = p.get("vy", 0)
    facing = props.get("facing", "right")
    pstate = p.get("state", "small_idle")

    gravity = 1200  # approx, derived from vy changes
    on_ground = (pstate in ("small_idle",)) and abs(vy) < 1e-6

    left_actions = ("move_left", "move_left_a")
    right_actions = ("move_right", "move_right_d")
    jump_actions = ("use_w", "use_z", "use_space", "use_arrowup")

    # Horizontal movement
    if action in left_actions:
        facing = "left"
        x = x - max_speed * dt * 0.243  # observed ~1.7 per step
        x = x - 0.0  # placeholder
    elif action in right_actions:
        facing = "right"

    # More accurate: observed horizontal delta ~1.7 units per move step
    # Recompute cleanly
    x = p.get("x", 0)
    if action in left_actions:
        facing = "left"
        x = x - 1.7
    elif action in right_actions:
        facing = "right"
        # moving right increases x by ~1.7 but if was at left mostly toward ground
        x = x + 1.7

    # Jump
    if action in jump_actions and abs(vy) < 1e-6 and abs(y - ground_y) < 1.0:
        vy = -157.0
        y = y + vy * 0.140 + 0.5 * gravity * 0.140 * 0.140
        pstate = "small_jump"
    else:
        # apply gravity if airborne
        if pstate in ("small_jump", "small_fall") or abs(y - ground_y) > 0.5:
            new_vy = vy + gravity * dt
            new_y = y + vy * dt + 0.5 * gravity * dt * dt
            if new_y >= ground_y:
                new_y = ground_y
                new_vy = 0
                pstate = "small_idle"
            elif new_vy >= 0:
                pstate = "small_fall"
            else:
                pstate = "small_jump"
            y = new_y
            vy = new_vy

    p["x"] = x
    p["y"] = y
    p["vy"] = vy
    p["state"] = pstate
    props["facing"] = facing

    return s
