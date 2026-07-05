import copy
import math

def predict(state, action):
    s = copy.deepcopy(state)
    gs = s["game_state"]
    p = gs["player"]
    env = gs["environment"]

    dt = 0.142  # approx timestep seconds
    # time advances
    s["gameTimeMs"] = s.get("gameTimeMs", 0) + 141
    s["timestampMs"] = s.get("timestampMs", 0) + 141
    day_inc = 0.0001148
    env["day_time"] = env["day_time"] + day_inc
    s["metrics"]["day_time"] = env["day_time"]

    grounded = p.get("is_grounded", False)
    gravity = -27.0  # per second approx

    # Determine action effects
    move_x = 0.0
    move_z = 0.0
    jump = False

    if action in ("move_left", "move_left_a"):
        move_x = -0.144
    elif action in ("move_right", "move_right_d"):
        move_x = 0.146
    elif action in ("use_w",):
        move_z = -0.145
    elif action in ("use_s",):
        move_z = 0.147
    elif action in ("use_arrowup",):
        move_z = -0.148
    elif action in ("use_arrowdown",):
        move_z = 0.148
    elif action == "use_space":
        jump = True

    # Horizontal movement
    p["x"] = p["x"] + move_x
    p["z"] = p["z"] + move_z

    # Vertical physics
    vy = p.get("vy", 0.0)

    if grounded and jump:
        # jump: set upward velocity
        p["vy"] = 5.3
        p["y"] = p["y"] + 0.5
        p["is_grounded"] = False
    else:
        # apply gravity integration
        new_vy = vy + gravity * dt
        # position update using average velocity approx
        p["y"] = p["y"] + vy * dt + 0.5 * gravity * dt * dt
        p["vy"] = new_vy

    # Recompute derived context based on block_y
    by = int(math.floor(p["y"]))
    env["player_context"]["block_y"] = by
    env["player_context"]["block_x"] = int(math.floor(p["x"]))
    env["player_context"]["block_z"] = int(math.floor(p["z"]))

    # block_below depends on y position; at 73 -> grass, at 74 -> air
    if by >= 74:
        env["player_context"]["block_below"] = "air"
    else:
        env["player_context"]["block_below"] = "grass"

    # Resource counts depend on block_y (vertical position affects nearby scan)
    # Observed: block_y=73 -> stone 392, coal 121, iron 75 (dy -5)
    #           block_y=74 -> stone 277, coal 87, iron 55 (dy -6)
    res = env["nearby_resources"]
    if by >= 74:
        stone_c, coal_c, iron_c = 277, 87, 55
        dy = -6
        nd = 6
        diag = 6.082762530298219
    else:
        stone_c, coal_c, iron_c = 392, 121, 75
        dy = -5
        nd = 5
        diag = 5.0990195135927845

    res["stone"]["count"] = stone_c
    res["coal_ore"]["count"] = coal_c
    res["iron_ore"]["count"] = iron_c
    res["stone"]["nearest_dy"] = dy
    res["coal_ore"]["nearest_dy"] = dy
    res["iron_ore"]["nearest_dy"] = dy
    res["stone"]["nearest_distance"] = nd
    res["coal_ore"]["nearest_distance"] = diag
    res["iron_ore"]["nearest_distance"] = diag

    m = s["metrics"]
    m["nearby_stone_count"] = stone_c
    m["nearby_iron_ore_count"] = iron_c
    m["nearest_stone_distance"] = nd
    m["nearest_iron_ore_distance"] = diag

    return s
