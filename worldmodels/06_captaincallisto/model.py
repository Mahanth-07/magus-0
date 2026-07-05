import copy

def predict(state, action):
    s = copy.deepcopy(state)
    gs = s.get("game_state", {})
    player = gs.get("player", {})

    x = player.get("x", 0)
    y = player.get("y", 0)
    z = player.get("z", 0)

    # Determine horizontal movement (x-axis)
    dx = 0.0
    if action.startswith("move_left"):
        dx = -0.22
    elif action.startswith("move_right"):
        # move_right base and various suffixes move right; but some suffixes
        # (arrowdown, arrowup, s, w, z, x) primarily affect z or y
        if action in ("move_right", "move_right_d"):
            dx = 0.22
        else:
            dx = -0.09  # slight drift observed with modifier keys
    # move_right_d clearly moves right
    if action == "move_right_d":
        dx = 0.22

    # Determine z movement (arrowup/arrowdown, w/s toward/away)
    dz = 0.0
    if action in ("move_right_arrowup", "move_right_w"):
        dz = 0.23  # move toward positive z
    elif action in ("move_right_arrowdown", "move_right_s"):
        dz = -0.24  # move toward negative z

    # Jump (z key) - vertical impulse
    # gravity: y falls toward ~3.987 (ground) unless jumping
    ground_y = 3.987

    new_x = x + dx
    new_z = z + dz

    # Vertical physics
    new_y = y
    if action == "move_right_z":
        # jump impulse
        if abs(y - ground_y) < 0.05:
            new_y = y + 1.057  # initial jump
        else:
            new_y = y + 0.5  # rising
    else:
        # apply gravity / settle
        if y > ground_y + 0.05:
            new_y = y - 0.23  # falling
            if new_y < ground_y:
                new_y = ground_y
        else:
            new_y = ground_y

    player["x"] = new_x
    player["y"] = new_y
    player["z"] = new_z

    return s
