import copy

def predict(state, action):
    s = copy.deepcopy(state)

    gs = s.get("game_state", {})
    env = gs.get("environment", {})
    player = gs.get("player", {})
    ents = gs.get("entities", [])

    # advance time by a typical frame
    dt = 140
    if "gameTimeMs" in s and isinstance(s["gameTimeMs"], (int, float)):
        s["gameTimeMs"] = s["gameTimeMs"] + dt
    if "timestampMs" in s and isinstance(s["timestampMs"], (int, float)):
        s["timestampMs"] = s["timestampMs"] + dt

    court_left = env.get("court_left", 145)
    court_top = env.get("court_top", 87)
    court_width = env.get("court_width", 510)
    court_height = env.get("court_height", 425)
    court_right = court_left + court_width
    court_bottom = court_top + court_height

    # player movement
    move = 0.0
    if action in ("move_left", "move_left_a"):
        move = -11.9
    elif action in ("move_right", "move_right_d"):
        move = 11.9

    if move != 0.0 and player:
        w = player.get("w", 0)
        newx = player.get("x", 0) + move
        # clamp so paddle stays in court
        minx = court_left
        maxx = court_right - w
        if newx < minx:
            newx = minx
        if newx > maxx:
            newx = maxx
        player["x"] = newx

    # ball movement (best-effort linear step with wall reflection)
    factor = 35.7
    for e in ents:
        if e.get("type") != "ball":
            continue
        st = e.get("state")
        if st == "moving":
            vx = e.get("vx", 0)
            vy = e.get("vy", 0)
            r = e.get("props", {}).get("radius", 5.1)
            nx = e.get("x", 0) + vx * factor
            ny = e.get("y", 0) + vy * factor
            # reflect off left/right walls
            if nx - r < court_left:
                nx = court_left + r + (court_left + r - nx)
                vx = -vx
            elif nx + r > court_right:
                nx = court_right - r - (nx + r - court_right)
                vx = -vx
            # reflect off top wall
            if ny - r < court_top:
                ny = court_top + r + (court_top + r - ny)
                vy = -vy
            e["x"] = nx
            e["y"] = ny
            e["vx"] = vx
            e["vy"] = vy

    return s
