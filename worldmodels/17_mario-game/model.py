import copy

AREA_WIDTH = 1792

def predict(state, action):
    s = copy.deepcopy(state)

    gt = s.get("gameTimeMs")
    if isinstance(gt, (int, float)):
        s["gameTimeMs"] = gt + 140
    ts = s.get("timestampMs")
    if isinstance(ts, (int, float)):
        s["timestampMs"] = ts + 140

    gs = s.get("game_state", {})
    p = gs.get("player", {})

    x = p.get("x", 0)
    y = p.get("y", 384)
    vy = p.get("vy", 0)

    GROUND = 384
    GRAVITY = 3.84
    DX = 0.6616266278342

    # Horizontal movement
    if action in ("move_left", "move_left_a"):
        x = x - DX
    elif action == "use_arrowright":
        x = x + DX

    # Jump initiation (only when on ground)
    on_ground = (y >= GROUND - 0.001 and abs(vy) < 0.001)
    if action in ("use_w", "use_arrowup") and on_ground:
        vy = -2.563854909329549
        y = y + vy

    else:
        # Apply gravity physics only if airborne (vy != 0 or y < ground)
        if not (y >= GROUND - 0.001 and abs(vy) < 0.001):
            new_vy = vy + GRAVITY
            new_y = y + new_vy
            if new_y >= GROUND:
                new_y = GROUND
                new_vy = 0
            y = new_y
            vy = new_vy

    p["x"] = x
    p["y"] = y
    p["vy"] = vy

    # completion_progress based on x
    # from data: x=63.338 -> 0.035345074426431804
    #           x=62.677 -> 0.0349758631385779
    #           x=64     -> 0.03571428571428571
    #           x=64.662 -> 0.03608349700213962
    #           x=62.015 -> 0.03460665185072399
    if isinstance(x, (int, float)):
        prog = x / AREA_WIDTH
        gs["completion_progress"] = prog
        m = s.get("metrics", {})
        m["level_progress_percent"] = int(round(prog * 100))

    return s
