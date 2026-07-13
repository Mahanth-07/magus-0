import copy

def predict(state, action):
    s = copy.deepcopy(state)
    gs = s["game_state"]
    player = gs["player"]
    ts = gs.get("environment", {}).get("tile_size", 40)

    # time advance ~ 136-140 ms
    dt = 137
    s["timestampMs"] = s["timestampMs"] + dt
    s["gameTimeMs"] = s["gameTimeMs"] + dt
    s["debug"]["game_timer_ms"] = s["debug"]["game_timer_ms"] + dt

    # Player movement
    dx = dy = 0
    if action in ("move_left", "move_left_a"):
        dx = -3
    elif action in ("move_right", "move_right_d"):
        dx = 3
    elif action in ("move_down", "move_down_s"):
        dy = 3
    elif action in ("move_up", "move_up_w"):
        dy = -3
    # bouncing observed: down can move +6 when tile crosses; approximate simple
    player["x"] = player["x"] + dx
    player["y"] = player["y"] + dy
    player["tile_x"] = round(player["x"] / ts - 0.0, 6)
    player["tile_x"] = player["x"] / ts
    player["tile_y"] = player["y"] / ts

    # Enemies: each moves along track by 35 per step, bouncing between bounds.
    ents = gs.get("entities", [])

    # Determine per-entity velocity from observed step size 35, with bounce points.
    # Left group entities move within [264..536] roughly; right group same range.
    # We track direction implicitly via distance to bounds.
    # From data: bounds appear to be ~[264, 536] (min/max seen).
    XMIN = 264
    XMAX = 536

    def next_x(x, vsign):
        return x + 35 * vsign

    # We don't have stored velocity, so infer from position relative to bounds:
    # If near max, go down; if near min, go up. But this is ambiguous mid-track.
    # Best: leave entities unchanged (unpredictable without velocity state).
    return s
