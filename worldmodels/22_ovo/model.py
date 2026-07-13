import copy
import math

def predict(state, action):
    s = copy.deepcopy(state)
    gs = s.get("game_state", {})
    player = gs.get("player", {})
    env = gs.get("environment", {})

    # Horizontal movement is the deterministic part
    dx_step = 1.7185  # approx per-step horizontal delta

    if action == "move_left":
        if isinstance(player.get("x"), (int, float)):
            player["x"] = player["x"] - dx_step
        if isinstance(env.get("scroll_x"), (int, float)):
            env["scroll_x"] = env["scroll_x"] - dx_step
        # facing left
        if isinstance(player.get("width"), (int, float)):
            player["width"] = -abs(player["width"])
    elif action == "move_right":
        if isinstance(player.get("x"), (int, float)):
            player["x"] = player["x"] + dx_step
        if isinstance(env.get("scroll_x"), (int, float)):
            env["scroll_x"] = env["scroll_x"] + dx_step
        if isinstance(player.get("width"), (int, float)):
            player["width"] = abs(player["width"])

    # Recompute distance to exit based on scroll position vs exit
    exit_ent = None
    for e in gs.get("entities", []):
        if e.get("type") == "exit":
            exit_ent = e
            break

    if exit_ent is not None:
        ex = exit_ent.get("x")
        ey = exit_ent.get("y")
        sx = env.get("scroll_x")
        sy = env.get("scroll_y")
        if all(isinstance(v, (int, float)) for v in (ex, ey, sx, sy)):
            dist = math.sqrt((ex - sx) ** 2 + (ey - sy) ** 2)
            gs["distance_to_exit"] = dist
            env.setdefault("closest_exit", {})
            if isinstance(env.get("closest_exit"), dict):
                env["closest_exit"]["distance"] = dist
            exit_ent.setdefault("props", {})
            exit_ent["props"]["distance"] = dist
            if isinstance(s.get("metrics"), dict):
                s["metrics"]["distance_to_goal"] = dist

    return s
