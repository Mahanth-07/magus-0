import copy


def predict(state, action):
    s = copy.deepcopy(state)

    gs = s.get("game_state", {})
    player = gs.get("player", {})
    env = gs.get("environment", {})
    top = env.get("top_stable_block", {})
    metrics = s.get("metrics", {})
    debug = s.get("debug", {})
    last = debug.get("last_block", {})

    # advance time
    s["gameTimeMs"] = s.get("gameTimeMs", 0) + 140
    s["timestampMs"] = s.get("timestampMs", 0) + 140

    level = gs.get("level", 1)
    plane = player.get("plane", "z")
    speed = player.get("speed", -0.11)
    aspeed = abs(speed)

    def set_camera(target):
        cam = env.get("camera_y", 0)
        if cam < target:
            newcam = cam + (target - cam) * 0.92
            if newcam > target:
                newcam = target
        else:
            newcam = target
        env["camera_y"] = newcam
        metrics["camera_y"] = newcam
        debug["camera_y"] = newcam

    if action in ("move_right", "move_right_d", "move_right_x"):
        step = 8 * aspeed
        if plane == "z":
            newz = player.get("z", 0) + step
            player["z"] = newz
            if "position" in last:
                last["position"]["z"] = newz
        else:
            newx = player.get("x", 0) + step
            player["x"] = newx
            if "position" in last:
                last["position"]["x"] = newx
        target = 2 * level + 6
        set_camera(target)
        return s

    if action == "use_space":
        # determine active moving axis and offset
        if plane == "z":
            below_size_A = top.get("size", {}).get("depth", 10)
            offset = player.get("z", 0)
        else:
            below_size_A = top.get("size", {}).get("width", 10)
            offset = player.get("x", 0)

        new_size = below_size_A - abs(offset)

        if new_size <= 0:
            # miss -> terminal fail
            if plane == "z":
                player["depth"] = new_size
                last.setdefault("size", {})["depth"] = new_size
            else:
                player["width"] = new_size
                last.setdefault("size", {})["width"] = new_size
            player["state"] = "missed"
            last["state"] = "missed"
            env["container_state"] = "ended"
            s["status"] = "terminal"
            debug["state"] = "ended"
            term = s.get("terminal", {})
            term["outcome"] = "fail"
            term["isTerminal"] = True
            term["reason"] = "missed_block"
            s["terminal"] = term
            metrics["deaths"] = metrics.get("deaths", 0) + 1
            metrics["attempts"] = metrics.get("attempts", 1) + 1
            debug["retries"] = debug.get("retries", 0) + 1
            s["is_actionable"] = False
            # camera may advance toward target
            set_camera(2 * level + 6)
            return s

        # success placement
        score = gs.get("score", 0) + 1
        gs["score"] = score
        metrics["score"] = score
        metrics["primary_score"] = metrics.get("primary_score", 0) + 1
        debug["ui_score"] = debug.get("ui_score", 0) + 1

        blocks = env.get("blocks_count", 2) + 1
        env["blocks_count"] = blocks
        metrics["blocks_count"] = blocks
        debug["blocks_count"] = blocks

        gs["level"] = level + 1

        old_dir = player.get("direction", 0.11)
        old_speed = player.get("speed", -0.11)
        old_axis = player.get("dimension_axis")
        old_plane = plane
        old_below_y = top.get("position", {}).get("y", 2)
        below_index = top.get("index", 1)

        # perpendicular size inherited on the new active block
        # placed block along-axis size = new_size

        # new top_stable = the placed active block
        top["index"] = below_index + 1
        top["plane"] = old_plane
        top["dimension_axis"] = old_axis
        top["direction"] = old_dir
        top["speed"] = old_speed
        top.setdefault("position", {})["y"] = old_below_y + 2
        top["position"]["x"] = 0
        top["position"]["z"] = 0
        if old_plane == "z":
            top.setdefault("size", {})["depth"] = new_size
        else:
            top.setdefault("size", {})["width"] = new_size

        # new active block
        new_dir = old_dir + 0.005
        new_speed = old_speed - 0.005
        player["block_index"] = player.get("block_index", 2) + 1
        player["direction"] = new_dir
        player["speed"] = new_speed
        player["y"] = player.get("y", 4) + 2
        player["state"] = "active"

        if old_plane == "z":
            # new active moves along x
            player["plane"] = "x"
            player["dimension_axis"] = "width"
            player["depth"] = new_size
            player["x"] = -11.31
            player["z"] = 0
        else:
            player["plane"] = "z"
            player["dimension_axis"] = "depth"
            player["width"] = new_size
            player["z"] = -11.28
            player["x"] = 0

        # last_block mirrors new active
        last["index"] = player["block_index"]
        last["plane"] = player["plane"]
        last["dimension_axis"] = player["dimension_axis"]
        last["direction"] = new_dir
        last["speed"] = new_speed
        last.setdefault("position", {})["y"] = player["y"]
        last["position"]["x"] = player["x"]
        last["position"]["z"] = player["z"]
        if player["plane"] == "x":
            last.setdefault("size", {})["depth"] = new_size
        else:
            last.setdefault("size", {})["width"] = new_size

        set_camera(2 * (level + 1) + 6)
        return s

    return s
