import copy

def predict(state, action):
    s = copy.deepcopy(state)
    if s.get("status") == "terminal":
        return s
    gs = s["game_state"]
    env = gs["environment"]
    m = s["metrics"]

    attached = env["attached_count"]
    new_attached = attached + 1
    total_required = env["total_required"]

    # score / queue
    gs["score"] += 1
    m["primary_score"] = gs["score"]
    env["attached_count"] = new_attached
    m["attached_count"] = new_attached
    env["queue_remaining"] -= 1
    m["queue_remaining"] = env["queue_remaining"]
    gs["completion_progress"] = gs["score"] / 8.0

    # terminal condition: attached reaches 8
    terminal = (new_attached >= 8)

    if terminal:
        env["core_angle"] += 6
        env["core_state"] = "fail"
        s["status"] = "terminal"
        s["terminal"]["isTerminal"] = True
        s["terminal"]["outcome"] = "fail"
        s["terminal"]["reason"] = "ball_collision"
        s["debug"]["raw_status"] = "fail"
        s["is_actionable"] = False
    else:
        env["core_angle"] += 12

    # time advances
    s["gameTimeMs"] += 139
    s["timestampMs"] += 139

    return s
