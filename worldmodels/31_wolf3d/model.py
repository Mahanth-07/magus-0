import copy

def predict(state, action):
    ns = copy.deepcopy(state)

    gs = ns.get("game_state", {})
    player = gs.get("player", {})

    # angle changes for arrow keys (deterministic ~7.03 deg step)
    if action == "use_arrowleft":
        a = player.get("angle_deg")
        if a is not None:
            player["angle_deg"] = round((a + 7.03) % 360, 2)
    elif action == "use_arrowright":
        a = player.get("angle_deg")
        if a is not None:
            player["angle_deg"] = round((a - 7.03) % 360, 2)

    return ns
