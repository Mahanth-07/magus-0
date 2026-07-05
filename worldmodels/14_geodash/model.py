import copy


def predict(state, action):
    s = copy.deepcopy(state)
    gs = s["game_state"]
    player = gs["player"]
    dbg = s["debug"]
    metrics = s["metrics"]

    ticker = dbg["ticker"]

    # Reset condition
    if action == "use_arrowleft" and ticker >= 28:
        s["gameTimeMs"] = 0
        dbg["ticker"] = 0
        dbg["player_x"] = 13
        player["stage_x"] = 13
        player["frame"] = 0
        gs["score"] = 0
        metrics["primary_score"] = 0
        return s

    dt = 3
    new_ticker = ticker + dt
    dbg["ticker"] = new_ticker
    s["gameTimeMs"] = new_ticker * 1000
    player["frame"] = new_ticker

    stage_x = player["stage_x"] + 27
    player["stage_x"] = stage_x
    dbg["player_x"] = stage_x

    scroll_x = gs["environment"]["camera"]["scroll_x"]
    if stage_x + 100 > scroll_x:
        scroll_x = stage_x + 100
    gs["environment"]["camera"]["scroll_x"] = scroll_x
    metrics["distance"] = scroll_x

    gs["score"] = gs["score"] + 1
    metrics["primary_score"] = metrics["primary_score"] + 1

    return s
