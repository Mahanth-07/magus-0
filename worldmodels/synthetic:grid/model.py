import copy

def predict(state, action):
    s = copy.deepcopy(state)
    
    player = s["game_state"]["player"]
    goal = s["game_state"]["environment"]["goal"]
    metrics = s["metrics"]
    
    # Grid bounds: 0-4
    GRID_MIN = 0
    GRID_MAX = 4
    
    # Apply movement
    new_x = player["x"]
    new_y = player["y"]
    
    if action == "move_right":
        new_x = min(player["x"] + 1, GRID_MAX)
    elif action == "move_left":
        new_x = max(player["x"] - 1, GRID_MIN)
    elif action == "move_down":
        new_y = min(player["y"] + 1, GRID_MAX)
    elif action == "move_up":
        new_y = max(player["y"] - 1, GRID_MIN)
    
    player["x"] = new_x
    player["y"] = new_y
    
    # Increment steps
    metrics["steps"] += 1
    
    # Check if player reached goal
    if player["x"] == goal["x"] and player["y"] == goal["y"]:
        metrics["score"] += 10
        # Goal relocates to unpredictable position - copy through unchanged
        # (the new goal position is random, we can't predict it)
    
    return s
