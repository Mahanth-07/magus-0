import copy

def predict(state, action):
    new_state = copy.deepcopy(state)
    
    if new_state["status"] != "playing":
        return new_state
    
    player = new_state["game_state"]["player"]
    goal = new_state["game_state"]["environment"]["goal"]
    
    # Grid bounds: 0 to 4 (5x5 grid)
    MIN_X, MAX_X = 0, 4
    MIN_Y, MAX_Y = 0, 4
    
    # Apply movement
    if action == "move_up":
        player["y"] = max(MIN_Y, player["y"] - 1)
    elif action == "move_down":
        player["y"] = min(MAX_Y, player["y"] + 1)
    elif action == "move_left":
        player["x"] = max(MIN_X, player["x"] - 1)
    elif action == "move_right":
        player["x"] = min(MAX_X, player["x"] + 1)
    
    # Increment steps
    new_state["metrics"]["steps"] += 1
    
    # Check if player reached the goal
    if player["x"] == goal["x"] and player["y"] == goal["y"]:
        new_state["metrics"]["score"] += 1
    
    return new_state
