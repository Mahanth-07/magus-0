import copy
import math

def predict(state, action):
    s = copy.deepcopy(state)
    
    if action == 'use_space':
        # Reset action - new random board state, cannot predict
        # Return state unchanged (unpredictable random spawn)
        return s
    
    gs = s['game_state']
    board = gs['environment']
    size = 4
    
    def slide_and_merge_left(row):
        tiles = [x for x in row if x != 0]
        merged = []
        skip = False
        score_gain = 0
        for i in range(len(tiles)):
            if skip:
                skip = False
                continue
            if i + 1 < len(tiles) and tiles[i] == tiles[i+1]:
                val = tiles[i] * 2
                merged.append(val)
                score_gain += val
                skip = True
            else:
                merged.append(tiles[i])
        while len(merged) < len(row):
            merged.append(0)
        return merged, score_gain
    
    total_score_gain = 0
    
    if action == 'use_arrowleft':
        new_board = []
        for r in range(size):
            row, gain = slide_and_merge_left(board[r])
            new_board.append(row)
            total_score_gain += gain
        gs['environment'] = new_board
    
    elif action == 'use_arrowdown':
        new_board = [row[:] for row in board]
        for c in range(size):
            col = [board[r][c] for r in range(size)]
            col_rev, gain = slide_and_merge_left(col[::-1])
            col_result = col_rev[::-1]
            total_score_gain += gain
            for r in range(size):
                new_board[r][c] = col_result[r]
        gs['environment'] = new_board
    else:
        return s
    
    gs['score'] += total_score_gain
    s['metrics']['primary_score'] = gs['score']
    
    filled = 0
    max_tile = 0
    for r in range(size):
        for c in range(size):
            v = gs['environment'][r][c]
            if v != 0:
                filled += 1
                if v > max_tile:
                    max_tile = v
    
    s['metrics']['filled_cells'] = filled
    s['metrics']['board_fill_ratio'] = filled / 16
    s['metrics']['max_tile'] = max_tile
    
    if max_tile >= 2:
        log_val = math.log2(max_tile)
        gs['level'] = int(log_val) * 2
        gs['completion_progress'] = log_val / 11
    
    # Build entities: x=row, y=col
    entities = []
    for r in range(size):
        for c in range(size):
            v = gs['environment'][r][c]
            if v != 0:
                entities.append({'type': 'tile', 'x': r, 'y': c, 'props': {'value': v}})
    gs['entities'] = entities
    
    return s
