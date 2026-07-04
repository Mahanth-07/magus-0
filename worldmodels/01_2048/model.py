import copy
import math

def predict(state, action):
    s = copy.deepcopy(state)
    
    if action == 'use_space':
        # Reset: score=0, level=2, 2 tiles with value 2
        # Board positions are random - copy through unchanged
        gs = s['game_state']
        gs['score'] = 0
        gs['level'] = 2
        gs['completion_progress'] = 1.0 / 11.0
        s['metrics']['primary_score'] = 0
        s['metrics']['max_tile'] = 2
        s['metrics']['filled_cells'] = 2
        s['metrics']['board_fill_ratio'] = 2 / 16.0
        # Leave environment and entities unchanged (random, unpredictable)
        return s
    
    gs = s['game_state']
    env = gs['environment']
    size = 4
    
    def slide_and_merge(row):
        tiles = [x for x in row if x != 0]
        merged = []
        score_gain = 0
        i = 0
        while i < len(tiles):
            if i + 1 < len(tiles) and tiles[i] == tiles[i+1]:
                val = tiles[i] * 2
                merged.append(val)
                score_gain += val
                i += 2
            else:
                merged.append(tiles[i])
                i += 1
        while len(merged) < size:
            merged.append(0)
        return merged, score_gain
    
    total_score_gain = 0
    
    if action == 'use_arrowleft':
        for r in range(size):
            row = env[r][:]
            new_row, gain = slide_and_merge(row)
            env[r] = new_row
            total_score_gain += gain
    
    elif action == 'use_arrowdown':
        for c in range(size):
            col = [env[r][c] for r in range(size)]
            col_rev = col[::-1]
            new_col_rev, gain = slide_and_merge(col_rev)
            new_col = new_col_rev[::-1]
            for r in range(size):
                env[r][c] = new_col[r]
            total_score_gain += gain
    
    gs['score'] = gs['score'] + total_score_gain
    s['metrics']['primary_score'] = gs['score']
    
    filled = sum(1 for r in range(size) for c in range(size) if env[r][c] != 0)
    max_tile = max((env[r][c] for r in range(size) for c in range(size) if env[r][c] != 0), default=2)
    s['metrics']['max_tile'] = max_tile
    s['metrics']['filled_cells'] = filled
    s['metrics']['board_fill_ratio'] = filled / 16.0
    
    # Rebuild entities from environment (x=row, y=col)
    new_entities = []
    for r in range(size):
        for c in range(size):
            if env[r][c] != 0:
                new_entities.append({
                    'type': 'tile',
                    'x': r,
                    'y': c,
                    'props': {'value': env[r][c]}
                })
    
    gs['entities'] = new_entities
    gs['environment'] = env
    
    if max_tile > 0:
        log_val = math.log2(max_tile)
        gs['level'] = max_tile
        gs['completion_progress'] = log_val / 11.0
    
    return s
