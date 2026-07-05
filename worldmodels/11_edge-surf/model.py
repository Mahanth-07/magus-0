import copy

def predict(state, action):
    s = copy.deepcopy(state)
    if action == 'use_space':
        if s.get('status') == 'playing':
            s['status'] = 'paused'
            s['is_actionable'] = False
            s['game_state']['environment']['session_state'] = 'pause'
    return s
