import copy

def predict(state, action):
    ns = copy.deepcopy(state)
    letter = action.split('_')[-1] if action.startswith('use_') else None
    env = ns['game_state']['environment']
    gl = env.get('current_guess_length', 0)

    if letter is not None:
        if gl < 5:
            # find current active row: the first row not fully committed
            # observed only row 0 used before submission
            row = 0
            env['letters'][row][gl] = letter
            env['current_guess_length'] = gl + 1
        else:
            # submission with full row: observed clears row 0, resets length,
            # status -> ready (invalid word case)
            row = 0
            for c in range(5):
                env['letters'][row][c] = ""
            env['current_guess_length'] = 0
            ns['status'] = "ready"

    return ns
