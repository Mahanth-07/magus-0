# tests/test_synthetic_games.py
"""Synthetic games — deterministic ground truth for onboarding tests."""

from ludus.synthetic import CounterGame, GridWorldGame


def _press(game, key):
    game.apply({"key": key, "duration_ms": 50})


# --- GridWorldGame -----------------------------------------------------------

def test_gridworld_arrows_move_player():
    g = GridWorldGame()
    x0 = g.raw_state()["game_state"]["player"]["x"]
    _press(g, "ArrowRight")
    assert g.raw_state()["game_state"]["player"]["x"] == x0 + 1
    _press(g, "ArrowLeft")
    assert g.raw_state()["game_state"]["player"]["x"] == x0
    y0 = g.raw_state()["game_state"]["player"]["y"]
    _press(g, "ArrowDown")
    assert g.raw_state()["game_state"]["player"]["y"] == y0 + 1


def test_gridworld_walls_clamp_movement():
    g = GridWorldGame()
    for _ in range(10):
        _press(g, "ArrowLeft")
    assert g.raw_state()["game_state"]["player"]["x"] == 0


def test_gridworld_reaching_goal_scores_and_respawns_goal():
    g = GridWorldGame()  # player (2,2), first goal (3,2)
    _press(g, "ArrowRight")  # onto the goal
    st = g.raw_state()
    assert st["metrics"]["score"] == 10
    goal = st["game_state"]["environment"]["goal"]
    assert (goal["x"], goal["y"]) != (3, 2)  # respawned elsewhere


def test_gridworld_quit_key_makes_state_terminal():
    g = GridWorldGame()
    _press(g, "q")
    assert g.raw_state()["status"] == "terminal"


def test_gridworld_noop_key_changes_nothing_but_step_counter():
    g = GridWorldGame()
    before = g.raw_state()
    _press(g, "z")
    after = g.raw_state()
    assert after["game_state"]["player"] == before["game_state"]["player"]
    assert after["metrics"]["score"] == before["metrics"]["score"]
    assert after["metrics"]["steps"] == before["metrics"]["steps"] + 1


def test_gridworld_is_deterministic():
    a, b = GridWorldGame(), GridWorldGame()
    for key in ["ArrowRight", "ArrowDown", "ArrowRight", "ArrowUp"]:
        _press(a, key); _press(b, key)
    assert a.raw_state() == b.raw_state()


# --- CounterGame -------------------------------------------------------------

def test_counter_x_increments_score():
    g = CounterGame()
    _press(g, "x")
    _press(g, "x")
    assert g.raw_state()["metrics"]["score"] == 2


def test_counter_z_drains_health_to_terminal():
    g = CounterGame()
    for _ in range(10):
        _press(g, "z")
    st = g.raw_state()
    assert st["metrics"]["health"] == 0
    assert st["status"] == "terminal"


# --- client-interface compliance (what run_episode/prober need) --------------

def test_synthetic_games_speak_the_client_interface():
    for g in (GridWorldGame(), CounterGame()):
        assert isinstance(g.screenshot(), bytes)
        m = g.metrics(["score"])
        assert m == {"score": 0.0}
        assert g.read_partner_actions() == []
        assert isinstance(g.raw_state(), dict)
        g.close()  # must not raise
