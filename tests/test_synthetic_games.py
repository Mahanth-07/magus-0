# tests/test_synthetic_games.py
"""Synthetic games — deterministic ground truth for onboarding tests."""

from ludus.synthetic import CounterGame, GridWorldGame, MenuGame, WallGame


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
    g = GridWorldGame()
    st0 = g.raw_state()
    assert (st0["game_state"]["player"]["x"], st0["game_state"]["player"]["y"]) == (2, 2)
    assert (st0["game_state"]["environment"]["goal"]["x"],
            st0["game_state"]["environment"]["goal"]["y"]) == (3, 2)
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


def test_gridworld_apply_after_terminal_is_a_frozen_noop():
    g = GridWorldGame()
    _press(g, "q")
    assert g.raw_state()["metrics"]["steps"] == 1  # the quit press itself counts
    frozen = g.raw_state()
    _press(g, "ArrowRight")
    _press(g, "x")
    assert g.raw_state() == frozen  # steps, score, player all frozen


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


def test_counter_apply_after_terminal_is_a_frozen_noop():
    g = CounterGame()
    for _ in range(10):
        _press(g, "z")
    assert g.terminal is True
    frozen = g.raw_state()
    _press(g, "x")
    assert g.raw_state() == frozen


def test_counter_noop_key_ticks_only_steps():
    g = CounterGame()
    before = g.raw_state()
    _press(g, "ArrowRight")
    after = g.raw_state()
    assert after["metrics"]["score"] == before["metrics"]["score"]
    assert after["metrics"]["health"] == before["metrics"]["health"]
    assert after["metrics"]["steps"] == before["metrics"]["steps"] + 1


# --- client-interface compliance (what run_episode/prober need) --------------

def test_synthetic_games_speak_the_client_interface():
    for g in (GridWorldGame(), CounterGame()):
        assert isinstance(g.screenshot(), bytes)
        m = g.metrics(["score"])
        assert m == {"score": 0.0}
        assert g.read_partner_actions() == []
        assert isinstance(g.raw_state(), dict)
        g.close()  # must not raise


def test_synthetic_games_expose_state_text():
    for g in (GridWorldGame(), CounterGame()):
        assert g.state_text() == ""


# --- WallGame ----------------------------------------------------------------

def test_wallgame_right_is_noop_at_wall_until_left():
    g = WallGame()
    _press(g, "ArrowRight")
    assert g.raw_state()["game_state"]["player"]["x"] == 4
    _press(g, "ArrowLeft")
    assert g.raw_state()["game_state"]["player"]["x"] == 3
    _press(g, "ArrowRight")
    assert g.raw_state()["game_state"]["player"]["x"] == 4


# --- MenuGame ----------------------------------------------------------------

def test_menu_game_starts_in_menu_status():
    g = MenuGame()
    assert g.raw_state()["status"] == "menu"


def test_menu_game_enter_flips_to_playing():
    g = MenuGame()
    _press(g, "Enter")
    assert g.raw_state()["status"] == "playing"


def test_menu_game_x_scores_once_playing():
    g = MenuGame()
    _press(g, "Enter")
    _press(g, "x")
    assert g.raw_state()["metrics"]["score"] == 1


def test_menu_game_non_enter_keys_ignored_in_menu():
    g = MenuGame()
    _press(g, "ArrowUp")
    assert g.raw_state()["status"] == "menu"
    _press(g, "Space")
    assert g.raw_state()["status"] == "menu"
