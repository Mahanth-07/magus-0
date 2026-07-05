# tests/test_inducer.py
"""Inducer — the synthesis/repair loop, offline with a scripted synthesizer."""

import json

import pytest

from ludus.onboarding.profile import GameProfile
from ludus.synthetic import GridWorldGame
from ludus.worldmodel.explore import explore_game
from ludus.worldmodel.inducer import extract_code, induce_world_model
from ludus.worldmodel.transitions import TransitionStore

# A CORRECT gridworld world model (we know the true dynamics).
GRID_MODEL = '''
import copy

MOVES = {"move_left": (-1, 0), "move_right": (1, 0),
         "move_up": (0, -1), "move_down": (0, 1)}
GOAL_CYCLE = [(3, 2), (0, 0), (4, 4), (0, 4), (4, 0)]

def predict(state, action):
    out = copy.deepcopy(state)
    p = out["game_state"]["player"]
    out["metrics"]["steps"] += 1
    dx, dy = MOVES.get(action, (0, 0))
    p["x"] = min(4, max(0, p["x"] + dx))
    p["y"] = min(4, max(0, p["y"] + dy))
    goal = out["game_state"]["environment"]["goal"]
    if (p["x"], p["y"]) == (goal["x"], goal["y"]):
        out["metrics"]["score"] += 10
        i = GOAL_CYCLE.index((goal["x"], goal["y"]))
        nx, ny = GOAL_CYCLE[(i + 1) % len(GOAL_CYCLE)]
        goal["x"], goal["y"] = nx, ny
    return out
'''

BROKEN_MODEL = '''
def predict(state, action):
    return state   # never changes anything
'''


def _grid_profile() -> GameProfile:
    return GameProfile(
        game_id="synthetic:grid",
        controls={"move_left": "ArrowLeft", "move_right": "ArrowRight",
                  "move_up": "ArrowUp", "move_down": "ArrowDown"},
        control_effects={}, metrics=["score", "steps"], primary_metric="score",
        higher_is_better=True, timing_ms=50, objective="test", state_schema={},
    )


def _store(tmp_path) -> TransitionStore:
    store = TransitionStore(tmp_path / "t.jsonl")
    explore_game(_grid_profile(), GridWorldGame, store=store,
                 episodes=4, steps_per_episode=15, seed=3)
    return store


def test_extract_code_from_fenced_response():
    text = "Here you go:\n```python\ndef predict(state, action):\n    return state\n```\nDone."
    assert extract_code(text).startswith("def predict")


def test_extract_code_bare_source_passthrough():
    src = "def predict(state, action):\n    return state\n"
    assert extract_code(src) == src


def test_extract_code_prefers_block_containing_predict():
    text = (
        "First, the key insight:\n```python\nMOVES = {\"a\": 1}\n```\n"
        "Full model:\n```python\ndef predict(state, action):\n    return state\n```\n"
    )
    assert extract_code(text).startswith("def predict")


def test_induce_succeeds_first_try_with_correct_model(tmp_path):
    calls = []

    def synthesizer(system, user):
        calls.append(user)
        return f"```python\n{GRID_MODEL}\n```"

    result = induce_world_model(
        "synthetic:grid", _store(tmp_path), profile=_grid_profile(),
        synthesizer=synthesizer, out_dir=tmp_path / "wm",
        max_iterations=3, threshold=0.9, seed=5,
    )
    assert result.status == "INDUCED"
    assert result.iterations == 1
    assert result.report.passed
    assert (tmp_path / "wm" / "synthetic:grid" / "model.py").exists()
    saved_report = json.loads(
        (tmp_path / "wm" / "synthetic:grid" / "report.json").read_text())
    assert saved_report["status"] == "INDUCED"
    assert saved_report["overall"] >= 0.9


def test_induce_repairs_with_counterexamples(tmp_path):
    responses = [f"```python\n{BROKEN_MODEL}\n```", f"```python\n{GRID_MODEL}\n```"]
    prompts = []

    def synthesizer(system, user):
        prompts.append(user)
        return responses[len(prompts) - 1]

    result = induce_world_model(
        "synthetic:grid", _store(tmp_path), profile=_grid_profile(),
        synthesizer=synthesizer, out_dir=tmp_path / "wm",
        max_iterations=3, threshold=0.9, seed=5,
    )
    assert result.status == "INDUCED"
    assert result.iterations == 2
    # the repair prompt must carry counterexamples from the failed attempt
    assert "predicted" in prompts[1] and "actual" in prompts[1]


def test_induce_fails_honestly_when_budget_exhausted(tmp_path):
    def synthesizer(system, user):
        return f"```python\n{BROKEN_MODEL}\n```"

    result = induce_world_model(
        "synthetic:grid", _store(tmp_path), profile=_grid_profile(),
        synthesizer=synthesizer, out_dir=tmp_path / "wm",
        max_iterations=2, threshold=0.9, seed=5,
    )
    assert result.status == "FAILED"
    assert result.iterations == 2
    # artifacts still saved for inspection, marked FAILED
    saved = json.loads((tmp_path / "wm" / "synthetic:grid" / "report.json").read_text())
    assert saved["status"] == "FAILED"


def test_report_n_cases_counts_holdout(tmp_path):
    # the saved report's n_cases must equal the HOLDOUT size (1 of 4 episodes)
    store = _store(tmp_path)
    train, holdout = store.split(holdout_frac=0.25, seed=5)

    def synthesizer(system, user):
        return f"```python\n{GRID_MODEL}\n```"

    result = induce_world_model(
        "synthetic:grid", store, profile=_grid_profile(),
        synthesizer=synthesizer, out_dir=tmp_path / "wm",
        max_iterations=1, threshold=0.9, seed=5,
    )
    assert result.report.n_cases == len(holdout)


def test_induce_refuses_single_episode_dataset(tmp_path):
    store = TransitionStore(tmp_path / "one.jsonl")
    explore_game(_grid_profile(), GridWorldGame, store=store,
                 episodes=1, steps_per_episode=5, seed=1)
    with pytest.raises(ValueError, match="not enough episodes"):
        induce_world_model(
            "synthetic:grid", store, profile=_grid_profile(),
            synthesizer=lambda s, u: "```python\ndef predict(s,a):\n    return s\n```",
            out_dir=tmp_path / "wm", max_iterations=1, threshold=0.9, seed=5,
        )


def test_prompt_sample_prioritizes_metric_changing_transitions():
    from ludus.worldmodel.inducer import MAX_PROMPT_TRANSITIONS, build_synthesis_prompt
    from ludus.worldmodel.transitions import Transition

    def noop(i):
        # steps ticks every press (like real games) — must NOT be treated as rare
        return Transition(game_id="g", episode_id=f"e{i}", step=i,
                          action="move_right", key="R",
                          before={"metrics": {"score": 0, "steps": i}, "game_state": {"player": {"x": i}}},
                          after={"metrics": {"score": 0, "steps": i + 1}, "game_state": {"player": {"x": i + 1}}})

    # rare event also ticks steps — only the score change should mark it rare
    rare = Transition(game_id="g", episode_id="rare", step=0,
                      action="collect", key="c",
                      before={"metrics": {"score": 0, "steps": 0}, "game_state": {"player": {"x": 0}}},
                      after={"metrics": {"score": 10, "steps": 1}, "game_state": {"player": {"x": 0}}})
    # rare event buried far beyond the sample cutoff
    train = [noop(i) for i in range(MAX_PROMPT_TRANSITIONS + 20)] + [rare]
    _, user = build_synthesis_prompt("g", _grid_profile(), train, [], "")
    assert "collect" in user, "metric-changing transition must reach the prompt"


def test_render_grid_shows_tile_values():
    from ludus.worldmodel.inducer import render_grid
    state = {"game_state": {"entities": [
        {"type": "tile", "x": 0, "y": 0, "props": {"value": 2}},
        {"type": "tile", "x": 3, "y": 1, "props": {"value": 4}},
    ]}}
    grid = render_grid(state)
    rows = grid.splitlines()
    assert len(rows) == 2                 # max y = 1
    assert "2" in rows[0] and "4" in rows[1]
    assert "." in rows[0]


def test_render_grid_noop_for_non_grid_states():
    from ludus.worldmodel.inducer import render_grid
    assert render_grid({"game_state": {"player": {"x": 1}}}) == ""
    assert render_grid({"game_state": {"entities": [{"x": 0.5, "y": 1}]}}) == ""


def test_prompt_includes_board_views_for_entity_grid_games():
    from ludus.worldmodel.inducer import build_synthesis_prompt
    from ludus.worldmodel.transitions import Transition
    ents = lambda *vals: [{"type": "tile", "x": i, "y": 0, "props": {"value": v}}
                          for i, v in enumerate(vals)]
    t = Transition(game_id="g", episode_id="e", step=0, action="left", key="L",
                   before={"metrics": {"score": 0}, "game_state": {"entities": ents(2, 2)}},
                   after={"metrics": {"score": 4}, "game_state": {"entities": ents(4)}})
    _, user = build_synthesis_prompt("g", _grid_profile(), [t], [], "")
    assert "Board views" in user
    assert "before:" in user and "after:" in user


def test_gate_requires_train_pass_too_rare_event_only_in_train(tmp_path):
    # Live-acceptance regression: a rare mechanic (score jump) observed ONLY
    # in train episodes must block INDUCED even when holdout passes cleanly.
    from ludus.worldmodel.transitions import Transition

    store = TransitionStore(tmp_path / "t.jsonl")

    def noop_transition(ep, step):
        s = {"status": "playing", "metrics": {"score": 0},
             "game_state": {"player": {"x": step}}}
        s2 = {"status": "playing", "metrics": {"score": 0},
              "game_state": {"player": {"x": step + 1}}}
        return Transition(game_id="g", episode_id=ep, step=step,
                          action="move_right", key="ArrowRight",
                          before=s, after=s2)

    episodes = ["a", "b", "c", "d"]
    for ep in episodes:
        for step in range(3):
            store.append(noop_transition(ep, step))
    # find which episode the (seeded) split holds out, then plant the rare
    # event in every TRAIN episode only
    _, holdout = store.split(holdout_frac=0.25, seed=5)
    holdout_ep = holdout[0].episode_id
    for ep in episodes:
        if ep == holdout_ep:
            continue
        store.append(Transition(
            game_id="g", episode_id=ep, step=99, action="collect", key="c",
            before={"status": "playing", "metrics": {"score": 0},
                    "game_state": {"player": {"x": 0}}},
            after={"status": "playing", "metrics": {"score": 10},
                   "game_state": {"player": {"x": 0}}},
        ))

    # model correct for move_right, WRONG for collect (predicts +1 not +10)
    wrong_rare = '''
import copy

def predict(state, action):
    out = copy.deepcopy(state)
    if action == "move_right":
        out["game_state"]["player"]["x"] += 1
    elif action == "collect":
        out["metrics"]["score"] += 1
    return out
'''
    result = induce_world_model(
        "g", store, profile=_grid_profile(),
        synthesizer=lambda s, u: f"```python\n{wrong_rare}\n```",
        out_dir=tmp_path / "wm", max_iterations=2, threshold=0.9, seed=5,
    )
    assert result.status == "FAILED"          # train evidence must block the gate
    saved = json.loads((tmp_path / "wm" / "g" / "report.json").read_text())
    assert saved["train_overall"] < 1.0
