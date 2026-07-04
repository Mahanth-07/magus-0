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


def test_induced_gate_is_on_holdout_not_train(tmp_path):
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
