# tests/test_cli_duel.py
"""Duel harness — planner vs a baseline provider, same game, same budget."""

import json

from ludus.cli import cmd_duel
from ludus.providers.mock import MockProvider

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


def _setup(tmp_path):
    from ludus.cli import cmd_onboard
    cmd_onboard(game="synthetic:grid", out_dir=tmp_path / "profiles")
    d = tmp_path / "wm" / "synthetic:grid"
    d.mkdir(parents=True)
    (d / "model.py").write_text(GRID_MODEL)
    (d / "report.json").write_text(json.dumps({"status": "INDUCED", "overall": 1.0}))
    return tmp_path


def test_duel_planner_beats_mock_on_grid(tmp_path, capsys):
    root = _setup(tmp_path)
    result = cmd_duel(
        game="synthetic:grid",
        profile_path=root / "profiles" / "synthetic:grid.json",
        worldmodels_dir=root / "wm", steps=6,
        baseline_provider=MockProvider(),
        runs_dir=root / "runs",
    )
    assert result["planner"]["score"] >= 10.0
    assert result["baseline"]["score"] <= result["planner"]["score"]
    assert result["winner"] == "planner"
    out = capsys.readouterr().out
    assert "DUEL" in out and "planner" in out and "winner" in out


def test_duel_result_is_persisted(tmp_path):
    root = _setup(tmp_path)
    result = cmd_duel(
        game="synthetic:grid",
        profile_path=root / "profiles" / "synthetic:grid.json",
        worldmodels_dir=root / "wm", steps=4,
        baseline_provider=MockProvider(),
        runs_dir=root / "runs",
    )
    saved = json.loads((root / "runs" / "duel-synthetic:grid.json").read_text())
    assert saved == result


# ---- repeats=3 tests ----

def test_duel_repeats_produces_three_scores_per_side(tmp_path):
    """repeats=3 must give three scores per side in the aggregated result."""
    root = _setup(tmp_path)
    result = cmd_duel(
        game="synthetic:grid",
        profile_path=root / "profiles" / "synthetic:grid.json",
        worldmodels_dir=root / "wm", steps=6,
        baseline_provider=MockProvider(),
        runs_dir=root / "runs",
        repeats=3,
    )
    assert len(result["planner"]["scores"]) == 3
    assert len(result["baseline"]["scores"]) == 3


def test_duel_repeats_correct_mean_std(tmp_path):
    """mean and std (population) are computed correctly."""
    import math
    root = _setup(tmp_path)
    result = cmd_duel(
        game="synthetic:grid",
        profile_path=root / "profiles" / "synthetic:grid.json",
        worldmodels_dir=root / "wm", steps=6,
        baseline_provider=MockProvider(),
        runs_dir=root / "runs",
        repeats=3,
    )
    for label in ("planner", "baseline"):
        scores = result[label]["scores"]
        expected_mean = sum(scores) / len(scores)
        expected_std = math.sqrt(sum((s - expected_mean) ** 2 for s in scores) / len(scores))
        assert abs(result[label]["mean"] - expected_mean) < 1e-9
        assert abs(result[label]["std"] - expected_std) < 1e-9


def test_duel_repeats_winner_by_mean(tmp_path):
    """winner is decided by means (planner should still win on grid)."""
    root = _setup(tmp_path)
    result = cmd_duel(
        game="synthetic:grid",
        profile_path=root / "profiles" / "synthetic:grid.json",
        worldmodels_dir=root / "wm", steps=6,
        baseline_provider=MockProvider(),
        runs_dir=root / "runs",
        repeats=3,
    )
    assert result["planner"]["mean"] >= result["baseline"]["mean"]
    assert result["winner"] == "planner"


def test_duel_repeats_episode_ids_unique(tmp_path):
    """Each repeat must produce a distinct episode_id in runs/."""
    root = _setup(tmp_path)
    cmd_duel(
        game="synthetic:grid",
        profile_path=root / "profiles" / "synthetic:grid.json",
        worldmodels_dir=root / "wm", steps=4,
        baseline_provider=MockProvider(),
        runs_dir=root / "runs",
        repeats=3,
    )
    runs_root = root / "runs"
    # 3 planner episodes + 3 baseline episodes = 6 unique dirs
    episode_dirs = [d for d in runs_root.iterdir() if d.is_dir()]
    ids = [d.name for d in episode_dirs]
    assert len(ids) == len(set(ids)), f"duplicate episode dirs: {ids}"
    # Each repeat id must contain the repeat index
    planner_ids = [i for i in ids if "planner" in i]
    baseline_ids = [i for i in ids if "baseline" in i]
    assert len(planner_ids) == 3
    assert len(baseline_ids) == 3
    for r in range(3):
        assert any(f"-r{r}" in i for i in planner_ids), f"missing planner repeat r{r}"
        assert any(f"-r{r}" in i for i in baseline_ids), f"missing baseline repeat r{r}"


def test_duel_repeats_json_has_repeats_key(tmp_path):
    """Persisted JSON must include 'repeats' and per-side 'scores' arrays."""
    root = _setup(tmp_path)
    cmd_duel(
        game="synthetic:grid",
        profile_path=root / "profiles" / "synthetic:grid.json",
        worldmodels_dir=root / "wm", steps=4,
        baseline_provider=MockProvider(),
        runs_dir=root / "runs",
        repeats=3,
    )
    saved = json.loads((root / "runs" / "duel-synthetic:grid.json").read_text())
    assert saved["repeats"] == 3
    assert "scores" in saved["planner"]
    assert "scores" in saved["baseline"]
    assert len(saved["planner"]["scores"]) == 3


def test_duel_n1_unchanged_shape(tmp_path):
    """repeats=1 (default) must produce the same shape as before (backward-compat)."""
    root = _setup(tmp_path)
    result = cmd_duel(
        game="synthetic:grid",
        profile_path=root / "profiles" / "synthetic:grid.json",
        worldmodels_dir=root / "wm", steps=4,
        baseline_provider=MockProvider(),
        runs_dir=root / "runs",
    )
    # existing keys must still be present
    assert "score" in result["planner"]
    assert "score" in result["baseline"]
    assert "winner" in result
    assert "game" in result
    # score == mean when repeats=1
    assert result["planner"]["score"] == result["planner"]["mean"]
    assert result["baseline"]["score"] == result["baseline"]["mean"]


def test_duel_repeats_printed_table_shows_mean_std(tmp_path, capsys):
    """With repeats>1, printed table must include mean±std and scores list."""
    root = _setup(tmp_path)
    cmd_duel(
        game="synthetic:grid",
        profile_path=root / "profiles" / "synthetic:grid.json",
        worldmodels_dir=root / "wm", steps=6,
        baseline_provider=MockProvider(),
        runs_dir=root / "runs",
        repeats=3,
    )
    out = capsys.readouterr().out
    assert "mean" in out.lower() or "±" in out
    assert "planner" in out
    assert "winner" in out
