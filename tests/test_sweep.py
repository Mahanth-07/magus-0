# tests/test_sweep.py
"""TDD for ludus/sweep.py — pure logic, no I/O beyond tmp_path fixtures."""

import json

import pytest

from ludus.sweep import (
    VERDICTS,
    append_result,
    load_results,
    render_table,
    sweep_game,
)

# ---------------------------------------------------------------------------
# load / append / resume
# ---------------------------------------------------------------------------


def test_load_results_missing_file_returns_empty(tmp_path):
    assert load_results(tmp_path / "nope.jsonl") == {}


def test_load_append_roundtrip(tmp_path):
    path = tmp_path / "results.jsonl"
    rec = {"game": "alpha", "verdict": "DUEL_WON"}
    append_result(path, rec)
    assert load_results(path) == {"alpha": rec}


def test_last_record_per_game_wins(tmp_path):
    path = tmp_path / "results.jsonl"
    append_result(path, {"game": "alpha", "verdict": "DUEL_LOST"})
    append_result(path, {"game": "alpha", "verdict": "DUEL_WON"})
    out = load_results(path)
    assert out["alpha"]["verdict"] == "DUEL_WON"


def test_multiple_games_all_loaded(tmp_path):
    path = tmp_path / "results.jsonl"
    append_result(path, {"game": "alpha", "verdict": "DUEL_WON"})
    append_result(path, {"game": "beta", "verdict": "DUEL_LOST"})
    out = load_results(path)
    assert set(out) == {"alpha", "beta"}


def test_append_result_creates_parent_dirs(tmp_path):
    path = tmp_path / "a" / "b" / "results.jsonl"
    append_result(path, {"game": "x", "verdict": "DUEL_WON"})
    assert path.exists()


def test_load_results_skips_blank_lines(tmp_path):
    path = tmp_path / "r.jsonl"
    path.write_text('\n{"game": "g", "verdict": "DUEL_WON"}\n\n')
    assert load_results(path) == {"g": {"game": "g", "verdict": "DUEL_WON"}}


# ---------------------------------------------------------------------------
# sweep_game happy path
# ---------------------------------------------------------------------------


def _stubs(*, onboard=None, explore=None, induce=None, duel=None):
    """Return a dict of stubs with call-tracking."""
    calls: dict[str, list] = {"onboard": [], "explore": [], "induce": [], "duel": []}

    def _onboard(game):
        calls["onboard"].append(game)
        return onboard(game) if onboard else {"up": "ArrowUp"}

    def _explore(game):
        calls["explore"].append(game)
        return explore(game) if explore else 120

    def _induce(game):
        calls["induce"].append(game)
        if induce is not None:
            return induce(game)
        return {"status": "INDUCED", "overall": 0.92, "primary_accuracy": 0.95,
                "iterations": 2}

    def _duel(game):
        calls["duel"].append(game)
        if duel is not None:
            return duel(game)
        return {"planner": {"score": 20.0}, "baseline": {"score": 5.0},
                "winner": "planner"}

    return {"onboard": _onboard, "explore": _explore,
            "induce": _induce, "duel": _duel, "_calls": calls}


def test_happy_path_returns_duel_won():
    s = _stubs()
    rec = sweep_game("mygame", **{k: v for k, v in s.items() if not k.startswith("_")})
    assert rec["game"] == "mygame"
    assert rec["verdict"] == "DUEL_WON"
    assert "controls" in rec
    assert "transitions" in rec
    assert "induction" in rec
    assert "duel" in rec


def test_happy_path_all_stages_called():
    s = _stubs()
    sweep_game("mygame", **{k: v for k, v in s.items() if not k.startswith("_")})
    assert s["_calls"]["onboard"] == ["mygame"]
    assert s["_calls"]["explore"] == ["mygame"]
    assert s["_calls"]["induce"] == ["mygame"]
    assert s["_calls"]["duel"] == ["mygame"]


def test_happy_path_duel_section_populated():
    s = _stubs()
    rec = sweep_game("g", **{k: v for k, v in s.items() if not k.startswith("_")})
    assert rec["duel"] == {"planner": 20.0, "baseline": 5.0, "winner": "planner"}


def test_happy_path_induction_section_populated():
    s = _stubs()
    rec = sweep_game("g", **{k: v for k, v in s.items() if not k.startswith("_")})
    assert rec["induction"]["status"] == "INDUCED"
    assert rec["induction"]["overall"] == pytest.approx(0.92)
    assert rec["induction"]["primary_accuracy"] == pytest.approx(0.95)
    assert rec["induction"]["iterations"] == 2


# ---------------------------------------------------------------------------
# failure stages — short-circuit + no later-stage calls
# ---------------------------------------------------------------------------


def test_onboard_failure_gives_onboard_failed():
    def bad_onboard(game):
        raise RuntimeError("no browser")

    s = _stubs(onboard=bad_onboard)
    rec = sweep_game("g", **{k: v for k, v in s.items() if not k.startswith("_")})
    assert rec["verdict"] == "ONBOARD_FAILED"
    assert "error" in rec


def test_onboard_failure_later_stages_not_called():
    def bad_onboard(game):
        raise RuntimeError("no browser")

    s = _stubs(onboard=bad_onboard)
    sweep_game("g", **{k: v for k, v in s.items() if not k.startswith("_")})
    assert s["_calls"]["explore"] == []
    assert s["_calls"]["induce"] == []
    assert s["_calls"]["duel"] == []


def test_explore_failure_gives_explore_failed():
    def bad_explore(game):
        raise IOError("disk full")

    s = _stubs(explore=bad_explore)
    rec = sweep_game("g", **{k: v for k, v in s.items() if not k.startswith("_")})
    assert rec["verdict"] == "EXPLORE_FAILED"
    assert s["_calls"]["induce"] == []
    assert s["_calls"]["duel"] == []


def test_explore_zero_transitions_gives_explore_failed():
    """Explore with missing/empty transitions file must surface as EXPLORE_FAILED."""
    def zero_explore(game):
        raise RuntimeError("explore produced no transitions")

    s = _stubs(explore=zero_explore)
    rec = sweep_game("g", **{k: v for k, v in s.items() if not k.startswith("_")})
    assert rec["verdict"] == "EXPLORE_FAILED"
    assert "no transitions" in rec["error"]


def test_induce_raises_gives_induction_failed():
    def bad_induce(game):
        raise ValueError("LLM error")

    s = _stubs(induce=bad_induce)
    rec = sweep_game("g", **{k: v for k, v in s.items() if not k.startswith("_")})
    assert rec["verdict"] == "INDUCTION_FAILED"
    assert s["_calls"]["duel"] == []


def test_induce_status_not_induced_gives_induction_failed_with_section():
    def failed_induce(game):
        return {"status": "FAILED", "overall": 0.5, "primary_accuracy": 0.3,
                "iterations": 4}

    s = _stubs(induce=failed_induce)
    rec = sweep_game("g", **{k: v for k, v in s.items() if not k.startswith("_")})
    assert rec["verdict"] == "INDUCTION_FAILED"
    # induction section is present even on failure (honest coverage claims)
    assert "induction" in rec
    assert rec["induction"]["status"] == "FAILED"
    assert s["_calls"]["duel"] == []


def test_duel_system_exit_gives_duel_refused():
    def refused_duel(game):
        raise SystemExit("model not INDUCED")

    s = _stubs(duel=refused_duel)
    rec = sweep_game("g", **{k: v for k, v in s.items() if not k.startswith("_")})
    assert rec["verdict"] == "DUEL_REFUSED"
    assert "error" in rec


def test_duel_runtime_error_gives_duel_error():
    def broken_duel(game):
        raise RuntimeError("browser crashed")

    s = _stubs(duel=broken_duel)
    rec = sweep_game("g", **{k: v for k, v in s.items() if not k.startswith("_")})
    assert rec["verdict"] == "DUEL_ERROR"
    assert "error" in rec


def test_duel_winner_baseline_gives_duel_lost():
    def baseline_wins(game):
        return {"planner": {"score": 3.0}, "baseline": {"score": 10.0},
                "winner": "baseline"}

    s = _stubs(duel=baseline_wins)
    rec = sweep_game("g", **{k: v for k, v in s.items() if not k.startswith("_")})
    assert rec["verdict"] == "DUEL_LOST"


def test_duel_winner_tie_gives_duel_tie():
    def tie_duel(game):
        return {"planner": {"score": 5.0}, "baseline": {"score": 5.0},
                "winner": "tie"}

    s = _stubs(duel=tie_duel)
    rec = sweep_game("g", **{k: v for k, v in s.items() if not k.startswith("_")})
    assert rec["verdict"] == "DUEL_TIE"


# ---------------------------------------------------------------------------
# render_table
# ---------------------------------------------------------------------------


def test_render_table_contains_row_per_game():
    results = {
        "alpha": {"game": "alpha", "verdict": "DUEL_WON",
                  "induction": {"overall": 0.92, "primary_accuracy": 0.95},
                  "duel": {"planner": 20, "baseline": 5, "winner": "planner"}},
        "beta": {"game": "beta", "verdict": "ONBOARD_FAILED"},
    }
    table = render_table(results)
    assert "alpha" in table
    assert "beta" in table
    assert "DUEL_WON" in table
    assert "ONBOARD_FAILED" in table


def test_render_table_contains_totals_line():
    results = {
        "alpha": {"game": "alpha", "verdict": "DUEL_WON"},
        "beta": {"game": "beta", "verdict": "DUEL_WON"},
        "gamma": {"game": "gamma", "verdict": "ONBOARD_FAILED"},
    }
    table = render_table(results)
    assert "Totals:" in table
    assert "DUEL_WON=2" in table
    assert "ONBOARD_FAILED=1" in table


def test_render_table_sorted_alphabetically():
    results = {
        "z_game": {"game": "z_game", "verdict": "DUEL_WON"},
        "a_game": {"game": "a_game", "verdict": "DUEL_LOST"},
    }
    table = render_table(results)
    # keep only data rows (skip header/separator which contain "verdict" or "---")
    data_lines = [l for l in table.splitlines()
                  if l.strip().startswith("|") and "verdict" not in l and "---" not in l]
    game_order = [l.split("|")[1].strip() for l in data_lines]
    assert game_order.index("a_game") < game_order.index("z_game")


def test_render_table_empty_induction_shows_dash():
    results = {"g": {"game": "g", "verdict": "ONBOARD_FAILED"}}
    table = render_table(results)
    # no induction data → dash placeholder
    assert "—" in table


def test_verdicts_constant_completeness():
    assert "DUEL_WON" in VERDICTS
    assert "ONBOARD_FAILED" in VERDICTS
    assert "INDUCTION_FAILED" in VERDICTS
    assert len(VERDICTS) == 8


def test_onboard_systemexit_becomes_verdict_not_crash():
    # onboard_game raises SystemExit on empty controls (CLI ergonomics);
    # SystemExit is not an Exception subclass — the sweep must survive it
    # (live bug: killed the re-sweep at game 4/14)
    from ludus.sweep import sweep_game

    def onboard(game):
        raise SystemExit("no controls discovered")

    record = sweep_game("g", onboard=onboard, explore=None, induce=None, duel=None)
    assert record["verdict"] == "ONBOARD_FAILED"
    assert "no controls" in record["error"]
