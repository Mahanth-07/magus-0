"""TDD tests for ludus/student/recap.py — RECAP-lite advantage-conditioned SFT.

Tests are written RED-first. They import from ludus.student.recap which does
not exist yet, so running them immediately confirms the failing baseline.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ludus.providers.insforge import _SYSTEM
from ludus.schemas import Decision, EpisodeResult, StepRecord


# ---------------------------------------------------------------------------
# Fixture helpers (mirror test_student_dataset.py exactly)
# ---------------------------------------------------------------------------

def _make_decision(**kwargs) -> Decision:
    defaults = dict(
        scene_summary="s", controlled_entity="c", current_subgoal="g",
        action="up", actions=["up"], action_args={"duration_ms": 200},
        expected_result="score increases", reason="r", confidence=0.9,
    )
    defaults.update(kwargs)
    return Decision(**defaults)


def _write_episode_dir(
    tmp_path: Path,
    episode_id: str,
    game: str,
    n_steps: int = 2,
    final_score: float = 50.0,
    primary_metric: str = "score",
    write_pngs: bool = True,
) -> Path:
    """Create a minimal runs/<episode_id>/ dir with steps.jsonl + episode.json + PNGs."""
    ep_dir = tmp_path / episode_id
    ep_dir.mkdir(parents=True)

    steps_lines = []
    for i in range(n_steps):
        step = StepRecord(
            episode_id=episode_id,
            step_index=i,
            mode="baseline",
            game=game,
            decision=_make_decision(),
            primary_metric=primary_metric,
            primary_delta=float(i + 1),
            improved=True,
            metric_delta={primary_metric: float(i + 1)},
            screenshot_ref=f"{episode_id}/step_{i:04d}.png",
            state_text="",
        )
        steps_lines.append(step.model_dump_json())
        if write_pngs:
            png_path = ep_dir / f"step_{i:04d}.png"
            png_path.write_bytes(b"\x89PNG fake")

    (ep_dir / "steps.jsonl").write_text("\n".join(steps_lines) + "\n")

    result = EpisodeResult(
        episode_id=episode_id,
        game=game,
        mode="baseline",
        steps=n_steps,
        legal_action_rate=1.0,
        final_metrics={primary_metric: final_score},
        rules=[],
        errored_steps=0,
    )
    (ep_dir / "episode.json").write_text(result.model_dump_json(indent=2))
    return ep_dir


def _write_profile(profiles_dir: Path, game: str, legal_actions=None, objective=None) -> None:
    """Write a minimal profiles/<game>.json so build_recap_dataset can load it."""
    from ludus.onboarding.profile import GameProfile
    legal_actions = legal_actions or ["up", "down", "left", "right"]
    objective = objective or f"maximize score in {game}"
    profile = GameProfile(
        game_id=game,
        controls={a: a for a in legal_actions},
        control_effects={a: [f"state.{a}"] for a in legal_actions},
        metrics=["score"],
        primary_metric="score",
        higher_is_better=True,
        timing_ms=120,
        objective=objective,
        state_schema={"score": "float"},
    )
    profiles_dir.mkdir(parents=True, exist_ok=True)
    profile.save(profiles_dir / f"{game}.json")


# ---------------------------------------------------------------------------
# Tests: compute_advantages
# ---------------------------------------------------------------------------

def test_compute_advantages_two_episodes_one_game(tmp_path):
    """Two episodes of the same game: one above baseline -> expert, one below -> novice."""
    from ludus.student.recap import compute_advantages

    runs_dir = tmp_path / "runs"
    profiles_dir = tmp_path / "profiles"
    _write_episode_dir(runs_dir, "ep-game1-high", game="game1", final_score=80.0)
    _write_episode_dir(runs_dir, "ep-game1-low", game="game1", final_score=20.0)
    _write_profile(profiles_dir, "game1")

    adv = compute_advantages(runs_dir, profiles_dir=profiles_dir)

    # baseline = mean(80, 20) = 50
    assert set(adv.keys()) == {"ep-game1-high", "ep-game1-low"}
    high = adv["ep-game1-high"]
    low = adv["ep-game1-low"]

    assert high["game"] == "game1"
    assert high["final_score"] == 80.0
    assert abs(high["baseline"] - 50.0) < 1e-9
    assert abs(high["advantage"] - 30.0) < 1e-9
    assert high["label"] == "expert"

    assert low["label"] == "novice"
    assert abs(low["advantage"] - (-30.0)) < 1e-9


def test_compute_advantages_negative_advantage_is_novice(tmp_path):
    """Episode with score below game mean is labelled novice."""
    from ludus.student.recap import compute_advantages

    runs_dir = tmp_path / "runs"
    profiles_dir = tmp_path / "profiles"
    _write_episode_dir(runs_dir, "ep-g-a", game="gX", final_score=10.0)
    _write_episode_dir(runs_dir, "ep-g-b", game="gX", final_score=90.0)
    _write_episode_dir(runs_dir, "ep-g-c", game="gX", final_score=50.0)
    _write_profile(profiles_dir, "gX")

    # baseline = mean(10, 90, 50) = 50
    adv = compute_advantages(runs_dir, profiles_dir=profiles_dir)
    assert adv["ep-g-a"]["label"] == "novice"
    assert adv["ep-g-b"]["label"] == "expert"
    assert adv["ep-g-c"]["label"] == "expert"  # advantage = 0 -> expert


def test_compute_advantages_zero_advantage_is_expert(tmp_path):
    """Advantage exactly 0 (on-baseline) is labelled expert."""
    from ludus.student.recap import compute_advantages

    runs_dir = tmp_path / "runs"
    profiles_dir = tmp_path / "profiles"
    _write_episode_dir(runs_dir, "ep-g-a", game="g2", final_score=50.0)
    _write_episode_dir(runs_dir, "ep-g-b", game="g2", final_score=50.0)
    _write_profile(profiles_dir, "g2")

    adv = compute_advantages(runs_dir, profiles_dir=profiles_dir)
    assert adv["ep-g-a"]["advantage"] == 0.0
    assert adv["ep-g-a"]["label"] == "expert"
    assert adv["ep-g-b"]["label"] == "expert"


def test_compute_advantages_single_episode_game_is_expert(tmp_path):
    """A game with exactly one episode gets advantage=0 and label='expert'."""
    from ludus.student.recap import compute_advantages

    runs_dir = tmp_path / "runs"
    profiles_dir = tmp_path / "profiles"
    _write_episode_dir(runs_dir, "ep-solo", game="solo_game", final_score=42.0)
    _write_profile(profiles_dir, "solo_game")

    adv = compute_advantages(runs_dir, profiles_dir=profiles_dir)
    solo = adv["ep-solo"]
    assert solo["advantage"] == 0.0
    assert solo["label"] == "expert"
    assert solo["baseline"] == 42.0


def test_compute_advantages_multi_game_independent(tmp_path):
    """Two games; baselines are computed independently."""
    from ludus.student.recap import compute_advantages

    runs_dir = tmp_path / "runs"
    profiles_dir = tmp_path / "profiles"
    # game_a: scores 10, 30 -> baseline 20
    _write_episode_dir(runs_dir, "ep-a1", game="game_a", final_score=10.0)
    _write_episode_dir(runs_dir, "ep-a2", game="game_a", final_score=30.0)
    # game_b: scores 100, 200 -> baseline 150
    _write_episode_dir(runs_dir, "ep-b1", game="game_b", final_score=100.0)
    _write_episode_dir(runs_dir, "ep-b2", game="game_b", final_score=200.0)
    _write_profile(profiles_dir, "game_a")
    _write_profile(profiles_dir, "game_b")

    adv = compute_advantages(runs_dir, profiles_dir=profiles_dir)
    assert adv["ep-a1"]["baseline"] == 20.0
    assert adv["ep-b1"]["baseline"] == 150.0
    assert adv["ep-a1"]["label"] == "novice"
    assert adv["ep-a2"]["label"] == "expert"
    assert adv["ep-b1"]["label"] == "novice"
    assert adv["ep-b2"]["label"] == "expert"


# ---------------------------------------------------------------------------
# Tests: condition_system and RECAP_CONDITION constant
# ---------------------------------------------------------------------------

def test_recap_condition_constant_and_helper():
    """RECAP_CONDITION constant exists; condition_system produces the right suffix."""
    from ludus.student.recap import RECAP_CONDITION, condition_system

    assert "{label}" in RECAP_CONDITION
    assert condition_system("expert") == "\nPlay quality: expert."
    assert condition_system("novice") == "\nPlay quality: novice."
    # Must start with newline so it appends cleanly to _SYSTEM
    assert condition_system("expert").startswith("\n")


# ---------------------------------------------------------------------------
# Tests: build_recap_dataset
# ---------------------------------------------------------------------------

def test_build_recap_dataset_keeps_novice_episodes(tmp_path):
    """Unlike build_dataset (which filters), RECAP keeps ALL episodes including novice."""
    from ludus.student.recap import build_recap_dataset

    runs_dir = tmp_path / "runs"
    profiles_dir = tmp_path / "profiles"
    _write_episode_dir(runs_dir, "ep-hi", game="mygame", final_score=80.0, n_steps=2)
    _write_episode_dir(runs_dir, "ep-lo", game="mygame", final_score=20.0, n_steps=2)
    _write_profile(profiles_dir, "mygame")

    out_path = tmp_path / "recap.jsonl"
    summary = build_recap_dataset(runs_dir, out_path, profiles_dir=profiles_dir)

    # Both episodes kept (novice is NOT filtered)
    assert summary["episodes"] == 2
    assert summary["expert_episodes"] == 1
    assert summary["novice_episodes"] == 1
    assert summary["rows"] == 4  # 2 steps * 2 episodes


def test_build_recap_dataset_expert_system_prompt(tmp_path):
    """Expert episode rows have 'Play quality: expert.' in the system field."""
    from ludus.student.recap import build_recap_dataset

    runs_dir = tmp_path / "runs"
    profiles_dir = tmp_path / "profiles"
    _write_episode_dir(runs_dir, "ep-hi", game="mygame", final_score=80.0, n_steps=1)
    _write_episode_dir(runs_dir, "ep-lo", game="mygame", final_score=20.0, n_steps=1)
    _write_profile(profiles_dir, "mygame")

    out_path = tmp_path / "recap.jsonl"
    build_recap_dataset(runs_dir, out_path, profiles_dir=profiles_dir)

    rows = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
    expert_rows = [r for r in rows if "Play quality: expert." in r["system"]]
    novice_rows = [r for r in rows if "Play quality: novice." in r["system"]]
    assert len(expert_rows) == 1
    assert len(novice_rows) == 1


def test_build_recap_dataset_novice_system_prompt(tmp_path):
    """Novice episode rows have 'Play quality: novice.' appended to the system field."""
    from ludus.student.recap import build_recap_dataset

    runs_dir = tmp_path / "runs"
    profiles_dir = tmp_path / "profiles"
    _write_episode_dir(runs_dir, "ep-hi", game="mygame", final_score=90.0, n_steps=1)
    _write_episode_dir(runs_dir, "ep-lo", game="mygame", final_score=10.0, n_steps=1)
    _write_profile(profiles_dir, "mygame")

    out_path = tmp_path / "recap.jsonl"
    build_recap_dataset(runs_dir, out_path, profiles_dir=profiles_dir)

    rows = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
    novice_rows = [r for r in rows if "ep-lo" in r["meta"]["episode_id"]]
    assert all("Play quality: novice." in r["system"] for r in novice_rows)


def test_build_recap_dataset_meta_gains_advantage_fields(tmp_path):
    """Each row's meta dict gains 'advantage', 'label', and 'baseline' keys."""
    from ludus.student.recap import build_recap_dataset

    runs_dir = tmp_path / "runs"
    profiles_dir = tmp_path / "profiles"
    _write_episode_dir(runs_dir, "ep-hi", game="mygame", final_score=80.0, n_steps=1)
    _write_episode_dir(runs_dir, "ep-lo", game="mygame", final_score=20.0, n_steps=1)
    _write_profile(profiles_dir, "mygame")

    out_path = tmp_path / "recap.jsonl"
    build_recap_dataset(runs_dir, out_path, profiles_dir=profiles_dir)

    rows = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
    for row in rows:
        meta = row["meta"]
        assert "advantage" in meta
        assert "label" in meta
        assert "baseline" in meta


def test_build_recap_dataset_system_starts_with_insforge_system(tmp_path):
    """System field must start with _SYSTEM (conditioning is a suffix, not a replacement)."""
    from ludus.student.recap import build_recap_dataset

    runs_dir = tmp_path / "runs"
    profiles_dir = tmp_path / "profiles"
    _write_episode_dir(runs_dir, "ep-hi", game="mygame", final_score=80.0, n_steps=1)
    _write_episode_dir(runs_dir, "ep-lo", game="mygame", final_score=20.0, n_steps=1)
    _write_profile(profiles_dir, "mygame")

    out_path = tmp_path / "recap.jsonl"
    build_recap_dataset(runs_dir, out_path, profiles_dir=profiles_dir)

    rows = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
    for row in rows:
        assert row["system"].startswith(_SYSTEM)


def test_build_recap_dataset_excludes_games_below_min_episodes(tmp_path):
    """Games with fewer than min_episodes_per_game episodes are excluded."""
    from ludus.student.recap import build_recap_dataset

    runs_dir = tmp_path / "runs"
    profiles_dir = tmp_path / "profiles"
    # game_a has 2 episodes -> included
    _write_episode_dir(runs_dir, "ep-a1", game="game_a", final_score=80.0, n_steps=1)
    _write_episode_dir(runs_dir, "ep-a2", game="game_a", final_score=20.0, n_steps=1)
    # game_b has only 1 episode -> excluded by min_episodes_per_game=2
    _write_episode_dir(runs_dir, "ep-b1", game="game_b", final_score=50.0, n_steps=1)
    _write_profile(profiles_dir, "game_a")
    _write_profile(profiles_dir, "game_b")

    out_path = tmp_path / "recap.jsonl"
    summary = build_recap_dataset(
        runs_dir, out_path, profiles_dir=profiles_dir, min_episodes_per_game=2
    )

    # Only game_a's 2 episodes kept; game_b's 1 episode excluded
    assert summary["episodes"] == 2
    rows = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
    assert all(r["meta"]["game"] == "game_a" for r in rows)


def test_build_recap_dataset_rows_export_compatible(tmp_path):
    """Rows have image/system/user/assistant keys, making them export_together-compatible."""
    from ludus.student.recap import build_recap_dataset

    runs_dir = tmp_path / "runs"
    profiles_dir = tmp_path / "profiles"
    _write_episode_dir(runs_dir, "ep-hi", game="mygame", final_score=80.0, n_steps=2)
    _write_episode_dir(runs_dir, "ep-lo", game="mygame", final_score=20.0, n_steps=2)
    _write_profile(profiles_dir, "mygame")

    out_path = tmp_path / "recap.jsonl"
    build_recap_dataset(runs_dir, out_path, profiles_dir=profiles_dir)

    rows = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
    for row in rows:
        assert "image" in row
        assert "system" in row
        assert "user" in row
        assert "assistant" in row
        assert "meta" in row


def test_build_recap_dataset_summary_counts(tmp_path):
    """Summary dict returns correct episode/row counts."""
    from ludus.student.recap import build_recap_dataset

    runs_dir = tmp_path / "runs"
    profiles_dir = tmp_path / "profiles"
    # 3 episodes: 2 above mean, 1 below
    _write_episode_dir(runs_dir, "ep-1", game="mygame", final_score=100.0, n_steps=2)
    _write_episode_dir(runs_dir, "ep-2", game="mygame", final_score=80.0, n_steps=3)
    _write_episode_dir(runs_dir, "ep-3", game="mygame", final_score=10.0, n_steps=1)
    _write_profile(profiles_dir, "mygame")

    out_path = tmp_path / "recap.jsonl"
    # mean = (100+80+10)/3 = 63.33; ep-1 expert, ep-2 expert, ep-3 novice
    summary = build_recap_dataset(runs_dir, out_path, profiles_dir=profiles_dir)

    assert summary["episodes"] == 3
    assert summary["expert_episodes"] == 2
    assert summary["novice_episodes"] == 1
    assert summary["rows"] == 6  # 2+3+1
