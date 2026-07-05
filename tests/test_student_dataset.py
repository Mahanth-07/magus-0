# tests/test_student_dataset.py
"""TDD tests for ludus/student/dataset.py — trajectory data layer (Stage 4 chunk 1).

Tests are written RED-first: they import from ludus.student.dataset which
does not exist yet, so the test file itself confirms the failing baseline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ludus.providers.insforge import _SYSTEM, build_user_text
from ludus.schemas import Decision, EpisodeResult, StepRecord


# ---------------------------------------------------------------------------
# Helpers: build a minimal on-disk episode directory
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
    episode_id: str = "traj-mygame-planner-000",
    game: str = "mygame",
    n_steps: int = 2,
    final_score: float = 50.0,
    state_texts: list[str] | None = None,
    primary_metric: str = "score",
    write_pngs: bool = True,
) -> Path:
    """Create a minimal runs/<episode_id>/ dir with steps.jsonl + episode.json + PNGs."""
    ep_dir = tmp_path / episode_id
    ep_dir.mkdir(parents=True)

    state_texts = state_texts or ["" for _ in range(n_steps)]

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
            state_text=state_texts[i] if i < len(state_texts) else "",
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
    """Write a minimal profiles/<game>.json so build_dataset can load it."""
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
# episode_to_sft_rows tests
# ---------------------------------------------------------------------------

def test_episode_to_sft_rows_returns_one_row_per_step(tmp_path):
    from ludus.student.dataset import episode_to_sft_rows

    ep_dir = _write_episode_dir(tmp_path, n_steps=3, final_score=30.0)
    rows = episode_to_sft_rows(ep_dir, objective="maximize score", legal_actions=["up", "down"])
    assert len(rows) == 3


def test_episode_to_sft_rows_required_keys_present(tmp_path):
    from ludus.student.dataset import episode_to_sft_rows

    ep_dir = _write_episode_dir(tmp_path, n_steps=2)
    rows = episode_to_sft_rows(ep_dir, objective="maximize score", legal_actions=["up", "down"])
    for row in rows:
        assert set(row.keys()) >= {"image", "system", "user", "assistant", "meta"}


def test_episode_to_sft_rows_system_is_insforge_system_prompt(tmp_path):
    from ludus.student.dataset import episode_to_sft_rows

    ep_dir = _write_episode_dir(tmp_path, n_steps=1)
    rows = episode_to_sft_rows(ep_dir, objective="obj", legal_actions=["up"])
    assert rows[0]["system"] == _SYSTEM


def test_episode_to_sft_rows_user_matches_runtime_build_user_text(tmp_path):
    """user text must be byte-for-byte identical to what build_user_text produces."""
    from ludus.student.dataset import episode_to_sft_rows

    state_text = "col heights: 1 2 3\nholes=1"
    ep_dir = _write_episode_dir(
        tmp_path, n_steps=1, state_texts=[state_text]
    )
    rows = episode_to_sft_rows(ep_dir, objective="maximize score", legal_actions=["up", "down"])
    expected_user = build_user_text(
        objective="maximize score",
        legal_actions=["up", "down"],
        state_text=state_text,
    )
    assert rows[0]["user"] == expected_user


def test_episode_to_sft_rows_assistant_is_valid_decision_json(tmp_path):
    from ludus.student.dataset import episode_to_sft_rows

    ep_dir = _write_episode_dir(tmp_path, n_steps=2)
    rows = episode_to_sft_rows(ep_dir, objective="obj", legal_actions=["up"])
    for row in rows:
        d = json.loads(row["assistant"])
        # Must contain all canonical Decision fields
        assert "scene_summary" in d
        assert "controlled_entity" in d
        assert "current_subgoal" in d
        assert "action" in d
        assert "actions" in d
        assert "action_args" in d
        assert "expected_result" in d
        assert "reason" in d
        assert "confidence" in d


def test_episode_to_sft_rows_meta_fields(tmp_path):
    from ludus.student.dataset import episode_to_sft_rows

    ep_dir = _write_episode_dir(
        tmp_path,
        episode_id="traj-mygame-planner-000",
        game="mygame",
        n_steps=2,
        final_score=75.0,
        primary_metric="score",
    )
    rows = episode_to_sft_rows(ep_dir, objective="obj", legal_actions=["up"])
    meta = rows[0]["meta"]
    assert meta["game"] == "mygame"
    assert meta["episode_id"] == "traj-mygame-planner-000"
    assert meta["step_index"] == 0
    assert "primary_delta" in meta
    assert meta["final_score"] == 75.0


def test_episode_to_sft_rows_image_is_absolute_path_to_png(tmp_path):
    from ludus.student.dataset import episode_to_sft_rows

    ep_dir = _write_episode_dir(tmp_path, n_steps=2)
    rows = episode_to_sft_rows(ep_dir, objective="obj", legal_actions=["up"])
    for row in rows:
        img_path = Path(row["image"])
        assert img_path.is_absolute()
        assert img_path.suffix == ".png"
        assert img_path.exists()


def test_episode_to_sft_rows_step_order_preserved(tmp_path):
    from ludus.student.dataset import episode_to_sft_rows

    ep_dir = _write_episode_dir(tmp_path, n_steps=4)
    rows = episode_to_sft_rows(ep_dir, objective="obj", legal_actions=["up"])
    for i, row in enumerate(rows):
        assert row["meta"]["step_index"] == i


# ---------------------------------------------------------------------------
# build_dataset tests
# ---------------------------------------------------------------------------

def test_build_dataset_writes_jsonl_and_returns_counts(tmp_path):
    from ludus.student.dataset import build_dataset

    runs_dir = tmp_path / "runs"
    profiles_dir = tmp_path / "profiles"
    _write_episode_dir(runs_dir, game="mygame", episode_id="ep-000", n_steps=3, final_score=50.0)
    _write_profile(profiles_dir, "mygame")

    out_path = tmp_path / "sft.jsonl"
    summary = build_dataset(runs_dir, out_path, profiles_dir=profiles_dir)

    assert out_path.exists()
    lines = [l for l in out_path.read_text().splitlines() if l.strip()]
    assert len(lines) == summary["rows"]
    assert summary["episodes_seen"] == 1
    assert summary["episodes_kept"] == 1
    assert summary["rows"] == 3


def test_build_dataset_filters_below_score_threshold(tmp_path):
    from ludus.student.dataset import build_dataset

    runs_dir = tmp_path / "runs"
    profiles_dir = tmp_path / "profiles"
    _write_episode_dir(runs_dir, game="mygame", episode_id="ep-high", n_steps=2, final_score=200.0)
    _write_episode_dir(runs_dir, game="mygame", episode_id="ep-low", n_steps=2, final_score=5.0)
    _write_profile(profiles_dir, "mygame")

    out_path = tmp_path / "sft.jsonl"
    summary = build_dataset(runs_dir, out_path, min_final_score=100.0, profiles_dir=profiles_dir)

    assert summary["episodes_seen"] == 2
    assert summary["episodes_kept"] == 1   # only ep-high
    assert summary["rows"] == 2            # 2 steps from the kept episode


def test_build_dataset_per_game_threshold(tmp_path):
    """min_final_score as a dict applies per-game thresholds."""
    from ludus.student.dataset import build_dataset

    runs_dir = tmp_path / "runs"
    profiles_dir = tmp_path / "profiles"
    _write_episode_dir(runs_dir, game="game_a", episode_id="ep-a", n_steps=2, final_score=30.0)
    _write_episode_dir(runs_dir, game="game_b", episode_id="ep-b", n_steps=2, final_score=30.0)
    _write_profile(profiles_dir, "game_a")
    _write_profile(profiles_dir, "game_b")

    out_path = tmp_path / "sft.jsonl"
    # game_a threshold 10 (keep), game_b threshold 50 (filter)
    summary = build_dataset(
        runs_dir, out_path,
        min_final_score={"game_a": 10.0, "game_b": 50.0},
        profiles_dir=profiles_dir,
    )
    assert summary["episodes_kept"] == 1
    assert summary["rows"] == 2


def test_build_dataset_skips_episodes_with_missing_profile(tmp_path):
    from ludus.student.dataset import build_dataset

    runs_dir = tmp_path / "runs"
    profiles_dir = tmp_path / "profiles"
    _write_episode_dir(runs_dir, game="known_game", episode_id="ep-known", n_steps=2, final_score=50.0)
    _write_episode_dir(runs_dir, game="unknown_game", episode_id="ep-unknown", n_steps=2, final_score=50.0)
    _write_profile(profiles_dir, "known_game")
    # intentionally NO profile for "unknown_game"

    out_path = tmp_path / "sft.jsonl"
    summary = build_dataset(runs_dir, out_path, profiles_dir=profiles_dir)

    # unknown_game episode is skipped; known_game is processed
    assert summary["episodes_kept"] == 1
    assert summary["rows"] == 2


def test_build_dataset_output_rows_are_valid_jsonl(tmp_path):
    from ludus.student.dataset import build_dataset

    runs_dir = tmp_path / "runs"
    profiles_dir = tmp_path / "profiles"
    _write_episode_dir(runs_dir, game="mygame", episode_id="ep-000", n_steps=2, final_score=50.0)
    _write_profile(profiles_dir, "mygame")

    out_path = tmp_path / "sft.jsonl"
    build_dataset(runs_dir, out_path, profiles_dir=profiles_dir)

    for line in out_path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            assert "image" in row
            assert "system" in row
            assert "user" in row
            assert "assistant" in row
            assert "meta" in row


def test_build_dataset_zero_threshold_keeps_all(tmp_path):
    """Default threshold of 0.0 keeps all episodes with non-negative scores."""
    from ludus.student.dataset import build_dataset

    runs_dir = tmp_path / "runs"
    profiles_dir = tmp_path / "profiles"
    for i in range(3):
        _write_episode_dir(
            runs_dir, game="mygame", episode_id=f"ep-{i:03d}",
            n_steps=1, final_score=float(i * 10)
        )
    _write_profile(profiles_dir, "mygame")

    out_path = tmp_path / "sft.jsonl"
    summary = build_dataset(runs_dir, out_path, min_final_score=0.0, profiles_dir=profiles_dir)
    assert summary["episodes_kept"] == 3
