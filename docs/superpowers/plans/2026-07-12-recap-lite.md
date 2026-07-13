# RECAP-lite: Advantage-Conditioned SFT Labeling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `ludus/student/recap.py` that computes per-episode advantages over a per-game baseline, labels episodes expert/novice, and produces conditioned SFT rows; gate `TogetherProvider` to append `"\nPlay quality: expert."` to the system message when `TOGETHER_RECAP=1`.

**Architecture:** Three layers — (1) pure-math advantage computation (`compute_advantages`), (2) RECAP dataset builder (`build_recap_dataset`) that keeps ALL episodes of qualified games and injects a conditioning suffix into the system field, (3) serving gate in `TogetherProvider._messages` toggled by `TOGETHER_RECAP` env var. A module constant `RECAP_CONDITION = "\nPlay quality: {label}."` and a `condition_system(label)` helper ensure exact byte-parity between training and inference conditioning strings.

**Tech Stack:** Python 3.11, pathlib, json, `ludus.schemas` (EpisodeResult/StepRecord), `ludus.providers.insforge._SYSTEM`, `ludus.student.dataset` helpers, pytest tmp_path fixtures.

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `ludus/student/recap.py` | `RECAP_CONDITION`, `condition_system`, `compute_advantages`, `build_recap_dataset` |
| Modify | `ludus/providers/together.py` | Append `condition_system("expert")` to system when `TOGETHER_RECAP=1` |
| Create | `tests/test_recap.py` | All `recap.py` tests (advantage math, labels, dataset builder, export compatibility) |
| Modify | `tests/test_together_provider.py` | Two new tests: RECAP gating on/off |

---

## Task 1: Write failing tests for `compute_advantages`

**Files:**
- Create: `tests/test_recap.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_recap.py` with the following content. These will fail because `ludus/student/recap.py` doesn't exist yet.

```python
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
```

- [ ] **Step 2: Run tests to verify they all fail**

```bash
cd /Users/mahanth/ludus && .venv/bin/pytest tests/test_recap.py -v 2>&1 | head -40
```

Expected: All tests fail with `ModuleNotFoundError: No module named 'ludus.student.recap'`

---

## Task 2: Write failing tests for `TogetherProvider` RECAP gating

**Files:**
- Modify: `tests/test_together_provider.py`

- [ ] **Step 1: Append the two new RECAP gating tests to `tests/test_together_provider.py`**

Open `tests/test_together_provider.py` and add these two tests at the bottom (after the existing `test_base_url_defaults_and_override` test):

```python
# ---------------------------------------------------------------------------
# Test 6: RECAP gating — TOGETHER_RECAP=1 appends conditioning suffix
# ---------------------------------------------------------------------------

def test_together_recap_off_by_default(monkeypatch):
    """Without TOGETHER_RECAP=1, system message must equal _SYSTEM exactly (no suffix)."""
    monkeypatch.delenv("TOGETHER_RECAP", raising=False)
    monkeypatch.setenv("TOGETHER_API_KEY", "test-key")
    monkeypatch.setenv("TOGETHER_MODEL", "test/model")
    captured = _capture_post(monkeypatch)

    ctx = PlannerContext(
        objective="win", legal_actions=["left", "right"],
        state_text="", screenshot_png=_make_png(),
    )
    TogetherProvider().decide(ctx)

    msgs = captured["json"]["messages"]
    sys_content = msgs[0]["content"]
    assert sys_content == _SYSTEM, (
        f"With TOGETHER_RECAP unset, system must be exactly _SYSTEM. Got: {sys_content!r}"
    )


def test_together_recap_on_appends_expert_conditioning(monkeypatch):
    """With TOGETHER_RECAP=1, system message is _SYSTEM + condition_system('expert')."""
    from ludus.student.recap import condition_system

    monkeypatch.setenv("TOGETHER_RECAP", "1")
    monkeypatch.setenv("TOGETHER_API_KEY", "test-key")
    monkeypatch.setenv("TOGETHER_MODEL", "test/model")
    captured = _capture_post(monkeypatch)

    ctx = PlannerContext(
        objective="win", legal_actions=["left", "right"],
        state_text="", screenshot_png=_make_png(),
    )
    TogetherProvider().decide(ctx)

    msgs = captured["json"]["messages"]
    sys_content = msgs[0]["content"]
    expected = _SYSTEM + condition_system("expert")
    assert sys_content == expected, (
        f"With TOGETHER_RECAP=1, system must be _SYSTEM + conditioning.\n"
        f"Expected: {expected!r}\nGot: {sys_content!r}"
    )
```

- [ ] **Step 2: Run the new tests to verify they fail**

```bash
cd /Users/mahanth/ludus && .venv/bin/pytest tests/test_together_provider.py::test_together_recap_off_by_default tests/test_together_provider.py::test_together_recap_on_appends_expert_conditioning -v 2>&1
```

Expected: `test_together_recap_off_by_default` PASSES (already satisfied by current code — system is _SYSTEM), `test_together_recap_on_appends_expert_conditioning` FAILS with `ModuleNotFoundError` or wrong system content.

---

## Task 3: Implement `ludus/student/recap.py`

**Files:**
- Create: `ludus/student/recap.py`

- [ ] **Step 1: Create the module**

```python
"""RECAP-lite: advantage-conditioned SFT labeling.

Design (π*0.6 RECAP adapted — pure SFT, no policy gradients):
  - compute_advantages: walk episode dirs, group by game, compute per-game mean
    final score (baseline), label each episode expert (advantage >= 0) or novice.
  - build_recap_dataset: keep ALL episodes of games with >= min_episodes_per_game
    episodes; inject conditioning suffix into system field of each row.
  - RECAP_CONDITION / condition_system: single source of truth for the suffix
    string so training and serving are byte-identical.

The conditioning suffix is appended to _SYSTEM (the InsForge preamble) at
training time; the serving provider (TogetherProvider with TOGETHER_RECAP=1)
must reproduce it exactly using condition_system("expert").
"""
from __future__ import annotations

import json
from pathlib import Path

from ludus.providers.insforge import _SYSTEM, build_user_text
from ludus.schemas import EpisodeResult, StepRecord
from ludus.student.dataset import episode_to_sft_rows, _get_primary_metric

# Module constant: the conditioning suffix template.
# {label} is filled in by condition_system(); the serving provider must use
# condition_system("expert") to reproduce the EXACT same string.
RECAP_CONDITION = "\nPlay quality: {label}."


def condition_system(label: str) -> str:
    """Return the conditioning suffix for a given label ('expert' or 'novice').

    The serving provider (TogetherProvider with TOGETHER_RECAP=1) calls this
    with label='expert' so inference exactly matches the training system prompt.
    """
    return RECAP_CONDITION.format(label=label)


def _read_episode(ep_dir: Path) -> EpisodeResult | None:
    """Read and parse episode.json; return None on any failure."""
    ep_path = ep_dir / "episode.json"
    steps_path = ep_dir / "steps.jsonl"
    if not ep_path.exists() or not steps_path.exists():
        return None
    try:
        return EpisodeResult.model_validate_json(ep_path.read_text())
    except Exception:
        return None


def _final_score(result: EpisodeResult, steps_path: Path) -> float:
    """Extract the primary-metric final score from an EpisodeResult.

    Mirrors the logic in build_dataset: try to find the primary_metric field
    name from the first step record, fall back to the first metric key.
    """
    primary_metric = _get_primary_metric(steps_path)
    if primary_metric and primary_metric in result.final_metrics:
        return float(result.final_metrics[primary_metric])
    # Fall back to the first metric value
    if result.final_metrics:
        return float(next(iter(result.final_metrics.values())))
    return 0.0


def compute_advantages(
    runs_dir: Path | str,
    *,
    profiles_dir: Path | str | None = None,
) -> dict[str, dict]:
    """Walk episode dirs, compute per-game baselines and per-episode advantages.

    Returns:
        {episode_id: {game, final_score, baseline, advantage, label}}

    Algorithm:
      1. Collect all valid episode dirs (has steps.jsonl + episode.json).
      2. Group by game; compute baseline = mean(final_score) per game.
      3. For each episode: advantage = final_score - baseline.
         label = "expert" if advantage >= 0 else "novice".
      4. Single-episode games: advantage = 0, label = "expert".

    profiles_dir is accepted for API compatibility but not used for filtering —
    compute_advantages processes ALL valid episode dirs regardless of profile.
    """
    runs_dir = Path(runs_dir)

    # Phase 1: collect (episode_id, game, final_score) for all valid episodes
    episodes: list[tuple[str, str, float, Path]] = []
    for ep_dir in sorted(runs_dir.iterdir()):
        if not ep_dir.is_dir():
            continue
        result = _read_episode(ep_dir)
        if result is None:
            continue
        steps_path = ep_dir / "steps.jsonl"
        score = _final_score(result, steps_path)
        episodes.append((result.episode_id, result.game, score, ep_dir))

    # Phase 2: per-game baseline (mean final score)
    game_scores: dict[str, list[float]] = {}
    for episode_id, game, score, _ in episodes:
        game_scores.setdefault(game, []).append(score)

    baselines: dict[str, float] = {
        game: sum(scores) / len(scores)
        for game, scores in game_scores.items()
    }

    # Phase 3: compute advantage and label per episode
    result_map: dict[str, dict] = {}
    for episode_id, game, score, ep_dir in episodes:
        baseline = baselines[game]
        advantage = score - baseline
        label = "expert" if advantage >= 0 else "novice"
        result_map[episode_id] = {
            "game": game,
            "final_score": score,
            "baseline": baseline,
            "advantage": advantage,
            "label": label,
        }

    return result_map


def build_recap_dataset(
    runs_dir: Path | str,
    out_path: Path | str,
    *,
    profiles_dir: Path | str | None = None,
    min_episodes_per_game: int = 2,
) -> dict:
    """Build advantage-conditioned SFT JSONL from all qualifying episodes.

    Unlike build_dataset (which FILTERS by score), build_recap_dataset:
      - Keeps ALL episodes of games with >= min_episodes_per_game episodes.
      - Conditions each row: system field becomes _SYSTEM + condition_system(label).
      - Adds {advantage, label, baseline} to each row's meta dict.

    Returns:
        {"episodes": N, "expert_episodes": E, "novice_episodes": V, "rows": R}

    Profile lookup:
        Loads profiles_dir/<game>.json for objective + legal_actions.
        Episodes whose profile is missing are excluded silently.
    """
    runs_dir = Path(runs_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if profiles_dir is not None:
        profiles_dir = Path(profiles_dir)

    # Step 1: compute advantages (all episodes, all games)
    advantages = compute_advantages(runs_dir, profiles_dir=profiles_dir)

    # Step 2: group by game to apply min_episodes_per_game filter
    game_episodes: dict[str, list[str]] = {}
    for episode_id, info in advantages.items():
        game_episodes.setdefault(info["game"], []).append(episode_id)

    # Set of episode_ids that qualify (game has >= min_episodes_per_game)
    qualified: set[str] = set()
    for game, ep_ids in game_episodes.items():
        if len(ep_ids) >= min_episodes_per_game:
            qualified.update(ep_ids)

    # Step 3: build episode_dir lookup
    ep_dir_map: dict[str, Path] = {}
    for ep_dir in sorted(runs_dir.iterdir()):
        if not ep_dir.is_dir():
            continue
        result = _read_episode(ep_dir)
        if result is not None:
            ep_dir_map[result.episode_id] = ep_dir

    # Step 4: write conditioned rows
    n_episodes = n_expert = n_novice = n_rows = 0

    with out_path.open("w") as fout:
        for episode_id in sorted(qualified):
            if episode_id not in ep_dir_map:
                continue

            ep_dir = ep_dir_map[episode_id]
            adv_info = advantages[episode_id]
            game = adv_info["game"]
            label = adv_info["label"]

            # Profile lookup for objective + legal_actions
            if profiles_dir is None:
                continue
            profile_path = profiles_dir / f"{game}.json"
            if not profile_path.exists():
                continue

            try:
                from ludus.onboarding.profile import GameProfile
                profile = GameProfile.load(profile_path)
            except Exception:
                continue

            objective = profile.objective
            legal_actions = list(profile.controls.keys())

            # Generate base SFT rows (identical to episode_to_sft_rows output)
            sft_rows = episode_to_sft_rows(
                ep_dir, objective=objective, legal_actions=legal_actions
            )

            # Condition each row: replace system with _SYSTEM + conditioning suffix
            conditioned_system = _SYSTEM + condition_system(label)
            for row in sft_rows:
                row["system"] = conditioned_system
                row["meta"]["advantage"] = adv_info["advantage"]
                row["meta"]["label"] = label
                row["meta"]["baseline"] = adv_info["baseline"]
                fout.write(json.dumps(row) + "\n")
                n_rows += 1

            n_episodes += 1
            if label == "expert":
                n_expert += 1
            else:
                n_novice += 1

    return {
        "episodes": n_episodes,
        "expert_episodes": n_expert,
        "novice_episodes": n_novice,
        "rows": n_rows,
    }
```

- [ ] **Step 2: Run the recap tests to verify they pass**

```bash
cd /Users/mahanth/ludus && .venv/bin/pytest tests/test_recap.py -v 2>&1
```

Expected: All tests in `tests/test_recap.py` PASS.

- [ ] **Step 3: Commit**

```bash
git add ludus/student/recap.py tests/test_recap.py
git commit -m "feat(student): add recap.py — advantage computation and conditioned SFT dataset builder"
```

---

## Task 4: Implement `TogetherProvider` RECAP gating

**Files:**
- Modify: `ludus/providers/together.py`

- [ ] **Step 1: Modify `_messages` in `TogetherProvider` to check `TOGETHER_RECAP`**

Open `ludus/providers/together.py`. The current `_messages` method returns:
```python
{"role": "system", "content": _SYSTEM},
```

Replace `_SYSTEM` with a local variable that conditionally appends the expert conditioning. Add this at the top of `_messages`:

```python
def _messages(self, ctx: PlannerContext) -> list[dict]:
    import os
    from ludus.student.recap import condition_system
    # Gate: only append expert conditioning when TOGETHER_RECAP=1
    recap_on = os.environ.get("TOGETHER_RECAP", "0").strip() == "1"
    system_text = _SYSTEM + condition_system("expert") if recap_on else _SYSTEM

    user_text = build_user_text(
        objective=ctx.objective,
        legal_actions=ctx.legal_actions,
        state_text=ctx.state_text,
        recent_outcomes=ctx.recent_outcomes,
        partner_recent_actions=ctx.partner_recent_actions,
        learned_rules=ctx.learned_rules,
    )
    # Image: downscaled to max 512px JPEG — EXACT same preprocessing as training.
    data_uri = _encode_image(ctx.screenshot_png)
    return [
        {"role": "system", "content": system_text},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ],
        },
    ]
```

- [ ] **Step 2: Run the together provider tests**

```bash
cd /Users/mahanth/ludus && .venv/bin/pytest tests/test_together_provider.py -v 2>&1
```

Expected: All 7 tests (5 original + 2 new) PASS.

- [ ] **Step 3: Verify the existing test that checks `sys_msg["content"] == _SYSTEM` still passes**

`test_together_payload_shape_system_string_user_array` checks `sys_msg["content"] == _SYSTEM`. Because the new test `test_together_recap_off_by_default` uses `monkeypatch.delenv("TOGETHER_RECAP", ...)`, and the existing test doesn't set `TOGETHER_RECAP`, this existing test will still pass — confirm it does.

```bash
cd /Users/mahanth/ludus && .venv/bin/pytest tests/test_together_provider.py::test_together_payload_shape_system_string_user_array -v 2>&1
```

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add ludus/providers/together.py tests/test_together_provider.py
git commit -m "feat(student): gate TogetherProvider to append expert conditioning when TOGETHER_RECAP=1"
```

---

## Task 5: Run full test suite and final commit

**Files:** None (verification only)

- [ ] **Step 1: Run the full test suite**

```bash
cd /Users/mahanth/ludus && .venv/bin/pytest -q --ignore=infra --ignore=gameworld 2>&1 | tail -10
```

Expected: `332 + <new tests> passed` with zero failures. The new tests are:
- `tests/test_recap.py`: ~14 tests
- `tests/test_together_provider.py`: 2 new tests (total 7)

- [ ] **Step 2: Create the final commit**

```bash
git add -p  # verify nothing unexpected
git commit -m "$(cat <<'EOF'
feat(student): RECAP-lite — advantage-conditioned SFT labeling + gated serving condition

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

### Spec coverage check

| Requirement | Task |
|---|---|
| `compute_advantages(runs_dir, *, profiles_dir)` | Task 3 |
| Group by game, baseline = mean final score | Task 3 |
| `advantage = final_score - baseline` | Task 3 |
| `label = "expert" if advantage >= 0 else "novice"` | Task 3 |
| Single-episode game: advantage=0, label="expert" | Task 3 + test in Task 1 |
| `build_recap_dataset` keeps ALL episodes (no score filter) | Task 3 |
| `min_episodes_per_game=2` excludes small games | Task 3 + test in Task 1 |
| System prompt conditioned with label suffix | Task 3 + test in Task 1 |
| Meta gains `{advantage, label, baseline}` | Task 3 + test in Task 1 |
| Summary `{episodes, expert_episodes, novice_episodes, rows}` | Task 3 + test in Task 1 |
| `RECAP_CONDITION` constant | Task 3 |
| `condition_system(label) -> str` helper | Task 3 + test in Task 1 |
| `TogetherProvider` appends expert conditioning when `TOGETHER_RECAP=1` | Task 4 |
| Gate is OFF by default (`TOGETHER_RECAP` unset) | Task 4 + test in Task 2 |
| Rows remain export_together-compatible (image/system/user/assistant keys) | Task 1 test |
| Tests in `tests/test_recap.py` | Tasks 1, 3 |
| Tests in `tests/test_together_provider.py` | Tasks 2, 4 |
| Full suite green (332 + new) | Task 5 |

### Placeholder scan

No TBD/TODO/implement-later present. All code steps are complete.

### Type consistency

- `compute_advantages` returns `dict[str, dict]` — used consistently in `build_recap_dataset`.
- `condition_system(label: str) -> str` — called as `condition_system("expert")` and `condition_system("novice")` in tests and implementation.
- `RECAP_CONDITION` is a `str` — `condition_system` calls `.format(label=label)` on it.
- `episode_to_sft_rows` is imported from `ludus.student.dataset` — signature unchanged.
- `_get_primary_metric` is imported from `ludus.student.dataset` — used identically.
