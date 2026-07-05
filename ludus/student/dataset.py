"""Trajectory dataset layer: runs/ episode dirs -> reward-filtered VLM-SFT JSONL.

Key design choices:
- Reuses _SYSTEM and build_user_text from ludus.providers.insforge byte-for-byte
  (M0 lesson: train/inference prompt must match exactly).
- state_text comes from StepRecord.state_text (persisted by run_episode since
  Stage 4 chunk 1 schema change); old records without it get "" -> prompt omits
  the Game state section, which is valid.
- Decision dict in assistant field uses Decision.model_dump() to get the
  canonical 9-field set the runtime parses (scene_summary, controlled_entity,
  current_subgoal, action, actions, action_args, expected_result, reason,
  confidence). No extra or missing keys.
- Reward filtering: keep episodes where final primary-metric score >= threshold.
  Threshold can be a scalar (applied to all games) or a per-game dict.
- Profile lookup: load GameProfile from profiles_dir/<game>.json to get
  objective + legal_actions. Episodes whose profile is missing are skipped
  (counted in episodes_seen but not episodes_kept/rows).
"""

from __future__ import annotations

import json
from pathlib import Path

from ludus.providers.insforge import _SYSTEM, build_user_text
from ludus.schemas import EpisodeResult, StepRecord


def episode_to_sft_rows(
    episode_dir: Path,
    *,
    objective: str,
    legal_actions: list[str],
) -> list[dict]:
    """Convert one on-disk episode directory into a list of VLM-SFT rows.

    Each row:
      image     : absolute path string to the step PNG
      system    : _SYSTEM (the InsForge preamble, identical to inference)
      user      : build_user_text(objective, legal_actions, state_text=<from step>)
      assistant : json.dumps of the canonical 9-field Decision dict
      meta      : {game, episode_id, step_index, primary_delta, final_score}

    Reads steps.jsonl (one StepRecord JSON per line) and episode.json
    (EpisodeResult). Steps without a PNG on disk are still included; the image
    path points to where the PNG should be (absolute, so callers can check).
    """
    episode_dir = Path(episode_dir)
    steps_path = episode_dir / "steps.jsonl"
    episode_path = episode_dir / "episode.json"

    # Load episode summary for final_score
    result = EpisodeResult.model_validate_json(episode_path.read_text())
    final_score = result.final_metrics.get(result.final_metrics and list(result.final_metrics)[0], 0.0)
    # Use the primary metric value: take the first numeric value from final_metrics
    # (EpisodeResult doesn't expose primary_metric directly, but StepRecords do)
    # We'll patch final_score per-step from the step's primary_metric below.

    rows: list[dict] = []
    steps_text = steps_path.read_text()
    for line in steps_text.splitlines():
        line = line.strip()
        if not line:
            continue
        step = StepRecord.model_validate_json(line)

        # final_score: primary metric value from EpisodeResult
        primary_score = result.final_metrics.get(step.primary_metric, 0.0)

        # Build the user text exactly as the runtime would at inference
        user_text = build_user_text(
            objective=objective,
            legal_actions=legal_actions,
            state_text=step.state_text,
        )

        # Decision dict: canonical 9-field set the runtime parses
        decision_dict = step.decision.model_dump()

        # Absolute image path
        if step.screenshot_ref:
            # screenshot_ref is stored as "episode_id/step_NNNN.png" (relative to runs/)
            # episode_dir IS runs/episode_id, so resolve relative to episode_dir's parent
            img_path = episode_dir.parent / step.screenshot_ref
        else:
            img_path = episode_dir / f"step_{step.step_index:04d}.png"
        img_path = img_path.resolve()

        rows.append({
            "image": str(img_path),
            "system": _SYSTEM,
            "user": user_text,
            "assistant": json.dumps(decision_dict),
            "meta": {
                "game": step.game,
                "episode_id": step.episode_id,
                "step_index": step.step_index,
                "primary_delta": step.primary_delta,
                "final_score": primary_score,
            },
        })

    return rows


def build_dataset(
    runs_dir: Path | str,
    out_path: Path | str,
    *,
    min_final_score: dict[str, float] | float = 0.0,
    profiles_dir: Path | str | None = None,
) -> dict:
    """Walk runs_dir, filter episodes by score, write reward-filtered VLM-SFT JSONL.

    Returns {"episodes_seen": N, "episodes_kept": K, "rows": R}.

    Episode directories are any subdirectory of runs_dir that contains both
    steps.jsonl and episode.json. The game name comes from episode.json "game"
    field.

    Filtering:
      - If min_final_score is a float, all games use that threshold.
      - If min_final_score is a dict, the per-game value is used (missing game
        key defaults to 0.0).

    Profile lookup:
      - Loads GameProfile from profiles_dir/<game>.json to get objective and
        legal_actions. Episodes whose profile file is missing are skipped but
        counted in episodes_seen.
    """
    runs_dir = Path(runs_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if profiles_dir is not None:
        profiles_dir = Path(profiles_dir)

    episodes_seen = 0
    episodes_kept = 0
    rows_written = 0

    with out_path.open("w") as fout:
        for ep_dir in sorted(runs_dir.iterdir()):
            if not ep_dir.is_dir():
                continue
            steps_path = ep_dir / "steps.jsonl"
            episode_path = ep_dir / "episode.json"
            if not steps_path.exists() or not episode_path.exists():
                continue

            try:
                result = EpisodeResult.model_validate_json(episode_path.read_text())
            except Exception:
                continue

            episodes_seen += 1
            game = result.game

            # Score threshold
            if isinstance(min_final_score, dict):
                threshold = min_final_score.get(game, 0.0)
            else:
                threshold = float(min_final_score)

            final_score = result.final_metrics.get(
                # pick the first metric that looks like the primary one
                next(iter(result.final_metrics), ""), 0.0
            )
            # Try to get the primary metric from the first step record
            primary_metric = _get_primary_metric(steps_path)
            if primary_metric and primary_metric in result.final_metrics:
                final_score = result.final_metrics[primary_metric]

            if final_score < threshold:
                continue

            # Profile lookup
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

            sft_rows = episode_to_sft_rows(ep_dir, objective=objective, legal_actions=legal_actions)
            for row in sft_rows:
                fout.write(json.dumps(row) + "\n")
                rows_written += 1

            episodes_kept += 1

    return {
        "episodes_seen": episodes_seen,
        "episodes_kept": episodes_kept,
        "rows": rows_written,
    }


def _get_primary_metric(steps_path: Path) -> str | None:
    """Read the primary_metric field from the first step in a steps.jsonl file."""
    try:
        with steps_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    return data.get("primary_metric")
    except Exception:
        pass
    return None
