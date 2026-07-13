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
