#!/usr/bin/env python3
"""Batch-collect trajectories from INDUCED games for student SFT training.

    python scripts/collect_trajectories.py [--games g1 g2 ...] [--episodes 3]
        [--steps 40] [--provider planner] [--runs-dir runs]
        [--profiles profiles] [--worldmodels worldmodels]

Default behavior: discovers INDUCED games by scanning worldmodels/*/report.json,
then runs --episodes episodes per game using --provider (default: planner).
Episode ids are unique: traj-{game}-{provider}-{NNN} so they never collide
with duel-* or each other across re-runs.

Prints per-episode final score. All persistence via LocalStore (same layout
as duel episodes — runs/<episode_id>/ with steps.jsonl + PNGs + episode.json).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ludus.cli import _load_dotenv
from ludus.onboarding.profile import GameProfile
from ludus.persistence.local import LocalStore
from ludus.providers import build_provider
from ludus.rulebook import Rulebook
from ludus.loop import run_episode
from ludus.adapters.generic import GenericAdapter


def _induced_games(worldmodels_dir: Path) -> list[str]:
    """Return game ids whose worldmodel report.json has status == 'INDUCED'."""
    games = []
    if not worldmodels_dir.exists():
        return games
    for report_path in sorted(worldmodels_dir.glob("*/report.json")):
        try:
            report = json.loads(report_path.read_text())
            if report.get("status") == "INDUCED":
                games.append(report_path.parent.name)
        except Exception:
            pass
    return games


def _collect_episode(
    game: str,
    provider_name: str,
    episode_index: int,
    *,
    steps: int,
    runs_dir: Path,
    profiles_dir: Path,
    worldmodels_dir: Path,
    headless: bool = True,
) -> dict:
    """Run one episode and return {episode_id, game, provider, final_score}."""
    episode_id = f"traj-{game}-{provider_name}-{episode_index:03d}"

    profile_path = profiles_dir / f"{game}.json"
    if not profile_path.exists():
        raise FileNotFoundError(f"profile not found: {profile_path}")
    profile = GameProfile.load(profile_path)
    adapter = GenericAdapter(profile.to_game_config())

    if provider_name == "planner":
        from ludus.planning.provider import PlannerProvider
        provider = PlannerProvider(profile, worldmodels_dir=str(worldmodels_dir))
    else:
        provider = build_provider(provider_name)

    from ludus.cli import _client_factory
    gw = _client_factory(game, timing_ms=profile.timing_ms, headless=headless)()
    store = LocalStore(root=runs_dir)
    try:
        result = run_episode(
            adapter=adapter,
            provider=provider,
            gameworld=gw,
            store=store,
            rulebook=Rulebook(),
            mode="baseline",
            max_steps=steps,
            episode_id=episode_id,
        )
    finally:
        if hasattr(gw, "close"):
            gw.close()

    final_score = result.final_metrics.get(profile.primary_metric, 0.0)
    return {
        "episode_id": episode_id,
        "game": game,
        "provider": provider_name,
        "final_score": final_score,
        "primary_metric": profile.primary_metric,
    }


def main() -> None:
    _load_dotenv()

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", nargs="*", default=None,
                    help="game ids to collect (default: all INDUCED from worldmodels/)")
    ap.add_argument("--episodes", type=int, default=3,
                    help="episodes per game (default: 3)")
    ap.add_argument("--steps", type=int, default=40,
                    help="steps per episode (default: 40)")
    ap.add_argument("--provider", default="planner",
                    help="provider name (default: planner)")
    ap.add_argument("--runs-dir", type=Path, default=REPO_ROOT / "runs",
                    help="where to persist episodes (default: runs/)")
    ap.add_argument("--profiles", type=Path, default=REPO_ROOT / "profiles",
                    help="profiles directory (default: profiles/)")
    ap.add_argument("--worldmodels", type=Path, default=REPO_ROOT / "worldmodels",
                    help="worldmodels directory (default: worldmodels/)")
    ap.add_argument("--headed", action="store_true",
                    help="show browser window")
    args = ap.parse_args()

    games = args.games or _induced_games(args.worldmodels)
    if not games:
        print("No INDUCED games found. Run: python -m ludus.cli induce <game> first.")
        sys.exit(1)

    print(f"Collecting {args.episodes} episode(s) × {len(games)} game(s) "
          f"× provider={args.provider} ({args.steps} steps each)")
    print(f"Games: {', '.join(games)}")
    print()

    total_rows = 0
    for game in games:
        for i in range(args.episodes):
            try:
                info = _collect_episode(
                    game, args.provider, i,
                    steps=args.steps,
                    runs_dir=args.runs_dir,
                    profiles_dir=args.profiles,
                    worldmodels_dir=args.worldmodels,
                    headless=not args.headed,
                )
                total_rows += args.steps
                print(
                    f"  {info['episode_id']}: "
                    f"{info['primary_metric']}={info['final_score']:g}"
                )
            except Exception as exc:
                print(f"  traj-{game}-{args.provider}-{i:03d}: ERROR — {exc}")

    print(f"\nDone. ~{total_rows} step records written to {args.runs_dir}/")


if __name__ == "__main__":
    main()
