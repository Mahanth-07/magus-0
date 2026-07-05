#!/usr/bin/env python3
"""Run the autonomous pipeline across GameWorld benchmark games.

    /opt/anaconda3/bin/python scripts/sweep_benchmark.py [--games g1 g2 ...]
        [--results docs/results/sweep.jsonl] [--episodes 6] [--steps 25]

Resumable: games with an existing record in --results are skipped. Induction
uses LUDUS_SYNTH_MODEL (default set here to claude-opus-4-8 — the sweep's
proven synthesis model). One line of progress per game stage on stdout."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("LUDUS_SYNTH_MODEL", "claude-opus-4-8")

from ludus.cli import _load_dotenv, cmd_duel, cmd_explore, cmd_induce, cmd_onboard
from ludus.onboarding.profile import GameProfile
from ludus.sweep import append_result, load_results, render_table, sweep_game

GAMES_DIR = REPO_ROOT / "gameworld" / "games" / "benchmark"
# co-op needs a human partner; excluded from the autonomous sweep
EXCLUDED = {"12_fireboy-and-watergirl"}


def _all_games() -> list[str]:
    return sorted(p.name for p in GAMES_DIR.iterdir()
                  if p.is_dir() and p.name not in EXCLUDED)


def main() -> None:
    _load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", nargs="*", default=None)
    ap.add_argument("--results", type=Path,
                    default=REPO_ROOT / "docs" / "results" / "sweep.jsonl")
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--duel-steps", type=int, default=15)
    ap.add_argument("--iterations", type=int, default=3)
    args = ap.parse_args()

    games = args.games or _all_games()
    done = load_results(args.results)

    def onboard(game):
        cmd_onboard(game, out_dir="profiles")
        return GameProfile.load(Path("profiles") / f"{game}.json").controls

    def explore(game):
        # fresh transitions per sweep run for a clean episode set
        t_path = Path("data") / game / "transitions.jsonl"
        if t_path.exists():
            t_path.unlink()
        cmd_explore(game, episodes=args.episodes, steps=args.steps,
                    duration_ms=30)
        return sum(1 for _ in t_path.open())

    def induce(game):
        cmd_induce(game, max_iterations=args.iterations)
        return json.loads((Path("worldmodels") / game / "report.json").read_text())

    def duel(game):
        return cmd_duel(game, steps=args.duel_steps)

    for i, game in enumerate(games, 1):
        if game in done:
            print(f"[{i}/{len(games)}] {game}: SKIP (already {done[game].get('verdict')})")
            continue
        print(f"[{i}/{len(games)}] {game}: running...")
        record = sweep_game(game, onboard=onboard, explore=explore,
                            induce=induce, duel=duel)
        append_result(args.results, record)
        done[game] = record
        print(f"[{i}/{len(games)}] {game}: {record.get('verdict')}")

    print()
    print(render_table(done))


if __name__ == "__main__":
    main()
