#!/usr/bin/env python3
# scripts/backfill_planning_grade.py
"""Backfill the planning-grade metric into existing worldmodels/*/report.json.

Validation-grade (the dual gate) says a model predicts observed transitions;
planning-grade says a forward planner rollout inside the model can GENERATE
scoring dynamics (CMA-ES round 1: doodle-jump passes the former at 0.0 on the
latter). For every worldmodels/<game>/report.json this recomputes the three
fields (mean_rollout_score, scoring_rollouts_frac, planning_grade) from the
saved model.py + data/<game>/transitions.jsonl (same holdout split as the
inducer: holdout_frac=0.25, seed=13) and patches them in place. Anything that
cannot run (missing model/profile/transitions, source-gate violation, sandbox
error) records zeros/False — a recorded zero beats a silent gap.

Usage: python scripts/backfill_planning_grade.py [--rollouts 6] [--steps 15]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ludus.onboarding.profile import GameProfile
from ludus.planning.tuning import planning_grade
from ludus.worldmodel.sandbox import check_source
from ludus.worldmodel.transitions import TransitionStore

ZEROS = {"mean_rollout_score": 0.0, "scoring_rollouts_frac": 0.0,
         "planning_grade": False}


def backfill(
    worldmodels_dir: Path | str = "worldmodels",
    data_dir: Path | str = "data",
    profiles_dir: Path | str = "profiles",
    *,
    rollouts: int = 6,
    steps: int = 15,
    depth: int = 2,
    beam: int = 8,
) -> list[dict]:
    """Patch every report.json; return one row of table data per model."""
    rows: list[dict] = []
    for report_path in sorted(Path(worldmodels_dir).glob("*/report.json")):
        game = report_path.parent.name
        report = json.loads(report_path.read_text())
        pg = dict(ZEROS)
        try:
            source = (report_path.parent / "model.py").read_text()
            if not check_source(source):
                profile = GameProfile.load(
                    Path(profiles_dir) / f"{game}.json")
                store = TransitionStore(
                    Path(data_dir) / game / "transitions.jsonl")
                try:  # mirror the inducer's holdout sampling
                    _, holdout = store.split(holdout_frac=0.25, seed=13)
                except Exception:
                    holdout = store.load()
                pg = planning_grade(
                    source, [t.before for t in holdout[:rollouts]],
                    sorted(profile.controls),
                    primary_metric=profile.primary_metric,
                    higher_is_better=profile.higher_is_better,
                    rollouts=rollouts, steps=steps, depth=depth, beam=beam)
        except Exception:
            pg = dict(ZEROS)
        report.update(pg)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        rows.append({"game": game,
                     "status": report.get("status", "?"),
                     "overall": report.get("overall", 0.0),
                     "primary": report.get("primary_accuracy", 0.0),
                     **pg})
    return rows


def format_table(rows: list[dict]) -> str:
    lines = [f"{'game':<32} {'status':<8} {'overall':>7} {'primary':>7} "
             f"{'plan?':>5} {'mean':>8} {'frac':>5}"]
    for r in rows:
        lines.append(
            f"{r['game']:<32} {r['status']:<8} {r['overall']:>7.3f} "
            f"{r['primary']:>7.2f} {'yes' if r['planning_grade'] else 'no':>5} "
            f"{r['mean_rollout_score']:>8.2f} {r['scoring_rollouts_frac']:>5.2f}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worldmodels-dir", default="worldmodels")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--profiles-dir", default="profiles")
    ap.add_argument("--rollouts", type=int, default=6)
    ap.add_argument("--steps", type=int, default=15)
    args = ap.parse_args()
    rows = backfill(args.worldmodels_dir, args.data_dir, args.profiles_dir,
                    rollouts=args.rollouts, steps=args.steps)
    print(format_table(rows))
    graded = sum(1 for r in rows if r["planning_grade"])
    print(f"\n{graded}/{len(rows)} models are planning-grade")


if __name__ == "__main__":
    main()
