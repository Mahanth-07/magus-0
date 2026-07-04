#!/usr/bin/env python3
"""Collect a distillation dataset from SIMULATED teacher self-play — no browser.

Uses ludus.teacher.tetris_sim: the teacher's own drop_piece engine plays the
game in pure Python; prompts are rendered by the live client's state-text
function (candidate-placement block included) and labels are candidate [0]
with _macro_live-corrected macros. `--epsilon` injects DAgger-style recovery
states (a random suboptimal placement is APPLIED while the LABEL stays the
teacher's best move).

Run (offline, seconds not hours):
    python scripts/collect_dataset_sim.py 2000 --epsilon 0.15 --seed 7

Output (gitignored): data/tetris/examples_sim.jsonl — same record shape as
scripts/collect_dataset.py minus images (text-only student). Export with:
    python scripts/export_nebius.py --mode text --in data/tetris/examples_sim.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ludus.config import load_game_config
from ludus.teacher.tetris_sim import simulate_examples

DEFAULT_OUT = REPO_ROOT / "data" / "tetris" / "examples_sim.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("n", nargs="?", type=int, default=2000, help="number of examples")
    ap.add_argument("--epsilon", type=float, default=0.15,
                    help="probability of applying a random suboptimal placement (recovery states)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "tetris.yaml")
    args = ap.parse_args()

    cfg = load_game_config(args.config)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    off_policy = 0
    holes: list[int] = []
    with args.out.open("w") as fout:
        for rec in simulate_examples(
            args.n, epsilon=args.epsilon, seed=args.seed,
            objective=cfg.objective, legal_actions=cfg.legal_actions,
            duration_ms=cfg.timing_ms,
        ):
            fout.write(json.dumps(rec) + "\n")
            n_written += 1
            off_policy += 0 if rec["applied"]["teacher"] else 1
            holes.append(rec["reward"]["holes_after"])

    print(f"examples written : {n_written} -> {args.out}")
    print(f"off-policy steps : {off_policy} ({100 * off_policy / max(1, n_written):.0f}%)")
    if holes:
        print(f"holes avg/max    : {sum(holes) / len(holes):.2f} / {max(holes)}")


if __name__ == "__main__":
    main()
