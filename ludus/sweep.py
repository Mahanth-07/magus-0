# ludus/sweep.py
"""Benchmark sweep: the full autonomous pipeline per game, honest verdicts.

Stages per game: onboard -> explore -> induce -> duel. Each stage can fail;
the failure BECOMES the verdict (spec: honest coverage claims). Results are
one JSON record per game in results_path (JSONL, append), keyed by game id —
re-running the sweep skips games that already have a record (resume)."""

from __future__ import annotations

import json
import traceback
from pathlib import Path

VERDICTS = ("ONBOARD_FAILED", "EXPLORE_FAILED", "INDUCTION_FAILED",
            "DUEL_REFUSED", "DUEL_ERROR", "DUEL_LOST", "DUEL_TIE", "DUEL_WON")


def load_results(results_path: Path | str) -> dict[str, dict]:
    """{game_id: record} from the JSONL (last record per game wins)."""
    path = Path(results_path)
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    with path.open() as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                out[rec["game"]] = rec
    return out


def append_result(results_path: Path | str, record: dict) -> None:
    path = Path(results_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def sweep_game(game: str, *, onboard, explore, induce, duel) -> dict:
    """Run one game through injected stage callables; convert failures to
    verdicts. Each callable raises on failure; induce returns the report
    dict (must contain 'status'); duel returns the duel result dict."""
    record: dict = {"game": game}
    try:
        record["controls"] = onboard(game)
    except Exception as exc:
        record.update(verdict="ONBOARD_FAILED", error=_err(exc))
        return record
    try:
        record["transitions"] = explore(game)
    except Exception as exc:
        record.update(verdict="EXPLORE_FAILED", error=_err(exc))
        return record
    try:
        report = induce(game)
        record["induction"] = {k: report.get(k) for k in
                               ("status", "overall", "primary_accuracy",
                                "iterations")}
    except Exception as exc:
        record.update(verdict="INDUCTION_FAILED", error=_err(exc))
        return record
    if report.get("status") != "INDUCED":
        record["verdict"] = "INDUCTION_FAILED"
        return record
    try:
        duel_result = duel(game)
    except SystemExit as exc:
        record.update(verdict="DUEL_REFUSED", error=str(exc))
        return record
    except Exception as exc:
        record.update(verdict="DUEL_ERROR", error=_err(exc))
        return record
    record["duel"] = {"planner": duel_result["planner"]["score"],
                      "baseline": duel_result["baseline"]["score"],
                      "winner": duel_result["winner"]}
    record["verdict"] = {"planner": "DUEL_WON", "baseline": "DUEL_LOST",
                         "tie": "DUEL_TIE"}[duel_result["winner"]]
    return record


def _err(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"[:300]


def render_table(results: dict[str, dict]) -> str:
    """Markdown coverage table, sweep's headline artifact."""
    lines = ["| game | verdict | induction (overall / primary) | duel (planner vs baseline) |",
             "|---|---|---|---|"]
    for game in sorted(results):
        r = results[game]
        ind = r.get("induction") or {}
        ind_s = (f"{ind.get('overall', 0):.2f} / {ind.get('primary_accuracy', 0):.2f}"
                 if ind else "—")
        duel = r.get("duel") or {}
        duel_s = (f"{duel.get('planner', 0):g} vs {duel.get('baseline', 0):g}"
                  if duel else "—")
        lines.append(f"| {game} | {r.get('verdict', '?')} | {ind_s} | {duel_s} |")
    counts: dict[str, int] = {}
    for r in results.values():
        counts[r.get("verdict", "?")] = counts.get(r.get("verdict", "?"), 0) + 1
    lines.append("")
    lines.append("Totals: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return "\n".join(lines)
