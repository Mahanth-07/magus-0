"""Render ranked macros into the candidate-block prompt format that M0 proved
(gateway 466, student 1022 on Tetris): a sorted best-first list whose entry
[0] the selector copies verbatim. Header text matches the Tetris block so the
distilled student's learned behavior ("copy [0]") transfers.

Not used by the duel (pure argmax needs no selector) — this is the Stage-4
exhaust surface: every planner turn can emit a (prompt, macro) SFT pair."""

from __future__ import annotations


def render_candidate_block(ranked: list[dict], *, primary_metric: str) -> str:
    if not ranked:
        return ""
    out = [
        "Candidate placements (each shows the predicted outcome if you pick "
        "it; choose the BEST and return its exact actions):"
    ]
    for i, c in enumerate(ranked):
        out.append(
            f"[{i}] predicted {primary_metric} {c['predicted_reward']:+.1f}, "
            f"actions={c['actions']}"
        )
    return "\n".join(out)
