"""Beam search over an induced world model — the general form of the Tetris
candidate-lookahead (rank_placements) that made Magus-0's results.

Level-synchronous expansion: at each depth, every (beam-state, action) pair is
batched into ONE sandbox subprocess call (run_predictions), so a depth-3 /
beam-16 / 4-action search costs 3 subprocess spawns, not 192. Branches are
scored by cumulative predicted primary-metric delta; terminal-status branches
stop expanding (their macro remains a candidate — dying with a high score can
be optimal; dying with none ranks last naturally).

Returns candidates best-first: reward desc, then SHORTER macro (a reward now
beats the same reward later), then lexicographic for determinism.
"""

from __future__ import annotations

from ludus.worldmodel.sandbox import run_predictions


def _metric(state: dict, primary_metric: str) -> float:
    v = (state.get("metrics") or {}).get(primary_metric, 0.0)
    return float(v) if isinstance(v, (int, float)) else 0.0


def rank_macros(
    model_source: str,
    state: dict,
    actions: list[str],
    *,
    primary_metric: str,
    higher_is_better: bool,
    depth: int = 3,
    beam: int = 16,
    top_k: int = 6,
    timeout_s: float = 30.0,
) -> list[dict]:
    """Rank action macros (length 1..depth) by predicted cumulative reward.

    Returns [{"actions": [...], "predicted_reward": float}, ...] best-first;
    [] if the model can't predict anything (broken/hanging source).
    """
    sign = 1.0 if higher_is_better else -1.0
    actions = sorted(actions)
    # beams: (state, macro, cum_reward)
    beams: list[tuple[dict, list[str], float]] = [(state, [], 0.0)]
    scored: dict[tuple[str, ...], float] = {}

    for _ in range(depth):
        if not beams:
            break
        cases = [{"state": s, "action": a}
                 for (s, _, _) in beams for a in actions]
        preds = run_predictions(model_source, cases, timeout_s=timeout_s)

        next_beams: list[tuple[dict, list[str], float]] = []
        i = 0
        for (s, macro, cum) in beams:
            for a in actions:
                pred = preds[i]
                i += 1
                if not isinstance(pred, dict):
                    continue
                reward = cum + sign * (_metric(pred, primary_metric)
                                       - _metric(s, primary_metric))
                new_macro = macro + [a]
                scored[tuple(new_macro)] = reward
                if str(pred.get("status", "playing")) == "playing":
                    next_beams.append((pred, new_macro, reward))
        next_beams.sort(key=lambda b: (-b[2], len(b[1]), b[1]))
        beams = next_beams[:beam]

    ranked = sorted(scored.items(), key=lambda kv: (-kv[1], len(kv[0]), kv[0]))
    return [{"actions": list(macro), "predicted_reward": reward}
            for macro, reward in ranked[:top_k]]
