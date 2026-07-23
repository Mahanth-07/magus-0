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

from ludus.onboarding.diff import diff_states, flatten_state
from ludus.worldmodel.canonical import canonicalize
from ludus.worldmodel.sandbox import run_predictions

# Weighted-score features over predicted next-states (CMA-ES tunes these).
# The defaults reproduce the pre-weights planner EXACTLY: score == cumulative
# sign-adjusted primary-metric delta, everything else off.
DEFAULT_WEIGHTS = {"primary": 1.0, "change": 0.0, "terminal": 0.0,
                   "secondary": 0.0, "length": 0.0}


def _metric(state: dict, primary_metric: str) -> float:
    v = (state.get("metrics") or {}).get(primary_metric, 0.0)
    return float(v) if isinstance(v, (int, float)) else 0.0


def _change_fraction(root_flat: dict, pred: dict) -> float:
    """Fraction of flattened paths that differ between root and pred."""
    pred_flat = flatten_state(pred)
    total = len(root_flat.keys() | pred_flat.keys())
    if not total:
        return 0.0
    return len(diff_states(root_flat, pred_flat)) / total


def _secondary_sum(root: dict, pred: dict, primary_metric: str) -> float:
    """Raw sum of deltas of non-primary numeric metrics (pred vs root)."""
    root_m = root.get("metrics") or {}
    pred_m = pred.get("metrics") or {}
    total = 0.0
    for k, v in pred_m.items():
        if k == primary_metric:
            continue
        r = root_m.get(k, 0.0)
        if (isinstance(v, (int, float)) and not isinstance(v, bool)
                and isinstance(r, (int, float)) and not isinstance(r, bool)):
            total += float(v) - float(r)
    return total


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
    weights: dict | None = None,
) -> list[dict]:
    """Rank action macros (length 1..depth) by weighted predicted score.

    `weights` (None == DEFAULT_WEIGHTS == pre-weights behavior) scores each
    branch as
        primary   * cumulative sign-adjusted primary-metric delta
      + change    * fraction of flattened paths changed vs the root state
      + terminal  * (1 if predicted status != "playing" else 0)
      + secondary * raw sum of non-primary numeric metric deltas vs root
      + length    * len(macro)

    Returns [{"actions": [...], "predicted_reward": float,
              "changes_state": bool}, ...] best-first;
    [] if the model can't predict anything (broken/hanging source).
    predicted_reward stays the raw primary reward (not the weighted score).

    Tie-break order (after weighted score): changes_state first → shorter
    macro → lexicographic.  The changes_state secondary key breaks the
    jammed-board deadlock (all rewards predict 0; a macro that moves a tile
    beats a pure no-op → board change → spawn → future opportunities).
    """
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)
    sign = 1.0 if higher_is_better else -1.0
    actions = sorted(actions)
    root_canonical = canonicalize(state)
    root_flat = flatten_state(state)
    # beams: (state, macro, cum_reward)
    beams: list[tuple[dict, list[str], float]] = [(state, [], 0.0)]
    # scored: macro_tuple -> (weighted_score, cum_reward, changes_state)
    scored: dict[tuple[str, ...], tuple[float, float, bool]] = {}

    for _ in range(depth):
        if not beams:
            break
        cases = [{"state": s, "action": a}
                 for (s, _, _) in beams for a in actions]
        preds = run_predictions(model_source, cases, timeout_s=timeout_s)

        # next_beams: (state, macro, cum_reward, weighted_score)
        next_beams: list[tuple[dict, list[str], float, float]] = []
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
                changes = canonicalize(pred) != root_canonical
                terminal = str(pred.get("status", "playing")) != "playing"
                score = w["primary"] * reward
                if w["change"]:
                    score += w["change"] * _change_fraction(root_flat, pred)
                if w["terminal"] and terminal:
                    score += w["terminal"]
                if w["secondary"]:
                    score += w["secondary"] * _secondary_sum(
                        state, pred, primary_metric)
                if w["length"]:
                    score += w["length"] * len(new_macro)
                scored[tuple(new_macro)] = (score, reward, changes)
                if not terminal:
                    next_beams.append((pred, new_macro, reward, score))
        next_beams.sort(key=lambda b: (-b[3], len(b[1]), b[1]))
        beams = [(s, m, r) for (s, m, r, _) in next_beams[:beam]]

    ranked = sorted(scored.items(),
                    key=lambda kv: (-kv[1][0], not kv[1][2], len(kv[0]), kv[0]))
    return [{"actions": list(macro), "predicted_reward": reward,
             "changes_state": changes}
            for macro, (_, reward, changes) in ranked[:top_k]]
