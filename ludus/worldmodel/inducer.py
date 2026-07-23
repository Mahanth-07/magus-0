# ludus/worldmodel/inducer.py
"""The induction loop: LLM writes predict(), validator grades it, failures
come back as counterexamples, repeat within budget. Gate on BOTH train and
holdout (a model must explain all observed data; rare mechanics may only
appear in train); counterexamples for repair come from TRAIN (no leak of the
gate set into the prompt). Artifacts (model.py + report.json) are always
saved — a FAILED verdict with evidence beats a silent nothing (spec: no stage
can lie)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from ludus.onboarding.diff import flatten_state, is_tick_path
from ludus.planning.tuning import planning_grade
from ludus.onboarding.profile import GameProfile
from ludus.worldmodel.canonical import canonicalize
from ludus.worldmodel.sandbox import ALLOWED_IMPORTS, check_source
from ludus.worldmodel.transitions import Transition, TransitionStore
from ludus.worldmodel.validate import ValidationReport, validate_model

MAX_PROMPT_TRANSITIONS = 30
MAX_FULL_STATES = 2
GRID_VIEW_TRANSITIONS = 10


def render_grid(state: dict) -> str:
    """ASCII grid view of game_state.entities when they carry small integer
    x/y coords (board games: 2048 tiles, minesweeper cells). Cell label =
    props.value if present, else first letter of type. "" when the state has
    no such entities (graceful no-op for non-grid games)."""
    ents = (state.get("game_state") or {}).get("entities")
    if not isinstance(ents, list) or not ents:
        return ""
    if not all(isinstance(e, dict) and isinstance(e.get("x"), int)
               and isinstance(e.get("y"), int) for e in ents):
        return ""
    xs = [e["x"] for e in ents]
    ys = [e["y"] for e in ents]
    if min(xs) < 0 or min(ys) < 0 or max(xs) > 15 or max(ys) > 15:
        return ""
    grid = [["." for _ in range(max(xs) + 1)] for _ in range(max(ys) + 1)]
    for e in ents:
        v = (e.get("props") or {}).get("value")
        grid[e["y"]][e["x"]] = str(v) if v is not None else str(e.get("type", "?"))[:1]
    return "\n".join(" ".join(f"{c:>4}" for c in row) for row in grid)


@dataclass
class InductionResult:
    status: str                 # "INDUCED" | "FAILED"
    source: str
    report: ValidationReport
    iterations: int
    model_path: Path | None = None
    planning: dict | None = None    # planning-grade metrics (see tuning.planning_grade)


def extract_code(text: str) -> str:
    """Pull the python source out of a fenced block; bare source passes
    through. With multiple blocks, prefer the one defining predict() —
    LLMs sometimes emit explanatory snippets before the real model."""
    blocks = re.findall(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if not blocks:
        return text
    for block in blocks:
        if "def predict" in block:
            return block.strip() + "\n"
    return blocks[0].strip() + "\n"


def _render_transition(t: Transition) -> str:
    before = flatten_state(canonicalize(t.before))
    after = flatten_state(canonicalize(t.after))
    changed = {p: (before.get(p), after.get(p))
               for p in before.keys() | after.keys()
               if before.get(p) != after.get(p)}
    return f"action={t.action!r} changed={json.dumps(changed, default=str)}"


def build_synthesis_prompt(
    game_id: str,
    profile: GameProfile,
    train: list[Transition],
    counterexamples: list[dict],
    prev_source: str,
) -> tuple[str, str]:
    """(system, user) for the synthesis LLM."""
    system = (
        "You write EXACT world models of simple games as pure Python. "
        "Reply with a single ```python``` block containing a module that "
        "defines predict(state, action) -> next_state. Rules: return a NEW "
        f"dict (deep-copy the input); imports limited to {sorted(ALLOWED_IMPORTS)}; "
        "no I/O, no randomness — if a field is genuinely unpredictable "
        "(random spawns), copy it through unchanged rather than guessing. "
        "Infer the game's mechanics (movement bounds, scoring, collisions, "
        "counters) from the observed transitions and reproduce them exactly. "
        "Entity lists are compared as SETS of identical dicts (order-free); "
        "unpredictable entities (random spawns) should be omitted rather than "
        "guessed — a guessed-wrong entity scores no better than an omitted one."
    )
    def _changes_metrics(t: Transition) -> bool:
        b = flatten_state(t.before.get("metrics", {}))
        a = flatten_state(t.after.get("metrics", {}))
        return any(b.get(p) != a.get(p)
                   for p in b.keys() | a.keys() if not is_tick_path(p))

    # Rare mechanics live in metric-changing transitions (scoring, line
    # clears, damage); random exploration makes them scarce, so they get
    # prompt priority — an unseen mechanic WILL be hallucinated (live M2
    # finding). Fill the rest with ordinary transitions in original order.
    rare = [t for t in train if _changes_metrics(t)]
    common = [t for t in train if not _changes_metrics(t)]
    sample = (rare + common)[:MAX_PROMPT_TRANSITIONS]
    lines = [
        f"Game: {game_id}",
        f"Actions: {sorted(profile.controls)}",
        "",
        "Example full states (before -> after) for shape reference:",
    ]
    for t in sample[:MAX_FULL_STATES]:
        lines.append(f"  action={t.action!r}")
        lines.append(f"  before={json.dumps(canonicalize(t.before), default=str)}")
        lines.append(f"  after ={json.dumps(canonicalize(t.after), default=str)}")
    lines.append("")
    lines.append("Observed transitions (flattened changed fields, (before, after)):")
    for t in sample:
        lines.append("  " + _render_transition(t))
    grid_pairs = [(t, render_grid(canonicalize(t.before)), render_grid(canonicalize(t.after)))
                  for t in sample[:GRID_VIEW_TRANSITIONS]]
    grid_pairs = [(t, gb, ga) for (t, gb, ga) in grid_pairs if gb and ga]
    if grid_pairs:
        lines.append("")
        lines.append("Board views (rows are y=0..N top-to-bottom, columns x=0..N "
                     "left-to-right; '.' = empty):")
        for t, gb, ga in grid_pairs:
            lines.append(f"--- action={t.action!r} ---")
            lines.append("before:")
            lines.append(gb)
            lines.append("after:")
            lines.append(ga)
    if counterexamples:
        lines.append("")
        lines.append("YOUR PREVIOUS ATTEMPT was wrong on these observed "
                     "training cases (field, action, predicted vs actual). "
                     "Fix the model:")
        for ce in counterexamples:
            lines.append(
                f"  field={ce['field']} action={ce['action']} "
                f"predicted={ce['predicted']!r} actual={ce['actual']!r}"
            )
        lines.append("")
        lines.append("Previous source:")
        lines.append("```python")
        lines.append(prev_source.rstrip())
        lines.append("```")
    return system, "\n".join(lines)


def induce_world_model(
    game_id: str,
    store: TransitionStore,
    *,
    profile: GameProfile,
    synthesizer,                       # (system, user) -> str
    out_dir: Path | str = "worldmodels",
    max_iterations: int = 4,
    threshold: float = 0.90,
    holdout_frac: float = 0.25,
    seed: int = 13,
) -> InductionResult:
    train, holdout = store.split(holdout_frac=holdout_frac, seed=seed)
    if not train or not holdout:
        raise ValueError(
            f"not enough episodes to split for induction: "
            f"{len(train)} train / {len(holdout)} holdout transitions "
            f"(need >= 2 episodes; explore more first)"
        )
    important = [f"metrics.{profile.primary_metric}", "game_state.player"]

    source, report = "", ValidationReport()
    train_report = ValidationReport()
    counterexamples: list[dict] = []
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        response = synthesizer(*build_synthesis_prompt(
            game_id, profile, train, counterexamples, source))
        source = extract_code(response)
        violations = check_source(source)
        if violations:
            counterexamples = [{"field": "(source gate)", "action": "-",
                                "predicted": v, "actual": "allowlisted code"}
                               for v in violations]
            report = ValidationReport()
            train_report = ValidationReport()
            continue
        # repair signal from TRAIN; the gate below requires BOTH splits to pass
        train_report = validate_model(source, train, important_paths=important,
                                      threshold=threshold,
                                      primary_metric=profile.primary_metric)
        report = validate_model(source, holdout, important_paths=important,
                                threshold=threshold,
                                primary_metric=profile.primary_metric)
        if train_report.passed and report.passed:
            break
        counterexamples = train_report.counterexamples

    status = "INDUCED" if (train_report.passed and report.passed) else "FAILED"
    # Planning-grade: validation asks "does it predict observed transitions?";
    # this asks "can the planner GENERATE scoring dynamics inside it?" (CMA-ES
    # round 1: doodle passed the former, scored 0.0 on the latter). Run
    # whenever the source passes the sandbox gate — even a FAILED model's
    # forward behavior is informative. Exceptions guard to zeros.
    planning = {"mean_rollout_score": 0.0, "scoring_rollouts_frac": 0.0,
                "planning_grade": False}
    if source and not check_source(source):
        try:
            planning = planning_grade(
                source, [t.before for t in holdout],
                sorted(profile.controls),
                primary_metric=profile.primary_metric,
                higher_is_better=profile.higher_is_better)
        except Exception:
            pass
    model_dir = Path(out_dir) / game_id
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.py").write_text(source or "# no source produced\n")
    (model_dir / "report.json").write_text(json.dumps({
        "status": status, "overall": report.overall,
        "train_overall": train_report.overall,
        "primary_accuracy": report.primary_accuracy,
        "train_primary_accuracy": train_report.primary_accuracy,
        "overall_floor": 0.75,
        "per_field": report.per_field, "n_cases": report.n_cases,
        "iterations": iterations, "threshold": threshold,
        "train_size": len(train), "holdout_size": len(holdout),
        **planning,
    }, indent=2) + "\n")
    return InductionResult(status=status, source=source, report=report,
                           iterations=iterations,
                           model_path=model_dir / "model.py",
                           planning=planning)
