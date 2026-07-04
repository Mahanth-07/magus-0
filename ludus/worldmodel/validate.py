"""Field-weighted prediction accuracy on held-out transitions (the promotion
gate from the spec): exact match for discrete fields, tolerance for floats,
important paths (primary metric, player state) weigh 3x. Counterexamples
(field, state, action, predicted, actual) feed the inducer's repair loop.

Comparison is over the flattened paths of the ACTUAL after-state — the model
must reproduce reality; extra predicted fields are ignored, missing ones are
wrong. Tick/wall-clock paths (steps, timestamps) are excluded from grading
— they are not game rules (prober parity)."""

from __future__ import annotations

from dataclasses import dataclass, field

from ludus.onboarding.diff import flatten_state, is_tick_path
from ludus.worldmodel.sandbox import run_predictions
from ludus.worldmodel.transitions import Transition

IMPORTANT_WEIGHT = 3.0
MAX_COUNTEREXAMPLES = 8


@dataclass
class ValidationReport:
    overall: float = 0.0
    per_field: dict = field(default_factory=dict)
    n_cases: int = 0
    passed: bool = False
    counterexamples: list = field(default_factory=list)


def _is_important(path: str, important_paths: list[str]) -> bool:
    return any(path == p or path.startswith(p + ".") or path.startswith(p + "[")
               for p in important_paths)


def _matches(predicted, actual, float_tol: float) -> bool:
    if isinstance(predicted, (int, float)) and isinstance(actual, (int, float)) \
            and not isinstance(predicted, bool) and not isinstance(actual, bool):
        return abs(predicted - actual) <= float_tol
    return predicted == actual


def validate_model(
    source: str,
    transitions: list[Transition],
    *,
    important_paths: list[str],
    threshold: float,
    float_tol: float = 1e-6,
    timeout_s: float = 30.0,
) -> ValidationReport:
    report = ValidationReport(n_cases=len(transitions))
    if not transitions:
        return report

    cases = [{"state": t.before, "action": t.action} for t in transitions]
    preds = run_predictions(source, cases, timeout_s=timeout_s)

    field_hits: dict[str, int] = {}
    field_total: dict[str, int] = {}
    weighted_hits = weighted_total = 0.0

    for t, pred in zip(transitions, preds):
        actual_flat = flatten_state(t.after)
        pred_flat = flatten_state(pred) if isinstance(pred, dict) else {}
        for path, actual in actual_flat.items():
            if is_tick_path(path):
                continue  # wall-clock/tick fields aren't game rules (prober parity)
            ok = path in pred_flat and _matches(pred_flat[path], actual, float_tol)
            weight = IMPORTANT_WEIGHT if _is_important(path, important_paths) else 1.0
            field_total[path] = field_total.get(path, 0) + 1
            weighted_total += weight
            if ok:
                field_hits[path] = field_hits.get(path, 0) + 1
                weighted_hits += weight
            elif len(report.counterexamples) < MAX_COUNTEREXAMPLES:
                report.counterexamples.append({
                    "field": path, "action": t.action,
                    "state": t.before,
                    "predicted": pred_flat.get(path), "actual": actual,
                })

    report.per_field = {p: field_hits.get(p, 0) / n for p, n in field_total.items()}
    report.overall = weighted_hits / weighted_total if weighted_total else 0.0
    report.passed = report.overall >= threshold
    return report
