# ludus/onboarding/metrics.py
"""MetricFinder — choose the game's metrics and primary metric.

GameWorld curates a `metrics.*` namespace in getState(), so metric discovery
is namespace filtering + primary selection. Selection order:
  1. Name priors (score/kills/points/... in priority order) — a field literally
     named `score` IS the score.
  2. Fallback: a metric some discovered control changes (probe evidence).
  3. Last resort: first non-vital numeric metric alphabetically.
Vitals (health/lives/ammo/time/steps) are relevant metrics but never primary
unless they are the only thing there.
"""

from __future__ import annotations

from dataclasses import dataclass, field

_NAME_PRIORS = ["score", "kills", "points", "coins", "diamonds", "gems",
                "progress", "level"]
_VITALS = ("health", "lives", "ammo", "time", "steps")

_NUMERIC_TYPES = {"int", "float"}


@dataclass
class MetricReport:
    metrics: list[str] = field(default_factory=list)
    primary_metric: str = ""
    higher_is_better: bool = True


def _is_vital(name: str) -> bool:
    return any(v in name.lower() for v in _VITALS)


def find_metrics(state_schema: dict[str, str],
                 control_effects: dict[str, list[str]]) -> MetricReport:
    metric_names = sorted(
        path.split(".", 1)[1]
        for path, tname in state_schema.items()
        if path.startswith("metrics.") and "." not in path.split(".", 1)[1]
        and tname in _NUMERIC_TYPES
    )
    report = MetricReport(metrics=metric_names)
    if not metric_names:
        return report

    # 1. name priors, in priority order
    for prior in _NAME_PRIORS:
        for name in metric_names:
            if prior in name.lower():
                report.primary_metric = name
                return report

    # 2. probe evidence: a metric some control changes
    affected = {p.split(".", 1)[1] for paths in control_effects.values()
                for p in paths if p.startswith("metrics.")}
    for name in metric_names:
        if name in affected and not _is_vital(name):
            report.primary_metric = name
            return report

    # 3. last resort: first non-vital, else first
    non_vital = [n for n in metric_names if not _is_vital(n)]
    report.primary_metric = (non_vital or metric_names)[0]
    return report
