# ludus/onboarding/onboard.py
"""Onboarding orchestrator: game id -> probe -> metrics -> objective -> GameProfile.

The generated objective is a deterministic template (LLM-refined objectives are
a later, optional pass — the template is offline-testable and good enough for
the selector prompt)."""

from __future__ import annotations

from pathlib import Path

from ludus.onboarding.metrics import find_metrics
from ludus.onboarding.prober import ControlProber, ProbeReport
from ludus.onboarding.profile import GameProfile

DEFAULT_TIMING_MS = 120


def build_objective(game_id: str, primary_metric: str, higher_is_better: bool,
                    controls: dict[str, str]) -> str:
    goal, gerund = (("Maximize", "maximizing") if higher_is_better
                    else ("Minimize", "minimizing"))
    control_list = ", ".join(f"{sem} ({key})" for sem, key in controls.items())
    return (
        f"You are playing {game_id}. {goal} {primary_metric}. "
        f"Available controls: {control_list}. "
        f"Each turn, return an \"actions\" sequence of 1-5 of these controls "
        f"that makes progress toward {gerund} {primary_metric}."
    )


def onboard_game(
    game_id: str,
    client_factory,
    *,
    probe_keys: list[str] | None = None,
    out_dir: Path | str = "profiles",
    timing_ms: int = DEFAULT_TIMING_MS,
) -> GameProfile:
    report: ProbeReport = ControlProber(client_factory, probe_keys=probe_keys).probe()
    metric_report = find_metrics(report.state_schema, report.control_effects)
    profile = GameProfile(
        game_id=game_id,
        controls=report.controls,
        control_effects=report.control_effects,
        metrics=metric_report.metrics,
        primary_metric=metric_report.primary_metric,
        higher_is_better=metric_report.higher_is_better,
        timing_ms=timing_ms,
        objective=build_objective(game_id, metric_report.primary_metric,
                                  metric_report.higher_is_better, report.controls),
        state_schema=report.state_schema,
    )
    profile.save(Path(out_dir) / f"{game_id}.json")
    return profile
