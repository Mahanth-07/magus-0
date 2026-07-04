# ludus/planning/provider.py
"""PlannerProvider — plays via beam search over the game's induced world model.

Loads worldmodels/<game_id>/model.py and REFUSES to run against a model whose
report.json is not INDUCED (spec: never plan against an unvalidated model —
the duel must not produce junk numbers). Slots into the existing loop as a
normal ModelProvider; no VLM, no network, decisions in milliseconds-to-tens-of-ms.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ludus.onboarding.profile import GameProfile
from ludus.planning.planner import rank_macros
from ludus.schemas import Decision, PlannerContext

log = logging.getLogger("ludus.planning")

MAX_MACRO_MOVES = 5  # Decision.actions contract (loop applies 1-5 moves)


class PlannerProvider:
    name = "planner"

    def __init__(self, profile: GameProfile,
                 worldmodels_dir: Path | str = "worldmodels",
                 depth: int = 3, beam: int = 16, top_k: int = 6) -> None:
        self._profile = profile
        self._depth = depth
        self._beam = beam
        self._top_k = top_k
        model_dir = Path(worldmodels_dir) / profile.game_id
        report_path = model_dir / "report.json"
        if not report_path.exists():
            raise SystemExit(
                f"no world model for {profile.game_id}; run: "
                f"python -m ludus.cli induce {profile.game_id}")
        report = json.loads(report_path.read_text())
        if report.get("status") != "INDUCED":
            raise SystemExit(
                f"world model for {profile.game_id} is not INDUCED "
                f"(status={report.get('status')!r}, overall="
                f"{report.get('overall')}); refusing to plan against an "
                f"unvalidated model")
        model_path = model_dir / "model.py"
        if not model_path.exists():
            raise SystemExit(
                f"model.py missing for {profile.game_id} (report says INDUCED "
                f"— corrupted artifacts?); re-run: python -m ludus.cli induce "
                f"{profile.game_id}")
        self._source = model_path.read_text()

    def available(self) -> bool:
        return True

    def _fallback(self, why: str) -> Decision:
        log.warning("PlannerProvider fallback: %s", why)
        action = sorted(self._profile.controls)[0]
        return self._decision([action], reason=f"planner fallback: {why}")

    def _decision(self, actions: list[str], reason: str) -> Decision:
        return Decision(
            scene_summary=f"{self._profile.game_id}: planning over induced model",
            controlled_entity="the player",
            current_subgoal=f"maximize {self._profile.primary_metric}",
            action=actions[0],
            actions=actions,
            expected_result=reason,
            reason=reason,
            confidence=1.0,
        )

    def decide(self, ctx: PlannerContext) -> Decision:
        if not isinstance(ctx.raw_state, dict) or not ctx.raw_state:
            return self._fallback("no raw_state in context")
        ranked = rank_macros(
            self._source, ctx.raw_state, sorted(self._profile.controls),
            primary_metric=self._profile.primary_metric,
            higher_is_better=self._profile.higher_is_better,
            depth=self._depth, beam=self._beam, top_k=self._top_k,
        )
        if not ranked:
            return self._fallback("world model produced no predictions")
        best = ranked[0]
        macro = best["actions"][:MAX_MACRO_MOVES]
        reason = (f"beam search depth={self._depth}: predicted "
                  f"{self._profile.primary_metric} "
                  f"{'+' if best['predicted_reward'] >= 0 else ''}"
                  f"{best['predicted_reward']:g}")
        decision = self._decision(macro, reason=reason)
        return decision
