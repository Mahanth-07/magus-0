# Magus-1 M3: Model-Based Planner + Duel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An agent that plans against its own induced world model — beam search over sandboxed `predict()` rollouts — exposed as a `PlannerProvider` the existing loop can drive, plus a `duel` harness that runs planner vs zero-shot frontier VLM on the same game and step budget. Milestone target: the planner **wins a duel** on a game it learned autonomously.

**Architecture:** The loop passes the raw state dict into `PlannerContext` (new optional field, hasattr-guarded — the same seam pattern as `state_text`). `rank_macros` does level-synchronous beam search: at each depth it batches every (beam-state × action) case into ONE sandbox subprocess call, scores branches by cumulative predicted primary-metric delta, and returns top-k macros best-first (reward desc, then shorter macro). `PlannerProvider` loads `worldmodels/<game>/model.py` — **refusing anything not gated INDUCED** — and returns the argmax macro as a normal `Decision`. `candidates.py` renders the ranked list into the proven M0 candidate-block prompt format (Stage-4 exhaust; the duel itself uses pure argmax). `cmd_duel` runs both sides and prints an honest table.

**Tech Stack:** Python 3.11+, existing ludus modules (sandbox, worldmodel artifacts, GenericAdapter, loop). No new dependencies.

**Spec:** `docs/specs/2026-07-04-magus-1-induced-world-models-design.md` (Stage 3). **Out of scope:** selector-model play and SFT exhaust collection (Stage 4), partner modeling (Stage 5), wall-clock/time-delta world models (M4 research), curiosity exploration.

**Conventions:** run tests with `.venv/bin/pytest` from `/Users/mahanth/ludus`, branch `magus-1-m1`, do not push. Commits end with `Co-Authored-By: Claude <noreply@anthropic.com>`.

---

## File structure (locked in)

```
ludus/
  schemas.py                 # MODIFY: PlannerContext.raw_state: dict | None = None
  loop.py                    # MODIFY: thread gameworld.raw_state() into ctx (hasattr guard)
  planning/
    __init__.py              # NEW: empty
    planner.py               # NEW: rank_macros — beam search over sandboxed predict()
    candidates.py            # NEW: render candidate block (M0 prompt format)
    provider.py              # NEW: PlannerProvider (loads model, gates on INDUCED, argmax)
  cli.py                     # MODIFY: --provider planner wiring + duel subcommand
tests/
  test_raw_state_threading.py  # NEW
  test_planner.py              # NEW
  test_candidates.py           # NEW
  test_planner_provider.py     # NEW
  test_cli_duel.py             # NEW
```

Shared test fixture note: several test files below use the exact `GRID_MODEL` source string already present in `tests/test_inducer.py` (a correct gridworld world model). Each file that needs it declares its own copy — tasks are read independently; do not import across test files.

---

### Task 1: Thread raw_state into PlannerContext

**Files:**
- Modify: `ludus/schemas.py` (PlannerContext)
- Modify: `ludus/loop.py` (ctx construction)
- Test: `tests/test_raw_state_threading.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_raw_state_threading.py
"""The loop must hand providers the raw state dict (planner seam)."""

from ludus.loop import run_episode
from ludus.persistence.local import LocalStore
from ludus.rulebook import Rulebook
from ludus.schemas import Decision, PlannerContext
from ludus.synthetic import GridWorldGame
from ludus.adapters.generic import GenericAdapter
from ludus.config import GameConfig


def _cfg() -> GameConfig:
    return GameConfig(
        name="gridworld", objective="test", legal_actions=["move_right"],
        primary_metric="score", higher_is_better=True, timing_ms=10,
        control_map={"move_right": "ArrowRight"}, relevant_metrics=["score"],
    )


class _CapturingProvider:
    name = "capture"

    def __init__(self):
        self.contexts = []

    def decide(self, ctx: PlannerContext) -> Decision:
        self.contexts.append(ctx)
        return Decision(
            scene_summary="s", controlled_entity="e", current_subgoal="g",
            action="move_right", actions=["move_right"],
            expected_result="r", reason="t", confidence=1.0,
        )


def test_planner_context_accepts_raw_state_default_none():
    ctx = PlannerContext(objective="o", legal_actions=["a"],
                         screenshot_png=b"x")
    assert ctx.raw_state is None


def test_loop_threads_raw_state_to_provider(tmp_path):
    provider = _CapturingProvider()
    run_episode(adapter=GenericAdapter(_cfg()), provider=provider,
                gameworld=GridWorldGame(), store=LocalStore(root=tmp_path),
                rulebook=Rulebook(), mode="baseline", max_steps=2,
                episode_id="rawstate-test")
    assert len(provider.contexts) == 2
    rs = provider.contexts[0].raw_state
    assert rs is not None
    assert rs["game_state"]["player"]["x"] == 2  # GridWorld spawn


def test_loop_survives_gameworld_without_raw_state(tmp_path):
    class NoRawState:
        def screenshot(self):
            return b"png"

        def metrics(self, keys):
            return {k: 0.0 for k in keys}

        def apply(self, command):
            pass

    provider = _CapturingProvider()
    run_episode(adapter=GenericAdapter(_cfg()), provider=provider,
                gameworld=NoRawState(), store=LocalStore(root=tmp_path),
                rulebook=Rulebook(), mode="baseline", max_steps=1,
                episode_id="norawstate-test")
    assert provider.contexts[0].raw_state is None
```

CAREFUL: check `ludus/schemas.py` for `Decision`'s exact required fields and `PlannerContext`'s existing fields before running — if `Decision` requires fields not set above (e.g. `action_args`), add them per the real schema; follow `tests/test_loop.py`'s existing pattern for constructing a scripted provider. Adapt the TEST to the real schemas, never the reverse.

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_raw_state_threading.py -q`
Expected: FAIL — `raw_state` not a PlannerContext field.

- [ ] **Step 3: Implement**

In `ludus/schemas.py`, add to `PlannerContext` (after `state_text`):

```python
    raw_state: dict | None = None   # full getState() dict (planner seam; None if unsupported)
```

In `ludus/loop.py`, in the `PlannerContext(...)` construction, add alongside the `state_text` line (same hasattr-guard idiom):

```python
            raw_state=(gameworld.raw_state() if hasattr(gameworld, "raw_state") else None),
```

- [ ] **Step 4: Verify green + full suite**

Run: `.venv/bin/pytest tests/test_raw_state_threading.py -q` — 3 passed.
Full suite: `.venv/bin/pytest -q --ignore=infra --ignore=gameworld` — expect 218, zero regressions.

- [ ] **Step 5: Commit**

```bash
git add ludus/schemas.py ludus/loop.py tests/test_raw_state_threading.py
git commit -m "feat(loop): thread raw_state into PlannerContext (planner seam)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: rank_macros — beam search over the induced model

**Files:**
- Create: `ludus/planning/__init__.py` (empty)
- Create: `ludus/planning/planner.py`
- Test: `tests/test_planner.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_planner.py
"""Beam search over a sandboxed induced model."""

from ludus.planning.planner import rank_macros

# Correct gridworld world model (same dynamics as ludus/synthetic.py).
GRID_MODEL = '''
import copy

MOVES = {"move_left": (-1, 0), "move_right": (1, 0),
         "move_up": (0, -1), "move_down": (0, 1)}
GOAL_CYCLE = [(3, 2), (0, 0), (4, 4), (0, 4), (4, 0)]

def predict(state, action):
    out = copy.deepcopy(state)
    p = out["game_state"]["player"]
    out["metrics"]["steps"] += 1
    dx, dy = MOVES.get(action, (0, 0))
    p["x"] = min(4, max(0, p["x"] + dx))
    p["y"] = min(4, max(0, p["y"] + dy))
    goal = out["game_state"]["environment"]["goal"]
    if (p["x"], p["y"]) == (goal["x"], goal["y"]):
        out["metrics"]["score"] += 10
        i = GOAL_CYCLE.index((goal["x"], goal["y"]))
        nx, ny = GOAL_CYCLE[(i + 1) % len(GOAL_CYCLE)]
        goal["x"], goal["y"] = nx, ny
    return out
'''

ACTIONS = ["move_left", "move_right", "move_up", "move_down"]


def _state(px, py, gx, gy, score=0):
    return {"status": "playing",
            "metrics": {"score": score, "steps": 0},
            "game_state": {"player": {"x": px, "y": py},
                           "environment": {"goal": {"x": gx, "y": gy}}}}


def test_finds_adjacent_goal_in_one_move():
    ranked = rank_macros(GRID_MODEL, _state(2, 2, 3, 2), ACTIONS,
                         primary_metric="score", higher_is_better=True,
                         depth=3, beam=8, top_k=4)
    best = ranked[0]
    assert best["actions"] == ["move_right"]
    assert best["predicted_reward"] == 10.0


def test_finds_two_move_goal_and_prefers_shortest_path():
    # goal at (4,3) from (2,2): reachable in 3 (right,right,down in some order)
    ranked = rank_macros(GRID_MODEL, _state(2, 2, 4, 3), ACTIONS,
                         primary_metric="score", higher_is_better=True,
                         depth=3, beam=16, top_k=4)
    best = ranked[0]
    assert best["predicted_reward"] == 10.0
    assert len(best["actions"]) == 3
    assert sorted(best["actions"]) == ["move_down", "move_right", "move_right"]


def test_no_reward_in_horizon_still_returns_ranked_macros():
    # goal unreachable within depth 2 from (0,0) at (4,4): all rewards 0
    ranked = rank_macros(GRID_MODEL, _state(0, 0, 4, 4), ACTIONS,
                         primary_metric="score", higher_is_better=True,
                         depth=2, beam=8, top_k=4)
    assert len(ranked) == 4
    assert all(r["predicted_reward"] == 0.0 for r in ranked)
    assert all(1 <= len(r["actions"]) <= 2 for r in ranked)


def test_broken_model_returns_empty():
    ranked = rank_macros("def predict(s, a):\n    raise RuntimeError()\n",
                         _state(2, 2, 3, 2), ACTIONS,
                         primary_metric="score", higher_is_better=True,
                         depth=2, beam=4, top_k=4)
    assert ranked == []


def test_terminal_branches_are_not_expanded():
    dying_model = '''
import copy

def predict(state, action):
    out = copy.deepcopy(state)
    if action == "move_right":
        out["metrics"]["score"] += 5
        out["status"] = "terminal"
    return out
'''
    ranked = rank_macros(dying_model, _state(2, 2, 3, 2), ["move_right", "move_left"],
                         primary_metric="score", higher_is_better=True,
                         depth=3, beam=8, top_k=10)
    # move_right yields +5 then terminal: macros extending past it must not exist
    for r in ranked:
        if r["actions"][0] == "move_right":
            assert len(r["actions"]) == 1
```

- [ ] **Step 2: Run to verify failure** — ModuleNotFoundError.

- [ ] **Step 3: Implement**

```python
# ludus/planning/__init__.py
```
(empty)

```python
# ludus/planning/planner.py
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
```

- [ ] **Step 4: Verify green + full suite**

Run: `.venv/bin/pytest tests/test_planner.py -q` — 5 passed. Full suite — 223.

DEBUGGING NOTE for `test_finds_two_move_goal_and_prefers_shortest_path`: from (2,2) to goal (4,3) minimum path length is 3 (dx=2, dy=1). If the test expects length 3 but rank_macros returns a shorter "winner", check the reward accounting — only macros whose LAST step lands on the goal collect +10 within depth 3.

- [ ] **Step 5: Commit**

```bash
git add ludus/planning/__init__.py ludus/planning/planner.py tests/test_planner.py
git commit -m "feat(planning): rank_macros — batched beam search over the induced model

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: Candidate-block rendering (Stage-4 exhaust, M0 format parity)

**Files:**
- Create: `ludus/planning/candidates.py`
- Test: `tests/test_candidates.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_candidates.py
"""Candidate-block rendering — the generalized M0 'copy candidate [0]' prompt."""

from ludus.planning.candidates import render_candidate_block


def test_renders_m0_compatible_block():
    ranked = [
        {"actions": ["move_right"], "predicted_reward": 10.0},
        {"actions": ["move_left", "move_up"], "predicted_reward": 0.0},
    ]
    block = render_candidate_block(ranked, primary_metric="score")
    lines = block.splitlines()
    assert lines[0].startswith("Candidate placements")
    assert "[0]" in lines[1] and "actions=['move_right']" in lines[1]
    assert "predicted score +10.0" in lines[1]
    assert "[1]" in lines[2] and "+0.0" in lines[2]


def test_empty_candidates_render_empty_string():
    assert render_candidate_block([], primary_metric="score") == ""
```

- [ ] **Step 2: Run to verify failure** — ModuleNotFoundError.

- [ ] **Step 3: Implement**

```python
# ludus/planning/candidates.py
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
            f"[{i}] predicted {primary_metric} +{c['predicted_reward']:g}, "
            f"actions={c['actions']}"
        )
    return "\n".join(out)
```

- [ ] **Step 4: Verify green + full suite** — 2 passed; suite 225.

- [ ] **Step 5: Commit**

```bash
git add ludus/planning/candidates.py tests/test_candidates.py
git commit -m "feat(planning): candidate-block rendering (M0 prompt parity, Stage-4 exhaust)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: PlannerProvider + CLI wiring

**Files:**
- Create: `ludus/planning/provider.py`
- Modify: `ludus/cli.py` (`--provider planner` special-case)
- Test: `tests/test_planner_provider.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_planner_provider.py
"""PlannerProvider — the induced model driving the real loop."""

import json

import pytest

from ludus.onboarding.profile import GameProfile
from ludus.planning.provider import PlannerProvider
from ludus.schemas import PlannerContext

GRID_MODEL = '''
import copy

MOVES = {"move_left": (-1, 0), "move_right": (1, 0),
         "move_up": (0, -1), "move_down": (0, 1)}
GOAL_CYCLE = [(3, 2), (0, 0), (4, 4), (0, 4), (4, 0)]

def predict(state, action):
    out = copy.deepcopy(state)
    p = out["game_state"]["player"]
    out["metrics"]["steps"] += 1
    dx, dy = MOVES.get(action, (0, 0))
    p["x"] = min(4, max(0, p["x"] + dx))
    p["y"] = min(4, max(0, p["y"] + dy))
    goal = out["game_state"]["environment"]["goal"]
    if (p["x"], p["y"]) == (goal["x"], goal["y"]):
        out["metrics"]["score"] += 10
        i = GOAL_CYCLE.index((goal["x"], goal["y"]))
        nx, ny = GOAL_CYCLE[(i + 1) % len(GOAL_CYCLE)]
        goal["x"], goal["y"] = nx, ny
    return out
'''


def _grid_profile() -> GameProfile:
    return GameProfile(
        game_id="synthetic:grid",
        controls={"move_left": "ArrowLeft", "move_right": "ArrowRight",
                  "move_up": "ArrowUp", "move_down": "ArrowDown"},
        control_effects={}, metrics=["score", "steps"], primary_metric="score",
        higher_is_better=True, timing_ms=10, objective="test", state_schema={},
    )


def _model_dir(tmp_path, status="INDUCED"):
    d = tmp_path / "synthetic:grid"
    d.mkdir(parents=True)
    (d / "model.py").write_text(GRID_MODEL)
    (d / "report.json").write_text(json.dumps({"status": status, "overall": 1.0}))
    return tmp_path


def _ctx(raw_state):
    return PlannerContext(objective="o", legal_actions=["move_right"],
                          screenshot_png=b"x", raw_state=raw_state)


def test_provider_refuses_non_induced_model(tmp_path):
    root = _model_dir(tmp_path, status="FAILED")
    with pytest.raises(SystemExit, match="not INDUCED"):
        PlannerProvider(_grid_profile(), worldmodels_dir=root)


def test_provider_plans_toward_goal(tmp_path):
    root = _model_dir(tmp_path)
    provider = PlannerProvider(_grid_profile(), worldmodels_dir=root, depth=3)
    state = {"status": "playing", "metrics": {"score": 0, "steps": 0},
             "game_state": {"player": {"x": 2, "y": 2},
                            "environment": {"goal": {"x": 3, "y": 2}}}}
    decision = provider.decide(_ctx(state))
    assert decision.actions == ["move_right"]
    assert decision.action == "move_right"
    assert "predicted score" in decision.reason


def test_provider_without_raw_state_falls_back_to_first_action(tmp_path):
    root = _model_dir(tmp_path)
    provider = PlannerProvider(_grid_profile(), worldmodels_dir=root)
    decision = provider.decide(_ctx(None))
    assert decision.actions == [sorted(_grid_profile().controls)[0]]


def test_provider_end_to_end_scores_in_real_episode(tmp_path):
    """The M3 offline acceptance: planner over an induced model beats random.
    From spawn (2,2) with goal (3,2), 5 steps must collect >= 10 score."""
    from ludus.adapters.generic import GenericAdapter
    from ludus.loop import run_episode
    from ludus.persistence.local import LocalStore
    from ludus.rulebook import Rulebook
    from ludus.synthetic import GridWorldGame

    root = _model_dir(tmp_path)
    profile = _grid_profile()
    provider = PlannerProvider(profile, worldmodels_dir=root, depth=3)
    result = run_episode(
        adapter=GenericAdapter(profile.to_game_config()), provider=provider,
        gameworld=GridWorldGame(), store=LocalStore(root=tmp_path / "runs"),
        rulebook=Rulebook(), mode="baseline", max_steps=5,
        episode_id="planner-e2e")
    assert result.final_metrics["score"] >= 10.0
```

(As always: check `Decision`/`EpisodeResult` field names against `ludus/schemas.py` — e.g. whether final metrics live at `result.final_metrics` — and follow `tests/test_loop.py`; adapt tests to reality.)

- [ ] **Step 2: Run to verify failure** — ModuleNotFoundError.

- [ ] **Step 3: Implement**

```python
# ludus/planning/provider.py
"""PlannerProvider — plays via beam search over the game's induced world model.

Loads worldmodels/<game_id>/model.py and REFUSES to run against a model whose
report.json is not INDUCED (spec: never plan against an unvalidated model —
the duel must not produce junk numbers). Slots into the existing loop as a
normal ModelProvider; no VLM, no network, decisions in milliseconds-to-tens-of-ms.
"""

from __future__ import annotations

import json
from pathlib import Path

from ludus.onboarding.profile import GameProfile
from ludus.planning.candidates import render_candidate_block
from ludus.planning.planner import rank_macros
from ludus.schemas import Decision, PlannerContext

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
        self._source = (model_dir / "model.py").read_text()

    def available(self) -> bool:
        return True

    def _fallback(self, why: str) -> Decision:
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
        # Stage-4 exhaust surface: the block a selector/student would see.
        decision.scene_summary = render_candidate_block(
            ranked, primary_metric=self._profile.primary_metric
        )[:400] or decision.scene_summary
        return decision
```

CAREFUL: check `Decision` in `ludus/schemas.py` — if `scene_summary` has a max length or `action_args` is required, adjust `_decision` accordingly (mirror how `tests/test_loop.py` or `MockProvider` builds Decisions). If overwriting `scene_summary` with the candidate block fails validation, drop that assignment and keep the plain summary (it is a nice-to-have, not load-bearing).

In `ludus/cli.py`:
1. Add `"planner"` to the `--provider` choices list.
2. After the adapter/profile resolution and BEFORE `provider = build_provider(args.provider)`, special-case:

```python
    if args.provider == "planner":
        if not args.profile:
            raise SystemExit("--provider planner requires --profile "
                             "(the planner needs the game's GameProfile)")
        from ludus.planning.provider import PlannerProvider
        provider = PlannerProvider(profile)
    else:
        provider = build_provider(args.provider)
```

(This requires keeping a reference to the loaded `profile` object in the `--profile` branch — it already exists as `profile` there; move its scope up if needed.)

- [ ] **Step 4: Verify green + full suite**

Run: `.venv/bin/pytest tests/test_planner_provider.py -q` — 4 passed. Full suite — 229.

- [ ] **Step 5: Commit**

```bash
git add ludus/planning/provider.py ludus/cli.py tests/test_planner_provider.py
git commit -m "feat(planning): PlannerProvider — induced-model play via the existing loop

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: duel subcommand

**Files:**
- Modify: `ludus/cli.py` (`duel` subcommand + `cmd_duel`)
- Test: `tests/test_cli_duel.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cli_duel.py
"""Duel harness — planner vs a baseline provider, same game, same budget."""

import json

from ludus.cli import cmd_duel
from ludus.providers.mock import MockProvider

GRID_MODEL = '''
import copy

MOVES = {"move_left": (-1, 0), "move_right": (1, 0),
         "move_up": (0, -1), "move_down": (0, 1)}
GOAL_CYCLE = [(3, 2), (0, 0), (4, 4), (0, 4), (4, 0)]

def predict(state, action):
    out = copy.deepcopy(state)
    p = out["game_state"]["player"]
    out["metrics"]["steps"] += 1
    dx, dy = MOVES.get(action, (0, 0))
    p["x"] = min(4, max(0, p["x"] + dx))
    p["y"] = min(4, max(0, p["y"] + dy))
    goal = out["game_state"]["environment"]["goal"]
    if (p["x"], p["y"]) == (goal["x"], goal["y"]):
        out["metrics"]["score"] += 10
        i = GOAL_CYCLE.index((goal["x"], goal["y"]))
        nx, ny = GOAL_CYCLE[(i + 1) % len(GOAL_CYCLE)]
        goal["x"], goal["y"] = nx, ny
    return out
'''


def _setup(tmp_path):
    from ludus.cli import cmd_onboard
    cmd_onboard(game="synthetic:grid", out_dir=tmp_path / "profiles")
    d = tmp_path / "wm" / "synthetic:grid"
    d.mkdir(parents=True)
    (d / "model.py").write_text(GRID_MODEL)
    (d / "report.json").write_text(json.dumps({"status": "INDUCED", "overall": 1.0}))
    return tmp_path


def test_duel_planner_beats_mock_on_grid(tmp_path, capsys):
    root = _setup(tmp_path)
    result = cmd_duel(
        game="synthetic:grid",
        profile_path=root / "profiles" / "synthetic:grid.json",
        worldmodels_dir=root / "wm", steps=6,
        baseline_provider=MockProvider(),
        runs_dir=root / "runs",
    )
    assert result["planner"]["score"] >= 10.0
    assert result["baseline"]["score"] <= result["planner"]["score"]
    assert result["winner"] == "planner"
    out = capsys.readouterr().out
    assert "DUEL" in out and "planner" in out and "winner" in out


def test_duel_result_is_persisted(tmp_path):
    root = _setup(tmp_path)
    result = cmd_duel(
        game="synthetic:grid",
        profile_path=root / "profiles" / "synthetic:grid.json",
        worldmodels_dir=root / "wm", steps=4,
        baseline_provider=MockProvider(),
        runs_dir=root / "runs",
    )
    saved = json.loads((root / "runs" / "duel-synthetic:grid.json").read_text())
    assert saved == result
```

- [ ] **Step 2: Run to verify failure** — ImportError (no cmd_duel).

- [ ] **Step 3: Implement in `ludus/cli.py`**

Add `cmd_duel` after `cmd_induce`:

```python
def cmd_duel(game: str, profile_path=None, worldmodels_dir="worldmodels",
             steps=15, baseline_provider=None, baseline_name="gateway",
             runs_dir="runs", headless: bool = True) -> dict:
    """Planner (induced model) vs a baseline provider: same game, same step
    budget, fresh client per side. Prints an honest table, persists the result."""
    from ludus.planning.provider import PlannerProvider

    profile = GameProfile.load(profile_path or Path("profiles") / f"{game}.json")
    adapter = GenericAdapter(profile.to_game_config())
    planner = PlannerProvider(profile, worldmodels_dir=worldmodels_dir)
    baseline = baseline_provider or build_provider(baseline_name)

    sides = {}
    for label, provider in (("planner", planner), ("baseline", baseline)):
        gw = _client_factory(game, timing_ms=profile.timing_ms,
                             headless=headless)()
        try:
            result = run_episode(
                adapter=adapter, provider=provider, gameworld=gw,
                store=LocalStore(root=runs_dir), rulebook=Rulebook(),
                mode="baseline", max_steps=steps,
                episode_id=f"duel-{game}-{label}")
        finally:
            if hasattr(gw, "close"):
                gw.close()
        sides[label] = {
            "provider": getattr(provider, "name", label),
            "score": result.final_metrics.get(profile.primary_metric, 0.0),
            "legal_action_rate": result.legal_action_rate,
            "steps": result.steps,
        }

    p, b = sides["planner"]["score"], sides["baseline"]["score"]
    better = p > b if profile.higher_is_better else p < b
    winner = "planner" if better else ("baseline" if p != b else "tie")
    out = {"game": game, "steps": steps,
           "primary_metric": profile.primary_metric,
           "planner": sides["planner"], "baseline": sides["baseline"],
           "winner": winner}

    print(f"==== DUEL {game} ({steps} steps, metric={profile.primary_metric}) ====")
    for label in ("planner", "baseline"):
        s = sides[label]
        print(f"  {label:9s} ({s['provider']:8s}): {profile.primary_metric}="
              f"{s['score']:g}  legal_rate={s['legal_action_rate']:.2f}")
    print(f"  winner: {winner}")

    dest = Path(runs_dir) / f"duel-{game}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n")
    return out
```

Add `import json` to cli.py's imports if absent; `run_episode`, `LocalStore`, `Rulebook`, `GenericAdapter`, `GameProfile`, `build_provider` are already imported.

Extend the subcommand dispatch tuple to `("onboard", "explore", "induce", "duel")` and add the branch:

```python
        elif cmd == "duel":
            ap.add_argument("--profile", type=Path, default=None)
            ap.add_argument("--worldmodels", default="worldmodels")
            ap.add_argument("--steps", type=int, default=15)
            ap.add_argument("--baseline", default="gateway")
            ap.add_argument("--headed", action="store_true")
            args = ap.parse_args(sys.argv[2:])
            cmd_duel(args.game, profile_path=args.profile,
                     worldmodels_dir=args.worldmodels, steps=args.steps,
                     baseline_name=args.baseline, headless=not args.headed)
```

- [ ] **Step 4: Verify green + full suite**

Run: `.venv/bin/pytest tests/test_cli_duel.py -q` — 2 passed. Full suite — 231. NOTE: MockProvider's fixed decision picks the first legal action (check `ludus/providers/mock.py`) — on grid that's `move_down` or `move_left` (alphabetical), wandering without reaching the goal cycle start; if Mock happens to score, the assertion `baseline <= planner` still holds.

- [ ] **Step 5: Commit**

```bash
git add ludus/cli.py tests/test_cli_duel.py
git commit -m "feat(cli): duel subcommand — planner vs baseline, same budget, honest table

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: Live acceptance — the milestone duels

Live env: gateway creds in `.env`; browser stack in `/opt/anaconda3/bin/python`; ANTHROPIC_API_KEY for any re-induction.

- [ ] **Step 1: Sanity duel — synthetic:grid, planner vs gateway VLM (offline game, live VLM)**

```bash
.venv/bin/python -m ludus.cli duel synthetic:grid --steps 10
```

Expected: planner wins decisively (it has the true dynamics; gemini-flash sees a game it has never encountered through a generic profile prompt). Record both scores.

- [ ] **Step 2: The real-game attempt — 01_2048 (turn-based, no wall-clock dynamics)**

2048's board merges are DETERMINISTIC given the action; only tile spawn is stochastic (copy-through per the synthesis policy). Its M1 profile has 3 controls (ArrowRight/ArrowUp were situational no-ops during probing) — re-onboard first with more repeats to give them more chances, then accept whatever is found, honestly:

```bash
/opt/anaconda3/bin/python - <<'EOF'
from ludus.cli import _client_factory
from ludus.onboarding.onboard import onboard_game
profile = onboard_game("01_2048", _client_factory("01_2048", 120, True))
print(profile.controls)
EOF
# then (anaconda python, live browser):
/opt/anaconda3/bin/python -m ludus.cli explore 01_2048 --episodes 8 --steps 30 --seed 11
/opt/anaconda3/bin/python -m ludus.cli induce 01_2048 --iterations 4
/opt/anaconda3/bin/python -m ludus.cli duel 01_2048 --steps 15
```

(If `onboard_game` needs different kwargs, use `python -m ludus.cli onboard 01_2048` instead — the point is a fresh probe.) Record every verdict verbatim. If induction FAILs, read the per-field report; the likely culprits are the board-list paths — if `board` flattens to `__len__` only, the model can't predict merges at all and the honest conclusion is that 2048 needs indexed board paths (raise `MAX_INDEXED_LIST` consideration for M4, do NOT hack it in now).

- [ ] **Step 3: Tetris duel (even with the FAILED model — refusal check)**

```bash
/opt/anaconda3/bin/python -m ludus.cli duel 29_tetris --steps 10 || echo "REFUSED (expected)"
```

Expected: SystemExit "not INDUCED ... refusing to plan against an unvalidated model" — the honesty gate working as designed. Record it.

- [ ] **Step 4: Commit artifacts + docs**

```bash
git add runs/duel-*.json profiles/ worldmodels/ 2>/dev/null; git add -u
git commit -m "feat(duel): first live duels — results recorded verbatim

Co-Authored-By: Claude <noreply@anthropic.com>"
```

(`runs/` is gitignored — if `git add runs/duel-*.json` fails, copy the duel JSONs into `docs/results/` and commit there instead; the duel numbers are the milestone evidence and must be versioned.)

Update README roadmap item 6 with an M3 line (duel results, verbatim numbers, honest caveats) and CLAUDE.md with the `duel` CLI usage + PlannerProvider description + the "refuses non-INDUCED models" behavior.

- [ ] **Step 5: Final suite + commit**

Run: `.venv/bin/pytest -q --ignore=infra --ignore=gameworld` — all pass.

```bash
git add README.md CLAUDE.md
git commit -m "docs: M3 planner + duel usage and results

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Self-review notes (already applied)

- **Spec coverage (Stage 3):** depth-limited rollout search over the induced model (Task 2), candidate rendering in the proven format (Task 3), duel harness with same-budget head-to-head + refusal of unvalidated models (Tasks 4-5), live head-to-heads (Task 6). Deliberately deferred: selector-model play + SFT exhaust logging (Stage 4), curiosity exploration, time-delta world models (M4).
- **Type consistency:** `rank_macros(source, state, actions, *, primary_metric, higher_is_better, depth, beam, top_k)` used identically in Tasks 2 and 4; `render_candidate_block(ranked, *, primary_metric)` in Tasks 3-4; `cmd_duel(game, profile_path, worldmodels_dir, steps, baseline_provider|baseline_name, runs_dir, headless)` in Task 5's tests and dispatch. GRID_MODEL duplicated per test file by design.
- **Known risk 1:** `Decision` schema constraints (field lengths, required `action_args`) may reject the provider's constructed decisions — Tasks 1 and 4 carry explicit adapt-the-test/check-the-schema instructions.
- **Known risk 2:** grid duel step economics — the planner reaches a goal roughly every 2-4 steps (+10 each); gemini-flash zero-shot on an unseen grid game may wander (~0). If the VLM ties, raise --steps before concluding; record whatever happens.
- **Known risk 3:** 2048 board representation — a 4x4 board (16 cells) sits exactly AT `MAX_INDEXED_LIST=16`: 4 rows of 4 → indexed fine at both levels. Verify during Task 6 by inspecting a transition; if the game exposes a flat 16-list it is still indexed (16 <= 16). The risk note in Task 6 covers the failure mode honestly.
