# Magus-1 M1: Auto-Onboarding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Point the system at a game id and get a machine-generated `GameProfile` (controls, metrics, objective) that the existing episode loop can play from — zero hand-written per-game config.

**Architecture:** A `ControlProber` presses candidate keys against a live game and attributes state changes to keys via flatten/diff utilities (with an ambient-drift baseline so gravity/timers aren't misattributed). A `MetricFinder` picks the primary metric from numeric state fields using name priors + observed changes. Both feed a `GameProfile` (pydantic, cached as JSON) that converts to the existing `GameConfig`, consumed by a new `GenericAdapter`. Synthetic pure-Python games with known ground truth make the whole pipeline testable offline in CI.

**Tech Stack:** Python 3.11+, pydantic v2, pytest, existing ludus package (`loop.py`, `GameConfig`, adapters, providers). No new dependencies.

**Spec:** `docs/specs/2026-07-04-magus-1-induced-world-models-design.md` (Milestone 1).

**Conventions for every task:** run tests with `.venv/bin/pytest` from the repo root `/Users/mahanth/ludus`. Commit messages end with:
```
Co-Authored-By: Claude <noreply@anthropic.com>
```

---

## File structure (locked in)

```
ludus/
  state_source.py            # NEW: StateSource Protocol — the pluggable state seam
  synthetic.py               # NEW: CounterGame, GridWorldGame (offline ground truth)
  onboarding/
    __init__.py              # NEW: empty
    diff.py                  # NEW: flatten_state, diff_states
    profile.py               # NEW: GameProfile (+ save/load/to_game_config)
    prober.py                # NEW: ControlProber + semantic naming heuristics
    metrics.py               # NEW: MetricFinder
    onboard.py               # NEW: orchestrator (probe → metrics → objective → profile)
  adapters/
    generic.py               # NEW: GenericAdapter (GameConfig-driven, no game class)
  gameworld_client.py        # MODIFY: add raw_state() to GameWorldClient + FakeGameWorld
  cli.py                     # MODIFY: `onboard` subcommand + `--profile` play path
tests/
  test_onboarding_diff.py    # NEW
  test_synthetic_games.py    # NEW
  test_raw_state.py          # NEW
  test_game_profile.py       # NEW
  test_prober.py             # NEW
  test_metric_finder.py      # NEW
  test_onboard_orchestrator.py  # NEW (includes offline end-to-end integration)
  test_generic_adapter.py    # NEW
profiles/                    # NEW dir: generated GameProfile JSONs (committed)
```

---

### Task 1: Flatten/diff utilities

The prober and metric finder both reason about "which fields changed." One shared module.

**Files:**
- Create: `ludus/onboarding/__init__.py`
- Create: `ludus/onboarding/diff.py`
- Test: `tests/test_onboarding_diff.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_onboarding_diff.py
"""flatten_state/diff_states — the change-attribution primitives for onboarding."""

from ludus.onboarding.diff import diff_states, flatten_state


def test_flatten_nested_dicts_to_dot_paths():
    state = {"metrics": {"score": 5}, "game_state": {"player": {"x": 2, "y": 3}}}
    flat = flatten_state(state)
    assert flat["metrics.score"] == 5
    assert flat["game_state.player.x"] == 2
    assert flat["game_state.player.y"] == 3


def test_flatten_indexes_small_lists():
    state = {"blocks": [{"x": 1}, {"x": 4}]}
    flat = flatten_state(state)
    assert flat["blocks[0].x"] == 1
    assert flat["blocks[1].x"] == 4


def test_flatten_summarizes_long_lists_as_length_only():
    # board grids (e.g. 20 rows) would create noise; record length, not elements
    state = {"board": [[0] * 10 for _ in range(20)]}
    flat = flatten_state(state)
    assert flat["board.__len__"] == 20
    assert not any(k.startswith("board[") for k in flat)


def test_flatten_keeps_scalars_and_none():
    flat = flatten_state({"status": "playing", "held": None})
    assert flat["status"] == "playing"
    assert flat["held"] is None


def test_diff_reports_changed_added_removed():
    before = {"metrics": {"score": 0, "lives": 3}, "phase": "a"}
    after = {"metrics": {"score": 10, "lives": 3}, "spawn": True}
    d = diff_states(flatten_state(before), flatten_state(after))
    assert d["metrics.score"] == (0, 10)
    assert d["phase"] == ("a", None)      # removed
    assert d["spawn"] == (None, True)     # added
    assert "metrics.lives" not in d       # unchanged


def test_diff_identical_states_is_empty():
    s = flatten_state({"a": {"b": 1}})
    assert diff_states(s, s) == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_onboarding_diff.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'ludus.onboarding'`

- [ ] **Step 3: Write the implementation**

```python
# ludus/onboarding/__init__.py
```
(empty file)

```python
# ludus/onboarding/diff.py
"""State flatten/diff primitives for onboarding.

flatten_state turns a nested getState()-shaped dict into {dot.path: scalar}.
Lists up to MAX_INDEXED_LIST elements are indexed ([i]); longer lists (board
grids) are summarized as `path.__len__` only — element-level noise from a
20x10 grid would swamp change attribution.
"""

from __future__ import annotations

MAX_INDEXED_LIST = 16

_SCALARS = (str, int, float, bool)


def flatten_state(value, prefix: str = "") -> dict:
    flat: dict = {}
    if isinstance(value, dict):
        for k, v in value.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            flat.update(flatten_state(v, key))
    elif isinstance(value, list):
        if len(value) > MAX_INDEXED_LIST:
            flat[f"{prefix}.__len__"] = len(value)
        else:
            for i, v in enumerate(value):
                flat.update(flatten_state(v, f"{prefix}[{i}]"))
    elif value is None or isinstance(value, _SCALARS):
        flat[prefix] = value
    else:  # unknown object; record its repr so changes are still visible
        flat[prefix] = repr(value)
    return flat


def diff_states(before: dict, after: dict) -> dict:
    """{path: (before_value, after_value)} for changed/added/removed paths.
    Missing-side values are None."""
    out: dict = {}
    for path in before.keys() | after.keys():
        b, a = before.get(path), after.get(path)
        if b != a:
            out[path] = (b, a)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_onboarding_diff.py -q`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add ludus/onboarding/__init__.py ludus/onboarding/diff.py tests/test_onboarding_diff.py
git commit -m "feat(onboarding): flatten/diff state primitives

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Synthetic games (offline ground truth)

Two pure-Python games that speak the GameWorld client interface (`raw_state`, `apply`, `metrics`, `screenshot`). They are the CI ground truth for the prober/metric-finder (and later for the inducer in M2 — that's why they live in `ludus/`, not `tests/`).

**Files:**
- Create: `ludus/synthetic.py`
- Test: `tests/test_synthetic_games.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_synthetic_games.py
"""Synthetic games — deterministic ground truth for onboarding tests."""

from ludus.synthetic import CounterGame, GridWorldGame


def _press(game, key):
    game.apply({"key": key, "duration_ms": 50})


# --- GridWorldGame -----------------------------------------------------------

def test_gridworld_arrows_move_player():
    g = GridWorldGame()
    x0 = g.raw_state()["game_state"]["player"]["x"]
    _press(g, "ArrowRight")
    assert g.raw_state()["game_state"]["player"]["x"] == x0 + 1
    _press(g, "ArrowLeft")
    assert g.raw_state()["game_state"]["player"]["x"] == x0
    y0 = g.raw_state()["game_state"]["player"]["y"]
    _press(g, "ArrowDown")
    assert g.raw_state()["game_state"]["player"]["y"] == y0 + 1


def test_gridworld_walls_clamp_movement():
    g = GridWorldGame()
    for _ in range(10):
        _press(g, "ArrowLeft")
    assert g.raw_state()["game_state"]["player"]["x"] == 0


def test_gridworld_reaching_goal_scores_and_respawns_goal():
    g = GridWorldGame()  # player (2,2), first goal (3,2)
    _press(g, "ArrowRight")  # onto the goal
    st = g.raw_state()
    assert st["metrics"]["score"] == 10
    goal = st["game_state"]["environment"]["goal"]
    assert (goal["x"], goal["y"]) != (3, 2)  # respawned elsewhere


def test_gridworld_quit_key_makes_state_terminal():
    g = GridWorldGame()
    _press(g, "q")
    assert g.raw_state()["status"] == "terminal"


def test_gridworld_noop_key_changes_nothing_but_step_counter():
    g = GridWorldGame()
    before = g.raw_state()
    _press(g, "z")
    after = g.raw_state()
    assert after["game_state"]["player"] == before["game_state"]["player"]
    assert after["metrics"]["score"] == before["metrics"]["score"]
    assert after["metrics"]["steps"] == before["metrics"]["steps"] + 1


def test_gridworld_is_deterministic():
    a, b = GridWorldGame(), GridWorldGame()
    for key in ["ArrowRight", "ArrowDown", "ArrowRight", "ArrowUp"]:
        _press(a, key); _press(b, key)
    assert a.raw_state() == b.raw_state()


# --- CounterGame -------------------------------------------------------------

def test_counter_x_increments_score():
    g = CounterGame()
    _press(g, "x")
    _press(g, "x")
    assert g.raw_state()["metrics"]["score"] == 2


def test_counter_z_drains_health_to_terminal():
    g = CounterGame()
    for _ in range(10):
        _press(g, "z")
    st = g.raw_state()
    assert st["metrics"]["health"] == 0
    assert st["status"] == "terminal"


# --- client-interface compliance (what run_episode/prober need) --------------

def test_synthetic_games_speak_the_client_interface():
    for g in (GridWorldGame(), CounterGame()):
        assert isinstance(g.screenshot(), bytes)
        m = g.metrics(["score"])
        assert m == {"score": 0.0}
        assert g.read_partner_actions() == []
        assert isinstance(g.raw_state(), dict)
        g.close()  # must not raise
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_synthetic_games.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'ludus.synthetic'`

- [ ] **Step 3: Write the implementation**

```python
# ludus/synthetic.py
"""Synthetic games — pure-Python stand-ins that speak the GameWorld client
interface (raw_state/apply/metrics/screenshot). Deterministic, no browser.

These are the CI ground truth for the onboarding pipeline (M1) and, later,
for world-model induction (M2): we KNOW the true controls/metrics/dynamics,
so tests can assert the pipeline recovers them exactly.

State dicts mimic GameWorld's getState() shape (docs/DISCOVERY.md):
top-level `status` in {"playing","terminal"}, `metrics`, nested `game_state`.
"""

from __future__ import annotations


class _SyntheticBase:
    """Shared client-interface plumbing (see GameWorldClient for the contract)."""

    def screenshot(self) -> bytes:
        return b"\x89PNG synthetic-frame"

    def metrics(self, keys: list[str]) -> dict[str, float]:
        m = self.raw_state().get("metrics", {})
        return {k: float(m.get(k, 0.0)) for k in keys}

    def read_partner_actions(self) -> list[str]:
        return []

    def raw_state(self) -> dict:  # pragma: no cover - overridden
        raise NotImplementedError

    def apply(self, command: dict) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def close(self) -> None:
        pass


class GridWorldGame(_SyntheticBase):
    """5x5 grid. Arrows move the player (clamped at walls). Reaching the goal
    scores +10 and the goal respawns along a fixed deterministic cycle.
    'q' ends the game (status -> terminal). Every apply() ticks `steps`."""

    SIZE = 5
    GOAL_CYCLE = [(3, 2), (0, 0), (4, 4), (0, 4), (4, 0)]
    MOVES = {"ArrowLeft": (-1, 0), "ArrowRight": (1, 0),
             "ArrowUp": (0, -1), "ArrowDown": (0, 1)}

    def __init__(self) -> None:
        self.x, self.y = 2, 2
        self._goal_i = 0
        self.score = 0
        self.steps = 0
        self.terminal = False

    @property
    def _goal(self) -> tuple[int, int]:
        return self.GOAL_CYCLE[self._goal_i % len(self.GOAL_CYCLE)]

    def apply(self, command: dict) -> None:
        if self.terminal:
            return
        self.steps += 1
        key = command.get("key")
        if key == "q":
            self.terminal = True
            return
        if key in self.MOVES:
            dx, dy = self.MOVES[key]
            self.x = min(self.SIZE - 1, max(0, self.x + dx))
            self.y = min(self.SIZE - 1, max(0, self.y + dy))
            if (self.x, self.y) == self._goal:
                self.score += 10
                self._goal_i += 1

    def raw_state(self) -> dict:
        gx, gy = self._goal
        return {
            "status": "terminal" if self.terminal else "playing",
            "metrics": {"score": self.score, "steps": self.steps},
            "game_state": {
                "player": {"x": self.x, "y": self.y},
                "environment": {"goal": {"x": gx, "y": gy}},
            },
        }


class CounterGame(_SyntheticBase):
    """'x' increments score. 'z' costs 10 health (floor 0 = terminal).
    Arrows do nothing. Every apply() ticks `steps`."""

    def __init__(self) -> None:
        self.score = 0
        self.health = 100
        self.steps = 0

    def apply(self, command: dict) -> None:
        if self.health <= 0:
            return
        self.steps += 1
        key = command.get("key")
        if key == "x":
            self.score += 1
        elif key == "z":
            self.health = max(0, self.health - 10)

    def raw_state(self) -> dict:
        return {
            "status": "terminal" if self.health <= 0 else "playing",
            "metrics": {"score": self.score, "health": self.health,
                        "steps": self.steps},
            "game_state": {"player": {"health": self.health}},
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_synthetic_games.py -q`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add ludus/synthetic.py tests/test_synthetic_games.py
git commit -m "feat(synthetic): deterministic ground-truth games for onboarding CI

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: StateSource protocol + raw_state() on real clients

The pluggable state seam from the spec. `GameWorldClient` gets a public `raw_state()` (it already fetches this internally for `state_text`); `FakeGameWorld` gets a minimal one so legacy `--fake` paths stay compatible.

**Files:**
- Create: `ludus/state_source.py`
- Modify: `ludus/gameworld_client.py` (two additions, locations below)
- Test: `tests/test_raw_state.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_raw_state.py
"""StateSource seam: every playable client exposes raw_state() -> dict."""

from ludus.gameworld_client import FakeGameWorld, GameWorldClient
from ludus.state_source import StateSource
from ludus.synthetic import CounterGame, GridWorldGame


def test_synthetic_and_fake_satisfy_statesource_protocol():
    for obj in (GridWorldGame(), CounterGame(), FakeGameWorld([{"score": 1.0}])):
        assert isinstance(obj, StateSource)


def test_fake_gameworld_raw_state_reflects_current_metrics():
    fake = FakeGameWorld([{"score": 3.0}, {"score": 7.0}])
    st = fake.raw_state()
    assert st["status"] == "playing"
    assert st["metrics"]["score"] == 3.0
    fake.apply({"key": "x", "duration_ms": 10})
    assert fake.raw_state()["metrics"]["score"] == 7.0


def test_gameworld_client_raw_state_delegates_to_manager(monkeypatch):
    # No browser: patch the async bridge and assert delegation + None-guard.
    client = GameWorldClient.__new__(GameWorldClient)  # skip __init__ (no threads)
    monkeypatch.setattr(client, "_ensure_started", lambda: None, raising=False)

    class _Mgr:
        async def get_game_state(self):
            return {"status": "playing", "metrics": {"score": 42}}

    client._mgr = _Mgr()
    monkeypatch.setattr(
        client, "_submit",
        lambda coro: __import__("asyncio").get_event_loop().run_until_complete(coro),
        raising=False,
    )
    assert client.raw_state()["metrics"]["score"] == 42
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_raw_state.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'ludus.state_source'`

- [ ] **Step 3: Write the implementation**

```python
# ludus/state_source.py
"""StateSource — the pluggable state-extraction seam (spec commitment #1).

Anything that can report the game's current structured state satisfies this.
Today: GameWorldClient (window.gameAPI.getState()), FakeGameWorld, synthetic
games. Later: a vision-based extractor for games with no state API — nothing
downstream changes.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class StateSource(Protocol):
    def raw_state(self) -> dict: ...
```

In `ludus/gameworld_client.py`, add to `GameWorldClient` directly after the existing `state_text` method (around line 492):

```python
    def raw_state(self) -> dict:
        """The raw getState() dict — the StateSource seam for onboarding/induction."""
        self._ensure_started()
        return self._submit(self._mgr.get_game_state()) or {}
```

In `ludus/gameworld_client.py`, add to `FakeGameWorld` after its `state_text` method (end of class):

```python
    def raw_state(self) -> dict:
        m = self._metrics[min(self._i, len(self._metrics) - 1)]
        return {"status": "playing", "metrics": dict(m), "game_state": {}}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_raw_state.py -q`
Expected: 3 passed

- [ ] **Step 5: Run the whole suite (no regressions)**

Run: `.venv/bin/pytest -q --ignore=infra --ignore=gameworld`
Expected: all pass (128 existing + new)

- [ ] **Step 6: Commit**

```bash
git add ludus/state_source.py ludus/gameworld_client.py tests/test_raw_state.py
git commit -m "feat(state): StateSource protocol + raw_state() on all clients

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: GameProfile model

The machine-generated replacement for `configs/*.yaml`. Bridges to the existing loop via `to_game_config()`.

**Files:**
- Create: `ludus/onboarding/profile.py`
- Test: `tests/test_game_profile.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_game_profile.py
"""GameProfile — machine-generated per-game config, cached as JSON."""

import json

from ludus.config import GameConfig
from ludus.onboarding.profile import GameProfile


def _profile() -> GameProfile:
    return GameProfile(
        game_id="gridworld",
        controls={"move_left": "ArrowLeft", "move_right": "ArrowRight"},
        control_effects={"move_left": ["game_state.player.x"],
                         "move_right": ["game_state.player.x"]},
        metrics=["score", "steps"],
        primary_metric="score",
        higher_is_better=True,
        timing_ms=120,
        objective="Maximize score. Controls: move_left (ArrowLeft), move_right (ArrowRight).",
        state_schema={"metrics.score": "int", "game_state.player.x": "int"},
    )


def test_roundtrips_through_json_file(tmp_path):
    p = _profile()
    path = tmp_path / "gridworld.json"
    p.save(path)
    loaded = GameProfile.load(path)
    assert loaded == p
    # file is plain JSON (inspectable artifact, spec commitment #2)
    assert json.loads(path.read_text())["game_id"] == "gridworld"


def test_to_game_config_bridges_to_existing_loop():
    cfg = _profile().to_game_config()
    assert isinstance(cfg, GameConfig)
    assert cfg.name == "gridworld"
    assert cfg.legal_actions == ["move_left", "move_right"]
    assert cfg.control_map == {"move_left": "ArrowLeft", "move_right": "ArrowRight"}
    assert cfg.primary_metric == "score"
    assert cfg.relevant_metrics == ["score", "steps"]
    assert cfg.timing_ms == 120
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_game_profile.py -q`
Expected: FAIL — `ModuleNotFoundError` (no `ludus.onboarding.profile`)

- [ ] **Step 3: Write the implementation**

```python
# ludus/onboarding/profile.py
"""GameProfile — the machine-generated replacement for configs/*.yaml.

Produced by the onboarding pipeline (prober + metric finder), cached as plain
JSON under profiles/<game>.json, and bridged into the existing episode loop
via to_game_config(). The profile is an inspectable artifact: everything the
system believes about a game's interface, with evidence (control_effects).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from ludus.config import GameConfig


class GameProfile(BaseModel):
    game_id: str
    controls: dict[str, str]                 # semantic action -> playwright key
    control_effects: dict[str, list[str]]    # semantic action -> changed state paths (evidence)
    metrics: list[str]                       # numeric metric names (from metrics.*)
    primary_metric: str
    higher_is_better: bool = True
    timing_ms: int = 120
    objective: str
    state_schema: dict[str, str]             # flattened path -> python type name

    def to_game_config(self) -> GameConfig:
        return GameConfig(
            name=self.game_id,
            objective=self.objective,
            legal_actions=list(self.controls),
            primary_metric=self.primary_metric,
            higher_is_better=self.higher_is_better,
            timing_ms=self.timing_ms,
            control_map=dict(self.controls),
            relevant_metrics=list(self.metrics),
        )

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.model_dump_json(indent=2) + "\n")

    @classmethod
    def load(cls, path: Path | str) -> "GameProfile":
        return cls.model_validate_json(Path(path).read_text())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_game_profile.py -q`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add ludus/onboarding/profile.py tests/test_game_profile.py
git commit -m "feat(onboarding): GameProfile model with GameConfig bridge

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: ControlProber

The heart of M1. Presses each candidate key several times, diffs state around each press, subtracts ambient drift (fields that change with NO input — gravity, timers), keeps effects that repeat consistently, names them heuristically, and survives quit/reset keys by relaunching via a client factory.

**Files:**
- Create: `ludus/onboarding/prober.py`
- Test: `tests/test_prober.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_prober.py
"""ControlProber against synthetic ground truth."""

from ludus.onboarding.prober import DEFAULT_PROBE_KEYS, ControlProber
from ludus.synthetic import CounterGame, GridWorldGame


def test_discovers_gridworld_movement_controls_with_semantic_names():
    prober = ControlProber(GridWorldGame, probe_keys=DEFAULT_PROBE_KEYS)
    report = prober.probe()
    controls = report.controls  # semantic -> key
    assert controls["move_left"] == "ArrowLeft"
    assert controls["move_right"] == "ArrowRight"
    assert controls["move_up"] == "ArrowUp"
    assert controls["move_down"] == "ArrowDown"


def test_noop_keys_are_excluded():
    report = ControlProber(GridWorldGame, probe_keys=["ArrowLeft", "z"]).probe()
    assert "ArrowLeft" in report.controls.values()
    assert "z" not in report.controls.values()


def test_ambient_step_counter_not_attributed_to_keys():
    # metrics.steps ticks on EVERY apply -> it must not appear as a key effect.
    report = ControlProber(GridWorldGame, probe_keys=["ArrowRight"]).probe()
    effects = report.control_effects["move_right"]
    assert "game_state.player.x" in effects
    assert "metrics.steps" not in effects


def test_counter_game_action_key_gets_use_name():
    report = ControlProber(CounterGame, probe_keys=["x", "ArrowLeft"]).probe()
    assert report.controls == {"use_x": "x"}  # arrows do nothing in CounterGame


def test_prober_survives_quit_key_via_relaunch():
    # 'q' makes GridWorld terminal; the prober must relaunch and keep probing.
    report = ControlProber(GridWorldGame, probe_keys=["q", "ArrowRight"]).probe()
    assert report.controls.get("move_right") == "ArrowRight"
    assert report.relaunches >= 1
    # 'q' itself only flips status -> excluded from controls
    assert "q" not in report.controls.values()


def test_state_schema_captured():
    report = ControlProber(GridWorldGame, probe_keys=["ArrowRight"]).probe()
    assert report.state_schema["metrics.score"] == "int"
    assert report.state_schema["game_state.player.x"] == "int"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_prober.py -q`
Expected: FAIL — `ModuleNotFoundError` (no `ludus.onboarding.prober`)

- [ ] **Step 3: Write the implementation**

```python
# ludus/onboarding/prober.py
"""ControlProber — discover a game's controls by pressing keys and diffing state.

Algorithm per probe() run:
  1. Ambient baseline: take consecutive raw_state() snapshots with NO input;
     any path that changes is "ambient" (gravity, timers, animations) and is
     never attributed to a key. metrics.steps-style tick counters are also
     excluded per-press (they change on every apply).
  2. Per key, `repeats` times: snapshot -> press -> snapshot -> diff. A path
     counts as a key effect if it changes in a majority of repeats and is not
     ambient. If the game leaves "playing" status (quit/reset keys, game
     over), relaunch via the client factory and continue.
  3. Heuristic semantic naming: player-position effects become move_left/
     right/up/down (screen convention: y grows downward); any other consistent
     effect becomes use_<key>. Keys with no consistent effect are dropped.

The prober needs a client FACTORY (not an instance) precisely because probing
can kill a game session. Works on anything with raw_state() + apply() —
GameWorldClient and the synthetic games alike.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ludus.onboarding.diff import diff_states, flatten_state

# GameWorld games draw from a small key vocabulary (docs/DISCOVERY.md + the
# three hand-written configs). Order matters: earlier keys claim semantic
# names first on collision.
DEFAULT_PROBE_KEYS = [
    "ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown",
    "Space", "x", "z", "a", "d", "w", "s",
]

# Tick counters change on every apply(); never attribute them to a key.
_TICK_HINTS = ("steps", "frame", "tick", "time")


@dataclass
class ProbeReport:
    controls: dict[str, str] = field(default_factory=dict)         # semantic -> key
    control_effects: dict[str, list[str]] = field(default_factory=dict)
    state_schema: dict[str, str] = field(default_factory=dict)     # path -> type name
    ambient_paths: list[str] = field(default_factory=list)
    relaunches: int = 0


def _status(state: dict) -> str:
    return str(state.get("status", "playing"))


def _is_tick(path: str) -> bool:
    leaf = path.rsplit(".", 1)[-1].lower()
    return any(h in leaf for h in _TICK_HINTS)


class ControlProber:
    def __init__(
        self,
        client_factory,
        probe_keys: list[str] | None = None,
        repeats: int = 3,
        duration_ms: int = 80,
    ) -> None:
        self._factory = client_factory
        self._keys = list(probe_keys or DEFAULT_PROBE_KEYS)
        self._repeats = repeats
        self._duration_ms = duration_ms

    def probe(self) -> ProbeReport:
        report = ProbeReport()
        client = self._factory()
        try:
            client, ambient = self._ambient_baseline(client, report)
            report.ambient_paths = sorted(ambient)
            report.state_schema = {
                p: type(v).__name__
                for p, v in flatten_state(client.raw_state()).items()
            }

            key_effects: dict[str, dict[str, list]] = {}
            for key in self._keys:
                client, effects = self._probe_key(client, key, ambient, report)
                if effects:
                    key_effects[key] = effects

            self._name_controls(key_effects, report)
            return report
        finally:
            if hasattr(client, "close"):
                client.close()

    # -- phases ---------------------------------------------------------------

    def _relaunch(self, client, report: ProbeReport):
        if hasattr(client, "close"):
            client.close()
        report.relaunches += 1
        return self._factory()

    def _ambient_baseline(self, client, report: ProbeReport):
        """Paths that change with NO input across `repeats` snapshot pairs."""
        ambient: set[str] = set()
        for _ in range(self._repeats):
            if _status(client.raw_state()) != "playing":
                client = self._relaunch(client, report)
            before = flatten_state(client.raw_state())
            after = flatten_state(client.raw_state())
            ambient |= set(diff_states(before, after))
        return client, ambient

    def _probe_key(self, client, key: str, ambient: set, report: ProbeReport):
        """Press `key` `repeats` times; return {path: [deltas]} for paths that
        changed in a majority of presses (ambient/tick paths excluded)."""
        changed_counts: dict[str, int] = {}
        deltas: dict[str, list] = {}
        for _ in range(self._repeats):
            if _status(client.raw_state()) != "playing":
                client = self._relaunch(client, report)
            before = flatten_state(client.raw_state())
            client.apply({"key": key, "duration_ms": self._duration_ms})
            after = flatten_state(client.raw_state())
            for path, (b, a) in diff_states(before, after).items():
                if path in ambient or _is_tick(path) or path == "status":
                    continue
                changed_counts[path] = changed_counts.get(path, 0) + 1
                if isinstance(b, (int, float)) and isinstance(a, (int, float)):
                    deltas.setdefault(path, []).append(a - b)
                else:
                    deltas.setdefault(path, []).append(None)
        majority = math.ceil(self._repeats / 2)
        return client, {
            p: deltas[p] for p, n in changed_counts.items() if n >= majority
        }

    # -- naming -----------------------------------------------------------------

    @staticmethod
    def _movement_name(effects: dict[str, list]) -> str | None:
        """move_left/right/up/down if the dominant effect is a player-position
        axis with a consistent sign. Screen convention: y grows DOWNWARD."""
        for axis, neg, pos in (("x", "move_left", "move_right"),
                               ("y", "move_up", "move_down")):
            paths = [p for p in effects
                     if "player" in p and p.endswith(f".{axis}")]
            for p in paths:
                nums = [d for d in effects[p] if isinstance(d, (int, float))]
                if nums and all(d < 0 for d in nums):
                    return neg
                if nums and all(d > 0 for d in nums):
                    return pos
        return None

    def _name_controls(self, key_effects: dict, report: ProbeReport) -> None:
        for key in self._keys:  # deterministic claim order
            effects = key_effects.get(key)
            if not effects:
                continue
            name = self._movement_name(effects) or f"use_{key.lower()}"
            if name in report.controls:  # collision: later key gets suffixed
                name = f"{name}_{key.lower()}"
            report.controls[name] = key
            report.control_effects[name] = sorted(effects)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_prober.py -q`
Expected: 6 passed

Note on `test_prober_survives_quit_key_via_relaunch`: 'q' flips only `status`
(excluded from effects) plus tick counters, so it yields no controls; the next
probe sees terminal status and relaunches. If the assertion on `relaunches`
fails, check that `_probe_key` re-reads status at the START of each repeat.

- [ ] **Step 5: Commit**

```bash
git add ludus/onboarding/prober.py tests/test_prober.py
git commit -m "feat(onboarding): ControlProber — key discovery with ambient-drift filter and relaunch safety

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: MetricFinder

Picks metrics + primary metric from probe evidence. Name priors first (a field literally called `score` IS the score — GameWorld exposes curated `metrics.*`), observed-change frequency as tiebreak and fallback.

**Files:**
- Create: `ludus/onboarding/metrics.py`
- Test: `tests/test_metric_finder.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_metric_finder.py
"""MetricFinder — primary-metric selection from state schema + probe evidence."""

from ludus.onboarding.metrics import find_metrics


def test_score_name_prior_wins():
    schema = {"metrics.score": "int", "metrics.steps": "int",
              "game_state.player.x": "int"}
    result = find_metrics(schema, control_effects={})
    assert result.primary_metric == "score"
    assert result.higher_is_better is True
    assert "score" in result.metrics and "steps" in result.metrics


def test_only_metrics_namespace_fields_become_metrics():
    schema = {"metrics.score": "int", "game_state.player.x": "int",
              "status": "str"}
    result = find_metrics(schema, control_effects={})
    assert result.metrics == ["score"]


def test_kills_prior_beats_generic_counter():
    schema = {"metrics.kills": "int", "metrics.steps": "int"}
    result = find_metrics(schema, control_effects={})
    assert result.primary_metric == "kills"


def test_fallback_prefers_key_affected_metric():
    # No name-prior match: prefer a metric some control actually changes.
    schema = {"metrics.blorbs": "int", "metrics.steps": "int"}
    effects = {"use_x": ["metrics.blorbs"]}
    result = find_metrics(schema, control_effects=effects)
    assert result.primary_metric == "blorbs"


def test_health_like_metrics_never_primary_when_alternative_exists():
    schema = {"metrics.health": "int", "metrics.score": "int"}
    result = find_metrics(schema, control_effects={})
    assert result.primary_metric == "score"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_metric_finder.py -q`
Expected: FAIL — `ModuleNotFoundError` (no `ludus.onboarding.metrics`)

- [ ] **Step 3: Write the implementation**

```python
# ludus/onboarding/metrics.py
"""MetricFinder — choose the game's metrics and primary metric.

GameWorld curates a `metrics.*` namespace in getState(), so metric discovery
is namespace filtering + primary selection. Selection order:
  1. Name priors (score/kills/points/... in priority order) — a field literally
     named `score` IS the score.
  2. Fallback: a metric some discovered control changes (probe evidence).
  3. Last resort: first non-vital numeric metric alphabetically.
Vitals (health/lives/ammo/time) are relevant metrics but never primary unless
they are the only thing there.
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_metric_finder.py -q`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add ludus/onboarding/metrics.py tests/test_metric_finder.py
git commit -m "feat(onboarding): MetricFinder — name priors + probe-evidence fallback

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 7: Onboard orchestrator (probe → metrics → objective → GameProfile)

**Files:**
- Create: `ludus/onboarding/onboard.py`
- Test: `tests/test_onboard_orchestrator.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_onboard_orchestrator.py
"""End-to-end offline onboarding: synthetic game -> GameProfile -> playable."""

from ludus.adapters.generic import GenericAdapter  # Task 8 provides this
from ludus.onboarding.onboard import onboard_game
from ludus.onboarding.profile import GameProfile
from ludus.synthetic import GridWorldGame


def test_onboard_gridworld_produces_complete_profile(tmp_path):
    profile = onboard_game(
        "gridworld", GridWorldGame,
        probe_keys=["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "z"],
        out_dir=tmp_path,
    )
    assert profile.game_id == "gridworld"
    assert profile.controls == {
        "move_left": "ArrowLeft", "move_right": "ArrowRight",
        "move_up": "ArrowUp", "move_down": "ArrowDown",
    }
    assert profile.primary_metric == "score"
    assert profile.higher_is_better is True
    assert "score" in profile.objective
    assert "move_left (ArrowLeft)" in profile.objective
    # cached to disk
    assert GameProfile.load(tmp_path / "gridworld.json") == profile


def test_onboarded_profile_drives_a_real_episode_offline(tmp_path):
    """The M1 acceptance test: onboard with zero hand-written config, then
    run the EXISTING episode loop against the same game via GenericAdapter."""
    from ludus.loop import run_episode
    from ludus.persistence.local import LocalStore
    from ludus.providers.mock import MockProvider
    from ludus.rulebook import Rulebook

    profile = onboard_game(
        "gridworld", GridWorldGame,
        probe_keys=["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"],
        out_dir=tmp_path,
    )
    adapter = GenericAdapter(profile.to_game_config())
    game = GridWorldGame()
    result = run_episode(
        adapter=adapter, provider=MockProvider(), gameworld=game,
        store=LocalStore(root=tmp_path / "runs"), rulebook=Rulebook(),
        mode="baseline", max_steps=3, episode_id="gridworld-onboard-test",
    )
    assert result.steps == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_onboard_orchestrator.py -q`
Expected: FAIL — `ModuleNotFoundError` (no `ludus.adapters.generic` / `ludus.onboarding.onboard`)

**Prerequisite check before Step 3:** open `ludus/providers/mock.py` and `ludus/persistence/local.py` and confirm the constructor signatures used above (`MockProvider()`, `LocalStore(root=...)`) and `run_episode`'s parameter names match `ludus/loop.py`. If they differ, adjust the TEST to the real signatures — the loop and stores are reused as-is, never modified for M1. MockProvider's fixed Decision uses an action that may not be in the profile's legal_actions; if `run_episode` requires legal actions for progress, construct `MockProvider` with an action from the profile (check `tests/test_loop.py` for the established pattern).

- [ ] **Step 3: Write the implementation**

```python
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
    goal = "Maximize" if higher_is_better else "Minimize"
    control_list = ", ".join(f"{sem} ({key})" for sem, key in controls.items())
    return (
        f"You are playing {game_id}. {goal} {primary_metric}. "
        f"Available controls: {control_list}. "
        f"Each turn, return an \"actions\" sequence of 1-5 of these controls "
        f"that makes progress toward {goal.lower()}ing {primary_metric}."
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
```

Task 8's `GenericAdapter` is also required by these tests — if executing strictly in order, write Task 8's adapter file now (it is 15 lines), then run both test files together.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_onboard_orchestrator.py tests/test_generic_adapter.py -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add ludus/onboarding/onboard.py tests/test_onboard_orchestrator.py
git commit -m "feat(onboarding): orchestrator — probe to saved GameProfile

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 8: GenericAdapter

**Files:**
- Create: `ludus/adapters/generic.py`
- Test: `tests/test_generic_adapter.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_generic_adapter.py
"""GenericAdapter — GameAdapter protocol from any GameConfig, no game class."""

from ludus.adapters.generic import GenericAdapter
from ludus.config import GameConfig
from ludus.schemas import ActionArgs


def _cfg() -> GameConfig:
    return GameConfig(
        name="gridworld", objective="Maximize score.",
        legal_actions=["move_left", "move_right"],
        primary_metric="score", higher_is_better=True, timing_ms=100,
        control_map={"move_left": "ArrowLeft", "move_right": "ArrowRight"},
        relevant_metrics=["score", "steps"],
    )


def test_exposes_adapter_protocol_fields():
    a = GenericAdapter(_cfg())
    assert a.name == "gridworld"
    assert a.legal_actions == ["move_left", "move_right"]
    assert a.primary_metric == "score"
    assert a.higher_is_better is True
    assert a.relevant_metrics == ["score", "steps"]
    assert a.timing_ms == 100


def test_semantic_to_gameworld_maps_key_and_duration():
    a = GenericAdapter(_cfg())
    cmd = a.semantic_to_gameworld("move_left", ActionArgs(duration_ms=100))
    assert cmd == {"key": "ArrowLeft", "duration_ms": 100}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_generic_adapter.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'ludus.adapters.generic'`
(May already exist if Task 7 required it early — then expect PASS and skip to Step 5.)

- [ ] **Step 3: Write the implementation**

```python
# ludus/adapters/generic.py
"""GenericAdapter — GameAdapter for machine-generated GameProfiles.

Identical shape to the hand-written adapters (see tetris.py): mapping +
metadata only, zero strategy. The difference: it needs no per-game class,
so onboarded games are playable without writing code."""

from ludus.config import GameConfig
from ludus.schemas import ActionArgs


class GenericAdapter:
    def __init__(self, cfg: GameConfig) -> None:
        self._cfg = cfg
        self.name = cfg.name
        self.objective = cfg.objective
        self.legal_actions = cfg.legal_actions
        self.primary_metric = cfg.primary_metric
        self.higher_is_better = cfg.higher_is_better
        self.relevant_metrics = cfg.relevant_metrics
        self.timing_ms = cfg.timing_ms

    def semantic_to_gameworld(self, action: str, args: ActionArgs) -> dict:
        return {"key": self._cfg.control_map[action], "duration_ms": args.duration_ms}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_generic_adapter.py -q`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add ludus/adapters/generic.py tests/test_generic_adapter.py
git commit -m "feat(adapters): GenericAdapter — play any onboarded profile without per-game code

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 9: CLI — `onboard` subcommand and `--profile` play path

Backward compatible: `python -m ludus.cli tetris baseline ...` keeps working. New: `python -m ludus.cli onboard <game>` and `python -m ludus.cli <game> <mode> --profile profiles/<game>.json`.

**Files:**
- Modify: `ludus/cli.py`
- Test: `tests/test_cli_onboard.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cli_onboard.py
"""CLI onboarding path — offline via synthetic games."""

from ludus.cli import cmd_onboard
from ludus.onboarding.profile import GameProfile


def test_cmd_onboard_synthetic_writes_profile(tmp_path, capsys):
    cmd_onboard(game="synthetic:grid", out_dir=tmp_path)
    profile = GameProfile.load(tmp_path / "synthetic:grid.json")
    assert profile.controls["move_right"] == "ArrowRight"
    assert profile.primary_metric == "score"
    out = capsys.readouterr().out
    assert "move_right" in out          # human-readable summary printed
    assert "score" in out


def test_cmd_onboard_counter_synthetic(tmp_path):
    cmd_onboard(game="synthetic:counter", out_dir=tmp_path)
    profile = GameProfile.load(tmp_path / "synthetic:counter.json")
    assert profile.controls == {"use_x": "x", "use_z": "z"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_cli_onboard.py -q`
Expected: FAIL — `ImportError: cannot import name 'cmd_onboard'`

- [ ] **Step 3: Modify `ludus/cli.py`**

Add imports at the top (after the existing imports, line 14):

```python
from ludus.adapters.generic import GenericAdapter
from ludus.onboarding.onboard import onboard_game
from ludus.onboarding.profile import GameProfile
```

Add after `_load_dotenv` (line 36):

```python
# Synthetic games are onboardable/playable offline via "synthetic:<name>" ids —
# used by tests and demos; real ids go through GameWorldClient.
def _client_factory(game: str, timing_ms: int, headless: bool):
    if game == "synthetic:grid":
        from ludus.synthetic import GridWorldGame
        return GridWorldGame
    if game == "synthetic:counter":
        from ludus.synthetic import CounterGame
        return CounterGame
    gw_id = GAMEWORLD_IDS.get(game, game)  # raw catalog ids (e.g. 05_snake) pass through
    return lambda: GameWorldClient(game_id=gw_id, timing_ms=timing_ms,
                                   headless=headless)


def cmd_onboard(game: str, out_dir="profiles", headless: bool = True) -> None:
    factory = _client_factory(game, timing_ms=120, headless=headless)
    profile = onboard_game(game, factory, out_dir=out_dir)
    print(f"onboarded {game} -> {out_dir}/{game}.json")
    print(f"  controls : {profile.controls}")
    print(f"  metrics  : {profile.metrics} (primary={profile.primary_metric})")
    print(f"  objective: {profile.objective}")
```

Replace the body of `main()` with subcommand routing (keeping the legacy play path byte-for-byte, plus the `--profile` option):

```python
def main() -> None:
    _load_dotenv()
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "onboard":
        ap = argparse.ArgumentParser(prog="ludus onboard")
        ap.add_argument("command")            # consumes the literal "onboard"
        ap.add_argument("game", help="adapter name, raw GameWorld id, or synthetic:<name>")
        ap.add_argument("--out", default="profiles")
        ap.add_argument("--headed", action="store_true")
        args = ap.parse_args()
        cmd_onboard(args.game, out_dir=args.out, headless=not args.headed)
        return

    ap = argparse.ArgumentParser()
    ap.add_argument("game")
    ap.add_argument("mode", choices=["baseline", "memory"])
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--provider", default="mock", choices=["mock", "gateway", "anthropic", "nebius", "fireworks", "fallback"])
    ap.add_argument("--fake", action="store_true", help="use FakeGameWorld (no browser)")
    ap.add_argument("--headed", action="store_true", help="show the browser window (great for demos)")
    ap.add_argument("--profile", type=Path, default=None,
                    help="play via a generated GameProfile instead of configs/<game>.yaml")
    args = ap.parse_args()

    if args.profile:
        cfg = GameProfile.load(args.profile).to_game_config()
        adapter = GenericAdapter(cfg)
    else:
        cfg = load_game_config(Path("configs") / f"{args.game}.yaml")
        adapter = ADAPTERS[args.game](cfg)
    provider = build_provider(args.provider)
    gw = FakeGameWorld([{cfg.primary_metric: 0.0}]) if args.fake else GameWorldClient(
        game_id=GAMEWORLD_IDS.get(cfg.name, cfg.name), timing_ms=cfg.timing_ms,
        headless=not args.headed)
    store = DualWriteStore(primary=LocalStore(), secondary=InsForgeStore())
    try:
        result = run_episode(adapter=adapter, provider=provider, gameworld=gw,
            store=store, rulebook=Rulebook(), mode=args.mode,
            max_steps=args.steps, episode_id=f"{args.game}-{args.mode}")
    finally:
        if hasattr(gw, "close"):
            gw.close()
    print(result.model_dump_json(indent=2))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_cli_onboard.py -q`
Expected: 2 passed

- [ ] **Step 5: Run the whole suite (legacy CLI path untouched)**

Run: `.venv/bin/pytest -q --ignore=infra --ignore=gameworld`
Expected: all pass

- [ ] **Step 6: Sanity-run the offline CLI end-to-end**

Run: `.venv/bin/python -m ludus.cli onboard synthetic:grid --out /tmp/profiles && cat /tmp/profiles/synthetic:grid.json`
Expected: printed control/metric summary; JSON profile with 4 movement controls, primary_metric "score".

- [ ] **Step 7: Commit**

```bash
git add ludus/cli.py tests/test_cli_onboard.py
git commit -m "feat(cli): onboard subcommand + --profile play path

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 10: Live acceptance + docs

M1 acceptance from the spec: `onboard` works on 5+ real GameWorld games with zero hand-written config, and a VLM plays via the generated profile. Requires the live env (Playwright lives in the anaconda python — see CLAUDE.md gotchas).

- [ ] **Step 1: Onboard 5 real games (live, anaconda python)**

Run (from repo root; pick ids present in `gameworld/games/benchmark/`):
```bash
for g in 29_tetris 05_snake 01_2048 31_wolf3d 03_astray; do
  /opt/anaconda3/bin/python -m ludus.cli onboard "$g" || echo "FAILED: $g"
done
ls profiles/
```
Expected: ≥5 profile JSONs. Inspect each: controls non-empty, primary_metric sensible. Record any failures verbatim — a game the prober can't onboard is a finding, not something to hide (spec: honest coverage claims).

- [ ] **Step 2: Play one onboarded game via the gateway VLM (live)**

Run: `/opt/anaconda3/bin/python -m ludus.cli 05_snake baseline --profile profiles/05_snake.json --provider gateway --steps 10`
Expected: episode completes, `legal_action_rate` > 0, score movement visible. This is the M1 demo.

- [ ] **Step 3: Commit generated profiles**

```bash
git add profiles/
git commit -m "feat(profiles): first auto-onboarded GameWorld profiles (M1 acceptance)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

- [ ] **Step 4: Update docs**

In `README.md`, under the Roadmap section, mark the M1 line (add below existing roadmap items):

```markdown
6. **Magus-1 (in progress)** — see `docs/specs/2026-07-04-magus-1-induced-world-models-design.md`.
   M1 auto-onboarding: ✅ `python -m ludus.cli onboard <game>` generates a playable
   GameProfile (controls, metrics, objective) with zero hand-written config.
```

In `CLAUDE.md`, add to the CLI section:

```markdown
- `python -m ludus.cli onboard <game>` — auto-discover controls/metrics -> `profiles/<game>.json`
  (`synthetic:grid` / `synthetic:counter` run offline). Play an onboarded game with
  `python -m ludus.cli <game> <mode> --profile profiles/<game>.json`.
```

- [ ] **Step 5: Final suite + commit**

Run: `.venv/bin/pytest -q --ignore=infra --ignore=gameworld`
Expected: all pass

```bash
git add README.md CLAUDE.md
git commit -m "docs: M1 auto-onboarding usage + roadmap status

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Self-review notes (already applied)

- **Spec coverage:** StateSource seam (Task 3), prober with safety relaunch (Task 5), MetricFinder (Task 6), GameProfile replacing configs (Tasks 4/8/9), SyntheticGame scaffold (Task 2), `onboard` CLI (Task 9), M1 acceptance (Task 10). Not in M1 (per spec staging): transitions store, inducer, sandbox, planner, `explore`/`induce`/`duel` subcommands.
- **Known coupling:** Task 7's integration test imports Task 8's `GenericAdapter`; Task 7 Step 3 says to create it early if executing in order. Both tests run together in Task 7 Step 4.
- **Signature risk:** Task 7's integration test uses `MockProvider()`, `LocalStore(root=...)`, and `run_episode(...)` signatures — the prerequisite check in Task 7 Step 2 requires verifying them against `tests/test_loop.py` before implementing, adjusting the test (never the production code) on mismatch.
- **`q` is not in `DEFAULT_PROBE_KEYS`** — quit-key relaunch is exercised in tests by passing it explicitly; real GameWorld games' quit/pause keys outside the vocabulary simply never get pressed.
