# Onboarding Robustness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix three sweep-failure clusters — games stuck at start screens (ONBOARD_FAILED TimeoutError), explore crashing on empty control profiles (IndexError), and sweep mis-counting missing transition files (FileNotFoundError) — so all 355 + new TDD tests pass.

**Architecture:** Three independent, minimally-scoped fixes: (1) wake-up phase injected into `ControlProber.probe()` and `explore_game()` before the existing logic; (2) empty-controls guardrails in `onboard_game()` and `explore_game()`; (3) file-existence guard in the sweep's `explore()` wrapper. Each fix is expressed as a new constant / field / raise-path — no restructuring.

**Tech Stack:** Python 3.11, pytest, existing synthetic game framework (`ludus/synthetic.py`), no new dependencies.

---

## File Map

| File | Role |
|---|---|
| `ludus/synthetic.py` | **Add** `MenuGame` (starts "menu", Enter flips to "playing", then CounterGame behavior) |
| `ludus/onboarding/prober.py` | **Add** `WAKE_KEYS` constant + `wake_key` field to `ProbeReport` + `_wake_up()` helper called at start of `probe()` |
| `ludus/worldmodel/explore.py` | **Add** `ValueError` early-exit on empty `profile.controls`; import + call `_wake_up_client()` per-episode |
| `ludus/onboarding/onboard.py` | **Add** `SystemExit` after saving if `report.controls` is empty |
| `scripts/sweep_benchmark.py` | **Guard** `t_path.exists()` before `t_path.open()`; raise `RuntimeError` on 0 transitions |

---

### Task 1: Add `MenuGame` synthetic game

**Files:**
- Modify: `ludus/synthetic.py`
- Test: `tests/test_synthetic_games.py`

- [ ] **Step 1: Write the failing test**

Open `tests/test_synthetic_games.py` and add at the bottom:

```python
from ludus.synthetic import MenuGame

def test_menu_game_starts_in_menu_status():
    g = MenuGame()
    assert g.raw_state()["status"] == "menu"


def test_menu_game_enter_flips_to_playing():
    g = MenuGame()
    g.apply({"key": "Enter"})
    assert g.raw_state()["status"] == "playing"


def test_menu_game_x_scores_once_playing():
    g = MenuGame()
    g.apply({"key": "Enter"})
    g.apply({"key": "x"})
    assert g.raw_state()["metrics"]["score"] == 1


def test_menu_game_non_enter_keys_ignored_in_menu():
    g = MenuGame()
    g.apply({"key": "ArrowUp"})
    assert g.raw_state()["status"] == "menu"
    g.apply({"key": "Space"})
    assert g.raw_state()["status"] == "menu"
```

- [ ] **Step 2: Run the failing tests**

```bash
.venv/bin/pytest tests/test_synthetic_games.py -k "menu_game" -v
```

Expected: `ImportError: cannot import name 'MenuGame'`

- [ ] **Step 3: Implement `MenuGame` in `ludus/synthetic.py`**

Add after the `CounterGame` class (at the very bottom of the file):

```python
class MenuGame(_SyntheticBase):
    """Starts with status 'menu'; pressing Enter flips to 'playing'.
    Once playing, behaves like CounterGame: 'x' increments score.
    Used to test the prober's wake-up phase."""

    def __init__(self) -> None:
        self._status = "menu"
        self.score = 0
        self.steps = 0

    def apply(self, command: dict) -> None:
        key = command.get("key")
        if self._status == "menu":
            if key == "Enter":
                self._status = "playing"
            return
        self.steps += 1
        if key == "x":
            self.score += 1

    def raw_state(self) -> dict:
        return {
            "status": self._status,
            "metrics": {"score": self.score, "steps": self.steps},
            "game_state": {"player": {"score": self.score}},
        }
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
.venv/bin/pytest tests/test_synthetic_games.py -k "menu_game" -v
```

Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add ludus/synthetic.py tests/test_synthetic_games.py
git commit -m "test(synthetic): add MenuGame for wake-up phase TDD"
```

---

### Task 2: Add `WAKE_KEYS` and `wake_key` field to prober

**Files:**
- Modify: `ludus/onboarding/prober.py`
- Test: `tests/test_prober.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_prober.py`:

```python
from ludus.onboarding.prober import WAKE_KEYS
from ludus.synthetic import MenuGame


def test_wake_keys_constant_exists_and_includes_enter():
    assert "Enter" in WAKE_KEYS
    assert "Space" in WAKE_KEYS


def test_probe_report_has_wake_key_field_default_none():
    from ludus.onboarding.prober import ProbeReport
    r = ProbeReport()
    assert r.wake_key is None


def test_prober_sets_wake_key_none_for_already_playing_game():
    report = ControlProber(GridWorldGame, probe_keys=["ArrowRight"],
                           ambient_window_s=0.0).probe()
    assert report.wake_key is None


def test_prober_discovers_controls_on_menu_game_with_enter_wake():
    report = ControlProber(MenuGame, probe_keys=["x"], ambient_window_s=0.0).probe()
    assert report.controls == {"use_x": "x"}
    assert report.wake_key == "Enter"


def test_prober_raises_runtime_error_when_no_wake_key_works():
    class AlwaysMenuGame(MenuGame):
        """Never transitions out of menu status regardless of input."""
        def apply(self, command: dict) -> None:
            pass  # always stays "menu"

    import pytest
    with pytest.raises(RuntimeError, match="wake keys"):
        ControlProber(AlwaysMenuGame, probe_keys=["x"],
                      ambient_window_s=0.0).probe()
```

- [ ] **Step 2: Run the failing tests**

```bash
.venv/bin/pytest tests/test_prober.py -k "wake" -v
```

Expected: `AttributeError: WAKE_KEYS` / `AttributeError: wake_key`

- [ ] **Step 3: Add `WAKE_KEYS` constant and `wake_key` field to prober**

In `ludus/onboarding/prober.py`, add the constant after `DEFAULT_PROBE_KEYS`:

```python
# Keys pressed during the wake-up phase to escape start/menu screens.
# Order matters: tried in sequence, first success wins.
WAKE_KEYS = ["Space", "Enter", "ArrowUp", "x"]
```

Add `wake_key` to `ProbeReport`:

```python
@dataclass
class ProbeReport:
    controls: dict[str, str] = field(default_factory=dict)
    control_effects: dict[str, list[str]] = field(default_factory=dict)
    state_schema: dict[str, str] = field(default_factory=dict)
    ambient_paths: list[str] = field(default_factory=list)
    relaunches: int = 0
    apply_errors: int = 0
    second_chance: list[str] = field(default_factory=list)
    wake_key: str | None = None
```

- [ ] **Step 4: Run the tests to verify the easy ones pass**

```bash
.venv/bin/pytest tests/test_prober.py -k "wake_keys_constant or probe_report_has_wake_key" -v
```

Expected: 2 PASSED

- [ ] **Step 5: Commit the constant + field**

```bash
git add ludus/onboarding/prober.py tests/test_prober.py
git commit -m "feat(prober): add WAKE_KEYS constant and wake_key field to ProbeReport"
```

---

### Task 3: Implement `_wake_up()` in `ControlProber.probe()`

**Files:**
- Modify: `ludus/onboarding/prober.py`

- [ ] **Step 1: Verify the three behavioral tests still fail**

```bash
.venv/bin/pytest tests/test_prober.py -k "wake_key_none_for or menu_game or runtime_error_when" -v
```

Expected: 3 FAIL (wake-up logic not implemented yet)

- [ ] **Step 2: Add `_wake_up()` helper method and call it in `probe()`**

Add `_wake_up()` as a method of `ControlProber` (add before `_ambient_baseline`):

```python
def _wake_up(self, client, report: ProbeReport):
    """If the game is not 'playing', try WAKE_KEYS one by one.

    On first success (status flips to 'playing'), record report.wake_key
    and return. If no key works, relaunch once and retry. If still not
    playing after all keys + relaunch, raise RuntimeError.

    No-op when the game is already 'playing'.
    """
    if _status(client.raw_state()) == "playing":
        return client  # fast path — existing games are unaffected

    for key in WAKE_KEYS:
        try:
            client.apply({"key": key, "duration_ms": self._duration_ms})
        except Exception as exc:
            report.apply_errors += 1
            log.warning("wake-up apply(%r) failed (%s)", key, exc)
            continue
        import time as _time
        _time.sleep(0.5)
        if _status(client.raw_state()) == "playing":
            report.wake_key = key
            return client

    # One relaunch, then retry wake keys
    client = self._relaunch(client, report)
    for key in WAKE_KEYS:
        try:
            client.apply({"key": key, "duration_ms": self._duration_ms})
        except Exception as exc:
            report.apply_errors += 1
            log.warning("wake-up apply(%r) failed after relaunch (%s)", key, exc)
            continue
        import time as _time
        _time.sleep(0.5)
        if _status(client.raw_state()) == "playing":
            report.wake_key = key
            return client

    raise RuntimeError(
        f"game never reached 'playing' status (tried wake keys {WAKE_KEYS})"
    )
```

At the start of `probe()`, insert the wake-up call **before** `_ambient_baseline`:

Replace the `probe()` method body so it reads:

```python
def probe(self) -> ProbeReport:
    report = ProbeReport()
    client = self._factory()
    try:
        client = self._wake_up(client, report)   # NEW — no-op if already playing
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

        effective = [k for k in self._keys if k in key_effects]
        retry = [k for k in self._keys if k not in key_effects]
        n_paths = max(1, len(report.state_schema))
        scramblers = [k for k in effective
                      if len(key_effects[k]) < RESET_EFFECT_FRACTION * n_paths]
        if scramblers and retry:
            for key in retry:
                client = self._scramble(client, scramblers, report)
                client, effects = self._probe_key(client, key, ambient, report)
                if effects:
                    key_effects[key] = effects
                    report.second_chance.append(key)

        self._name_controls(key_effects, report)
        return report
    finally:
        if hasattr(client, "close"):
            client.close()
```

- [ ] **Step 3: Run all wake-up tests**

```bash
.venv/bin/pytest tests/test_prober.py -k "wake" -v
```

Expected: all 5 wake-related tests PASSED

- [ ] **Step 4: Run the full prober test suite to ensure no regressions**

```bash
.venv/bin/pytest tests/test_prober.py tests/test_prober_hardening.py -v
```

Expected: all tests PASSED

- [ ] **Step 5: Commit**

```bash
git add ludus/onboarding/prober.py
git commit -m "feat(prober): implement _wake_up() phase — exits menu screens before probing"
```

---

### Task 4: Thread wake-up into `explore_game()`

**Files:**
- Modify: `ludus/worldmodel/explore.py`
- Test: `tests/test_explore.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_explore.py`:

```python
from ludus.synthetic import MenuGame


def _menu_profile() -> GameProfile:
    return GameProfile(
        game_id="synthetic:menu",
        controls={"use_x": "x"},
        control_effects={}, metrics=["score"], primary_metric="score",
        higher_is_better=True, timing_ms=50, objective="test",
        state_schema={},
    )


def test_explore_wakes_menu_game_before_episode(tmp_path):
    """explore_game must press Enter to leave the menu before each episode."""
    store = TransitionStore(tmp_path / "t.jsonl")
    summary = explore_game(_menu_profile(), MenuGame, store=store,
                           episodes=2, steps_per_episode=3, seed=1)
    ts = store.load()
    # All before-states must be "playing" (menu was exited)
    assert all(t.before["status"] == "playing" for t in ts)
    assert summary["transitions"] == 6


def test_explore_raises_value_error_on_empty_controls(tmp_path):
    import pytest
    from ludus.onboarding.profile import GameProfile
    empty_profile = GameProfile(
        game_id="bad", controls={}, control_effects={},
        metrics=[], primary_metric="score", higher_is_better=True,
        timing_ms=50, objective="test", state_schema={},
    )
    store = TransitionStore(tmp_path / "t.jsonl")
    with pytest.raises(ValueError, match="no controls"):
        explore_game(empty_profile, MenuGame, store=store,
                     episodes=1, steps_per_episode=1, seed=0)
```

- [ ] **Step 2: Run the failing tests**

```bash
.venv/bin/pytest tests/test_explore.py -k "menu_game or empty_controls" -v
```

Expected: FAILED

- [ ] **Step 3: Update `explore_game()` with empty-controls guard and wake-up**

Replace the `explore_game()` function body in `ludus/worldmodel/explore.py`:

```python
from __future__ import annotations

import random
import time

from ludus.onboarding.prober import WAKE_KEYS
from ludus.onboarding.profile import GameProfile
from ludus.worldmodel.transitions import Transition, TransitionStore


def explore_game(
    profile: GameProfile,
    client_factory,
    *,
    store: TransitionStore,
    episodes: int,
    steps_per_episode: int,
    seed: int,
    duration_ms: int | None = None,
) -> dict:
    """Returns {"episodes": n, "transitions": n, "terminals": n}.

    Pass duration_ms to override profile.timing_ms — use short "tap" presses
    for turn-based games where key-repeat would fire multiple moves per press,
    contaminating induction transitions with multi-move dynamics.
    """
    if not profile.controls:
        raise ValueError(f"profile has no controls for {profile.game_id!r}")

    rng = random.Random(seed)
    actions = sorted(profile.controls)
    n_transitions = n_terminals = 0
    press_duration = duration_ms if duration_ms is not None else profile.timing_ms

    for ep in range(episodes):
        episode_id = f"{profile.game_id}-explore-{seed}-{ep}"
        client = client_factory()
        try:
            # Wake-up phase: if game is not playing yet, try WAKE_KEYS
            for wk in WAKE_KEYS:
                if str(client.raw_state().get("status", "playing")) == "playing":
                    break
                try:
                    client.apply({"key": wk, "duration_ms": press_duration})
                except Exception:
                    pass
                time.sleep(0.05)

            for step in range(steps_per_episode):
                before = client.raw_state()
                if str(before.get("status", "playing")) != "playing":
                    break
                action = rng.choice(actions)
                client.apply({"key": profile.controls[action],
                              "duration_ms": press_duration})
                after = client.raw_state()
                terminal = str(after.get("status", "playing")) != "playing"
                store.append(Transition(
                    game_id=profile.game_id, episode_id=episode_id, step=step,
                    action=action, key=profile.controls[action],
                    before=before, after=after, terminal_after=terminal,
                ))
                n_transitions += 1
                if terminal:
                    n_terminals += 1
                    break
        finally:
            if hasattr(client, "close"):
                client.close()

    return {"episodes": episodes, "transitions": n_transitions,
            "terminals": n_terminals}
```

- [ ] **Step 4: Run explore tests**

```bash
.venv/bin/pytest tests/test_explore.py -v
```

Expected: all tests PASSED (including new ones)

- [ ] **Step 5: Commit**

```bash
git add ludus/worldmodel/explore.py tests/test_explore.py
git commit -m "feat(explore): wake-up phase per episode + ValueError on empty controls"
```

---

### Task 5: Empty-controls guardrail in `onboard_game()`

**Files:**
- Modify: `ludus/onboarding/onboard.py`
- Test: `tests/test_onboard_orchestrator.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_onboard_orchestrator.py`:

```python
import pytest
from ludus.synthetic import CounterGame


def test_onboard_empty_controls_saves_profile_then_raises(tmp_path):
    """When probing discovers no controls, profile is saved (honest artifact)
    but onboard_game raises SystemExit with a clear message."""
    # ArrowLeft has no effect in CounterGame -> empty controls
    with pytest.raises(SystemExit, match="no controls discovered"):
        onboard_game(
            "counter-empty", CounterGame,
            probe_keys=["ArrowLeft"],   # no-op key only
            out_dir=tmp_path,
        )
    # Profile must exist on disk (honest artifact)
    saved = GameProfile.load(tmp_path / "counter-empty.json")
    assert saved.controls == {}
```

- [ ] **Step 2: Run the failing test**

```bash
.venv/bin/pytest tests/test_onboard_orchestrator.py -k "empty_controls" -v
```

Expected: FAILED (no SystemExit raised)

- [ ] **Step 3: Add the guardrail to `onboard_game()`**

Replace the `onboard_game()` function in `ludus/onboarding/onboard.py`:

```python
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
    if not report.controls:
        raise SystemExit(
            f"no controls discovered for {game_id!r} — profile saved but unplayable"
        )
    return profile
```

- [ ] **Step 4: Run the full onboard test suite**

```bash
.venv/bin/pytest tests/test_onboard_orchestrator.py tests/test_cli_onboard.py -v
```

Expected: all tests PASSED

- [ ] **Step 5: Commit**

```bash
git add ludus/onboarding/onboard.py tests/test_onboard_orchestrator.py
git commit -m "feat(onboard): raise SystemExit after saving if no controls discovered"
```

---

### Task 6: Sweep counter resilience — guard missing `transitions.jsonl`

**Files:**
- Modify: `scripts/sweep_benchmark.py`
- Test: `tests/test_sweep.py`

- [ ] **Step 1: Write the failing test**

The sweep's `explore()` wrapper is tested indirectly via `sweep_game()` in `test_sweep.py`, but the file-check logic lives in `sweep_benchmark.py` (a script, not a module). We test the relevant behavior by checking that the `sweep_game()` callable can return `EXPLORE_FAILED` with a clear message when `explore()` raises `RuntimeError`.

First verify the existing test covers `RuntimeError -> EXPLORE_FAILED` already:

```bash
.venv/bin/pytest tests/test_sweep.py -k "explore_failure" -v
```

Expected: PASSED (it tests `IOError`, but sweep catches all `Exception`)

Now add a test specifically for the "no transitions" scenario:

Add to `tests/test_sweep.py`:

```python
def test_explore_zero_transitions_gives_explore_failed():
    """Explore returning 0 transitions (missing file) must surface as EXPLORE_FAILED."""
    def zero_explore(game):
        raise RuntimeError("explore produced no transitions")

    s = _stubs(explore=zero_explore)
    rec = sweep_game("g", **{k: v for k, v in s.items() if not k.startswith("_")})
    assert rec["verdict"] == "EXPLORE_FAILED"
    assert "no transitions" in rec["error"]
```

- [ ] **Step 2: Run the test**

```bash
.venv/bin/pytest tests/test_sweep.py -k "zero_transitions" -v
```

Expected: PASSED (sweep already catches all exceptions — the test validates the contract)

- [ ] **Step 3: Guard the file-existence check in `scripts/sweep_benchmark.py`**

In `scripts/sweep_benchmark.py`, replace the `explore()` closure:

```python
    def explore(game):
        # fresh transitions per sweep run for a clean episode set
        t_path = Path("data") / game / "transitions.jsonl"
        if t_path.exists():
            t_path.unlink()
        cmd_explore(game, episodes=args.episodes, steps=args.steps,
                    duration_ms=30)
        if not t_path.exists():
            raise RuntimeError(f"explore produced no transitions for {game!r}")
        n = sum(1 for _ in t_path.open())
        if n == 0:
            raise RuntimeError(f"explore produced no transitions for {game!r}")
        return n
```

- [ ] **Step 4: Run full suite**

```bash
.venv/bin/pytest -q --ignore=infra
```

Expected: 355 + new tests PASSED (exact count: at least 367+)

- [ ] **Step 5: Commit**

```bash
git add scripts/sweep_benchmark.py tests/test_sweep.py
git commit -m "fix(sweep): guard missing transitions.jsonl; raise RuntimeError on 0 transitions"
```

---

### Task 7: Final integration — single commit for the feature

- [ ] **Step 1: Run the complete suite**

```bash
.venv/bin/pytest -v --ignore=infra 2>&1 | tail -20
```

Expected: All tests PASSED

- [ ] **Step 2: Verify the wake-up is a no-op for all existing games**

```bash
.venv/bin/pytest tests/test_prober.py tests/test_prober_hardening.py tests/test_onboard_orchestrator.py -v
```

Expected: All PASSED, `wake_key is None` for GridWorldGame tests

- [ ] **Step 3: Create the final commit**

```bash
git add -p  # stage any remaining unstaged changes
git commit -m "fix(onboarding): wake-up phase for menu games, empty-controls guardrails, sweep resilience

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**

| Requirement | Task |
|---|---|
| `WAKE_KEYS` constant exported from prober | Task 2 |
| `report.wake_key` field | Task 2 |
| `_wake_up()` in `probe()` before ambient baseline | Task 3 |
| Wake-up is no-op for already-playing games | Task 3 test |
| `MenuGame` synthetic | Task 1 |
| MenuGame + Enter -> playing test | Task 1 |
| Prober discovers `use_x` on MenuGame, `wake_key == "Enter"` | Task 3 test |
| Game never playing -> RuntimeError mentioning wake keys | Task 3 test |
| GridWorldGame probe unchanged, `wake_key None` | Task 3 test |
| Wake-up in `explore_game()` per episode | Task 4 |
| `explore_game()` raises `ValueError` on empty controls | Task 4 |
| `onboard_game()` saves profile then raises `SystemExit` on empty controls | Task 5 |
| Sweep `explore()` guards missing file | Task 6 |
| Sweep `explore()` raises `RuntimeError` on 0 transitions | Task 6 |

**Placeholder scan:** No TBD / TODO / "similar to" patterns found.

**Type consistency:**
- `WAKE_KEYS: list[str]` used consistently in prober and explore
- `report.wake_key: str | None` set only in `_wake_up()`
- `_wake_up()` takes `(self, client, report: ProbeReport) -> client` — same signature pattern as `_relaunch()`
- `explore_game()` imports `WAKE_KEYS` from `ludus.onboarding.prober` — no circular import (explore already imports from onboarding)
