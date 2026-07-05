# Magus-1 M2: World-Model Induction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Given an onboarded game, autonomously collect (state, action, next-state) transitions, have an LLM write the game's `predict()` function as Python, and validate it against held-out real transitions — producing `worldmodels/<game>/model.py` with an honest INDUCED/FAILED verdict.

**Architecture:** An `Explorer` plays uniform-random episodes via the GameProfile and logs per-press `Transition`s to a JSONL `TransitionStore` (episode-keyed train/holdout split). The `Inducer` prompts an LLM (pluggable `chat` seam: Anthropic default, InsForge gateway fallback, mock for CI) with the state schema + sample transitions to synthesize a pure-Python `predict(state, action) -> next_state`; candidate code passes an AST allowlist check, runs batched in a subprocess `Sandbox` (timeout, crash isolation), and is scored by a field-weighted `Validator` (flatten/diff reuse); validation counterexamples feed a bounded repair loop. Stochastic fields (piece spawns) are expected to score low — the gate weighs player/primary-metric fields highest and reports the rest honestly.

**Tech Stack:** Python 3.11+, pydantic v2, pytest, httpx (already a dep), stdlib `ast`/`subprocess`. No new dependencies.

**Spec:** `docs/specs/2026-07-04-magus-1-induced-world-models-design.md` (Stage 2). **Out of scope here (next plan):** planner, candidate rendering, `duel` CLI; curiosity-weighted exploration; multi-state probing for 2048's no-op keys.

**Conventions:** run tests with `.venv/bin/pytest` from `/Users/mahanth/ludus`, branch `magus-1-m1`, do not push. Commits end with `Co-Authored-By: Claude <noreply@anthropic.com>`.

---

## File structure (locked in)

```
ludus/
  synthetic.py               # MODIFY: add AutoRunnerGame (wall-clock drift game)
  onboarding/prober.py       # MODIFY: press-window ambient baseline; apply() resilience
  worldmodel/
    __init__.py              # NEW: empty
    transitions.py           # NEW: Transition model + TransitionStore (JSONL, episode split)
    explore.py               # NEW: uniform-random exploration -> TransitionStore
    sandbox.py               # NEW: AST allowlist + subprocess batch runner
    validate.py              # NEW: field-weighted accuracy vs held-out transitions
    llm.py                   # NEW: chat(system, user) seam (anthropic | gateway)
    inducer.py               # NEW: synthesize -> check -> validate -> repair loop
  cli.py                     # MODIFY: `explore` + `induce` subcommands
tests/
  test_prober_hardening.py   # NEW
  test_transitions.py        # NEW
  test_explore.py            # NEW
  test_sandbox.py            # NEW
  test_wm_validate.py        # NEW
  test_inducer.py            # NEW  (mock synthesizer; offline end-to-end)
worldmodels/                 # NEW dir: model.py + report.json per game (committed)
```

---

### Task 1: Prober hardening (M1 carry-forwards the explorer depends on)

Two live-env findings from M1: (a) continuous games (snake/flappy) scramble control names because the ambient baseline diffs two *instant* snapshots while a press spans ~200ms of world motion; (b) one Playwright hiccup in `apply()` kills a whole probe run.

**Files:**
- Modify: `ludus/synthetic.py` (add `AutoRunnerGame`)
- Modify: `ludus/onboarding/prober.py`
- Test: `tests/test_prober_hardening.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_prober_hardening.py
"""M1 live findings: press-window ambient sampling + apply() resilience."""

import pytest

from ludus.onboarding.prober import ControlProber
from ludus.synthetic import AutoRunnerGame, GridWorldGame


def test_autorunner_drifts_with_wall_clock():
    g = AutoRunnerGame(drift_hz=50)
    x0 = g.raw_state()["game_state"]["player"]["x"]
    import time
    time.sleep(0.1)
    assert g.raw_state()["game_state"]["player"]["x"] > x0


def test_press_window_ambient_filters_continuous_motion():
    # AutoRunner's player.x advances with TIME, not input. With press-window
    # ambient sampling the drift lands in ambient_paths and 'z' (a real no-op)
    # is not credited with movement. This is the snake/flappy scramble fix.
    prober = ControlProber(lambda: AutoRunnerGame(drift_hz=50),
                           probe_keys=["z"], duration_ms=80)
    report = prober.probe()
    assert any(p.endswith("player.x") for p in report.ambient_paths)
    assert report.controls == {}


def test_press_window_does_not_break_static_games():
    # GridWorld has no wall-clock dynamics; discovery must be unchanged.
    report = ControlProber(GridWorldGame, probe_keys=["ArrowRight"]).probe()
    assert report.controls == {"move_right": "ArrowRight"}


class _FlakyGrid(GridWorldGame):
    """apply() raises once (simulated Playwright timeout), then recovers."""

    def __init__(self):
        super().__init__()
        self.raised = False

    def apply(self, command):
        if not self.raised and command.get("key") == "ArrowRight":
            self.raised = True
            raise TimeoutError("simulated Playwright hiccup")
        super().apply(command)


def test_apply_exception_skips_press_not_probe():
    report = ControlProber(_FlakyGrid, probe_keys=["ArrowRight", "ArrowLeft"]).probe()
    # one press lost, but 2/3 remaining still make the majority -> discovered
    assert report.controls.get("move_right") == "ArrowRight"
    assert report.controls.get("move_left") == "ArrowLeft"
    assert report.apply_errors == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_prober_hardening.py -q`
Expected: FAIL — `ImportError: cannot import name 'AutoRunnerGame'`

- [ ] **Step 3: Implement**

Add to `ludus/synthetic.py` (after `CounterGame`):

```python
class AutoRunnerGame(_SyntheticBase):
    """Continuous-motion regression game: player.x advances with WALL-CLOCK
    time (like snake/flappy), independent of input. 'x' bumps score; all other
    keys are no-ops. Used to pin the press-window ambient-sampling fix — an
    instant-snapshot ambient baseline misses the drift and misattributes it
    to whatever key was pressed."""

    def __init__(self, drift_hz: int = 50) -> None:
        import time as _time
        self._time = _time
        self._t0 = _time.monotonic()
        self._drift_hz = drift_hz
        self.score = 0
        self.steps = 0

    def apply(self, command: dict) -> None:
        self.steps += 1
        if command.get("key") == "x":
            self.score += 1

    def raw_state(self) -> dict:
        x = int((self._time.monotonic() - self._t0) * self._drift_hz)
        return {
            "status": "playing",
            "metrics": {"score": self.score, "steps": self.steps},
            "game_state": {"player": {"x": x, "y": 0}},
        }
```

In `ludus/onboarding/prober.py`:

1. Add `import time` and `import logging` up top; `log = logging.getLogger("ludus.onboarding")`.
2. Add field to `ProbeReport`: `apply_errors: int = 0`.
3. In `_ambient_baseline`, replace the instant snapshot pair with a press-window pair:

```python
    def _ambient_baseline(self, client, report: ProbeReport):
        """Paths that change with NO input across `repeats` PRESS-SIZED windows.

        The window sleeps as long as a real key press occupies (duration_ms +
        the client's post-press settle), so drift from continuously-moving
        games (snake, flappy) is captured as ambient instead of being
        misattributed to whichever key happens to be pressed (M1 finding)."""
        window_s = (self._duration_ms + 100) / 1000.0
        ambient: set[str] = set()
        for _ in range(self._repeats):
            if _status(client.raw_state()) != "playing":
                client = self._relaunch(client, report)
            before = flatten_state(client.raw_state())
            time.sleep(window_s)
            after = flatten_state(client.raw_state())
            ambient |= set(diff_states(before, after))
        return client, ambient
```

4. In `_probe_key`, wrap the press in a try/except so one flaky apply skips the press, not the probe:

```python
            before = flatten_state(client.raw_state())
            try:
                client.apply({"key": key, "duration_ms": self._duration_ms})
            except Exception as exc:
                report.apply_errors += 1
                log.warning("apply(%r) failed (%s); skipping press", key, exc)
                continue
            after = flatten_state(client.raw_state())
```

(`report` is not currently a parameter of `_probe_key` — add it: change the signature to `_probe_key(self, client, key, ambient, report)`; the call site already passes `report`.)

- [ ] **Step 4: Run to verify green**

Run: `.venv/bin/pytest tests/test_prober_hardening.py tests/test_prober.py tests/test_synthetic_games.py -q`
Expected: all pass (the existing prober tests must be unaffected — GridWorld/Counter have no wall-clock dynamics, and the ambient sleep only adds ~0.5s total).

- [ ] **Step 5: Full suite + commit**

Run: `.venv/bin/pytest -q --ignore=infra --ignore=gameworld` — expect 177 passed.

```bash
git add ludus/synthetic.py ludus/onboarding/prober.py tests/test_prober_hardening.py
git commit -m "fix(onboarding): press-window ambient baseline + apply() resilience

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Transition model + TransitionStore

**Files:**
- Create: `ludus/worldmodel/__init__.py` (empty)
- Create: `ludus/worldmodel/transitions.py`
- Test: `tests/test_transitions.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_transitions.py
"""TransitionStore — the induction dataset: per-press (state, action, state')."""

from ludus.worldmodel.transitions import Transition, TransitionStore


def _t(episode: str, step: int, score: int) -> Transition:
    return Transition(
        game_id="g", episode_id=episode, step=step,
        action="move_right", key="ArrowRight",
        before={"status": "playing", "metrics": {"score": score}},
        after={"status": "playing", "metrics": {"score": score + 1}},
        terminal_after=False,
    )


def test_append_and_load_roundtrip(tmp_path):
    store = TransitionStore(tmp_path / "g" / "transitions.jsonl")
    store.append(_t("ep0", 0, 0))
    store.append(_t("ep0", 1, 1))
    loaded = store.load()
    assert len(loaded) == 2
    assert loaded[0].after["metrics"]["score"] == 1
    assert loaded[1].step == 1


def test_load_missing_file_is_empty(tmp_path):
    assert TransitionStore(tmp_path / "nope.jsonl").load() == []


def test_split_holds_out_whole_episodes(tmp_path):
    store = TransitionStore(tmp_path / "t.jsonl")
    for ep in ("ep0", "ep1", "ep2", "ep3"):
        for step in range(5):
            store.append(_t(ep, step, step))
    train, holdout = store.split(holdout_frac=0.25, seed=7)
    assert len(train) == 15 and len(holdout) == 5
    train_eps = {t.episode_id for t in train}
    holdout_eps = {t.episode_id for t in holdout}
    assert not (train_eps & holdout_eps)  # split by EPISODE, never within one


def test_split_deterministic_for_seed(tmp_path):
    store = TransitionStore(tmp_path / "t.jsonl")
    for ep in ("a", "b", "c", "d"):
        store.append(_t(ep, 0, 0))
    h1 = {t.episode_id for t in store.split(holdout_frac=0.25, seed=3)[1]}
    h2 = {t.episode_id for t in store.split(holdout_frac=0.25, seed=3)[1]}
    assert h1 == h2
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_transitions.py -q`
Expected: FAIL — ModuleNotFoundError

- [ ] **Step 3: Implement**

```python
# ludus/worldmodel/__init__.py
```
(empty)

```python
# ludus/worldmodel/transitions.py
"""Transition dataset for world-model induction.

One Transition per key press: the full raw state before and after (the
induction target is `predict(before, action) ~= after`). JSONL on disk so
exploration can append across sessions. Train/holdout split is by EPISODE —
within-episode transitions are heavily correlated, so a within-episode split
would leak and inflate validation accuracy.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from pydantic import BaseModel


class Transition(BaseModel):
    game_id: str
    episode_id: str
    step: int
    action: str          # semantic action name (from GameProfile.controls)
    key: str             # the raw key actually pressed
    before: dict
    after: dict
    terminal_after: bool = False


class TransitionStore:
    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)

    def append(self, t: Transition) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a") as f:
            f.write(t.model_dump_json() + "\n")

    def load(self) -> list[Transition]:
        if not self._path.exists():
            return []
        out = []
        with self._path.open() as f:
            for line in f:
                if line.strip():
                    out.append(Transition.model_validate_json(line))
        return out

    def split(self, holdout_frac: float, seed: int) -> tuple[list[Transition], list[Transition]]:
        """Episode-keyed split: whole episodes go to one side or the other."""
        transitions = self.load()
        episodes = sorted({t.episode_id for t in transitions})
        rng = random.Random(seed)
        rng.shuffle(episodes)
        n_hold = max(1, round(len(episodes) * holdout_frac)) if episodes else 0
        holdout_eps = set(episodes[:n_hold])
        train = [t for t in transitions if t.episode_id not in holdout_eps]
        holdout = [t for t in transitions if t.episode_id in holdout_eps]
        return train, holdout
```

- [ ] **Step 4: Verify green, full suite, commit**

Run: `.venv/bin/pytest tests/test_transitions.py -q` — 4 passed; full suite — 181 passed.

```bash
git add ludus/worldmodel/__init__.py ludus/worldmodel/transitions.py tests/test_transitions.py
git commit -m "feat(worldmodel): Transition model + episode-split TransitionStore

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: Explorer

**Files:**
- Create: `ludus/worldmodel/explore.py`
- Test: `tests/test_explore.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_explore.py
"""Explorer — uniform-random self-play that logs induction transitions."""

from ludus.onboarding.profile import GameProfile
from ludus.synthetic import GridWorldGame
from ludus.worldmodel.explore import explore_game
from ludus.worldmodel.transitions import TransitionStore


def _grid_profile() -> GameProfile:
    return GameProfile(
        game_id="synthetic:grid",
        controls={"move_left": "ArrowLeft", "move_right": "ArrowRight",
                  "move_up": "ArrowUp", "move_down": "ArrowDown"},
        control_effects={}, metrics=["score", "steps"], primary_metric="score",
        higher_is_better=True, timing_ms=50, objective="test",
        state_schema={},
    )


def test_explore_logs_requested_transitions(tmp_path):
    store = TransitionStore(tmp_path / "t.jsonl")
    summary = explore_game(_grid_profile(), GridWorldGame, store=store,
                           episodes=3, steps_per_episode=10, seed=5)
    ts = store.load()
    assert len(ts) == 30
    assert summary["transitions"] == 30 and summary["episodes"] == 3
    t0 = ts[0]
    assert t0.action in _grid_profile().controls
    assert t0.key == _grid_profile().controls[t0.action]
    assert t0.before["status"] == "playing"
    assert t0.game_id == "synthetic:grid"


def test_explore_episode_ids_distinct_and_stable(tmp_path):
    store = TransitionStore(tmp_path / "t.jsonl")
    explore_game(_grid_profile(), GridWorldGame, store=store,
                 episodes=2, steps_per_episode=4, seed=1)
    eps = [t.episode_id for t in store.load()]
    assert len(set(eps)) == 2
    assert eps[0] != eps[-1]


def test_explore_deterministic_for_seed(tmp_path):
    s1 = TransitionStore(tmp_path / "a.jsonl")
    s2 = TransitionStore(tmp_path / "b.jsonl")
    explore_game(_grid_profile(), GridWorldGame, store=s1,
                 episodes=2, steps_per_episode=6, seed=9)
    explore_game(_grid_profile(), GridWorldGame, store=s2,
                 episodes=2, steps_per_episode=6, seed=9)
    a = [(t.action, t.after) for t in s1.load()]
    b = [(t.action, t.after) for t in s2.load()]
    assert a == b


def test_explore_ends_episode_on_terminal(tmp_path):
    # add the quit key as a "control": pressing it ends the episode early;
    # the explorer must record terminal_after=True and start a fresh episode.
    profile = _grid_profile()
    profile.controls = {"use_q": "q"}
    store = TransitionStore(tmp_path / "t.jsonl")
    explore_game(profile, GridWorldGame, store=store,
                 episodes=2, steps_per_episode=5, seed=2)
    ts = store.load()
    assert len(ts) == 2                      # each episode ends on press 1
    assert all(t.terminal_after for t in ts)
    assert len({t.episode_id for t in ts}) == 2
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_explore.py -q` — ModuleNotFoundError.

- [ ] **Step 3: Implement**

```python
# ludus/worldmodel/explore.py
"""Exploration: uniform-random self-play over a GameProfile's controls,
logging one Transition per press. This is the induction dataset generator —
random policy is deliberate (diverse states beat competent play for learning
dynamics; curiosity-weighting is a future refinement).

Uses a client FACTORY (probing precedent): terminal states end the episode
and the next episode gets a fresh client."""

from __future__ import annotations

import random

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
) -> dict:
    """Returns {"episodes": n, "transitions": n, "terminals": n}."""
    rng = random.Random(seed)
    actions = sorted(profile.controls)
    n_transitions = n_terminals = 0

    for ep in range(episodes):
        episode_id = f"{profile.game_id}-explore-{seed}-{ep}"
        client = client_factory()
        try:
            for step in range(steps_per_episode):
                before = client.raw_state()
                if str(before.get("status", "playing")) != "playing":
                    break
                action = rng.choice(actions)
                client.apply({"key": profile.controls[action],
                              "duration_ms": profile.timing_ms})
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

- [ ] **Step 4: Verify green, full suite, commit**

Run: `.venv/bin/pytest tests/test_explore.py -q` — 4 passed; full suite — 185 passed.

```bash
git add ludus/worldmodel/explore.py tests/test_explore.py
git commit -m "feat(worldmodel): uniform-random explorer -> TransitionStore

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: Sandbox

**Files:**
- Create: `ludus/worldmodel/sandbox.py`
- Test: `tests/test_sandbox.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sandbox.py
"""Sandbox — induced code runs contained: AST allowlist + subprocess + timeout."""

from ludus.worldmodel.sandbox import check_source, run_predictions

GOOD = '''
import copy

def predict(state, action):
    out = copy.deepcopy(state)
    if action == "inc":
        out["metrics"]["score"] += 1
    return out
'''


def test_check_source_accepts_allowlisted_imports():
    assert check_source(GOOD) == []


def test_check_source_rejects_forbidden_import():
    bad = "import os\n\ndef predict(state, action):\n    return state\n"
    violations = check_source(bad)
    assert violations and "os" in violations[0]


def test_check_source_rejects_dangerous_names():
    bad = "def predict(state, action):\n    return eval('state')\n"
    assert check_source(bad)


def test_check_source_rejects_syntax_errors():
    assert check_source("def predict(state action):")


def test_run_predictions_happy_path():
    cases = [
        {"state": {"metrics": {"score": 0}}, "action": "inc"},
        {"state": {"metrics": {"score": 5}}, "action": "noop"},
    ]
    preds = run_predictions(GOOD, cases)
    assert preds[0]["metrics"]["score"] == 1
    assert preds[1]["metrics"]["score"] == 5


def test_run_predictions_per_case_error_is_none():
    src = '''
def predict(state, action):
    if action == "boom":
        raise ValueError("nope")
    return state
'''
    preds = run_predictions(src, [
        {"state": {"a": 1}, "action": "boom"},
        {"state": {"a": 1}, "action": "ok"},
    ])
    assert preds[0] is None
    assert preds[1] == {"a": 1}


def test_run_predictions_timeout_returns_all_none():
    src = '''
def predict(state, action):
    while True:
        pass
'''
    preds = run_predictions(src, [{"state": {}, "action": "x"}], timeout_s=2.0)
    assert preds == [None]


def test_run_predictions_missing_predict_returns_all_none():
    preds = run_predictions("x = 1\n", [{"state": {}, "action": "a"}])
    assert preds == [None]
```

- [ ] **Step 2: Run to verify failure** — ModuleNotFoundError.

- [ ] **Step 3: Implement**

```python
# ludus/worldmodel/sandbox.py
"""Contained execution for LLM-written world models.

Two layers (spec: "the agent runs code it wrote — contain it"):
  1. check_source: AST gate — imports outside the allowlist, dangerous names
     (eval/exec/open/...), and syntax errors are rejected BEFORE execution.
  2. run_predictions: the source runs in a SUBPROCESS with a hard timeout;
     a hang or crash marks the candidate failed, never the agent. Cases are
     batched per subprocess call (one interpreter spawn per validation pass,
     not per transition).

Honest limits: this is crash/hang isolation plus a static gate, not a
security boundary against a genuinely adversarial payload."""

from __future__ import annotations

import ast
import json
import subprocess
import sys

ALLOWED_IMPORTS = {"math", "copy", "itertools", "collections"}
FORBIDDEN_NAMES = {"eval", "exec", "open", "compile", "input", "__import__",
                   "globals", "locals", "vars", "breakpoint"}

# Runs inside the subprocess: reads {"source", "cases"} JSON on stdin, execs
# the model source, calls predict per case, writes a JSON list on stdout
# (per-case result or null on per-case error).
_RUNNER = r"""
import json, sys
payload = json.load(sys.stdin)
ns = {}
exec(payload["source"], ns)
predict = ns.get("predict")
out = []
for case in payload["cases"]:
    if predict is None:
        out.append(None)
        continue
    try:
        out.append(predict(case["state"], case["action"]))
    except Exception:
        out.append(None)
print(json.dumps(out))
"""


def check_source(source: str) -> list[str]:
    """Static gate. Returns a list of violation strings ([] = clean)."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"syntax error: {exc}"]
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_IMPORTS:
                    violations.append(f"forbidden import: {root}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in ALLOWED_IMPORTS:
                violations.append(f"forbidden import: {root}")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            violations.append(f"forbidden name: {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_NAMES:
            violations.append(f"forbidden attribute: {node.attr}")
    return violations


def run_predictions(source: str, cases: list[dict], timeout_s: float = 30.0) -> list:
    """predict(state, action) per case, batched in one subprocess.
    Returns a list aligned with `cases`; None marks a per-case error. A
    timeout, crash, or gate violation returns all-None (candidate failed)."""
    if check_source(source):
        return [None] * len(cases)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _RUNNER],
            input=json.dumps({"source": source, "cases": cases}),
            capture_output=True, text=True, timeout=timeout_s,
        )
        if proc.returncode != 0:
            return [None] * len(cases)
        out = json.loads(proc.stdout)
        if not isinstance(out, list) or len(out) != len(cases):
            return [None] * len(cases)
        return out
    except (subprocess.TimeoutExpired, json.JSONDecodeError):
        return [None] * len(cases)
```

- [ ] **Step 4: Verify green, full suite, commit**

Run: `.venv/bin/pytest tests/test_sandbox.py -q` — 8 passed; full suite — 193 passed.

```bash
git add ludus/worldmodel/sandbox.py tests/test_sandbox.py
git commit -m "feat(worldmodel): sandbox — AST allowlist gate + subprocess batch runner

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: Validator

**Files:**
- Create: `ludus/worldmodel/validate.py`
- Test: `tests/test_wm_validate.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_wm_validate.py
"""Field-weighted validation of induced models against real transitions."""

from ludus.worldmodel.transitions import Transition
from ludus.worldmodel.validate import validate_model

PERFECT = '''
import copy

def predict(state, action):
    out = copy.deepcopy(state)
    if action == "inc":
        out["metrics"]["score"] += 1
    return out
'''

WRONG_SCORE = '''
import copy

def predict(state, action):
    return copy.deepcopy(state)   # never updates score
'''


def _transitions(n=4):
    return [Transition(
        game_id="g", episode_id=f"ep{i}", step=0, action="inc", key="x",
        before={"status": "playing", "metrics": {"score": i},
                "game_state": {"player": {"x": 1}}},
        after={"status": "playing", "metrics": {"score": i + 1},
               "game_state": {"player": {"x": 1}}},
    ) for i in range(n)]


def test_perfect_model_scores_one_and_passes():
    report = validate_model(PERFECT, _transitions(),
                            important_paths=["metrics.score"], threshold=0.9)
    assert report.overall == 1.0
    assert report.passed is True
    assert report.per_field["metrics.score"] == 1.0
    assert report.n_cases == 4
    assert report.counterexamples == []


def test_wrong_model_fails_with_counterexamples():
    report = validate_model(WRONG_SCORE, _transitions(),
                            important_paths=["metrics.score"], threshold=0.9)
    assert report.passed is False
    assert report.per_field["metrics.score"] == 0.0
    assert report.per_field["game_state.player.x"] == 1.0   # untouched field fine
    ce = report.counterexamples[0]
    assert ce["field"] == "metrics.score"
    assert ce["predicted"] != ce["actual"]


def test_important_paths_weigh_more():
    # score wrong (weight 3), player.x + status right (weight 1 each):
    # weighted accuracy = 2/5, NOT the unweighted 2/3.
    report = validate_model(WRONG_SCORE, _transitions(),
                            important_paths=["metrics.score"], threshold=0.9)
    assert abs(report.overall - 2 / 5) < 1e-9


def test_float_tolerance():
    src = '''
import copy

def predict(state, action):
    out = copy.deepcopy(state)
    out["metrics"]["score"] = state["metrics"]["score"] + 1 + 1e-9
    return out
'''
    report = validate_model(src, _transitions(),
                            important_paths=["metrics.score"], threshold=0.9)
    assert report.per_field["metrics.score"] == 1.0


def test_crashed_prediction_counts_all_fields_wrong():
    src = 'def predict(state, action):\n    raise RuntimeError("boom")\n'
    report = validate_model(src, _transitions(),
                            important_paths=["metrics.score"], threshold=0.9)
    assert report.overall == 0.0
    assert report.passed is False
```

- [ ] **Step 2: Run to verify failure** — ModuleNotFoundError.

- [ ] **Step 3: Implement**

```python
# ludus/worldmodel/validate.py
"""Field-weighted prediction accuracy on held-out transitions (the promotion
gate from the spec): exact match for discrete fields, tolerance for floats,
important paths (primary metric, player state) weigh 3x. Counterexamples
(field, state, action, predicted, actual) feed the inducer's repair loop.

Comparison is over the flattened paths of the ACTUAL after-state — the model
must reproduce reality; extra predicted fields are ignored, missing ones are
wrong. `steps`/tick-style paths still count: they are trivially learnable
(+1 per press) and a model that can't learn them deserves the penalty."""

from __future__ import annotations

from dataclasses import dataclass, field

from ludus.onboarding.diff import flatten_state
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
```

- [ ] **Step 4: Verify green, full suite, commit**

Run: `.venv/bin/pytest tests/test_wm_validate.py -q` — 5 passed; full suite — 198 passed.

```bash
git add ludus/worldmodel/validate.py tests/test_wm_validate.py
git commit -m "feat(worldmodel): field-weighted validator with counterexamples

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: LLM chat seam

**Files:**
- Create: `ludus/worldmodel/llm.py`
- Test: `tests/test_wm_llm.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_wm_llm.py
"""chat() seam — payload construction + response parsing, no network."""

import httpx
import pytest

from ludus.worldmodel import llm


def test_anthropic_payload_and_parse(monkeypatch):
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured.update(url=url, headers=headers, json=json)
        return httpx.Response(200, json={"content": [{"text": "SOURCE"}]},
                              request=httpx.Request("POST", url))

    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setattr(llm.httpx, "post", fake_post)
    out = llm.chat("sys", "user text", backend="anthropic")
    assert out == "SOURCE"
    assert captured["url"].endswith("/v1/messages")
    assert captured["json"]["system"] == "sys"
    assert captured["json"]["messages"][0]["content"] == "user text"


def test_gateway_payload_and_parse(monkeypatch):
    def fake_post(url, headers=None, json=None, timeout=None):
        assert url.endswith("/chat/completion")
        assert json["messages"][0]["role"] == "system"
        return httpx.Response(200, json={"text": "GW"},
                              request=httpx.Request("POST", url))

    monkeypatch.setenv("INSFORGE_GATEWAY_URL", "https://x/api/ai")
    monkeypatch.setenv("INSFORGE_API_KEY", "k")
    monkeypatch.setenv("INSFORGE_VISION_MODEL", "m")
    monkeypatch.setattr(llm.httpx, "post", fake_post)
    assert llm.chat("sys", "u", backend="gateway") == "GW"


def test_backend_default_prefers_anthropic_when_key_present(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    assert llm.default_backend() == "anthropic"
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    monkeypatch.setenv("INSFORGE_GATEWAY_URL", "https://x")
    assert llm.default_backend() == "gateway"


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        llm.chat("s", "u", backend="nope")
```

- [ ] **Step 2: Run to verify failure** — ModuleNotFoundError.

- [ ] **Step 3: Implement**

```python
# ludus/worldmodel/llm.py
"""Plain-text chat seam for code synthesis (the inducer's LLM).

The Decision-shaped providers in ludus/providers are the wrong interface for
synthesis — this is system+user string in, raw text out. Backends:
  anthropic — best-in-class code synthesis (default when key present).
  gateway   — InsForge model gateway (same endpoint the vision provider uses).
Model/backend via env: LUDUS_SYNTH_BACKEND, LUDUS_SYNTH_MODEL."""

from __future__ import annotations

import os

import httpx

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"


def default_backend() -> str:
    if os.environ.get("LUDUS_SYNTH_BACKEND"):
        return os.environ["LUDUS_SYNTH_BACKEND"]
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "gateway"


def chat(system: str, user: str, *, backend: str | None = None,
         max_tokens: int = 8192, timeout: float = 180.0) -> str:
    backend = backend or default_backend()
    if backend == "anthropic":
        model = os.environ.get("LUDUS_SYNTH_MODEL", DEFAULT_ANTHROPIC_MODEL)
        r = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": os.environ.get("ANTHROPIC_API_KEY", ""),
                     "anthropic-version": "2023-06-01"},
            json={"model": model, "max_tokens": max_tokens, "system": system,
                  "messages": [{"role": "user", "content": user}]},
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json()["content"][0]["text"]
    if backend == "gateway":
        base = os.environ.get("INSFORGE_GATEWAY_URL", "")
        r = httpx.post(
            f"{base}/chat/completion",
            headers={"Authorization": f"Bearer {os.environ.get('INSFORGE_API_KEY', '')}"},
            json={"model": os.environ.get("LUDUS_SYNTH_MODEL",
                                          os.environ.get("INSFORGE_VISION_MODEL", "")),
                  "messages": [{"role": "system", "content": system},
                               {"role": "user", "content": user}],
                  "temperature": 0},
            timeout=timeout,
        )
        r.raise_for_status()
        body = r.json()
        text = body.get("text")
        if text is None:
            text = body["choices"][0]["message"]["content"]
        return text
    raise ValueError(f"unknown synthesis backend {backend!r}")
```

- [ ] **Step 4: Verify green, full suite, commit**

Run: `.venv/bin/pytest tests/test_wm_llm.py -q` — 4 passed; full suite — 202 passed.

```bash
git add ludus/worldmodel/llm.py tests/test_wm_llm.py
git commit -m "feat(worldmodel): plain-text chat seam (anthropic | gateway)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 7: Inducer (synthesize → check → validate → repair)

**Files:**
- Create: `ludus/worldmodel/inducer.py`
- Test: `tests/test_inducer.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_inducer.py
"""Inducer — the synthesis/repair loop, offline with a scripted synthesizer."""

import json

from ludus.onboarding.profile import GameProfile
from ludus.synthetic import GridWorldGame
from ludus.worldmodel.explore import explore_game
from ludus.worldmodel.inducer import extract_code, induce_world_model
from ludus.worldmodel.transitions import TransitionStore

# A CORRECT gridworld world model (we know the true dynamics).
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

BROKEN_MODEL = '''
def predict(state, action):
    return state   # never changes anything
'''


def _grid_profile() -> GameProfile:
    return GameProfile(
        game_id="synthetic:grid",
        controls={"move_left": "ArrowLeft", "move_right": "ArrowRight",
                  "move_up": "ArrowUp", "move_down": "ArrowDown"},
        control_effects={}, metrics=["score", "steps"], primary_metric="score",
        higher_is_better=True, timing_ms=50, objective="test", state_schema={},
    )


def _store(tmp_path) -> TransitionStore:
    store = TransitionStore(tmp_path / "t.jsonl")
    explore_game(_grid_profile(), GridWorldGame, store=store,
                 episodes=4, steps_per_episode=15, seed=3)
    return store


def test_extract_code_from_fenced_response():
    text = "Here you go:\n```python\ndef predict(state, action):\n    return state\n```\nDone."
    assert extract_code(text).startswith("def predict")


def test_extract_code_bare_source_passthrough():
    src = "def predict(state, action):\n    return state\n"
    assert extract_code(src) == src


def test_induce_succeeds_first_try_with_correct_model(tmp_path):
    calls = []

    def synthesizer(system, user):
        calls.append(user)
        return f"```python\n{GRID_MODEL}\n```"

    result = induce_world_model(
        "synthetic:grid", _store(tmp_path), profile=_grid_profile(),
        synthesizer=synthesizer, out_dir=tmp_path / "wm",
        max_iterations=3, threshold=0.9, seed=5,
    )
    assert result.status == "INDUCED"
    assert result.iterations == 1
    assert result.report.passed
    assert (tmp_path / "wm" / "synthetic:grid" / "model.py").exists()
    saved_report = json.loads(
        (tmp_path / "wm" / "synthetic:grid" / "report.json").read_text())
    assert saved_report["status"] == "INDUCED"
    assert saved_report["overall"] >= 0.9


def test_induce_repairs_with_counterexamples(tmp_path):
    responses = [f"```python\n{BROKEN_MODEL}\n```", f"```python\n{GRID_MODEL}\n```"]
    prompts = []

    def synthesizer(system, user):
        prompts.append(user)
        return responses[len(prompts) - 1]

    result = induce_world_model(
        "synthetic:grid", _store(tmp_path), profile=_grid_profile(),
        synthesizer=synthesizer, out_dir=tmp_path / "wm",
        max_iterations=3, threshold=0.9, seed=5,
    )
    assert result.status == "INDUCED"
    assert result.iterations == 2
    # the repair prompt must carry counterexamples from the failed attempt
    assert "predicted" in prompts[1] and "actual" in prompts[1]


def test_induce_fails_honestly_when_budget_exhausted(tmp_path):
    def synthesizer(system, user):
        return f"```python\n{BROKEN_MODEL}\n```"

    result = induce_world_model(
        "synthetic:grid", _store(tmp_path), profile=_grid_profile(),
        synthesizer=synthesizer, out_dir=tmp_path / "wm",
        max_iterations=2, threshold=0.9, seed=5,
    )
    assert result.status == "FAILED"
    assert result.iterations == 2
    # artifacts still saved for inspection, marked FAILED
    saved = json.loads((tmp_path / "wm" / "synthetic:grid" / "report.json").read_text())
    assert saved["status"] == "FAILED"


def test_induced_gate_is_on_holdout_not_train(tmp_path):
    # the saved report's n_cases must equal the HOLDOUT size (1 of 4 episodes)
    store = _store(tmp_path)
    train, holdout = store.split(holdout_frac=0.25, seed=5)

    def synthesizer(system, user):
        return f"```python\n{GRID_MODEL}\n```"

    result = induce_world_model(
        "synthetic:grid", store, profile=_grid_profile(),
        synthesizer=synthesizer, out_dir=tmp_path / "wm",
        max_iterations=1, threshold=0.9, seed=5,
    )
    assert result.report.n_cases == len(holdout)
```

- [ ] **Step 2: Run to verify failure** — ModuleNotFoundError.

- [ ] **Step 3: Implement**

```python
# ludus/worldmodel/inducer.py
"""The induction loop: LLM writes predict(), validator grades it, failures
come back as counterexamples, repeat within budget. Gate on HOLDOUT episodes;
counterexamples for repair come from TRAIN (no leak of the gate set into the
prompt). Artifacts (model.py + report.json) are always saved — a FAILED
verdict with evidence beats a silent nothing (spec: no stage can lie)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from ludus.onboarding.diff import flatten_state
from ludus.onboarding.profile import GameProfile
from ludus.worldmodel.sandbox import ALLOWED_IMPORTS, check_source
from ludus.worldmodel.transitions import Transition, TransitionStore
from ludus.worldmodel.validate import ValidationReport, validate_model

MAX_PROMPT_TRANSITIONS = 30
MAX_FULL_STATES = 2


@dataclass
class InductionResult:
    status: str                 # "INDUCED" | "FAILED"
    source: str
    report: ValidationReport
    iterations: int
    model_path: Path | None = None


def extract_code(text: str) -> str:
    """Pull the python source out of a fenced block; bare source passes through."""
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    return m.group(1).strip() + "\n" if m else text


def _render_transition(t: Transition) -> str:
    before = flatten_state(t.before)
    after = flatten_state(t.after)
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
        "counters) from the observed transitions and reproduce them exactly."
    )
    sample = train[:MAX_PROMPT_TRANSITIONS]
    lines = [
        f"Game: {game_id}",
        f"Actions: {sorted(profile.controls)}",
        "",
        "Example full states (before -> after) for shape reference:",
    ]
    for t in sample[:MAX_FULL_STATES]:
        lines.append(f"  action={t.action!r}")
        lines.append(f"  before={json.dumps(t.before, default=str)}")
        lines.append(f"  after ={json.dumps(t.after, default=str)}")
    lines.append("")
    lines.append("Observed transitions (flattened changed fields, (before, after)):")
    for t in sample:
        lines.append("  " + _render_transition(t))
    if counterexamples:
        lines.append("")
        lines.append("YOUR PREVIOUS ATTEMPT was wrong on these held-out cases "
                     "(field, action, predicted vs actual). Fix the model:")
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
    important = [f"metrics.{profile.primary_metric}", "game_state.player"]

    source, report = "", ValidationReport()
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
            continue
        # repair signal from TRAIN; the gate below is on HOLDOUT
        train_report = validate_model(source, train, important_paths=important,
                                      threshold=threshold)
        report = validate_model(source, holdout, important_paths=important,
                                threshold=threshold)
        if report.passed:
            break
        counterexamples = train_report.counterexamples

    status = "INDUCED" if report.passed else "FAILED"
    model_dir = Path(out_dir) / game_id
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.py").write_text(source or "# no source produced\n")
    (model_dir / "report.json").write_text(json.dumps({
        "status": status, "overall": report.overall,
        "per_field": report.per_field, "n_cases": report.n_cases,
        "iterations": iterations, "threshold": threshold,
        "train_size": len(train), "holdout_size": len(holdout),
    }, indent=2) + "\n")
    return InductionResult(status=status, source=source, report=report,
                           iterations=iterations,
                           model_path=model_dir / "model.py")
```

- [ ] **Step 4: Verify green, full suite, commit**

Run: `.venv/bin/pytest tests/test_inducer.py -q` — 6 passed; full suite — 208 passed.

Note on `test_induce_succeeds_first_try_with_correct_model`: GRID_MODEL replicates GridWorldGame exactly, so holdout accuracy must be 1.0. If it isn't, diff the mismatch — the likely culprit is the goal-cycle index logic; do NOT relax the threshold.

```bash
git add ludus/worldmodel/inducer.py tests/test_inducer.py
git commit -m "feat(worldmodel): inducer — synthesize/check/validate/repair loop

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 8: CLI — `explore` and `induce` subcommands

**Files:**
- Modify: `ludus/cli.py`
- Test: `tests/test_cli_worldmodel.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cli_worldmodel.py
"""CLI explore/induce paths — offline via synthetic games + scripted synthesizer."""

import json

from ludus.cli import cmd_explore, cmd_induce
from ludus.worldmodel.transitions import TransitionStore

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


def _onboard_grid(tmp_path):
    from ludus.cli import cmd_onboard
    cmd_onboard(game="synthetic:grid", out_dir=tmp_path / "profiles")
    return tmp_path / "profiles" / "synthetic:grid.json"


def test_cmd_explore_writes_transitions(tmp_path, capsys):
    profile_path = _onboard_grid(tmp_path)
    cmd_explore(game="synthetic:grid", profile_path=profile_path,
                data_dir=tmp_path / "data", episodes=2, steps=8, seed=4)
    ts = TransitionStore(tmp_path / "data" / "synthetic:grid" / "transitions.jsonl").load()
    assert len(ts) == 16
    assert "transitions" in capsys.readouterr().out


def test_cmd_induce_produces_artifacts(tmp_path, capsys):
    profile_path = _onboard_grid(tmp_path)
    cmd_explore(game="synthetic:grid", profile_path=profile_path,
                data_dir=tmp_path / "data", episodes=4, steps=15, seed=4)
    cmd_induce(game="synthetic:grid", profile_path=profile_path,
               data_dir=tmp_path / "data", out_dir=tmp_path / "wm",
               synthesizer=lambda s, u: f"```python\n{GRID_MODEL}\n```",
               max_iterations=2)
    report = json.loads((tmp_path / "wm" / "synthetic:grid" / "report.json").read_text())
    assert report["status"] == "INDUCED"
    out = capsys.readouterr().out
    assert "INDUCED" in out
```

- [ ] **Step 2: Run to verify failure** — ImportError (no cmd_explore).

- [ ] **Step 3: Modify `ludus/cli.py`**

Add imports (with the other ludus imports):

```python
from ludus.worldmodel.explore import explore_game
from ludus.worldmodel.inducer import induce_world_model
from ludus.worldmodel.transitions import TransitionStore
```

Add after `cmd_onboard`:

```python
def cmd_explore(game: str, profile_path=None, data_dir="data", episodes=5,
                steps=40, seed=7, headless: bool = True) -> None:
    profile = GameProfile.load(profile_path or Path("profiles") / f"{game}.json")
    factory = _client_factory(game, timing_ms=profile.timing_ms, headless=headless)
    store = TransitionStore(Path(data_dir) / game / "transitions.jsonl")
    summary = explore_game(profile, factory, store=store, episodes=episodes,
                           steps_per_episode=steps, seed=seed)
    print(f"explored {game}: {summary['transitions']} transitions over "
          f"{summary['episodes']} episodes ({summary['terminals']} terminals) "
          f"-> {Path(data_dir) / game / 'transitions.jsonl'}")


def cmd_induce(game: str, profile_path=None, data_dir="data",
               out_dir="worldmodels", synthesizer=None, max_iterations=4) -> None:
    from ludus.worldmodel.llm import chat, default_backend
    profile = GameProfile.load(profile_path or Path("profiles") / f"{game}.json")
    store = TransitionStore(Path(data_dir) / game / "transitions.jsonl")
    if not store.load():
        raise SystemExit(f"no transitions for {game}; run: python -m ludus.cli explore {game}")
    if synthesizer is None:
        print(f"[induce] synthesis backend: {default_backend()}")
        synthesizer = chat
    result = induce_world_model(game, store, profile=profile,
                                synthesizer=synthesizer, out_dir=out_dir,
                                max_iterations=max_iterations)
    print(f"induction {result.status} after {result.iterations} iteration(s): "
          f"holdout accuracy {result.report.overall:.3f} "
          f"({result.report.n_cases} cases) -> {result.model_path}")
    worst = sorted(result.report.per_field.items(), key=lambda kv: kv[1])[:5]
    for path, acc in worst:
        print(f"  worst field: {path} = {acc:.2f}")
```

In `main()`, extend the subcommand routing (replace the single `onboard` peek with a small dispatch — keep the legacy path untouched):

```python
    if len(sys.argv) > 1 and sys.argv[1] in ("onboard", "explore", "induce"):
        cmd = sys.argv[1]
        ap = argparse.ArgumentParser(prog=f"ludus {cmd}")
        ap.add_argument("game")
        if cmd == "onboard":
            ap.add_argument("--out", default="profiles")
            ap.add_argument("--headed", action="store_true")
            args = ap.parse_args(sys.argv[2:])
            cmd_onboard(args.game, out_dir=args.out, headless=not args.headed)
        elif cmd == "explore":
            ap.add_argument("--profile", type=Path, default=None)
            ap.add_argument("--data-dir", default="data")
            ap.add_argument("--episodes", type=int, default=5)
            ap.add_argument("--steps", type=int, default=40)
            ap.add_argument("--seed", type=int, default=7)
            ap.add_argument("--headed", action="store_true")
            args = ap.parse_args(sys.argv[2:])
            cmd_explore(args.game, profile_path=args.profile, data_dir=args.data_dir,
                        episodes=args.episodes, steps=args.steps, seed=args.seed,
                        headless=not args.headed)
        else:
            ap.add_argument("--profile", type=Path, default=None)
            ap.add_argument("--data-dir", default="data")
            ap.add_argument("--out", default="worldmodels")
            ap.add_argument("--iterations", type=int, default=4)
            args = ap.parse_args(sys.argv[2:])
            cmd_induce(args.game, profile_path=args.profile, data_dir=args.data_dir,
                       out_dir=args.out, max_iterations=args.iterations)
        return
```

- [ ] **Step 4: Verify green, full suite, commit**

Run: `.venv/bin/pytest tests/test_cli_worldmodel.py tests/test_cli_onboard.py -q` — 4 passed; full suite — 210 passed.

```bash
git add ludus/cli.py tests/test_cli_worldmodel.py
git commit -m "feat(cli): explore + induce subcommands

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 9: Live acceptance — real LLM induces a synthetic game, then the Tetris gold experiment

Live env: `.env` has ANTHROPIC_API_KEY (synthesis) and the browser stack lives in `/opt/anaconda3/bin/python` (see CLAUDE.md gotchas).

- [ ] **Step 1: Real-LLM induction of synthetic:grid (cheap end-to-end smoke)**

```bash
.venv/bin/python -m ludus.cli explore synthetic:grid --episodes 6 --steps 25 --seed 11
.venv/bin/python -m ludus.cli induce synthetic:grid --iterations 4
```

Expected: `induction INDUCED ... holdout accuracy >= 0.9`. This is the headline mechanism working for real: an LLM wrote the game's rules as code from observed transitions and the code predicts held-out reality. If FAILED, read `worldmodels/synthetic:grid/report.json` worst fields + the saved model.py, diagnose (prompt? validator? extraction?), fix, re-run. Do not proceed until this passes.

- [ ] **Step 2: Tetris gold experiment (live browser, honest outcome)**

```bash
/opt/anaconda3/bin/python -m ludus.cli explore 29_tetris --episodes 6 --steps 40 --seed 11
/opt/anaconda3/bin/python -m ludus.cli induce 29_tetris --iterations 4
```

Record the verdict either way. Expectations to note in the commit/README: piece-spawn fields (`preview_queue`, next piece) are genuinely random — a correct model copies them through and eats the miss; the board-list `.__len__` collapse (MAX_INDEXED_LIST) makes full-board prediction coarse. INDUCED on important fields is a strong result; FAILED with a clear per-field report is a publishable finding about where code-induction stops. Do NOT tune the threshold to force a pass.

- [ ] **Step 3: Commit artifacts**

```bash
git add worldmodels/ data/*/transitions.jsonl 2>/dev/null || git add worldmodels/
git commit -m "feat(worldmodels): first real-LLM induction runs (synthetic:grid + 29_tetris experiment)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

(If `data/` is gitignored, commit only `worldmodels/`; note transition counts in the commit message.)

- [ ] **Step 4: Docs**

README roadmap item 6: append one line — M2 induction status with the two verdicts and holdout accuracies. CLAUDE.md CLI section: add `explore` and `induce` usage lines mirroring the onboard entry, plus a "worldmodels/ layout" note (model.py + report.json per game).

- [ ] **Step 5: Final suite + commit**

Run: `.venv/bin/pytest -q --ignore=infra --ignore=gameworld` — all pass.

```bash
git add README.md CLAUDE.md
git commit -m "docs: M2 world-model induction usage + status

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Self-review notes (already applied)

- **Spec coverage (Stage 2):** TransitionStore (Task 2), exploration (Task 3), synthesis loop w/ counterexample repair (Task 7), sandbox w/ timeout + allowlist (Task 4), field-weighted validation gating promotion (Task 5), INDUCED/FAILED honest verdicts + saved artifacts (Task 7), `explore`/`induce` CLI (Task 8), live proof (Task 9). M1 carry-forwards folded in (Task 1). Planner/duel deliberately deferred to the Stage-3 plan.
- **Type consistency:** `validate_model(source, transitions, *, important_paths, threshold)` is used identically in Tasks 5 and 7; `TransitionStore.split(holdout_frac, seed)` in Tasks 2 and 7; `explore_game(profile, factory, *, store, episodes, steps_per_episode, seed)` in Tasks 3, 8. `GRID_MODEL` appears in Tasks 7 and 8 test files verbatim (duplicated by design — tasks are read independently).
- **Known risk:** GridWorld's goal-cycle logic in GRID_MODEL must match `ludus/synthetic.py` exactly (index-by-current-goal, then +1 mod 5). Verified against the Task 2 (M1) implementation: `_goal_i` increments and the property applies modulo — equivalent to GOAL_CYCLE.index(current)+1 as long as the cycle has no duplicate positions (it doesn't).
- **Stochasticity policy** is explicit in the synthesis system prompt ("copy unpredictable fields through unchanged") and in Task 9's expectations — no silent threshold tuning.
