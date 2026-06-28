# GameWorld API Discovery

Reverse-engineered from source at `/Users/mahanth/ludus/gameworld/` (no source modified). All file paths below are repo-relative to that root unless absolute.

GameWorld benchmarks multimodal game agents over **34 browser games / 170 tasks** using outcome-based, state-verifiable evaluation. Games are plain HTML5/JS served over a local `http.server`, driven through **Playwright (Chromium)**. Each game exposes a uniform `window.gameAPI` JS object that the Python layer reads via `page.evaluate(...)`.

> **Repo layout caveat:** The *game library* (`games/benchmark/`) is a **separate repo** (`github.com/gameworld-dev/gameworld-games`) cloned into `games/benchmark/`. Locally only **`01_2048` and `10_doodle-jump`** are present. The *catalog* (`catalog/games/*.yaml`, `catalog/tasks/<game>/*.yaml`) defines all 34 games and is fully present. To launch Tetris/Wolf3D/Fireboy/Rocket-League you must first clone the games repo so the matching `games/benchmark/<game_name>/index.html` exists.

---

## 0. How GameWorld is driven (package vs scripts)

**Run-as-scripts-from-repo-root.** There is no installed package / console entry point (`pyproject.toml` declares no `[project.scripts]`). All imports are absolute against repo-root modules (`from catalog import ...`, `from env.game_launcher import ...`, `from runtime import ...`). To drive it programmatically from `ludus`, **add the gameworld repo root to `sys.path`** (or run with `cwd` = repo root) and import the modules directly. The async entry points (`env/`, `runtime/`) are all `asyncio`-based.

Interpreter: **base anaconda python has `playwright`** (verified: `python -c "import playwright"` → ok). Browser binary must be installed once: `playwright install chromium`.

The reusable building blocks (all importable, no CLI needed):
- `env.game_launcher.GameLauncher` — serves a game dir over HTTP.
- `env.browser_manager.BrowserGameManager` / `BrowserConfig` — Playwright Chromium + the `gameAPI` plumbing. **Owns `.page` (the Playwright `Page`).**
- `env.game_state_tracker.build_game_state_tracker()` → `GameAPIStateTracker` — reads `window.gameAPI.getState()`.
- `env.action_executor.ActionExecutor` — turns a normalized action dict into Playwright keyboard/mouse ops.
- `env.task_evaluator.build_task_evaluator(...)` — computes outcome metrics from serialized state.
- `agents.harness.semantic_controls.map_semantic_controls_output(...)` — maps a semantic action to a low-level action dict.
- `catalog.games.load_game`, `catalog.tasks.load_task`, `catalog.build_runtime_config`.

---

## 1. Game launch

### Manual CLI commands (read from argparse)

`main.py` — full agent run (`argparse` in `main.py:102-144`):
```bash
# preset = game_id+task_id+model1,model2
python main.py --config 29_tetris+29_01+gpt-5.2 --headed
python main.py --config 12_fireboy-and-watergirl+12_01+gpt-5.2,gpt-5.2   # 2 roles -> 2 models
# flags: --port INT, --log-root DIR, --headless / --headed (default: auto-detect display)
```

`play.py` — launch + validate a game without an agent (`argparse` in `play.py:25-91`). Default subcommand is `stream-state` (a bare game arg is auto-prefixed, `play.py:94-102`):
```bash
python play.py --game 10_doodle-jump                 # stream gameAPI state once/sec
python play.py stream-state --game 29_tetris --port 8101 [--headless] [--suffix '?level=1']
python play.py capture-task --game 12_fireboy-and-watergirl --task 12_01 [--headless]
#   capture-task writes results/play/<game>/<task>/{capture.png,manifest.json}
```

`run_suite.py` — batch (`run_suite.py:25-31`): `python run_suite.py --suite benchmark/suites/<x>.yaml --port 8101 --max-parallel 5`.

### Programmatic launch path

The cleanest low-level launch (mirrors `play.py:_stream_state`, `play.py:120-154`):
```python
from catalog.games import resolve_game_id, load_game        # catalog/games/__init__.py
from env.game_launcher import GameLauncher, append_url_suffix  # env/game_launcher.py
from env.browser_manager import BrowserConfig, BrowserGameManager  # env/browser_manager.py
from env.game_state_tracker import build_game_state_tracker

game_id = resolve_game_id("29_tetris")     # validates against catalog YAML stems
game = load_game(game_id)                  # -> GameDefinition (game_name,width,height,speed_multiplier,...)

launcher = GameLauncher(game_name=game.game_name, port=8101)  # serves games/benchmark/<game_name>/index.html
game_url = launcher.start()                # -> "http://127.0.0.1:8101/index.html"
game_url = append_url_suffix(game_url, "?level=1")   # optional task suffix

mgr = BrowserGameManager(BrowserConfig(
    game_url=game_url, width=game.width, height=game.height,
    headless=True, speed_multiplier=game.speed_multiplier, random_seed=42,
))
# async context:
async with mgr:                            # launches Chromium, navigates, calls gameAPI.init()
    await mgr.wait_until_actionable(stage="boot")   # blocks until status in {"ready","playing"}
    ...
launcher.stop()
```

- `GameLauncher` (`env/game_launcher.py:38-129`): config = constructor args. `DEFAULT_PORT=8101`, `DEFAULT_HTML="index.html"`, `DEFAULT_BASE_DIR = <repo>/games/benchmark`. `resolve_game_directory(game_name)` → `games/benchmark/<game_name>/`. `.start()` spawns `python -m http.server <port> --bind 127.0.0.1` with cwd = the game dir; raises `FileNotFoundError` if the dir is absent (the case for the 3 missing target games locally).
- High-level alternative: `runtime.env.GameEnv(runtime_config, headless=, port=)` (`runtime/env.py:30`) wraps launcher+browser+tracker and adds `start()/capture_screenshot()/execute_action()/capture_state()/reset_game()`. Built from a `RuntimeConfig` via `catalog.build_runtime_config(preset)` (`catalog/builder.py:51`).

### Config format

- **Preset string:** `"<game_id>+<task_id>+<model1,model2,...>"` parsed in `catalog/builder.py:_parse_preset_parts` (14-34). One model per role; a single model is auto-duplicated across roles (`_resolve_model_ids`, 37-48).
- **Game YAML** (`catalog/games/<id>.yaml`, schema `catalog/games/_base.py:GameDefinition`): `game_name`, `game_rules`, `player_mode` (single/cooperative/competitive), `speed_multiplier`, `width`/`height` (default 1280×720), optional `url`, and `game_roles[]`. Each role has `computer_use_controls` (`allowed_keys`, `allow_clicks`, `hold_duration`), `semantic_controls[]`, and `prompt`.
- **Task YAML** (`catalog/tasks/<id>/<task>.yaml`, loader `catalog/tasks/_base.py`): `task_prompt`, `game_url_suffix`, `evaluator_id`, `evaluator_config`, `task_start_score_field`, `task_target_score_field`, `max_steps`, `pause_during_inference`, `continue_on_fail`.

### Exact IDs for the 4 target games (all exist in catalog; HTML present only if games repo cloned)

| Target | `game_id` (catalog YAML stem == `game_name` == games/benchmark dir) | player_mode | tasks |
|---|---|---|---|
| **Tetris** | `29_tetris` | single | `29_01`..`29_05` |
| **Wolfenstein-3D** | `31_wolf3d` | single | `31_01`..`31_05` |
| **Fireboy & Watergirl** | `12_fireboy-and-watergirl` | **cooperative** (2 roles: `watergirl`, `fireboy`) | `12_01`..`12_05` |
| **Rocket League (2D)** | `25_rocket-league-2d` | (see YAML) | `25_01`..`25_05` |

**All 34 game_ids** (`catalog/games/*.yaml` stems): `01_2048, 02_another-gentlemans-adventure, 03_astray, 04_boxel-rebound, 05_breakout, 06_captaincallisto, 07_chrome-dino, 08_core-ball, 09_cubefield, 10_doodle-jump, 11_edge-surf, 12_fireboy-and-watergirl, 13_flappy-bird, 14_geodash, 15_google-snake, 16_hextris, 17_mario-game, 18_minecraft-clone-glm, 19_minesweeper, 20_monkey-mart, 21_ns-shaft, 22_ovo, 23_pacman, 24_restless-wing-syndrome, 25_rocket-league-2d, 26_run-3, 27_stack, 28_temple-run-2, 29_tetris, 30_vex-3, 31_wolf3d, 32_wordle, 33_worlds-hardest-game, 34_worlds-hardest-game-2`.

**Missing locally:** all except `01_2048` and `10_doodle-jump` — including all 4 targets. `git clone https://github.com/gameworld-dev/gameworld-games.git games/benchmark` to fetch them.

---

## 2. Semantic action space (generalist agent)

A "generalist" agent emits a **semantic tool call**, not raw keys. The wire format (parsed by `agents/harness/semantic_controls.py`):

```json
{ "tool_name": "<semantic_action_id>", "arguments": { ... optional params ... } }
```

(arguments may also be inlined as sibling keys; `_extract_arguments`, `semantic_controls.py:25-41`). The agent client returns this dict from `get_action()` (`agents/mm_agents/base/base_client.py:_finalize_tool_action` 515-529 enforces a non-empty `tool_name`).

**Parsing call (the exact one to use):**
```python
from agents.harness.semantic_controls import map_semantic_controls_output
low_level = map_semantic_controls_output(
    {"tool_name": "hard_drop"},          # raw semantic call
    semantic_controls_map,               # {action_id: binding_dict}
)
# -> e.g. {"action": "press_key", "key": "Space", "semantic_controls": "hard_drop"}
```
Validation-only variant: `inspect_semantic_controls_output(raw, map)` → `{is_valid, reason, control_id, mapped_action}`.

**Where legal semantic names live:** per-game, per-role under `semantic_controls:` in `catalog/games/<game>.yaml`. Each entry = `{id, description, binding}` where `binding` is the low-level action (e.g. `{action: press_key, key: "x", duration: 0.2}` or `{action: press_keys, keys: ["w","a"]}` or `{action: wait, duration: 0.5}`). Parsed into `SemanticControls` (`catalog/games/_base.py:71-114`).

The runtime lookup map is built by `agents.harness.prompting.build_semantic_controls_map(role.semantic_controls)` (`prompting.py:55-68`) → `{action_id: binding}`. In a full run it lives at `RuntimeConfig.semantic_controls_maps[role_idx]` (`catalog/builder.py:81`), attached to each `Agent.semantic_controls_map` (`main.py:50-59`). The Coordinator maps then executes: `_resolve_action` → `map_semantic_controls_output` → `env.execute_action` (`runtime/coordinator.py:288-309`).

**Legal semantic action ids per target game** (from the YAMLs):
- **`29_tetris`** (role `player`): `wait, move_left, move_right, soft_drop, rotate_cw, rotate_ccw, hard_drop, hold_piece`. (Bindings: x=rotate_cw, z=rotate_ccw, Space=hard_drop, Shift=hold.)
- **`31_wolf3d`** (role `player`): `wait, move_forward, move_backward, turn_left, turn_right, shoot, open_door`. (ArrowUp/Down/Left/Right, x=shoot, Space=open_door; each `duration: 0.2`.)
- **`12_fireboy-and-watergirl`** — two roles:
  - `watergirl`: `wait, move_left, move_right, jump_left, jump_right` (keys `a`/`d`, combos `["w","a"]`/`["w","d"]`).
  - `fireboy`: `wait, move_left, move_right, jump_left, jump_right` (Arrow keys, combos `["ArrowUp","ArrowLeft"]`/`["ArrowUp","ArrowRight"]`).
- **`25_rocket-league-2d`**: not read in detail here — read `catalog/games/25_rocket-league-2d.yaml` `semantic_controls[]` for the exact ids (same schema).

> "computer_use" agents instead emit raw low-level action dicts directly (`{"action":"press_key","key":"x"}`), bypassing the semantic map. The semantic map applies only when `agent_type=="generalist"` and a map is present (`coordinator.py:289`).

### Low-level action dict schema (what executes on the page)

`env.action_executor.ActionExecutor` (`env/action_executor.py`). Canonical `action` types: `press_key`, `press_keys`, `click`, `click_hold`, `mouse_move`, `drag`, `scroll`, `type`, `wait` (`CANONICAL_ACTIONS`, line 64). Keys are normalized via `KEY_ALIASES` (line 21) and gated by the role's `allowed_keys` / `allow_clicks`. Examples:
```python
{"action": "press_key",  "key": "ArrowLeft", "duration": 0.2}
{"action": "press_keys", "keys": ["ArrowUp","ArrowRight"], "duration": 0.2}  # held simultaneously
{"action": "wait", "duration": 0.5}
{"action": "click", "x": 100, "y": 200, "button": "left"}
```
Execution: `await ActionExecutor(page, controls).execute(action)` or `.execute_actions([...])`. `press_key` does `keyboard.down(key)` → `sleep(hold)` → `keyboard.up(key)` (`_execute_press_key`, 481-490). Hold defaults to the role's `hold_duration` (0.2s) unless the action carries `duration`.

---

## 3. Screenshot capture

Capture goes through a persistent CDP session (avoids headed-mode viewport flash). It writes a PNG file and returns its `Path`; normalized to `width×height`.

- **Low-level:** `await BrowserGameManager.capture_screenshot(name: str) -> Path` (`env/browser_manager.py:340-347`). Internally `CDPScreenshotter.capture` runs CDP `Page.captureScreenshot` `{format:"png", fromSurface:true, captureBeyondViewport:false}`, base64-decodes, resizes to target via PIL, **writes bytes to `screenshot_dir/name`** (`browser_manager.py:81-117`). It returns a path, not bytes — to get **bytes**: `path.read_bytes()`, or call CDP yourself for in-memory.
- **Runtime wrapper:** `await GameEnv.capture_screenshot(agent_id) -> Path` (`runtime/env.py:137-144`) auto-names `frame_NNNNNN_<agent_id>.png` under a browser lock.
- **Raw Playwright fallback (bytes directly):** `await mgr.page.screenshot()` returns PNG bytes (the `page` is public, §5). The CDP path is preferred by GameWorld for fidelity/normalization.

`screenshot_dir` defaults to `.screenshots_temp/<pid>_<uuid>` (`browser_manager.py:63-64`) and is **deleted on `mgr.close()`** (`browser_manager.py:446-450`) — copy out anything you keep (as `play.py:_capture_task` does, 243-247).

---

## 4. Evaluator metrics / serialized state

### Reading raw serialized state (the numbers)

`window.gameAPI.getState()` is the single source of truth, read by `GameAPIStateTracker`:
```python
tracker = build_game_state_tracker()                 # env/game_state_tracker.py:196
snap = await tracker.snapshot(mgr.page)              # -> GameStateSnapshot(state=dict, summary=str)
state = snap.state
# or directly:
state = await mgr.get_game_state()                  # browser_manager.py:349-357  (None on failure)
# or raw JS:  state = await mgr.page.evaluate("() => window.gameAPI.getState()")
```
`get_game_state` runs `GET_GAME_STATE_SCRIPT` = `() => window.gameAPI.getState()` (`game_state_tracker.py:27-34`). The tracker also probes child frames (`_candidate_pages`, 176-193) for iframe games.

**Verified state shape** (from `games/benchmark/10_doodle-jump/game_api.js:288-322`, uniform across games):
```js
{
  gameId, schemaVersion, timestampMs,
  status: "loading"|"ready"|"playing"|"terminal",
  is_actionable: bool,
  terminal: { isTerminal: bool, outcome: "success"|"fail"|... },
  game_state: { score, level, completion_progress, ... per-game fields },
  metrics: { ...per-game numeric metrics },   // e.g. kills, health, ammo, lives
  // plus per-game extras (player pos, platforms, attempts, etc.)
}
gameAPI also exposes: init({}), reset({}), supports_inplace_reset, supports_reload_reset.
```

### Computing outcome-based task metrics

`env.task_evaluator.build_task_evaluator(...)` (`task_evaluator.py:422`) returns an async closure; the only registered `evaluator_id` is **`game_api_metric`** (`_TASK_EVALUATORS`, 417-419). Call it per step:
```python
from env import build_task_evaluator
ev = build_task_evaluator(
    evaluator_id=task.evaluator_id,            # "game_api_metric"
    evaluator_config=task.evaluator_config,    # {score_field, aggregate_score_fields, metrics_fields, end_field,...}
    start_score=task.task_start_score_field,
    target_score=task.task_target_score_field,
    max_steps=task.max_steps,
    continue_on_fail=task.continue_on_fail,
)
result = await ev(state=state, step_index=i, metrics=running_metrics)   # TaskEvaluationResult
```
`TaskEvaluationResult` (`task_evaluator.py:9-19`): `status` ("success"/"fail"/"unknown"/"error"), `summary`, `metrics` (dict), `should_stop`, `should_reset`, `stop_reason`, `finalized`.

How metrics are computed (`_finalize_task_evaluation`, 285-392):
- **Primary score** = `evaluator_config.score_field` (dot path into state, `_get_nested_value`), or sum of `aggregate_score_fields`.
- Tracks `score_current, score_start, score_best, score_run_best, score`.
- **Progress** normalized to `[0,1]`: `(score_best - score_start) / (target - score_start)` → `progress_current`, `progress_best`, `progress` (`_update_progress_metrics`, 140-171). `target_reached` when `score_best >= target`.
- `metrics_fields` are copied verbatim into the result for reporting (`_copy_extra_metrics`, 174-182).
- Terminal/stop logic reads `state.terminal.isTerminal` + `terminal.outcome` and `end_field`.

Runtime wrapper: `runtime.evaluator.Evaluator(runtime_config)` with `.evaluate(agent, state)` / `.summarize(...)` (`runtime/evaluator.py`).

### Metric names per target game (from task YAMLs)

- **Tetris** (`catalog/tasks/29_tetris/29_01.yaml`): primary **`game_state.score`** (target 200). Extra `metrics_fields`: `game_state.score`, `game_state.level`, `game_state.completion_progress`, `metrics.lines_remaining`, `metrics.lines_target`, `terminal.outcome`, `terminal.isTerminal`.
- **Wolf3D** (`catalog/tasks/31_wolf3d/31_01.yaml`): primary **`metrics.kills`** (target 2). Extra: `metrics.kills`, `metrics.max_kills`, `metrics.total_monsters`, `metrics.health`, `metrics.ammo`, `metrics.lives`, `terminal.outcome`, `terminal.isTerminal`.
- **Fireboy & Watergirl** (`12_01.yaml`): primary = **sum of** `game_state.diamonds.fireboy.collected` + `game_state.diamonds.watergirl.collected` (`aggregate_score_fields`, target 2). Extra: `game_state.level`, `game_state.score`, `game_state.completion_progress`, both `.collected` and `.total` diamond counts, `metrics.coins`, `metrics.coins_total`, terminal fields.

(Other tasks `*_02`..`*_05` vary the target/threshold; same field families.)

---

## 5. Playwright handle (raw key injection + JS listeners)

**Yes — the Playwright objects are public on `BrowserGameManager`** (`env/browser_manager.py:228-234`):
```python
mgr.page      # playwright.async_api.Page   (PUBLIC attribute)
mgr.browser   # Browser
mgr.context   # BrowserContext
```
In a full run, reachable via `coordinator.env.game_manager.page` (`runtime/env.py:_require_game_manager`, 89-92).

**(a) Inject raw keyboard events** — three ways, all on `mgr.page`:
- Trusted OS-level (what GameWorld uses): `await mgr.page.keyboard.down("ArrowUp")` / `.up(...)` / `.press("x")` / `.type("...")` (see `ActionExecutor._execute_press_key`, `action_executor.py:481-501`). These dispatch real `keydown`/`keyup` with `isTrusted=true`.
- Or build an action dict and run it through `ActionExecutor(mgr.page, controls).execute({...})`.
- Or `await mgr.page.evaluate(...)` to dispatch synthetic events (note: `isTrusted=false`).

**(b) Attach a JS event listener (record human keypresses)** — yes, this is the recommended approach for the human-vs-agent co-op case. Inject a recorder that pushes into a `window` buffer, then poll it from Python:
```python
await mgr.page.add_init_script("""
  window.__humanKeys = [];
  window.addEventListener('keydown', e => window.__humanKeys.push({key:e.key, t:Date.now()}), true);
""")  # add BEFORE navigation; or page.evaluate(...) after load for a one-shot install
# drain from Python each step:
events = await mgr.page.evaluate("() => { const b = window.__humanKeys; window.__humanKeys = []; return b; }")
```
GameWorld already uses `page.add_init_script(...)` for its own hooks (`browser_manager.py:290-301`: WebGL preserve, speed control, deterministic RNG) — so init-script injection is a supported, proven pattern. You can also subscribe to Playwright page events (`mgr.page.on("console", ...)`) if your recorder logs via `console.log`.

> Co-op note: Fireboy & Watergirl is `player_mode: cooperative` with two key sets (Watergirl = `w/a/d`, Fireboy = arrows). A human can drive one role's keys at the real keyboard **only in headed mode** with the window focused; a remote/headless human's keypresses must be re-injected via `mgr.page.keyboard`. The two roles target the *same* `page` (single browser tab), so both agent and human input land on one page.

---

## 6. Timing (safe wait after an action)

There is **no per-game explicit "step delay"** beyond these knobs:

- **Action hold duration** = per-role `computer_use_controls.hold_duration` (default **0.2s**; `RoleControls.hold_duration`, `catalog/games/_base.py:46`). A `press_key` already blocks for this hold (down→sleep(hold)→up). Per-action `duration` overrides it; per-key overrides via `key_durations`.
  - Tetris/Wolf3D/Fireboy all use `hold_duration: 0.2`. Wolf3D and Fireboy semantic bindings also set `duration: 0.2` explicitly. Tetris `wait`=0.5s, Wolf3D `wait`=0.5s, Fireboy `wait`=0.2s.
- **Inter-step sleep** in the live loop: `AGENT_LOOP_SLEEP_S = 0.05` (`runtime/coordinator.py:19`) between agent steps.
- **Readiness wait** (post-launch / post-reset): `BrowserGameManager.wait_until_actionable(stage, timeout_s=60, actionable_statuses=("ready","playing"), extra_wait_after_actionable_s=0.1)` (`browser_manager.py:359-373`). Poll interval `DEFAULT_READINESS_POLL_S=0.3` (line 41). **Always call this before the first action** and after any reset.
- **`speed_multiplier`** (per game, default 1.0) scales the in-page clock via the injected `dynamic_speed_control.js`, so real-time waits should be interpreted relative to it.
- **`pause_during_inference`** (task flag, default true): the coordinator calls `pause_game()` while the model thinks and `resume_game()` before executing (`coordinator.py:274-286`), so the world doesn't advance during inference. If you replicate this, call `await mgr.pause_game()` / `await mgr.resume_game()` (`browser_manager.py:386-404`).

**Recommended external pacing:** after an `apply()`, the key is already held `hold_duration` (~0.2s); then sleep a small extra (e.g. 0.05–0.2s, matching `AGENT_LOOP_SLEEP_S`) before reading state/screenshot so the frame reflects the input. For reset/boot, gate on `wait_until_actionable`, not a fixed sleep.

---

## INTEGRATION NOTES FOR LUDUS

Recommended `GameWorldClient` wrapping one `BrowserGameManager` + a baked task evaluator. Construct it like `play.py:_stream_state`/`_capture_task`. It is async (GameWorld is asyncio-only); expose async methods or wrap with `asyncio.run`/a loop.

**Setup (once):** add gameworld repo root to `sys.path`; `playwright install chromium` done.

```python
# pseudo-shape; cite the real calls
import sys; sys.path.insert(0, "/Users/mahanth/ludus/gameworld")
from catalog.games import resolve_game_id, load_game
from catalog.tasks import load_task
from agents.harness.prompting import build_semantic_controls_map
from agents.harness.semantic_controls import map_semantic_controls_output
from env.game_launcher import GameLauncher, append_url_suffix
from env.browser_manager import BrowserConfig, BrowserGameManager
from env.game_state_tracker import build_game_state_tracker
from env.action_executor import ActionExecutor
from env import build_task_evaluator

class GameWorldClient:
    async def open(self, game_id, task_id, *, role_idx=0, headless=True):
        gid = resolve_game_id(game_id)
        self.game = load_game(gid)
        self.task = load_task(gid, task_id)
        self.role = self.game.game_roles[role_idx]
        self.sem_map = build_semantic_controls_map(self.role.semantic_controls)  # {id: binding}
        self.launcher = GameLauncher(self.game.game_name, port=8101)
        url = append_url_suffix(self.launcher.start(), self.task.game_url_suffix)
        self.mgr = BrowserGameManager(BrowserConfig(
            game_url=url, width=self.game.width, height=self.game.height,
            headless=headless, speed_multiplier=self.game.speed_multiplier))
        await self.mgr.start()
        await self.mgr.wait_until_actionable(stage="boot")
        self.tracker = build_game_state_tracker()
        self.executor = ActionExecutor(self.mgr.page, controls=self.role.controls)
        self.ev = build_task_evaluator(
            self.task.evaluator_id, self.task.evaluator_config,
            self.task.task_start_score_field, self.task.task_target_score_field,
            self.task.max_steps, self.task.continue_on_fail)
        self._m = {}            # running evaluator metrics
        self._step = 0
        # human-action recorder (co-op):
        await self.mgr.page.evaluate(
            "() => { if(!window.__humanKeys){window.__humanKeys=[];"
            "window.addEventListener('keydown',e=>window.__humanKeys.push(e.key),true);} }")

    # screenshot() -> bytes
    async def screenshot(self) -> bytes:
        p = await self.mgr.capture_screenshot("frame.png")   # CDP png, normalized
        return p.read_bytes()
        # (or: return await self.mgr.page.screenshot()  -> raw bytes, no temp file)

    # metrics(keys) -> dict[str,float]
    async def metrics(self, keys) -> dict:
        snap = await self.tracker.snapshot(self.mgr.page)
        self._step += 1
        res = await self.ev(state=snap.state, step_index=self._step, metrics=self._m)
        self._m = dict(res.metrics)
        # res.metrics has score/score_best/progress/target_reached + the task's metrics_fields
        return {k: self._m.get(k) for k in keys}
        # for raw state fields instead: read snap.state[...] directly (dot paths)

    # apply(command) -> None     command = semantic id (str) OR low-level dict
    async def apply(self, command) -> None:
        if isinstance(command, str):
            command = {"tool_name": command}
        if "tool_name" in command:                            # semantic -> low-level
            action = map_semantic_controls_output(command, self.sem_map)
        else:
            action = command                                  # already low-level dict
        await self.executor.execute(action)                   # holds hold_duration internally

    # read_partner_actions() -> list[str]   (human keypresses since last drain)
    async def read_partner_actions(self) -> list[str]:
        return await self.mgr.page.evaluate(
            "() => { const b=window.__humanKeys||[]; window.__humanKeys=[]; return b; }")

    async def close(self):
        await self.mgr.close()        # also wipes the temp screenshot dir
        self.launcher.stop()
```

Mapping of the requested API → GameWorld calls:
- `screenshot() -> bytes` → `BrowserGameManager.capture_screenshot(name).read_bytes()` (CDP, normalized) **or** `page.screenshot()` for direct bytes.
- `metrics(keys) -> dict[str,float]` → `GameAPIStateTracker.snapshot(page)` then `build_task_evaluator(...)(state, step, metrics)`; pull `keys` from `result.metrics` (computed: `score`, `score_best`, `progress`, `target_reached`) or from raw `snapshot.state` dot-paths (e.g. `game_state.score`, `metrics.kills`).
- `apply(command) -> None` → if semantic: `map_semantic_controls_output(command, sem_map)` → `ActionExecutor.execute(action)`. If already a low-level dict, pass straight to `execute`. (Use `build_semantic_controls_map(role.semantic_controls)` for the map.)
- `read_partner_actions() -> list[str]` → inject a `keydown` listener via `page.add_init_script`/`page.evaluate` writing to `window.__humanKeys`, then drain it with `page.evaluate`. The Playwright `page` is `mgr.page` (public).

**Gotchas for the integration team:**
1. The **games library must be cloned** (`games/benchmark/<game_name>/index.html`); only `01_2048`,`10_doodle-jump` exist locally → all 4 target games will `FileNotFoundError` until then.
2. GameWorld is **asyncio + run-from-repo-root** (no pip package). Drive everything inside an event loop with the repo on `sys.path`.
3. Screenshot files live in a temp dir that `mgr.close()` deletes — read bytes before closing, or use `page.screenshot()`.
4. Always `wait_until_actionable` after launch and after `reset_game()`; status flows `loading → ready/playing → terminal`.
5. The evaluator is **stateful** — pass the same running `metrics` dict each step so `score_best`/`progress` accumulate; `reset_task_evaluator_episode_metrics(m)` after an in-episode reset.
6. For co-op, both roles share one `page`; a human at the physical keyboard only works **headed + focused**, otherwise re-inject via `page.keyboard`.
7. `random_seed=42` (default) + `speed_multiplier` + deterministic-RNG init script make runs reproducible — keep them consistent for comparability.
