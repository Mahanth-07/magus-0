"""GameWorld driving client.

FakeGameWorld is the offline test double (no browser). GameWorldClient (Phase 5)
wraps the REAL GameWorld API discovered in docs/DISCOVERY.md and exposes the same
synchronous methods used by ludus.loop.run_episode:

    screenshot() -> bytes
    metrics(keys) -> dict[str, float]
    apply(command) -> None        # command = {"key": <playwright key>, "duration_ms": int}
    read_partner_actions() -> list[str]

GameWorld is asyncio-only (Playwright + an http.server-served HTML5 game). We bridge
async -> sync by owning a single private event loop running on a dedicated background
thread; every public method submits a coroutine to that loop via
asyncio.run_coroutine_threadsafe(...).result(). Keeping ONE long-lived loop (rather
than asyncio.run per call) is required: all Playwright objects (browser/context/page)
are bound to the loop that created them, so the browser must live for the loop's life.
"""

import asyncio
import logging
import math
import os
import sys
import threading
from pathlib import Path

log = logging.getLogger("ludus.gameworld")

# GameWorld is run-as-scripts-from-repo-root (no pip package); put its root on sys.path
# so `from env... import ...` / `from catalog... import ...` resolve. See DISCOVERY.md §0.
# Resolution order: $GAMEWORLD_ROOT, else `<repo-root>/gameworld` (repo root = parent of
# the `ludus` package dir). Override the env var if you clone GameWorld elsewhere.
_GAMEWORLD_ROOT = os.environ.get("GAMEWORLD_ROOT") or str(
    Path(__file__).resolve().parent.parent / "gameworld"
)
if _GAMEWORLD_ROOT not in sys.path:
    sys.path.insert(0, _GAMEWORLD_ROOT)


def _find_free_port() -> int:
    """Bind to port 0 and let the OS hand back a free TCP port."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _get_nested(state: dict, dotted: str):
    """Resolve a dot-path (e.g. 'game_state.score') against a nested dict; None if missing."""
    cur = state
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


# Where in window.gameAPI.getState() each logical metric key lives. We try the dot-path
# variants in order and take the first that resolves to a finite number. Covers the
# Tetris state shape (game_state.score / metrics.score / metrics.* / completion_progress)
# while staying generic enough that unknown keys fall back to a top-level / metrics lookup.
_METRIC_PATHS: dict[str, list[str]] = {
    "score": ["game_state.score", "metrics.score", "metrics.primary_score"],
    "level": ["game_state.level", "metrics.level"],
    "lines": ["metrics.lines_remaining"],
    "lines_remaining": ["metrics.lines_remaining"],
    "lines_target": ["metrics.lines_target"],
    "height": ["metrics.occupied_cells"],
    "occupied_cells": ["metrics.occupied_cells"],
    "completion_progress": ["game_state.completion_progress"],
}


def _tetris_state_text(state: dict) -> str:
    """Render a compact board summary from a Tetris getState() dict.

    Derives per-column heights + hole count from the raw board grid
    (game_state.environment.board: rows x cols of 0/1, row 0 = top). Also reports
    current/next/held piece and key metrics when present. Returns "" if there's no
    board grid (e.g. non-Tetris game, or a loading state), making it a safe no-op.
    """
    env = _get_nested(state, "game_state.environment") or {}
    if not isinstance(env, dict):
        return ""
    board = env.get("board")
    if not (isinstance(board, list) and board and isinstance(board[0], list)):
        return ""

    rows = len(board)
    cols = len(board[0])

    # The board grid includes the ACTIVE falling piece. For settled-stack analysis
    # (column heights + holes) we exclude the active piece's cells so the numbers
    # describe what's already locked down, not the in-flight piece.
    player = _get_nested(state, "game_state.player") or {}
    active = set()
    if isinstance(player, dict):
        for b in (player.get("blocks") or []):
            if isinstance(b, dict):
                x, y = b.get("x"), b.get("y")
                if isinstance(x, int) and isinstance(y, int):
                    active.add((y, x))

    heights = [0] * cols
    holes = 0
    for c in range(cols):
        seen_block = False
        col_holes = 0
        for r in range(rows):
            cell = board[r][c] and (r, c) not in active
            if cell:
                if not seen_block:
                    heights[c] = rows - r  # height measured from the floor
                    seen_block = True
            elif seen_block:
                col_holes += 1
        holes += col_holes

    lines: list[str] = []

    cur = player.get("shape") if isinstance(player, dict) else None
    preview = env.get("preview_queue")
    held = env.get("held_piece")
    piece_bits = []
    if cur:
        piece_bits.append(f"current={cur}")
        if active:
            xs = sorted({x for (_, x) in active})
            piece_bits.append(f"at_cols={'-'.join(str(x) for x in xs)}")
    if isinstance(preview, list) and preview:
        nxt = ",".join(str(p) for p in preview[:3] if p is not None)
        if nxt:
            piece_bits.append(f"next=[{nxt}]")
    if held:
        piece_bits.append(f"held={held}")
    if piece_bits:
        lines.append(" ".join(piece_bits))

    lines.append(f"Board {cols}x{rows} (col heights, 0=empty, left->right):")
    lines.append(" ".join(str(h) for h in heights))
    lines.append(f"Aggregate height={sum(heights)} max={max(heights)} holes={holes}")

    metrics = state.get("metrics") or {}
    metric_bits = []
    for key in ("score", "level", "lines_remaining"):
        v = metrics.get(key)
        if isinstance(v, (int, float)):
            metric_bits.append(f"{key}={int(v)}")
    if metric_bits:
        lines.append(" ".join(metric_bits))

    # --- LOOKAHEAD: enumerate candidate placements in code, let the model pick ---
    # The VLM can't simulate piece interlock from a board summary, so we hand it the
    # ranked candidate placements (rotation x column hard-drops) with their simulated
    # results + the exact action macro that reaches each. Omitted gracefully if we
    # can't read the active piece pose for this frame.
    candidate_block = _tetris_candidate_block(board, player, active)
    if candidate_block:
        lines.append(candidate_block)

    return "\n".join(lines)


def _active_rotation_index(shape: str, active: set) -> int | None:
    """Infer the active piece's rotation index by matching its normalized cells
    against the ROTATIONS table. `active` is a set of (row, col). Returns the
    rotation index, or None if it doesn't match any known orientation."""
    from ludus.teacher.tetris_heuristic import ROTATIONS

    shape = (shape or "").lower()
    if shape not in ROTATIONS or not active:
        return None
    min_r = min(r for r, _ in active)
    min_c = min(c for _, c in active)
    norm = frozenset((c - min_c, r - min_r) for (r, c) in active)
    for idx, cells in enumerate(ROTATIONS[shape]):
        if frozenset(cells) == norm:
            return idx
    return None


def _tetris_candidate_block(board, player, active) -> str:
    """Build the ranked-candidate-placements text for the Tetris prompt.

    Strips the active piece, infers its current leftmost col + rotation, runs the
    lookahead, and renders a compact top-6 list. Returns "" (graceful no-op) if the
    active piece pose can't be determined or anything goes wrong."""
    try:
        from ludus.teacher.tetris_heuristic import rank_placements, strip_active_piece

        shape = player.get("shape") if isinstance(player, dict) else None
        blocks = player.get("blocks") if isinstance(player, dict) else None
        if not (shape and active and isinstance(blocks, list)):
            return ""

        current_left = min(c for _, c in active)
        current_rotation = _active_rotation_index(shape, active)
        if current_rotation is None:
            return ""

        stack = strip_active_piece(board, blocks)
        cands = rank_placements(
            stack, shape,
            current_left=current_left,
            current_rotation=current_rotation,
            top_k=6,
        )
        if not cands:
            return ""

        out = [
            "Candidate placements (each shows the resulting board if you pick it; "
            "choose the BEST and return its exact actions):"
        ]
        for i, c in enumerate(cands):
            out.append(
                f"[{i}] rotate x{c['rotation']}, col {c['target_col']} -> "
                f"holes {c['holes']}, maxH {c['max_height']}, "
                f"lines {c['lines_cleared']}, actions={c['actions']}"
            )
        return "\n".join(out)
    except Exception:  # never let the lookahead crash the prompt build
        return ""


def _relative_bearing(px: float, py: float, angle_deg: float, ex: float, ey: float) -> float:
    """Relative bearing (degrees, [-180,180]) from the player's facing to an enemy tile.

    CALIBRATED LIVE against 31_wolf3d (see git history): the player's forward vector is
    (cos(angle_deg), sin(angle_deg)) in tile space where +x is right and +y is DOWN
    (screen coords), so the absolute angle toward the enemy is atan2(ey-py, ex-px).
    `turn_left` (ArrowLeft) INCREASES angle_deg; `turn_right` (ArrowRight) DECREASES it.

    Returns angle_to_enemy - angle_deg normalized to [-180,180]:
      > 0  -> enemy is counter-clockwise of facing -> turn LEFT to face it.
      < 0  -> enemy is clockwise of facing         -> turn RIGHT to face it.
    Verified live: turning RIGHT on a -63deg bearing drove |bearing| 63 -> 9 (centered).
    """
    enemy_ang = math.degrees(math.atan2(ey - py, ex - px))
    return ((enemy_ang - angle_deg + 180) % 360) - 180


def _bearing_phrase(bearing: float) -> str:
    """Translate a relative bearing into turn guidance + a coarse direction word."""
    mag = abs(bearing)
    if mag <= 15:
        return "it is directly AHEAD; attack now if close"
    if mag >= 150:
        side = "RIGHT" if bearing < 0 else "LEFT"
        return f"it is BEHIND you -> turn {side} ~{mag:.0f}deg to face it"
    side = "RIGHT" if bearing < 0 else "LEFT"
    if bearing < 0:
        zone = "AHEAD-RIGHT" if mag < 90 else "BEHIND-RIGHT"
    else:
        zone = "AHEAD-LEFT" if mag < 90 else "BEHIND-LEFT"
    return f"turn {side} ~{mag:.0f}deg to face it (it is {zone})"


def _wolf3d_state_text(state: dict) -> str:
    """Render a compact spatial summary from a Wolf3D getState() dict.

    Reports the player pose (tile, facing), health/ammo/weapon, and the nearest enemy's
    bearing + distance with explicit turn guidance, then tactics. Returns "" if this
    doesn't look like a Wolf3D state (no player.angle_deg), so it's a safe no-op.
    """
    player = _get_nested(state, "game_state.player") or {}
    if not isinstance(player, dict) or not isinstance(player.get("angle_deg"), (int, float)):
        return ""

    lines: list[str] = []

    ptx, pty = player.get("tile_x"), player.get("tile_y")
    ang = player.get("angle_deg")
    hp = player.get("health")
    ammo = player.get("ammo")
    weapon = player.get("weapon")
    pose = f"Player tile=({ptx},{pty}) facing={ang:.0f}deg."
    stat_bits = []
    if isinstance(hp, (int, float)):
        stat_bits.append(f"health={int(hp)}")
    if isinstance(ammo, (int, float)):
        stat_bits.append(f"ammo={int(ammo)}")
    if weapon:
        stat_bits.append(f"weapon={weapon}")
    if stat_bits:
        pose += " " + " ".join(stat_bits) + "."
    lines.append(pose)

    env = _get_nested(state, "game_state.environment") or {}
    enemy = env.get("nearest_enemy") if isinstance(env, dict) else None
    if isinstance(enemy, dict) and isinstance(enemy.get("tile_x"), (int, float)):
        etx, ety = enemy.get("tile_x"), enemy.get("tile_y")
        etype = enemy.get("type") or "enemy"
        estate = enemy.get("state")
        dist = enemy.get("distance_tiles")
        bearing = _relative_bearing(ptx, pty, ang, etx, ety)
        head = f"Nearest enemy: {etype}"
        if estate:
            head += f" ({estate})"
        if isinstance(dist, (int, float)):
            head += f", {dist:.1f} tiles away"
        head += f" at tile ({etx},{ety}) -> {_bearing_phrase(bearing)}."
        lines.append(head)
    else:
        lines.append("Nearest enemy: NONE listed (none in view) -> turn to scan, then move_forward to explore.")

    metrics = state.get("metrics") or {}
    rem = metrics.get("monsters_remaining")
    total = metrics.get("total_monsters") or metrics.get("max_kills")
    if isinstance(rem, (int, float)):
        if isinstance(total, (int, float)):
            lines.append(f"Monsters remaining: {int(rem)} of {int(total)}.")
        else:
            lines.append(f"Monsters remaining: {int(rem)}.")

    lines.append(
        "Tactics: turn toward the enemy using the LEFT/RIGHT guidance until it is "
        "centered (AHEAD), move_forward to close distance, attack only when facing it "
        "and close; back off / strafe if health is low. If no enemy is listed, turn to "
        "scan, then move_forward to explore."
    )
    return "\n".join(lines)


class GameWorldClient:
    """Real GameWorld driving client. Same interface as FakeGameWorld."""

    def __init__(
        self,
        game_id: str,
        timing_ms: int,
        headless: bool = True,
        port: int | None = None,
        call_timeout: float = 60.0,
    ) -> None:
        self.game_id = game_id
        self.timing_ms = timing_ms
        self.headless = headless
        self._port = port
        self._call_timeout = call_timeout

        self._launcher = None
        self._mgr = None
        self._started = False

        # Dedicated event loop on a background thread (the async bridge).
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    # --- async bridge ---------------------------------------------------------

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _submit(self, coro):
        """Run a coroutine on the private loop from the (sync) caller thread.

        Bounded by self._call_timeout so a stalled GameWorld surfaces a TimeoutError
        instead of hanging the episode forever; the TimeoutError is allowed to propagate.
        """
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(
            timeout=self._call_timeout
        )

    # --- lifecycle ------------------------------------------------------------

    async def _connect(self) -> None:
        from catalog.games import resolve_game_id, load_game
        from env.game_launcher import GameLauncher, append_url_suffix
        from env.browser_manager import BrowserConfig, BrowserGameManager

        gid = resolve_game_id(self.game_id)
        game = load_game(gid)
        port = self._port or _find_free_port()

        self._launcher = GameLauncher(game_name=game.game_name, port=port)
        game_url = self._launcher.start()
        # Fireboy & Watergirl boots to a splash/level-menu that never reaches the
        # actionable "playing" state on its own; its game_api exposes a direct-start
        # path gated on ?autostart=1 (game_api.js isDirectStartEnabled). Append it so
        # the co-op level loads automatically. Other games are unaffected.
        if "fireboy" in game.game_name.lower():
            game_url = append_url_suffix(game_url, "?autostart=1")

        self._mgr = BrowserGameManager(
            BrowserConfig(
                game_url=game_url,
                width=game.width,
                height=game.height,
                headless=self.headless,
                speed_multiplier=game.speed_multiplier,
            )
        )
        await self._mgr.start()
        # Gate on real readiness (status in {ready, playing}) before any input.
        # Use a shorter initial wait (15s) so we can attempt wake keys quickly.
        ready = await self._mgr.wait_until_actionable(
            stage="boot", timeout_s=15.0
        )
        if not ready:
            # Game is stuck in menu/loading — try wake keys + mouse clicks.
            # Mirrors the prober's _wake_up logic but in async context.
            from ludus.onboarding.prober import WAKE_KEYS
            log.info("boot: game not ready after 15s; attempting wake sequence")
            for wake_key in WAKE_KEYS:
                try:
                    await self._apply({"key": wake_key})
                except Exception as exc:  # noqa: BLE001
                    log.debug("boot wake apply(%r) failed: %s", wake_key, exc)
                await asyncio.sleep(0.5)
                state = await self._mgr.get_game_state() or {}
                status = state.get("status", "")
                if status in ("ready", "playing"):
                    log.info("boot: woke to %r via %r", status, wake_key)
                    break
            else:
                # All wake keys failed — log and continue; if the game truly can't
                # start, the first raw_state() / apply() call will surface the error.
                log.warning(
                    "boot: all wake keys exhausted and game still in status=%s; "
                    "proceeding anyway (apply calls may fail)",
                    (await self._mgr.get_game_state() or {}).get("status", "unknown"),
                )

        # Partner-action recorder: capture ONLY the human partner's (Fireboy's) keys
        # — the arrow keys. Watergirl's keys (a/d/w), which the agent injects via
        # Playwright, are deliberately excluded so read_partner_actions() reflects the
        # human partner's actions, not the agent's own.
        await self._mgr.page.evaluate(
            "() => { if (!window.__humanKeys) { window.__humanKeys = [];"
            " var FIREBOY_KEYS = ['ArrowUp','ArrowDown','ArrowLeft','ArrowRight'];"
            " window.addEventListener('keydown',"
            " e => { if (FIREBOY_KEYS.includes(e.key)) window.__humanKeys.push(e.key); },"
            " true); } }"
        )
        self._started = True

    def _ensure_started(self) -> None:
        if not self._started:
            self._submit(self._connect())

    # --- public sync API (matches FakeGameWorld) ------------------------------

    def screenshot(self) -> bytes:
        self._ensure_started()
        return self._submit(self._screenshot())

    async def _screenshot(self) -> bytes:
        # Raw Playwright path returns PNG bytes directly (no temp file to clean up).
        return await self._mgr.page.screenshot()

    def metrics(self, keys: list[str]) -> dict[str, float]:
        self._ensure_started()
        return self._submit(self._metrics(keys))

    async def _metrics(self, keys: list[str]) -> dict[str, float]:
        state = await self._mgr.get_game_state() or {}
        out: dict[str, float] = {}
        for key in keys:
            value = None
            for path in _METRIC_PATHS.get(key, [key, f"metrics.{key}", f"game_state.{key}"]):
                value = _get_nested(state, path)
                if isinstance(value, (int, float)):
                    break
            if isinstance(value, (int, float)):
                out[key] = float(value)
            else:
                log.warning("metric %r not found in game state; defaulting 0.0", key)
                out[key] = 0.0
        return out

    def apply(self, command: dict) -> None:
        self._ensure_started()
        self._submit(self._apply(command))

    async def _apply(self, command: dict) -> None:
        key = command.get("key")
        if not key:
            log.warning("apply() got command with no 'key': %r", command)
            return
        duration_ms = command.get("duration_ms", self.timing_ms)

        # Mouse pseudo-commands: "mouse:center", "mouse:top", "mouse:lower", "mouse:3q"
        # Dispatch to page.mouse.click at canvas fractions; then settle.
        if key.startswith("mouse:"):
            await self._mouse_apply(key)
            return

        kb = self._mgr.page.keyboard
        # Press the key held for duration_ms (down -> wait -> up), as the real game's
        # jaws.js input loop polls held keys; then let the game tick a moment so the
        # next screenshot/state read reflects the input.
        await kb.down(key)
        await asyncio.sleep(duration_ms / 1000.0)
        await kb.up(key)
        await asyncio.sleep(0.1)

    async def _mouse_apply(self, key: str) -> None:
        """Handle mouse:* pseudo-commands by clicking at canvas fractions.

        Supported positions:
          mouse:center   — (w/2, h/2)   canvas center
          mouse:top      — (w/2, h/3)   upper third (covers run-3 play button)
          mouse:lower    — (w/2, 2h/3)  lower third
          mouse:3q       — (w/2, 3h/4)  three-quarter height (covers vex-3)
          mouse:<x>x<y>  — absolute pixel coords, e.g. "mouse:853x360"
        """
        page = self._mgr.page
        vp = page.viewport_size or {}
        w = vp.get("width") or 1280
        h = vp.get("height") or 720
        sub = key[len("mouse:"):]
        if sub == "center":
            cx, cy = w // 2, h // 2
        elif sub == "top":
            cx, cy = w // 2, h // 3
        elif sub == "lower":
            cx, cy = w // 2, 2 * h // 3
        elif sub == "3q":
            cx, cy = w // 2, 3 * h // 4
        elif "x" in sub:
            parts = sub.split("x", 1)
            try:
                cx, cy = int(parts[0]), int(parts[1])
            except ValueError:
                cx, cy = w // 2, h // 2
        else:
            cx, cy = w // 2, h // 2
        await page.mouse.click(cx, cy)
        await asyncio.sleep(0.15)

    def pause(self) -> None:
        """Freeze time-based game hooks (gravity etc.) during model inference.

        Mirrors GameWorld's coordinator pause_during_inference behavior so a
        falling Tetris piece doesn't lock before a positioning sequence lands.
        """
        self._ensure_started()
        self._submit(self._mgr.pause_game())

    def resume(self) -> None:
        self._ensure_started()
        self._submit(self._mgr.resume_game())

    def state_text(self) -> str:
        """Compact textual state summary from getState(), for the planner prompt.

        Tetris: column heights, holes + candidate placements. Wolf3D: player pose +
        nearest-enemy bearing/distance + tactics. Tries Tetris first (board grid), then
        Wolf3D (player.angle_deg). Empty string if neither shape is recognizable, so
        it's a safe no-op for other games / loading states.
        """
        self._ensure_started()
        state = self._submit(self._mgr.get_game_state()) or {}
        tetris = _tetris_state_text(state)
        if tetris:
            return tetris
        return _wolf3d_state_text(state)

    def raw_state(self) -> dict:
        """The raw getState() dict — the StateSource seam for onboarding/induction."""
        self._ensure_started()
        return self._submit(self._mgr.get_game_state()) or {}

    def read_partner_actions(self) -> list[str]:
        self._ensure_started()
        return self._submit(self._read_partner_actions())

    async def _read_partner_actions(self) -> list[str]:
        # Drain the human-keypress buffer installed in _connect(): read it, then
        # clear it so each step sees only keys pressed since the last read. The
        # recorder pushes raw KeyboardEvent.key names (e.g. "ArrowRight", "w").
        keys = await self._mgr.page.evaluate(
            "() => { const k = window.__humanKeys || []; window.__humanKeys = []; return k; }"
        )
        return [str(k) for k in (keys or [])]

    def __enter__(self) -> "GameWorldClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if self._started and self._mgr is not None:
            try:
                self._submit(self._mgr.close())
            except Exception:
                pass
        if self._launcher is not None:
            try:
                self._launcher.stop()
            except Exception:
                pass
        self._started = False
        # Tear down the private loop + thread.
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)
        except Exception:
            pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class FakeGameWorld:
    def __init__(self, metric_script: list[dict[str, float]]) -> None:
        self._metrics = metric_script
        self._i = 0
        self.applied: list[dict] = []

    def screenshot(self) -> bytes:
        return b"\x89PNG fake-frame"

    def metrics(self, keys: list[str]) -> dict[str, float]:
        m = self._metrics[min(self._i, len(self._metrics) - 1)]
        return {k: m.get(k, 0.0) for k in keys}

    def apply(self, command: dict) -> None:
        self.applied.append(command)
        self._i = min(self._i + 1, len(self._metrics) - 1)

    def read_partner_actions(self) -> list[str]:
        return []  # FakeGameWorld: no human partner.

    def state_text(self) -> str:
        return ""

    def raw_state(self) -> dict:
        m = self._metrics[min(self._i, len(self._metrics) - 1)]
        return {"status": "playing", "metrics": dict(m), "game_state": {}}
