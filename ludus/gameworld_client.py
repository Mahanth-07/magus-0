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
        await self._mgr.wait_until_actionable(stage="boot")

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
        kb = self._mgr.page.keyboard
        # Press the key held for duration_ms (down -> wait -> up), as the real game's
        # jaws.js input loop polls held keys; then let the game tick a moment so the
        # next screenshot/state read reflects the input.
        await kb.down(key)
        await asyncio.sleep(duration_ms / 1000.0)
        await kb.up(key)
        await asyncio.sleep(0.1)

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
