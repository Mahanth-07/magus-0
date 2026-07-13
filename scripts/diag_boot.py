#!/usr/bin/env python3
"""Diagnose boot-hang games: what status do they report during startup?

Usage:
    /opt/anaconda3/bin/python scripts/diag_boot.py --game 04_boxel-rebound

Connects to the game, polls gameAPI.getState() every second for 30s
WITHOUT waiting for actionable status, and prints what it sees.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Load gameworld onto path
import os
GAMEWORLD_ROOT = os.environ.get("GAMEWORLD_ROOT") or str(REPO_ROOT / "gameworld")
sys.path.insert(0, GAMEWORLD_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("diag_boot")


async def _probe_game(game_id: str, poll_secs: int = 20, click_after: int = 5):
    from catalog.games import resolve_game_id, load_game
    from env.game_launcher import GameLauncher
    from env.browser_manager import BrowserConfig, BrowserGameManager
    import socket

    # Find free port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    gid = resolve_game_id(game_id)
    game = load_game(gid)
    log.info("Launching %s (id=%s) on port %d", game.game_name, gid, port)

    launcher = GameLauncher(game_name=game.game_name, port=port)
    game_url = launcher.start()
    log.info("Game URL: %s", game_url)

    mgr = BrowserGameManager(
        BrowserConfig(
            game_url=game_url,
            width=game.width,
            height=game.height,
            headless=True,
            speed_multiplier=game.speed_multiplier,
        )
    )

    results = []
    try:
        await mgr.start()
        log.info("Browser started, polling state for %ds...", poll_secs)
        start = time.monotonic()
        clicked = False

        while time.monotonic() - start < poll_secs:
            elapsed = time.monotonic() - start
            state = await mgr.get_game_state()

            # After `click_after` seconds, try clicking canvas center
            if not clicked and elapsed >= click_after:
                log.info("Trying canvas click at %.1fs...", elapsed)
                try:
                    w, h = game.width, game.height
                    await mgr.page.mouse.click(w // 2, h // 2)
                    clicked = True
                    log.info("Canvas click sent")
                except Exception as e:
                    log.warning("Click failed: %s", e)

            if state is None:
                status = "NO_GAME_API"
                snapshot = None
            else:
                status = state.get("status", "NO_STATUS")
                snapshot = {k: v for k, v in state.items()
                            if k not in ("raw",) and not isinstance(v, dict)}

            results.append({"t": round(elapsed, 1), "status": status, "snap": snapshot})
            log.info("t=%.1f status=%s is_actionable=%s", elapsed,
                     status, state.get("is_actionable") if state else None)

            if status in ("ready", "playing"):
                log.info("GAME IS ACTIONABLE at t=%.1f", elapsed)
                # Try a key press and see if it responds
                await mgr.page.keyboard.press("Space")
                await asyncio.sleep(0.5)
                state2 = await mgr.get_game_state()
                log.info("After Space press: status=%s",
                         state2.get("status") if state2 else "N/A")
                break

            await asyncio.sleep(1.0)
        else:
            log.warning("Game did NOT reach actionable status in %ds", poll_secs)

    finally:
        await mgr.close()
        launcher.stop()

    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--game", required=True, help="GameWorld game id (e.g. 04_boxel-rebound)")
    ap.add_argument("--poll-secs", type=int, default=20)
    ap.add_argument("--click-after", type=int, default=5,
                    help="seconds to wait before trying canvas click")
    args = ap.parse_args()

    results = asyncio.run(_probe_game(args.game, args.poll_secs, args.click_after))
    print("\n=== STATUS TIMELINE ===")
    for r in results:
        print(f"t={r['t']:5.1f}s  status={r['status']:15s}  snap={json.dumps(r['snap'])}")


if __name__ == "__main__":
    main()
