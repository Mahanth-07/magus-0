import argparse
import os
from pathlib import Path
from ludus.config import load_game_config
from ludus.adapters.tetris import TetrisAdapter
from ludus.adapters.wolf3d import Wolf3DAdapter
from ludus.adapters.fireboy_watergirl import FireboyWatergirlAdapter
from ludus.providers import build_provider
from ludus.persistence.local import LocalStore
from ludus.persistence.dual import DualWriteStore
from ludus.persistence.insforge import InsForgeStore
from ludus.rulebook import Rulebook
from ludus.gameworld_client import GameWorldClient, FakeGameWorld
from ludus.loop import run_episode

ADAPTERS = {"tetris": TetrisAdapter, "wolf3d": Wolf3DAdapter,
            "fireboy_watergirl": FireboyWatergirlAdapter}
# Map the logical adapter name (config `name`) to the GameWorld catalog game_id
# (catalog YAML stem). resolve_game_id() requires an exact stem match.
GAMEWORLD_IDS = {"tetris": "29_tetris", "wolf3d": "31_wolf3d",
                 "fireboy_watergirl": "12_fireboy-and-watergirl"}


def _load_dotenv(path: str = ".env") -> None:
    """Load KEY=VALUE lines from a local .env into os.environ (without overriding
    already-set vars). No external dependency."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip())


def main() -> None:
    _load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("game")
    ap.add_argument("mode", choices=["baseline", "memory"])
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--provider", default="mock", choices=["mock", "gateway", "anthropic", "nebius", "fireworks", "fallback"])
    ap.add_argument("--fake", action="store_true", help="use FakeGameWorld (no browser)")
    ap.add_argument("--headed", action="store_true", help="show the browser window (great for demos)")
    args = ap.parse_args()

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


if __name__ == "__main__":
    main()
