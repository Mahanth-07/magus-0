import argparse
import os
import sys
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
from ludus.adapters.generic import GenericAdapter
from ludus.onboarding.onboard import onboard_game
from ludus.onboarding.profile import GameProfile

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


def main() -> None:
    _load_dotenv()

    if len(sys.argv) > 1 and sys.argv[1] == "onboard":
        ap = argparse.ArgumentParser(prog="ludus onboard")
        ap.add_argument("game", help="adapter name, raw GameWorld id, or synthetic:<name>")
        ap.add_argument("--out", default="profiles")
        ap.add_argument("--headed", action="store_true")
        args = ap.parse_args(sys.argv[2:])
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
        profile = GameProfile.load(args.profile)
        if profile.game_id != args.game:
            print(f"WARNING: --profile game_id {profile.game_id!r} != positional game "
                  f"{args.game!r}; playing {profile.game_id!r}, logging as {args.game!r}")
        cfg = profile.to_game_config()
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


if __name__ == "__main__":
    main()
