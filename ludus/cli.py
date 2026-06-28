import argparse
from pathlib import Path
from ludus.config import load_game_config
from ludus.adapters.tetris import TetrisAdapter
from ludus.providers import build_provider
from ludus.persistence.local import LocalStore
from ludus.rulebook import Rulebook
from ludus.gameworld_client import GameWorldClient, FakeGameWorld
from ludus.loop import run_episode

ADAPTERS = {"tetris": TetrisAdapter}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("game")
    ap.add_argument("mode", choices=["baseline", "memory"])
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--provider", default="mock", choices=["mock", "gateway", "anthropic", "fallback"])
    ap.add_argument("--fake", action="store_true", help="use FakeGameWorld (no browser)")
    args = ap.parse_args()

    cfg = load_game_config(Path("configs") / f"{args.game}.yaml")
    adapter = ADAPTERS[args.game](cfg)
    provider = build_provider(args.provider)
    gw = FakeGameWorld([{cfg.primary_metric: 0.0}]) if args.fake else GameWorldClient(game_id=cfg.name, timing_ms=cfg.timing_ms)
    result = run_episode(adapter=adapter, provider=provider, gameworld=gw,
        store=LocalStore(), rulebook=Rulebook(), mode=args.mode,
        max_steps=args.steps, episode_id=f"{args.game}-{args.mode}")
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
