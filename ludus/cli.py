import argparse
import json
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
from ludus.worldmodel.explore import explore_game
from ludus.worldmodel.inducer import induce_world_model
from ludus.worldmodel.transitions import TransitionStore

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


def cmd_explore(game: str, profile_path=None, data_dir="data", episodes=5,
                steps=40, seed=7, headless: bool = True,
                duration_ms=None) -> None:
    profile = GameProfile.load(profile_path or Path("profiles") / f"{game}.json")
    factory = _client_factory(game, timing_ms=profile.timing_ms, headless=headless)
    store = TransitionStore(Path(data_dir) / game / "transitions.jsonl")
    summary = explore_game(profile, factory, store=store, episodes=episodes,
                           steps_per_episode=steps, seed=seed,
                           duration_ms=duration_ms)
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
          f"(primary_score {result.report.primary_accuracy:.2f}) "
          f"({result.report.n_cases} cases) -> {result.model_path}")
    worst = sorted(result.report.per_field.items(), key=lambda kv: kv[1])[:5]
    for path, acc in worst:
        print(f"  worst field: {path} = {acc:.2f}")


def _run_one_side(game: str, label: str, provider, adapter, profile,
                  steps: int, runs_dir: str, headless: bool,
                  repeat_idx: int = 0) -> dict:
    """Run a single episode for one side of a duel. Returns per-episode info."""
    episode_id = f"duel-{game}-{label}-r{repeat_idx}"
    gw = _client_factory(game, timing_ms=profile.timing_ms, headless=headless)()
    try:
        result = run_episode(
            adapter=adapter, provider=provider, gameworld=gw,
            store=LocalStore(root=runs_dir), rulebook=Rulebook(),
            mode="baseline", max_steps=steps,
            episode_id=episode_id)
    finally:
        if hasattr(gw, "close"):
            gw.close()
    return {
        "score": result.final_metrics.get(profile.primary_metric, 0.0),
        "legal_action_rate": result.legal_action_rate,
        "steps": result.steps,
    }


def _aggregate(episode_results: list[dict], provider_name: str, channel: str) -> dict:
    """Aggregate multiple episode results into per-side stats."""
    import math
    scores = [r["score"] for r in episode_results]
    mean = sum(scores) / len(scores)
    std = math.sqrt(sum((s - mean) ** 2 for s in scores) / len(scores))
    avg_legal = sum(r["legal_action_rate"] for r in episode_results) / len(episode_results)
    avg_steps = sum(r["steps"] for r in episode_results) / len(episode_results)
    return {
        "provider": provider_name,
        "score": mean,          # backward-compat: score == mean
        "mean": mean,
        "std": std,
        "min": min(scores),
        "max": max(scores),
        "scores": scores,
        "legal_action_rate": avg_legal,
        "steps": avg_steps,
        "channel": channel,
    }


def cmd_duel(game: str, profile_path=None, worldmodels_dir="worldmodels",
             steps=15, baseline_provider=None, baseline_name="gateway",
             runs_dir="runs", headless: bool = True,
             repeats: int = 1) -> dict:
    """Planner (induced model) vs a baseline provider: same game, same step
    budget, fresh client per side. Prints an honest table, persists the result.

    With repeats>1 each side is run `repeats` times; winner is decided on means.
    """
    from ludus.planning.provider import PlannerProvider

    profile = GameProfile.load(profile_path or Path("profiles") / f"{game}.json")
    adapter = GenericAdapter(profile.to_game_config())
    planner = PlannerProvider(profile, worldmodels_dir=worldmodels_dir)
    baseline = baseline_provider or build_provider(baseline_name)

    _channels = {"planner": "raw_state", "baseline": "screenshot"}
    sides = {}
    for label, provider in (("planner", planner), ("baseline", baseline)):
        episodes = [
            _run_one_side(game, label, provider, adapter, profile,
                          steps, runs_dir, headless, repeat_idx=r)
            for r in range(repeats)
        ]
        sides[label] = _aggregate(
            episodes,
            provider_name=getattr(provider, "name", label),
            channel=_channels[label],
        )

    p_mean = sides["planner"]["mean"]
    b_mean = sides["baseline"]["mean"]
    better = p_mean > b_mean if profile.higher_is_better else p_mean < b_mean
    winner = "planner" if better else ("baseline" if p_mean != b_mean else "tie")

    out = {"game": game, "steps": steps, "repeats": repeats,
           "primary_metric": profile.primary_metric,
           "planner": sides["planner"], "baseline": sides["baseline"],
           "winner": winner}

    channels = {"planner": "[raw_state]", "baseline": "[screenshot]"}
    print(f"==== DUEL {game} ({steps} steps x{repeats}, metric={profile.primary_metric}) ====")
    for label in ("planner", "baseline"):
        s = sides[label]
        if repeats > 1:
            print(f"  {label:9s} ({s['provider']:8s}): {profile.primary_metric}="
                  f"{s['mean']:g}±{s['std']:.2f}  scores={s['scores']}  "
                  f"legal_rate={s['legal_action_rate']:.2f}  {channels[label]}")
        else:
            print(f"  {label:9s} ({s['provider']:8s}): {profile.primary_metric}="
                  f"{s['score']:g}  legal_rate={s['legal_action_rate']:.2f}  "
                  f"{channels[label]}")
    print(f"  winner: {winner}")

    dest = Path(runs_dir) / f"duel-{game}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n")
    return out


def main() -> None:
    _load_dotenv()

    if len(sys.argv) > 1 and sys.argv[1] in ("onboard", "explore", "induce", "duel"):
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
            ap.add_argument("--duration-ms", type=int, default=None)
            args = ap.parse_args(sys.argv[2:])
            cmd_explore(args.game, profile_path=args.profile, data_dir=args.data_dir,
                        episodes=args.episodes, steps=args.steps, seed=args.seed,
                        headless=not args.headed, duration_ms=args.duration_ms)
        elif cmd == "induce":
            ap.add_argument("--profile", type=Path, default=None)
            ap.add_argument("--data-dir", default="data")
            ap.add_argument("--out", default="worldmodels")
            ap.add_argument("--iterations", type=int, default=4)
            args = ap.parse_args(sys.argv[2:])
            cmd_induce(args.game, profile_path=args.profile, data_dir=args.data_dir,
                       out_dir=args.out, max_iterations=args.iterations)
        else:
            ap.add_argument("--profile", type=Path, default=None)
            ap.add_argument("--worldmodels", default="worldmodels")
            ap.add_argument("--steps", type=int, default=15)
            ap.add_argument("--baseline", default="gateway")
            ap.add_argument("--headed", action="store_true")
            ap.add_argument("--repeats", type=int, default=1,
                            help="run each side N times; winner decided on means")
            args = ap.parse_args(sys.argv[2:])
            cmd_duel(args.game, profile_path=args.profile,
                     worldmodels_dir=args.worldmodels, steps=args.steps,
                     baseline_name=args.baseline, headless=not args.headed,
                     repeats=args.repeats)
        return

    ap = argparse.ArgumentParser()
    ap.add_argument("game")
    ap.add_argument("mode", choices=["baseline", "memory"])
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--provider", default="mock", choices=["mock", "gateway", "anthropic", "nebius", "fireworks", "together", "fallback", "planner"])
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
    if args.provider == "planner":
        if not args.profile:
            raise SystemExit("--provider planner requires --profile "
                             "(the planner needs the game's GameProfile)")
        from ludus.planning.provider import PlannerProvider
        provider = PlannerProvider(profile)
    else:
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
