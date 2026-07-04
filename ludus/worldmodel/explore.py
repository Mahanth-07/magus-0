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
