import logging

from ludus.schemas import PlannerContext, StepRecord, EpisodeResult
from ludus.outcome import OutcomeDetector
from ludus.reflection import Reflector

log = logging.getLogger("ludus.loop")


def run_episode(
    *, adapter, provider, gameworld, store, rulebook,
    mode: str, max_steps: int, episode_id: str,
    detector: OutcomeDetector | None = None,
    reflector: Reflector | None = None,
) -> EpisodeResult:
    detector = detector or OutcomeDetector()
    reflector = reflector or Reflector()
    use_memory = mode == "memory"

    legal_count = 0
    errored = 0
    recent_outcomes: list[str] = []
    last_metrics: dict[str, float] = {}

    metric_keys = adapter.relevant_metrics if hasattr(adapter, "relevant_metrics") else [adapter.primary_metric]
    can_pause = hasattr(gameworld, "pause") and hasattr(gameworld, "resume")

    for i in range(max_steps):
        # Freeze the world while the model thinks so gravity (or any time-based
        # hook) doesn't advance the game during the slow vision round-trip and
        # undo a positioning sequence before it's applied. No-op for worlds that
        # don't support pausing (e.g. FakeGameWorld).
        if can_pause:
            gameworld.pause()

        png = gameworld.screenshot()
        pre = gameworld.metrics(metric_keys)

        ctx = PlannerContext(
            objective=adapter.objective,
            legal_actions=adapter.legal_actions,
            recent_outcomes=recent_outcomes[-5:],
            learned_rules=rulebook.render() if use_memory else "",
            screenshot_png=png,
            partner_recent_actions=(gameworld.read_partner_actions()
                                    if hasattr(gameworld, "read_partner_actions") else []),
            state_text=(gameworld.state_text() if hasattr(gameworld, "state_text") else ""),
            raw_state=(gameworld.raw_state() if hasattr(gameworld, "raw_state") else None),
        )
        # A single transient model-call failure must not kill the whole episode
        # (long runs / slow reasoning models / dedicated endpoints occasionally
        # hiccup). Skip the step and continue; `finally` always unpauses.
        try:
            decision = provider.decide(ctx)
        except Exception as exc:
            log.warning("step %d: provider.decide() failed (%s); skipping step", i, exc)
            errored += 1
            recent_outcomes.append(f"(step {i} errored: {type(exc).__name__})")
            continue
        finally:
            if can_pause:
                gameworld.resume()

        # `actions` is an optional ordered macro (1-5 moves to position then
        # commit); fall back to the single primary `action`. Outcome attribution
        # and legal_action_rate stay keyed on the primary `action` (unchanged).
        moves = decision.actions or [decision.action]
        is_legal = decision.action in adapter.legal_actions
        if is_legal:
            legal_count += 1
        for move in moves:
            if move in adapter.legal_actions:
                gameworld.apply(adapter.semantic_to_gameworld(move, decision.action_args))

        post = gameworld.metrics(metric_keys)
        outcome = detector.detect(pre, post, adapter.primary_metric, adapter.higher_is_better)
        last_metrics = post

        rule_added = None
        if use_memory and is_legal:
            rule_added = reflector.reflect(decision, outcome, adapter.name)
            rulebook.add(rule_added)

        ref = store.save_screenshot(episode_id, i, png)
        store.save_step(StepRecord(
            episode_id=episode_id, step_index=i, mode=mode, game=adapter.name,
            decision=decision, primary_metric=outcome.primary_metric,
            primary_delta=outcome.primary_delta, improved=outcome.improved,
            metric_delta=outcome.delta, rule_added=rule_added, screenshot_ref=ref,
            state_text=ctx.state_text,
        ))
        recent_outcomes.append(f"{decision.action} -> {outcome.summary}")

    # legal-action rate is over ATTEMPTED steps (those where the model returned a
    # decision); errored/skipped steps are excluded so a flaky API doesn't look
    # like illegal play.
    attempted = max_steps - errored
    result = EpisodeResult(
        episode_id=episode_id, game=adapter.name, mode=mode, steps=max_steps,
        legal_action_rate=(legal_count / attempted) if attempted > 0 else 0.0,
        final_metrics=last_metrics, rules=rulebook.rules(),
        errored_steps=errored,
    )
    store.save_episode(result)
    return result
