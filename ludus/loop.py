from ludus.schemas import PlannerContext, StepRecord, EpisodeResult
from ludus.outcome import OutcomeDetector
from ludus.reflection import Reflector


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
    recent_outcomes: list[str] = []
    last_metrics: dict[str, float] = {}

    for i in range(max_steps):
        png = gameworld.screenshot()
        pre = gameworld.metrics(adapter.relevant_metrics if hasattr(adapter, "relevant_metrics") else [adapter.primary_metric])

        ctx = PlannerContext(
            objective=adapter.objective,
            legal_actions=adapter.legal_actions,
            recent_outcomes=recent_outcomes[-5:],
            learned_rules=rulebook.render() if use_memory else "",
            screenshot_png=png,
            partner_recent_actions=(gameworld.read_partner_actions()
                                    if hasattr(gameworld, "read_partner_actions") else []),
        )
        decision = provider.decide(ctx)

        is_legal = decision.action in adapter.legal_actions
        if is_legal:
            legal_count += 1
            gameworld.apply(adapter.semantic_to_gameworld(decision.action, decision.action_args))

        post = gameworld.metrics(adapter.relevant_metrics if hasattr(adapter, "relevant_metrics") else [adapter.primary_metric])
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
        ))
        recent_outcomes.append(f"{decision.action} -> {outcome.summary}")

    result = EpisodeResult(
        episode_id=episode_id, game=adapter.name, mode=mode, steps=max_steps,
        legal_action_rate=(legal_count / max_steps) if max_steps else 0.0,
        final_metrics=last_metrics, rules=rulebook.rules(),
    )
    store.save_episode(result)
    return result
