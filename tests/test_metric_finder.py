# tests/test_metric_finder.py
"""MetricFinder — primary-metric selection from state schema + probe evidence."""

from ludus.onboarding.metrics import find_metrics


def test_score_name_prior_wins():
    schema = {"metrics.score": "int", "metrics.steps": "int",
              "game_state.player.x": "int"}
    result = find_metrics(schema, control_effects={})
    assert result.primary_metric == "score"
    assert result.higher_is_better is True
    assert "score" in result.metrics and "steps" in result.metrics


def test_only_metrics_namespace_fields_become_metrics():
    schema = {"metrics.score": "int", "game_state.player.x": "int",
              "status": "str"}
    result = find_metrics(schema, control_effects={})
    assert result.metrics == ["score"]


def test_kills_prior_beats_generic_counter():
    schema = {"metrics.kills": "int", "metrics.steps": "int"}
    result = find_metrics(schema, control_effects={})
    assert result.primary_metric == "kills"


def test_fallback_prefers_key_affected_metric():
    # No name-prior match: prefer a metric some control actually changes.
    schema = {"metrics.blorbs": "int", "metrics.steps": "int"}
    effects = {"use_x": ["metrics.blorbs"]}
    result = find_metrics(schema, control_effects=effects)
    assert result.primary_metric == "blorbs"


def test_probe_evidence_beats_alphabetical_fallback():
    # Two non-vital no-prior metrics: step 3 alone would pick "aarbs"
    # (alphabetical); probe evidence must steer selection to "zorps".
    schema = {"metrics.aarbs": "int", "metrics.zorps": "int"}
    effects = {"use_x": ["metrics.zorps"]}
    result = find_metrics(schema, control_effects=effects)
    assert result.primary_metric == "zorps"


def test_health_like_metrics_never_primary_when_alternative_exists():
    schema = {"metrics.health": "int", "metrics.score": "int"}
    result = find_metrics(schema, control_effects={})
    assert result.primary_metric == "score"


def test_gameworlds_curated_primary_score_beats_other_score_fields():
    # best_score is a high-water mark (delta ~always 0); GameWorld curates
    # primary_score as THE score — it must win.
    schema = {"metrics.best_score": "int", "metrics.primary_score": "float",
              "metrics.attempts": "int"}
    result = find_metrics(schema, control_effects={})
    assert result.primary_metric == "primary_score"
