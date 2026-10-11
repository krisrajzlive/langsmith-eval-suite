"""Composite quality score: a weighted average of per-example evaluator feedback."""

from __future__ import annotations

import math

COMPOSITE_KEY = "composite_quality"

# Correctness dominates; grounding and citation discipline protect against fluent but unsupported answers.
DEFAULT_WEIGHTS: dict[str, float] = {
    "correctness": 0.40,
    "groundedness": 0.25,
    "helpfulness": 0.10,
    "citations_grounded": 0.15,
    "answer_concise": 0.10,
}


def weighted_score(scores: dict[str, float | None], weights: dict[str, float] | None = None) -> float:
    """Weighted mean over the metrics that are present, with weights renormalized.

    Metrics skipped for an example (None) are excluded rather than counted as zero.
    Returns NaN if none of the weighted metrics are available.
    """
    weights = weights or DEFAULT_WEIGHTS
    present = {k: w for k, w in weights.items() if scores.get(k) is not None}
    total = sum(present.values())
    if total == 0:
        return math.nan
    return sum(float(scores[k]) * w for k, w in present.items()) / total


def write_composite_feedback(experiment_name: str, weights: dict[str, float] | None = None, client=None) -> int:
    """Attach COMPOSITE_KEY feedback to every run of a finished experiment. Returns the number of runs updated."""
    from langsmith import Client

    client = client or Client()
    results = client.get_experiment_results(name=experiment_name)
    written = 0
    for example_with_runs in results["examples_with_runs"]:
        for run in example_with_runs.runs:
            stats = run.feedback_stats or {}
            scores = {k: v.get("avg") for k, v in stats.items() if v.get("n", 0) > 0}
            score = weighted_score(scores, weights)
            if not math.isnan(score):
                client.create_feedback(run_id=run.id, key=COMPOSITE_KEY, score=score, session_id=run.session_id)
                written += 1
    return written
