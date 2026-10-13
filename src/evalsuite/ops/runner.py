"""Experiment runner for the order-operations suite."""

from __future__ import annotations

from statistics import mean

from langsmith import evaluate

from ..config import get_settings
from ..datasets import DATASET_NAMES
from ..runners import _app_model, _guard
from .agent import OPS_PROMPT_VERSION, run_ops_task
from .evaluators import OPS_EVALUATORS, reliability, reliability_summary


def run_ops(
    prompt_version: str | None = None,
    prefix: str | None = None,
    judges: dict | None = None,
    app: str | None = None,
    judge: str | None = None,
    guard: str | None = None,
    repetitions: int | None = None,
):
    """Run every operations task against fresh environments. Evaluators are deterministic, so `judge` is unused.

    The signature matches the RAG suites so the model matrix can run this suite unchanged.
    """
    s = get_settings()
    model, input_guard = _app_model(app), _guard(guard)

    def target(inputs: dict) -> dict:
        return run_ops_task(inputs, model=model, guard=input_guard)

    return evaluate(
        target,
        data=DATASET_NAMES["ops_tasks"],
        evaluators=OPS_EVALUATORS,
        summary_evaluators=[reliability_summary],
        experiment_prefix=prefix or "ops",
        metadata={
            "eval_type": "ops",
            "ops_prompt": OPS_PROMPT_VERSION,
            "app_model": app or f"{s.app_provider}:{s.app_model}",
            "guard_model": guard or s.guard_model or None,
        },
        max_concurrency=s.eval_max_concurrency,
        num_repetitions=repetitions or s.eval_num_repetitions,
        disable_evaluator_tracing=not s.eval_trace_evaluators,
    )


def ops_metrics(results) -> dict[str, float]:
    """Mean of every per-example score over all runs, plus pass@1 and pass^k across repetitions."""
    scores: dict[str, list[float]] = {}
    pairs: list[tuple[str, bool]] = []
    for item in results:
        row = {r.key: r.score for r in item["evaluation_results"]["results"]}
        for key, value in row.items():
            if value is not None:
                scores.setdefault(key, []).append(float(value))
        pairs.append((str(item["example"].id), bool(row.get("task_success"))))
    metrics = {k: mean(v) for k, v in scores.items()}
    metrics.update({f"ops_{k}": v for k, v in reliability(pairs).items()})
    metrics["pass_hat_k"] = metrics.get("ops_pass_hat_k", 0.0)
    return metrics
