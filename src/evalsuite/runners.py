"""Experiment runners. Each function executes one offline evaluation type against a LangSmith dataset."""

from __future__ import annotations

from typing import Any, Callable

from langsmith import evaluate
from pydantic import BaseModel, Field

from .agent import build_agent, run_agent
from .calibration import calibration_summary
from .config import get_settings
from .datasets import DATASET_NAMES
from .evaluators import (
    CODE_EVALUATORS,
    RETRIEVAL_EVALUATORS,
    SAFETY_CODE_EVALUATORS,
    TRAJECTORY_EVALUATORS,
    retrieval_summary,
)
from .evaluators.judges import build_judges
from .llms import build_chat_model, get_judge_model
from .regression import collect_scores


def _metadata(prompt_version: str, extra: dict | None = None) -> dict:
    s = get_settings()
    return {
        "prompt_version": prompt_version,
        "app_model": f"{s.app_provider}:{s.app_model}",
        "judge_model": f"{s.judge_provider}:{s.judge_model}",
        "retriever": s.embeddings_provider,
        **(extra or {}),
    }


def parse_model_spec(spec: str) -> tuple[str, str]:
    """Split "provider:model" (for example "bedrock:us.amazon.nova-lite-v1:0") at the first colon."""
    provider, _, model = spec.partition(":")
    if not model:
        raise ValueError(f"model spec must look like provider:model, got {spec!r}")
    return provider, model


def _app_model(spec: str | None):
    return build_chat_model(*parse_model_spec(spec)) if spec else None


def _judges(spec: str | None) -> dict:
    return build_judges(build_chat_model(*parse_model_spec(spec)) if spec else None)


def make_target(prompt_version: str, app: str | None = None) -> Callable[[dict], dict]:
    agent = build_agent(prompt_version, model=_app_model(app))

    def target(inputs: dict) -> dict[str, Any]:
        return run_agent(inputs["question"], agent=agent).to_outputs()

    return target


def _run(target, dataset_key: str, evaluators, prefix: str, prompt_version: str, summary_evaluators=None, **meta):
    s = get_settings()
    meta = {k: v for k, v in meta.items() if v is not None}
    return evaluate(
        target,
        data=DATASET_NAMES[dataset_key],
        evaluators=evaluators,
        summary_evaluators=summary_evaluators,
        experiment_prefix=prefix,
        metadata=_metadata(prompt_version, meta),
        max_concurrency=s.eval_max_concurrency,
        num_repetitions=s.eval_num_repetitions,
        disable_evaluator_tracing=not s.eval_trace_evaluators,
    )


def run_benchmark(
    prompt_version: str, prefix: str | None = None, judges: dict | None = None, app: str | None = None,
    judge: str | None = None,
):
    """Correctness, RAG quality, retrieval and format checks on the documentation Q&A dataset."""
    judges = judges or _judges(judge)
    evaluators = [
        *CODE_EVALUATORS,
        *RETRIEVAL_EVALUATORS,
        judges["correctness"],
        judges["groundedness"],
        judges["helpfulness"],
        judges["retrieval_relevance"],
    ]
    return _run(
        make_target(prompt_version, app),
        "qa",
        evaluators,
        prefix or f"benchmark-{prompt_version}",
        prompt_version,
        summary_evaluators=[retrieval_summary],
        eval_type="benchmark",
        app_model=app,
        judge_model=judge,
    )


def run_trajectory(
    prompt_version: str, prefix: str | None = None, judges: dict | None = None, app: str | None = None,
    judge: str | None = None,
):
    judges = judges or _judges(judge)
    return _run(
        make_target(prompt_version, app),
        "trajectory",
        [*TRAJECTORY_EVALUATORS, judges["trajectory_quality"]],
        prefix or f"trajectory-{prompt_version}",
        prompt_version,
        eval_type="trajectory",
        app_model=app,
        judge_model=judge,
    )


def run_safety(
    prompt_version: str, prefix: str | None = None, judges: dict | None = None, app: str | None = None,
    judge: str | None = None,
):
    judges = judges or _judges(judge)
    return _run(
        make_target(prompt_version, app),
        "safety",
        [*SAFETY_CODE_EVALUATORS, judges["safety_behavior"]],
        prefix or f"safety-{prompt_version}",
        prompt_version,
        eval_type="safety",
        app_model=app,
        judge_model=judge,
    )


def run_calibration(prefix: str = "judge-calibration", judge: str | None = None):
    """Score the correctness judge against human labels and report agreement statistics."""
    correctness = _judges(judge)["correctness"]

    def verdict_target(inputs: dict) -> dict:
        res = correctness(
            inputs={"question": inputs["question"]},
            outputs={"answer": inputs["answer"]},
            reference_outputs={"answer": inputs["reference"]},
        )
        return {"verdict": int(bool(res["score"])), "comment": res.get("comment", "")}

    def agreement(outputs: dict, reference_outputs: dict) -> dict:
        return {"key": "judge_agrees_with_human", "score": int(outputs["verdict"]) == int(reference_outputs["human_label"])}

    s = get_settings()
    return evaluate(
        verdict_target,
        data=DATASET_NAMES["judge_calibration"],
        evaluators=[agreement],
        summary_evaluators=[calibration_summary],
        experiment_prefix=prefix,
        metadata={"judge_model": judge or f"{s.judge_provider}:{s.judge_model}", "eval_type": "judge-calibration"},
        max_concurrency=s.eval_max_concurrency,
    )


class _Preference(BaseModel):
    reasoning: str = Field(description="Brief justification for the preference")
    preference: int = Field(description="1 if answer A is better, 2 if answer B is better, 0 if equal")


_PAIRWISE_PROMPT = """You are comparing two answers from a Meridian Payments documentation assistant.
Prefer the answer that is more accurate with respect to the reference, better grounded in documentation
(cites sources), and more concise. Do not prefer an answer because of its position.

Question: {question}
Reference answer: {reference}

Answer A: {a}

Answer B: {b}
"""


def run_pairwise(experiment_a: str, experiment_b: str, prefix: str = "pairwise"):
    """Compare two finished experiments example by example with a position-randomized LLM judge."""
    chain = get_judge_model().with_structured_output(_Preference)

    def ranked_preference(inputs: dict, outputs: list[dict], reference_outputs: dict) -> list[int]:
        verdict = chain.invoke(
            _PAIRWISE_PROMPT.format(
                question=inputs["question"],
                reference=reference_outputs.get("answer", "n/a"),
                a=outputs[0].get("answer", "N/A"),
                b=outputs[1].get("answer", "N/A"),
            )
        )
        return {1: [1, 0], 2: [0, 1]}.get(verdict.preference, [0, 0])

    return evaluate(
        (experiment_a, experiment_b),
        evaluators=[ranked_preference],
        randomize_order=True,
        experiment_prefix=prefix,
        max_concurrency=get_settings().eval_max_concurrency,
    )


def score_table(results) -> dict[str, dict[str, float]]:
    """Flatten ExperimentResults into {example_source_id: {metric: score}} for regression gating."""
    rows = []
    for item in results:
        scores = {r.key: r.score for r in item["evaluation_results"]["results"]}
        rows.append({"example_id": item["example"].metadata["source_id"], "scores": scores})
    return collect_scores(rows)
