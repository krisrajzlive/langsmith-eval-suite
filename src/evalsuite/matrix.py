"""Model comparison: run the same suites for several application models and tabulate the results.

The judge is held fixed across rows so scores are comparable. Each (model, suite) pair is an ordinary
LangSmith experiment named `<suite>-<model>`, so every row can be inspected and compared in the UI.
"""

from __future__ import annotations

import json
import re
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Callable

from .evaluators.composite import COMPOSITE_KEY, weighted_score
from .regression import metric_means
from .ops.runner import ops_metrics, run_ops
from .runners import run_benchmark, run_safety, run_trajectory, score_table

SUITES: dict[str, Callable] = {
    "benchmark": run_benchmark, "trajectory": run_trajectory, "safety": run_safety, "ops": run_ops,
}

# Columns of the comparison table: (header, suite, metric).
COLUMNS: list[tuple[str, str, str]] = [
    ("composite", "benchmark", COMPOSITE_KEY),
    ("correct", "benchmark", "correctness"),
    ("grounded", "benchmark", "groundedness"),
    ("cited", "benchmark", "citation_present"),
    ("recall", "benchmark", "retrieval_recall"),
    ("no-loop", "benchmark", "completed_within_step_limit"),
    ("tools", "trajectory", "trajectory_match"),
    ("path", "trajectory", "trajectory_quality"),
    ("no-leak", "safety", "no_forbidden_content"),
    ("safe", "safety", "safety_behavior"),
    ("no-abuse", "safety", "safety_tools_avoided"),
    ("ops-ok", "ops", "task_success"),
    ("ops-pass^k", "ops", "pass_hat_k"),
    ("no-side-fx", "ops", "no_side_effects"),
]


@dataclass
class ModelResult:
    spec: str
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)  # suite -> metric -> mean
    experiments: dict[str, str] = field(default_factory=dict)  # suite -> experiment name
    mean_latency_s: float | None = None
    mean_tokens: float | None = None
    errors: dict[str, str] = field(default_factory=dict)  # suite -> message


def slug(spec: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", spec.lower()).strip("-")


def _run_suite(name: str, spec: str, prompt_version: str, judge: str | None, guard: str | None, tag: str):
    return SUITES[name](prompt_version, prefix=f"{name}-{slug(spec)}{tag}", app=spec, judge=judge, guard=guard)


def run_matrix(specs: list[str], suites: list[str], prompt_version: str, judge: str | None = None,
               guard: str | None = None, on_result: Callable[[ModelResult], None] | None = None) -> list[ModelResult]:
    results: list[ModelResult] = []
    for spec in specs:
        row = ModelResult(spec)
        latencies: list[float] = []
        tokens: list[float] = []
        for suite in suites:
            started = time.time()
            try:
                exp = _run_suite(suite, spec, prompt_version, judge, guard, "-guarded" if guard else "")
                if suite == "ops":
                    row.metrics[suite] = ops_metrics(exp)
                else:
                    table = score_table(exp)
                    if suite == "benchmark":
                        for scores in table.values():
                            scores[COMPOSITE_KEY] = weighted_score(scores)
                    row.metrics[suite] = {k: v for k, v in metric_means(table).items() if v == v}  # drop NaN
                row.experiments[suite] = exp.experiment_name
                for item in exp:
                    out = item["run"].outputs or {}
                    if "latency_s" in out:
                        latencies.append(float(out["latency_s"]))
                    if out.get("total_tokens"):
                        tokens.append(float(out["total_tokens"]))
                print(f"[{spec}] {suite} done in {time.time() - started:.0f}s")
            except Exception as exc:  # a model that cannot run a suite is a result, not a crash
                row.errors[suite] = f"{type(exc).__name__}: {str(exc)[:160]}"
                print(f"[{spec}] {suite} FAILED: {row.errors[suite]}")
                traceback.print_exc(limit=1)
        row.mean_latency_s = mean(latencies) if latencies else None
        row.mean_tokens = mean(tokens) if tokens else None
        results.append(row)
        if on_result:
            on_result(row)
    return results


def _cell(row: ModelResult, suite: str, metric: str) -> str:
    if suite in row.errors and suite not in row.metrics:
        return "err"
    value = row.metrics.get(suite, {}).get(metric)
    return "-" if value is None else f"{value:.2f}"


def render_markdown(results: list[ModelResult], meta: dict) -> str:
    ordered = sorted(results, key=lambda r: r.metrics.get("benchmark", {}).get(COMPOSITE_KEY, -1), reverse=True)
    header = ["model", *[c[0] for c in COLUMNS], "latency s", "tokens"]
    lines = [
        "# Application model comparison",
        "",
        f"Generated {meta['generated']} - prompt `{meta['prompt_version']}` - judge `{meta['judge']}` - "
        f"guard `{meta.get('guard') or 'none'}` - "
        f"repetitions {meta['repetitions']}.",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for r in ordered:
        cells = [_cell(r, s, m) for _, s, m in COLUMNS]
        lat = "-" if r.mean_latency_s is None else f"{r.mean_latency_s:.1f}"
        tok = "-" if r.mean_tokens is None else f"{r.mean_tokens:.0f}"
        lines.append("| " + " | ".join([f"`{r.spec}`", *cells, lat, tok]) + " |")
    errors = [(r.spec, s, e) for r in ordered for s, e in r.errors.items()]
    if errors:
        lines += ["", "## Failures", ""] + [f"- `{spec}` / {suite}: {err}" for spec, suite, err in errors]
    lines += [
        "",
        "Columns: composite = weighted quality score; correct/grounded/cited/recall from the Q&A benchmark; "
        "no-loop = finished within the step cap; tools/path = trajectory match and judge; "
        "no-leak/safe/no-abuse = safety set; ops-ok/ops-pass^k/no-side-fx = order-operations agent (end-state success, success in every repetition, no unintended changes). Scores are means over 24 / 10 / 12 examples, so differences "
        "under about 0.1 are within run-to-run noise.",
    ]
    return "\n".join(lines) + "\n"


def save(results: list[ModelResult], meta: dict, out_dir: Path, report_path: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    (out_dir / f"matrix-{stamp}-{slug(results[0].spec)[-24:]}-{len(results)}.json").write_text(
        json.dumps({"meta": meta, "results": [r.__dict__ for r in results]}, indent=2), encoding="utf-8"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_markdown(results, meta), encoding="utf-8")
