"""Regression gating: compare per-example scores of a candidate experiment against a stored baseline."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean

GATE_METRICS = ("correctness", "groundedness", "citations_grounded", "retrieval_recall")
PASS_THRESHOLD = 0.5


@dataclass
class GateConfig:
    max_mean_drop: float = 0.05  # tolerated drop in a metric's mean score
    max_regressed_examples: int = 1  # tolerated count of examples that passed before and fail now
    metrics: tuple[str, ...] = GATE_METRICS


@dataclass
class GateReport:
    passed: bool
    mean_deltas: dict[str, float] = field(default_factory=dict)
    regressed_examples: dict[str, list[str]] = field(default_factory=dict)
    improved_examples: dict[str, list[str]] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = ["Regression gate: " + ("PASS" if self.passed else "FAIL")]
        for metric, delta in sorted(self.mean_deltas.items()):
            reg = len(self.regressed_examples.get(metric, []))
            imp = len(self.improved_examples.get(metric, []))
            lines.append(f"  {metric:<22} mean delta {delta:+.3f}   regressed={reg} improved={imp}")
        lines += [f"  ! {f}" for f in self.failures]
        return "\n".join(lines)


def collect_scores(rows: list[dict]) -> dict[str, dict[str, float]]:
    """Normalize experiment rows to {example_source_id: {metric: score}}, skipping unscored feedback."""
    table: dict[str, dict[str, float]] = {}
    for row in rows:
        example_id = row["example_id"]
        for key, score in row["scores"].items():
            if score is not None:
                table.setdefault(example_id, {})[key] = float(score)
    return table


def metric_means(table: dict[str, dict[str, float]]) -> dict[str, float]:
    by_metric: dict[str, list[float]] = {}
    for scores in table.values():
        for key, value in scores.items():
            by_metric.setdefault(key, []).append(value)
    return {k: mean(v) for k, v in by_metric.items()}


def compare(baseline: dict, candidate: dict, config: GateConfig | None = None) -> GateReport:
    config = config or GateConfig()
    base_means, cand_means = metric_means(baseline), metric_means(candidate)
    report = GateReport(passed=True)
    for metric in config.metrics:
        if metric not in base_means or metric not in cand_means:
            continue
        delta = cand_means[metric] - base_means[metric]
        report.mean_deltas[metric] = delta
        regressed, improved = [], []
        for example_id, base_scores in baseline.items():
            before, after = base_scores.get(metric), candidate.get(example_id, {}).get(metric)
            if before is None or after is None:
                continue
            if before >= PASS_THRESHOLD > after:
                regressed.append(example_id)
            elif after >= PASS_THRESHOLD > before:
                improved.append(example_id)
        report.regressed_examples[metric], report.improved_examples[metric] = regressed, improved
        if delta < -config.max_mean_drop:
            report.failures.append(f"{metric}: mean dropped {delta:+.3f} (limit -{config.max_mean_drop})")
        if len(regressed) > config.max_regressed_examples:
            report.failures.append(
                f"{metric}: {len(regressed)} examples regressed (limit {config.max_regressed_examples}): {regressed}"
            )
    report.passed = not report.failures
    return report


def save_baseline(path: Path, table: dict, meta: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta or {}, "scores": table}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_baseline(path: Path) -> dict[str, dict[str, float]]:
    return json.loads(path.read_text(encoding="utf-8"))["scores"]
