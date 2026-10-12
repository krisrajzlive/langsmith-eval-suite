"""Command line entry point: `python -m evalsuite.cli <command>`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import get_settings
from .datasets import DATASET_NAMES, seed_dataset
from .regression import GateConfig, compare, load_baseline, save_baseline


def _report(results) -> None:
    """Print the experiment name and mean score per metric, with pass counts for boolean checks."""
    from .regression import metric_means
    from .runners import score_table

    table = score_table(results)
    print(f"experiment: {results.experiment_name}  ({len(table)} examples)")
    for metric, value in sorted(metric_means(table).items()):
        n = sum(1 for scores in table.values() if metric in scores)
        print(f"  {metric:<26} {value:6.3f}   (n={n})")
    binary = {m for m in metric_means(table) if all(s[m] in (0.0, 1.0) for s in table.values() if m in s)}
    failing = sorted((m, ex) for ex, scores in table.items() for m, v in scores.items() if m in binary and v < 0.5)
    for metric, example in failing:
        print(f"  FAIL {metric}: {example}")


def _cmd_seed(args) -> int:
    for key in DATASET_NAMES:
        print(f"seeded {seed_dataset(key)}")
    return 0


def _cmd_benchmark(args) -> int:
    from .runners import run_benchmark

    _report(run_benchmark(args.prompt_version, args.prefix))
    return 0


def _cmd_trajectory(args) -> int:
    from .runners import run_trajectory

    _report(run_trajectory(args.prompt_version, args.prefix))
    return 0


def _cmd_safety(args) -> int:
    from .runners import run_safety

    _report(run_safety(args.prompt_version, args.prefix))
    return 0


def _cmd_calibrate(args) -> int:
    from .runners import run_calibration

    results = run_calibration(prefix=args.prefix or "judge-calibration", judge=args.judge)
    rows = [i["evaluation_results"]["results"][0].score for i in results]
    print(f"experiment: {results.experiment_name}")
    print(f"  judge agrees with humans on {sum(map(bool, rows))}/{len(rows)}")
    return 0


def _cmd_matrix(args) -> int:
    from datetime import datetime, timezone

    from .matrix import run_matrix, save

    s = get_settings()
    specs = [m.strip() for m in args.models.split(",") if m.strip()]
    suites = [m.strip() for m in args.suites.split(",") if m.strip()]
    meta = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "prompt_version": args.prompt_version,
        "judge": args.judge or f"{s.judge_provider}:{s.judge_model}",
        "repetitions": s.eval_num_repetitions,
    }
    done: list = []

    def checkpoint(row) -> None:
        done.append(row)
        save(done, meta, s.baselines_dir.parent / "results", args.report)  # partial results survive a crash

    run_matrix(specs, suites, args.prompt_version, args.judge, on_result=checkpoint)
    print(f"report written to {args.report}")
    return 0


def _cmd_regression(args) -> int:
    """Run the benchmark for a candidate prompt and gate it against the stored baseline.

    With --update-baseline the new scores replace the baseline after the comparison is printed.
    """
    from .evaluators.composite import COMPOSITE_KEY, weighted_score
    from .runners import run_benchmark, score_table

    baseline_path = get_settings().baselines_dir / "qa_baseline.json"
    if not args.update_baseline and not baseline_path.exists():
        print(f"no baseline at {baseline_path}; run with --update-baseline first", file=sys.stderr)
        return 2
    results = run_benchmark(args.prompt_version, args.prefix or f"regression-{args.prompt_version}")
    _report(results)
    table = score_table(results)
    for scores in table.values():
        scores[COMPOSITE_KEY] = weighted_score(scores)
    exit_code = 0
    if baseline_path.exists():
        report = compare(load_baseline(baseline_path), table, GateConfig(max_mean_drop=args.max_drop))
        print(report.render())
        exit_code = 0 if report.passed else 1
    if args.update_baseline:
        save_baseline(baseline_path, table, {"prompt_version": args.prompt_version, "experiment": results.experiment_name})
        print(f"baseline written to {baseline_path}")
        return 0
    return exit_code


def _cmd_pairwise(args) -> int:
    from .runners import run_pairwise

    print(f"experiment: {run_pairwise(args.experiment_a, args.experiment_b).experiment_name}")
    return 0


def _cmd_composite(args) -> int:
    from .evaluators.composite import write_composite_feedback

    print(f"composite feedback written for {write_composite_feedback(args.experiment)} runs")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evalsuite")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("seed", help="Create or update LangSmith datasets from datasets/*.jsonl").set_defaults(fn=_cmd_seed)

    for name, fn, help_text in [
        ("benchmark", _cmd_benchmark, "Correctness, RAG quality, retrieval and format checks"),
        ("trajectory", _cmd_trajectory, "Agent tool-selection and trajectory quality"),
        ("safety", _cmd_safety, "Prompt-injection, leakage, PII and scope checks"),
        ("regression", _cmd_regression, "Benchmark a candidate prompt and gate it against the baseline"),
    ]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--prompt-version", default=get_settings().agent_prompt_version)
        p.add_argument("--prefix", default=None)
        if name == "regression":
            p.add_argument("--update-baseline", action="store_true")
            p.add_argument("--max-drop", type=float, default=GateConfig().max_mean_drop)
        p.set_defaults(fn=fn)

    p = sub.add_parser("calibrate", help="Measure LLM-judge agreement with human labels")
    p.add_argument("--judge", default=None, help="provider:model to calibrate (default: configured judge)")
    p.add_argument("--prefix", default=None)
    p.set_defaults(fn=_cmd_calibrate)

    p = sub.add_parser("matrix", help="Compare several application models on the same suites")
    p.add_argument("--models", required=True, help="comma-separated provider:model specs")
    p.add_argument("--suites", default="benchmark,trajectory,safety")
    p.add_argument("--prompt-version", default=get_settings().agent_prompt_version)
    p.add_argument("--judge", default=None, help="provider:model held fixed across rows")
    p.add_argument("--report", type=Path, default=Path("docs/model-comparison.md"))
    p.set_defaults(fn=_cmd_matrix)

    p = sub.add_parser("pairwise", help="Compare two finished experiments with a pairwise judge")
    p.add_argument("experiment_a")
    p.add_argument("experiment_b")
    p.set_defaults(fn=_cmd_pairwise)

    p = sub.add_parser("composite", help="Write weighted composite feedback onto a finished experiment")
    p.add_argument("experiment")
    p.set_defaults(fn=_cmd_composite)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
