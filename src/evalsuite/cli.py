"""Command line entry point: `python -m evalsuite.cli <command>`."""

from __future__ import annotations

import argparse
import sys

from .config import get_settings
from .datasets import DATASET_NAMES, seed_dataset
from .regression import GateConfig, compare, load_baseline, save_baseline


def _cmd_seed(args) -> int:
    for key in DATASET_NAMES:
        print(f"seeded {seed_dataset(key)}")
    return 0


def _cmd_benchmark(args) -> int:
    from .runners import run_benchmark

    results = run_benchmark(args.prompt_version, args.prefix)
    print(f"experiment: {results.experiment_name}")
    return 0


def _cmd_trajectory(args) -> int:
    from .runners import run_trajectory

    print(f"experiment: {run_trajectory(args.prompt_version, args.prefix).experiment_name}")
    return 0


def _cmd_safety(args) -> int:
    from .runners import run_safety

    print(f"experiment: {run_safety(args.prompt_version, args.prefix).experiment_name}")
    return 0


def _cmd_calibrate(args) -> int:
    from .runners import run_calibration

    print(f"experiment: {run_calibration().experiment_name}")
    return 0


def _cmd_regression(args) -> int:
    """Run the benchmark for a candidate prompt and gate it against the stored baseline."""
    from .evaluators.composite import COMPOSITE_KEY, weighted_score
    from .runners import run_benchmark, score_table

    baseline_path = get_settings().baselines_dir / "qa_baseline.json"
    if not args.update_baseline and not baseline_path.exists():
        print(f"no baseline at {baseline_path}; run with --update-baseline first", file=sys.stderr)
        return 2
    results = run_benchmark(args.prompt_version, args.prefix or f"regression-{args.prompt_version}")
    table = score_table(results)
    for scores in table.values():
        scores[COMPOSITE_KEY] = weighted_score(scores)
    if args.update_baseline:
        save_baseline(baseline_path, table, {"prompt_version": args.prompt_version, "experiment": results.experiment_name})
        print(f"baseline written to {baseline_path}")
        return 0
    report = compare(load_baseline(baseline_path), table, GateConfig(max_mean_drop=args.max_drop))
    print(report.render())
    return 0 if report.passed else 1


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

    sub.add_parser("calibrate", help="Measure LLM-judge agreement with human labels").set_defaults(fn=_cmd_calibrate)

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
