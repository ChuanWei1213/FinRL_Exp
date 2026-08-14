"""Terminal entry point for multi-window Synthetic-vs-Real experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from finrl.experiments.notebook_utils import find_project_root
from finrl.experiments.synthetic_vs_real import load_experiment_config
from finrl.experiments.synthetic_vs_real import run_experiment_suite


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate real-only, synthetic-only, and balanced "
            "real+synthetic PPO policies across chronological time windows."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/synthetic_vs_real.json",
        help="JSON experiment config, relative to the project root by default.",
    )
    parser.add_argument(
        "--mode",
        choices=("smoke", "full"),
        default="smoke",
        help="Training budget defined in the config (default: smoke).",
    )
    parser.add_argument(
        "--stage",
        choices=("all", "train", "evaluate"),
        default="all",
        help=(
            "Run the complete train/evaluate pipeline, training only, or "
            "evaluation from completed training artifacts (default: all)."
        ),
    )
    parser.add_argument(
        "--workers",
        type=positive_int,
        default=4,
        help="Parallel PPO training processes, each limited to one CPU (default: 4).",
    )
    parser.add_argument(
        "--window",
        action="append",
        dest="windows",
        help="Run only this named time split; repeat to select multiple splits.",
    )
    parser.add_argument(
        "--ticker-group",
        action="append",
        dest="ticker_groups",
        help=(
            "Run only this named ticker group; repeat to select multiple groups. "
            "Defaults to active_ticker_groups in the config."
        ),
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help=(
            "Disable both training and evaluation caches; training artifacts "
            "are written below the experiment result directory."
        ),
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help=(
            "Retrain in all/train stages and rebuild evaluation outputs; in "
            "evaluate stage, keep models but force reevaluation."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable tqdm progress bars (normal status lines remain).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = find_project_root(Path.cwd())
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = project_root / config_path
    config = load_experiment_config(config_path)
    suite = run_experiment_suite(
        project_root=project_root,
        config=config,
        mode=args.mode,
        split_names=args.windows,
        ticker_group_names=args.ticker_groups,
        use_cache=not args.no_cache,
        force_retrain=args.force_retrain,
        show_progress=not args.quiet,
        stage=args.stage,
        workers=args.workers,
    )
    cases = suite.window_results or suite.training_cases
    summary = {
        "stage": suite.stage,
        "workers": args.workers,
        "manifest": str(suite.manifest_path) if suite.manifest_path else None,
        "suite_root": str(suite.suite_root) if suite.suite_root else None,
        "cases": [
            {
                "window_id": result.split.name,
                "ticker_group": result.ticker_group,
                "tickers": list(result.tickers),
            }
            for result in cases
        ],
        "test_rows": len(suite.test_metrics),
        "continuous_metric_rows": len(suite.continuous_metrics),
        "paired_delta_rows": len(suite.paired_deltas),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
