"""Command-line entry points for a CSV audit and the prespecified experiments."""

import argparse
import hashlib
import sys
from pathlib import Path

from . import __version__
from .audit import audit_returns
from .io import read_returns_csv
from .report import write_audit_report, write_experiment_report


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="strategy-inference",
        description="Mean-return inference under time dependence and candidate selection.",
    )
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="Audit the candidate family in a numeric CSV.")
    audit.add_argument("input", type=Path)
    audit.add_argument("--output", type=Path, default=Path("results/audit"))
    audit.add_argument("--benchmark", help="Subtract this column from every candidate.")
    audit.add_argument("--n-resamples", type=int, default=999)
    audit.add_argument("--block-length", type=float)
    audit.add_argument("--lags", type=int)
    audit.add_argument("--alpha", type=float, default=0.05)
    audit.add_argument("--seed", type=int, default=0)
    complete = audit.add_mutually_exclusive_group()
    complete.add_argument(
        "--complete-search", dest="search_complete", action="store_const", const=True
    )
    complete.add_argument(
        "--incomplete-search", dest="search_complete", action="store_const", const=False
    )
    audit.set_defaults(search_complete=None)
    reproduce = commands.add_parser(
        "reproduce", help="Run the three prespecified simulation figures."
    )
    reproduce.add_argument("--profile", choices=("quick", "full"), default="full")
    reproduce.add_argument("--output", type=Path, default=Path("results/full"))
    reproduce.add_argument("--seed", type=int, default=20261002)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "audit":
            table = read_returns_csv(args.input, benchmark=args.benchmark)
            result = audit_returns(
                table.values,
                names=table.names,
                n_resamples=args.n_resamples,
                block_length=args.block_length,
                lags=args.lags,
                alpha=args.alpha,
                seed=args.seed,
                search_complete=args.search_complete,
            )
            report = write_audit_report(
                result,
                args.output,
                provenance={
                    "input_name": args.input.name,
                    "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
                    "benchmark_column": args.benchmark,
                    "date_column_present": table.dates is not None,
                    "seed": args.seed,
                    "package_version": __version__,
                },
            )
            print(f"Selected: {result.selected_name}; family p = {result.global_pvalue:.4f}")
            print(f"Report: {report.resolve()}")
            for warning in result.warnings:
                print(f"Note: {warning}", file=sys.stderr)
        else:
            from .experiments import run_experiments

            metadata = run_experiments(args.output, profile=args.profile, seed=args.seed)
            report = write_experiment_report(args.output)
            print(f"Status: {metadata['status']}; elapsed: {metadata['elapsed_seconds']:.1f}s")
            print(f"Report: {report.resolve()}")
    except ImportError as exc:
        print(
            f"Missing optional dependency: {exc}. Install 'strategy-inference[figures]'.",
            file=sys.stderr,
        )
        return 2
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0
