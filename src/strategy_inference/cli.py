"""Command-line entry points for a CSV audit and the prespecified experiments."""

import argparse
import csv
import hashlib
import io
import json
import sys
from pathlib import Path

from . import __version__


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
    audit.add_argument("--studentization", choices=("fixed", "resampled"), default="fixed")
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
    test = commands.add_parser("test", help="Test return means; export compact JSON or CSV.")
    test.add_argument("input", type=Path)
    test.add_argument("--method", choices=("bootstrap", "gaussian_ar"), default="bootstrap")
    test.add_argument("--output", default="-", help="Output path, or '-' for stdout (default).")
    test.add_argument("--format", choices=("json", "csv"), help="Default: CSV for .csv, JSON otherwise.")
    test.add_argument("--benchmark", help="Subtract this numeric column from every candidate.")
    test.add_argument("--alpha", type=float, default=0.05)
    test.add_argument("--n-resamples", type=int)
    test.add_argument("--seed", type=int)
    test.add_argument("--batch-size", type=int)
    test.add_argument("--block-length", type=float)
    test.add_argument("--lags", type=int)
    test.add_argument("--studentization", choices=("fixed", "resampled"))
    test.add_argument("--beta", type=float)
    test.add_argument("--max-dimension", type=int)
    test.add_argument("--block-lengths", nargs="+", type=int)
    reproduce = commands.add_parser(
        "reproduce", help="Run the three prespecified simulation figures."
    )
    reproduce.add_argument("--profile", choices=("quick", "full"), default="full")
    reproduce.add_argument("--study", choices=("baseline", "calibration"), default="baseline")
    reproduce.add_argument("--output", type=Path)
    reproduce.add_argument("--seed", type=int)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "audit":
            from .audit import audit_returns
            from .io import read_returns_csv
            from .report import write_audit_report

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
                studentization=args.studentization,
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
        elif args.command == "test":
            from .io import read_returns_csv
            from .testing import test_returns

            if args.output != "-":
                path = Path(args.output)
                if path.resolve() == args.input.resolve() or (
                    path.exists() and args.input.exists() and path.samefile(args.input)
                ):
                    raise ValueError("Output must differ from the input CSV.")
            options = {name: getattr(args, name) for name in (
                "n_resamples", "seed", "batch_size", "block_length", "lags", "studentization",
                "beta", "max_dimension", "block_lengths",
            ) if getattr(args, name) is not None}
            table = read_returns_csv(args.input, benchmark=args.benchmark)
            result = test_returns(table, method=args.method, alpha=args.alpha, **options)
            output_format = args.format or ("csv" if args.output.endswith(".csv") else "json")
            if output_format == "csv":
                stream = io.StringIO(newline="")
                records = result.records()
                writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
                contents = stream.getvalue()
            else:
                record = result.to_dict()
                record["provenance"] = {
                    "input_name": args.input.name,
                    "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
                    "benchmark_column": args.benchmark,
                    "package_version": __version__,
                    "seed": options.get("seed", 0) if args.method == "bootstrap" else None,
                }
                contents = json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
            if args.output == "-":
                sys.stdout.write(contents)
            else:
                path = Path(args.output)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(contents, encoding="utf-8")
                print(f"Result: {path.resolve()}")
        else:
            output = args.output or (
                Path("results/calibration") / args.profile
                if args.study == "calibration"
                else Path("results") / args.profile
            )
            if args.study == "calibration":
                from .calibration import run_calibration
                from .calibration_plotting import plot_calibration
                from .calibration_report import write_calibration_report

                metadata = run_calibration(output, profile=args.profile, seed=args.seed)
                paths = plot_calibration(output, metadata)
                report = write_calibration_report(output)
                paths.append(report)
                for path in paths:
                    metadata["outputs"].append(path.name)
                    metadata["file_sha256"][path.name] = hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest()
                (output / "run-metadata.json").write_text(
                    json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8",
                )
            else:
                from .experiments import run_experiments
                from .report import write_experiment_report

                metadata = run_experiments(
                    output, profile=args.profile, seed=20261002 if args.seed is None else args.seed
                )
                report = write_experiment_report(output)
            print(f"Status: {metadata['status']}; elapsed: {metadata['elapsed_seconds']:.1f}s")
            print(f"Report: {report.resolve()}")
    except ImportError as exc:
        print(
            f"Missing optional dependency: {exc}. Install 'strategy-inference[figures]'.",
            file=sys.stderr,
        )
        return 2
    except (TypeError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0
