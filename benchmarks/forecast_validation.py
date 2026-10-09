"""Prespecified multi-series forecasts and same-target inference validation.

Install the package and statsmodels in a validation environment, then run:
  python benchmarks/forecast_validation.py --mode quick --output result.json
Commit this runner and its sibling protocol before running the benchmark.
Only JSON is written; the output must be a new file. Full mode changes only
the simulation replication count, not models, data selection or inference.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy.stats import norm

import strategy_inference
from strategy_inference import (
    Autoregression,
    SeasonalNaive,
    backtest,
    compare_forecasts,
    drift_forecast,
    evaluate_forecasts,
    load_dataset,
    naive_forecast,
    select_forecaster,
)

PROTOCOL_PATH = Path(__file__).with_name("forecast_validation_protocol.json")
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def wilson(decisions):
    """A 95% binomial interval, including all-success and all-failure cases."""
    n = len(decisions)
    if not n:
        raise ValueError("At least one decision is required.")
    p, z = float(np.mean(decisions)), float(norm.isf(0.025))
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    radius = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [max(0.0, float(center - radius)), min(1.0, float(center + radius))]


def mean_variance(n, phi):
    """Exact variance of a stationary unit-marginal-variance AR(1) mean."""
    offsets = np.arange(1, n)
    return float((n + 2 * np.sum((n - offsets) * phi**offsets)) / n**2)


def _reference_ar(histories, lags, options):
    from statsmodels.tsa.ar_model import AutoReg

    library = Autoregression(lags=lags)
    leads = np.array([1])
    maximum = lags if isinstance(lags, int) else max(lags)

    def own():
        return np.array([library(history, leads)[0] for history in histories])

    def reference():
        return np.array(
            [
                AutoReg(history, lags=lags, trend="c", hold_back=maximum)
                .fit()
                .predict(start=len(history), end=len(history), dynamic=False)[0]
                for history in histories
            ]
        )

    # Warm-up is excluded from the timed repetitions.
    own()
    reference()
    timings = {"library": [], "statsmodels": []}
    results = {}
    functions = (("library", own), ("statsmodels", reference))
    for repeat in range(options["repetitions"]):
        for name, function in functions[:: 1 if repeat % 2 == 0 else -1]:
            started = time.perf_counter()
            results[name] = function()
            timings[name].append(time.perf_counter() - started)
    expected, actual = results["statsmodels"], results["library"]
    return dict(
        lags=lags,
        n_origins=len(histories),
        passed=bool(np.allclose(actual, expected, rtol=options["rtol"], atol=options["atol"])),
        max_absolute_difference=float(np.max(np.abs(actual - expected))),
        median_seconds={name: float(np.median(times)) for name, times in timings.items()},
        timing_seconds=timings,
    )


def _panel_case(dataset, name, options=PROTOCOL["panel"]):
    values = dataset[name].values
    if len(values) != options["n_observations"] or not np.isfinite(values).all():
        raise ValueError(
            f"{name}: unexpected length or missing/nonfinite values; no series replaced."
        )
    train_stop, validation_stop = [p * len(values) // 100 for p in options["split_percent"]]
    candidates, configurations = {}, {}
    for raw_lags in options["lag_sets"]:
        lags = tuple(raw_lags) if isinstance(raw_lags, list) else raw_lags
        for ridge in options["ridge_grid"]:
            label = f"lags={lags}/ridge={ridge:g}"
            candidates[label] = Autoregression(lags=lags, ridge=ridge)
            configurations[label] = dict(lags=raw_lags, ridge=ridge)
    choice = select_forecaster(
        values[:train_stop],
        values[train_stop:validation_stop],
        candidates,
        window=options["window"],
        loss=options["selection_loss"],
    )
    models = dict(
        naive=naive_forecast,
        seasonal=SeasonalNaive(options["seasonal_period"]),
        drift=drift_forecast,
        selected_ar=choice.forecaster,
    )
    run = backtest(
        values,
        models,
        initial_train_size=validation_stop,
        window=options["window"],
        horizon=1,
    )
    mae = evaluate_forecasts(run, loss="absolute").mean_loss[0]
    rmse = np.sqrt(evaluate_forecasts(run, loss="squared").mean_loss[0])
    period = options["seasonal_period"]
    denominator = float(np.mean(np.abs(values[period:train_stop] - values[: train_stop - period])))
    histories = [values[split.train_start : split.train_stop] for split in run.splits]
    references = [
        _reference_ar(
            histories, tuple(lags) if isinstance(lags, list) else lags, options["reference"]
        )
        for lags in options["lag_sets"]
    ]
    comparison = compare_forecasts(
        run,
        baseline=options["comparison_baseline"],
        lead_time=1,
        loss=options["comparison_loss"],
        seed=PROTOCOL["seed"],
        **PROTOCOL["inference"],
    )
    return dict(
        series=name,
        train=[0, train_stop],
        validation=[train_stop, validation_stop],
        test=[validation_stop, len(values)],
        n_test_targets=run.n_folds,
        selection=choice.to_dict(),
        configurations=configurations,
        mase_training_denominator=denominator,
        scores=[
            dict(
                model=model,
                mae=float(mae[i]),
                rmse=float(rmse[i]),
                mase=None if denominator == 0 else float(mae[i] / denominator),
            )
            for i, model in enumerate(run.names)
        ],
        comparison=comparison.to_dict(),
        references=references,
    )


def panel(*, cache_dir=None, offline=False):
    options = PROTOCOL["panel"]
    dataset = load_dataset(options["dataset"], cache_dir=cache_dir, offline=offline)
    if dataset.frequency != options["frequency"]:
        raise ValueError("Unexpected archive frequency.")
    return dict(
        dataset=dataset.name,
        source=dataset.source,
        revision=dataset.revision,
        sha256=dataset.sha256,
        license=dataset.license,
        series=[_panel_case(dataset, name) for name in options["series"]],
    )


def simulation(mode="quick"):
    options = PROTOCOL["simulation"]
    n, initial = options["n_observations"], options["initial_train_size"]
    replications = options["replications"][mode]
    forecasts = {
        name: (lambda train, leads, value=value: np.full(len(leads), value))
        for name, value in options["forecasts"].items()
    }
    cells = []
    for phi_index, phi in enumerate(options["phi"]):
        decisions = {delta: {"bootstrap": [], "oracle": []} for delta in options["delta"]}
        variance = mean_variance(n, phi)
        for replicate in range(replications):
            rng = np.random.default_rng(
                np.random.SeedSequence([PROTOCOL["seed"], 1, phi_index, replicate])
            )
            innovations = rng.normal(size=n + initial)
            centered = np.empty_like(innovations)
            centered[0] = innovations[0]
            for t in range(1, len(centered)):
                centered[t] = phi * centered[t - 1] + np.sqrt(1 - phi * phi) * innovations[t]
            seed = int(
                np.random.SeedSequence([PROTOCOL["seed"], 2, phi_index, replicate]).generate_state(
                    1
                )[0]
            )
            for delta in options["delta"]:
                values = centered - delta
                run = backtest(values, forecasts, initial_train_size=initial, horizon=1)
                result = compare_forecasts(
                    run,
                    baseline=options["baseline"],
                    lead_time=1,
                    loss=options["loss"],
                    seed=seed,
                    **PROTOCOL["inference"],
                )
                decisions[delta]["bootstrap"].append(bool(result.inference.global_reject))
                statistic = float(np.mean(-values[initial:])) / np.sqrt(variance)
                decisions[delta]["oracle"].append(
                    bool(statistic > norm.isf(PROTOCOL["inference"]["alpha"]))
                )
        for delta, paired in decisions.items():
            difference = np.asarray(paired["bootstrap"], dtype=float) - np.asarray(
                paired["oracle"], dtype=float
            )
            cells.append(
                dict(
                    phi=phi,
                    delta=delta,
                    target="boundary_null" if delta == 0 else "alternative",
                    replications=replications,
                    oracle_mean_variance=variance,
                    methods={
                        name: dict(
                            rejections=sum(bits), rate=float(np.mean(bits)), wilson_95=wilson(bits)
                        )
                        for name, bits in paired.items()
                    },
                    paired_rate_difference=float(np.mean(difference)),
                    paired_mcse=float(np.std(difference, ddof=1) / np.sqrt(replications)),
                    decisions=paired,
                )
            )
    return cells


def _source():
    root = PROTOCOL_PATH.parent.parent
    package_path = Path(strategy_inference.__file__).resolve().parent
    if package_path != (root / "src" / "strategy_inference").resolve():
        raise RuntimeError(
            "Benchmark requires this checkout's editable package; "
            "run python -m pip install -e '.[dev]'. "
            f"Imported package from {package_path}."
        )

    def git(*arguments):
        return subprocess.check_output(["git", *arguments], cwd=root, text=True).strip()

    paths = [str(Path(__file__).relative_to(root)), str(PROTOCOL_PATH.relative_to(root))]
    if git("status", "--porcelain", "--", *paths):
        raise RuntimeError("Commit the runner and protocol before running the benchmark.")
    return dict(
        commit=git("rev-parse", "HEAD"),
        protocol_commit=git("log", "-1", "--format=%H", "--", paths[1]),
        sha256={path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in paths},
        core_sha256={
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((root / "src" / "strategy_inference").rglob("*.py"))
        },
        python=platform.python_version(),
        platform=platform.platform(),
        library=strategy_inference.__version__,
        package_path=str(package_path),
        dependencies={name: importlib.metadata.version(name) for name in ("numpy", "scipy")},
        thread_environment={
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--suite", choices=("all", "panel", "simulation"), default="all")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    started = time.perf_counter()
    record = dict(schema_version=1, mode=args.mode, protocol=PROTOCOL, source=_source())
    if args.suite != "simulation":
        record["source"]["dependencies"]["statsmodels"] = importlib.metadata.version("statsmodels")
        record["panel"] = panel(cache_dir=args.cache_dir, offline=args.offline)
    if args.suite != "panel":
        record["simulation"] = simulation(args.mode)
    record["elapsed_seconds"] = time.perf_counter() - started
    contents = json.dumps(record, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(contents)
    print(args.output.resolve())
    if any(
        not check["passed"]
        for row in record.get("panel", {}).get("series", [])
        for check in row["references"]
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
