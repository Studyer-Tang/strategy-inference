"""Reproduce local time-series API timings and incremental allocation peaks.

Run from a source checkout:
    python benchmarks/time_series.py --output benchmarks/results/time-series-0.6.json

Each case runs in its own serial process with BLAS thread settings fixed to one.
Imports, input generation and the comparison's existing backtest are excluded.
These are engineering measurements, not statistical calibration experiments.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import tracemalloc
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPEATS = 5
THREAD_ENV = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
}
CASES = (
    "squared_loss_10000x20",
    "backtest_4096",
    "compare_forecasts_958x2",
    "adaptive_intervals_100000",
    "adaptive_stream_100000",
    "adaptive_stream_1000",
)
SEEDS = {"squared_loss": 20261220, "backtest": 20261221, "conformal": 20261222}


def _source_hashes():
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((ROOT / "src/strategy_inference").glob("*.py"))
    }


def _array_record(values):
    return {
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "sha256": hashlib.sha256(values.tobytes(order="C")).hexdigest(),
    }


def _encode(value, np):
    if isinstance(value, np.ndarray):
        return _array_record(value)
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return {field.name: _encode(getattr(value, field.name), np) for field in fields(value)}
    if isinstance(value, Mapping):
        return {key: _encode(item, np) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_encode(item, np) for item in value]
    return value


def _fingerprint(value, np):
    encoded = json.dumps(_encode(value, np), sort_keys=True, allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _seasonal_ar(np, n: int, seed: int):
    """A fixed seasonal signal plus stationary AR noise; generated before timing."""
    rng = np.random.default_rng(seed)
    innovations = rng.normal(size=n)
    noise = np.empty(n)
    phi = 0.65
    noise[0] = innovations[0]
    for i in range(1, n):
        noise[i] = phi * noise[i - 1] + (1 - phi**2) ** 0.5 * innovations[i]
    position = np.arange(n)
    season = 2 * np.sin(2 * np.pi * position / 12) + 0.5 * np.cos(2 * np.pi * position / 24)
    actual = season + noise
    predicted = season + phi * np.r_[0.0, noise[:-1]]
    return actual, predicted


def _case(name, si, np):
    if name == "squared_loss_10000x20":
        actual, mean_prediction = _seasonal_ar(np, 10_000, SEEDS["squared_loss"])
        rng = np.random.default_rng(SEEDS["squared_loss"] + 1)
        predicted = mean_prediction[:, None] + rng.normal(size=(10_000, 20)) * 0.25
        function = si.forecast_loss
        return lambda: function(actual, predicted, loss="squared"), {
            "n_obs": 10_000,
            "n_models": 20,
            "seed": SEEDS["squared_loss"],
            "prediction_noise_seed": SEEDS["squared_loss"] + 1,
            "loss": "squared",
            "inputs": {"actual": _array_record(actual), "predicted": _array_record(predicted)},
        }
    if name in ("backtest_4096", "compare_forecasts_958x2"):
        actual, _ = _seasonal_ar(np, 4096, SEEDS["backtest"])
        models = {
            "naive": si.naive_forecast,
            "drift": si.drift_forecast,
            "seasonal_naive_12": si.SeasonalNaive(12),
        }
        options = dict(initial_train_size=256, window=256, horizon=12, step=4)
        settings = {
            "n_obs": 4096,
            "seed": SEEDS["backtest"],
            "models": list(models),
            "backtest_options": options,
            "input": _array_record(actual),
        }
        function = si.backtest
        if name == "backtest_4096":
            return lambda: function(actual, models, **options), settings
        existing = function(actual, models, **options)
        loss = si.forecast_loss(existing.actuals[:, 0], existing.forecasts[:, 0], loss="squared")
        improvements = loss[:, 0, None] - loss[:, 1:]
        variances = improvements.var(axis=0)
        if np.any(variances <= 0) or not np.isfinite(variances).all():
            raise ValueError("The fixed comparison family has a degenerate loss difference.")
        comparison = si.compare_forecasts
        comparison_options = dict(baseline="naive", lead_time=1, n_resamples=999, seed=17)
        settings.update(
            {
                "n_origins": existing.n_folds,
                "n_candidates": 2,
                "comparison_options": comparison_options,
                "loss_difference_variances": variances.tolist(),
                "existing_backtest_excluded": True,
            }
        )
        return lambda: comparison(existing, **comparison_options), settings
    n = 1000 if name == "adaptive_stream_1000" else 100_000
    actual, predicted = _seasonal_ar(np, n, SEEDS["conformal"])
    options = dict(alpha=0.1, step_size=0.1, decay=0.6, scale=1.0, initial_quantile=0.5)
    settings = {
        "n_obs": n,
        "seed": SEEDS["conformal"],
        "options": options,
        "inputs": {"actual": _array_record(actual), "predicted": _array_record(predicted)},
    }
    if name == "adaptive_intervals_100000":
        function = si.adaptive_intervals
        settings["retained_output"] = "Arrays for every observation."
        return lambda: function(actual, predicted, **options), settings
    tracker_class = si.AdaptiveConformal

    def stream():
        tracker = tracker_class(**options)
        for label, point in zip(actual, predicted, strict=True):
            tracker.predict(point)
            tracker.update(label)
        return {
            "n_updates": tracker.n_updates,
            "misses": tracker.misses,
            "next_quantile": tracker.quantile,
            "coverage": tracker.coverage,
            "pending": tracker.pending is not None,
        }

    settings["retained_output"] = "One tracker and its terminal scalar summary; no trajectory."
    settings["purpose"] = "Memory-scaling probe." if n == 1000 else "Streaming timing and memory."
    return stream, settings


def _worker(name):
    sys.path.insert(0, str(ROOT / "src"))
    import numpy as np

    import strategy_inference as si

    if Path(si.__file__).resolve().parent != (ROOT / "src/strategy_inference").resolve():
        raise ValueError("Benchmarking requires this checkout's library.")
    function, settings = _case(name, si, np)
    warmup = function()  # Includes any remaining lazy imports; not measured.
    fingerprint = _fingerprint(warmup, np)
    del warmup
    samples = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        result = function()
        samples.append(time.perf_counter() - start)
        if _fingerprint(result, np) != fingerprint:
            raise ValueError(f"Repeated calls changed the result in {name}.")
        del result
    tracemalloc.start()
    result = function()
    current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    if _fingerprint(result, np) != fingerprint:
        raise ValueError(f"The allocation measurement changed the result in {name}.")
    summary = result if name.startswith("adaptive_stream") else None
    runtime = io.StringIO()
    with contextlib.redirect_stdout(runtime):
        np.show_runtime()
    return {
        "case": name,
        "settings": settings,
        "warm_seconds": samples,
        "warm_median_seconds": statistics.median(samples),
        "traced_current_bytes": current_bytes,
        "traced_peak_bytes": peak_bytes,
        "result_sha256": fingerprint,
        "terminal_state": summary,
        "package_version": si.__version__,
        "numpy_version": np.__version__,
        "thread_environment": {key: os.environ.get(key) for key in THREAD_ENV},
        "numpy_runtime": runtime.getvalue(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", choices=CASES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(_worker(args.worker), allow_nan=False))
        return
    if args.output is None:
        parser.error("--output is required")
    if args.output.exists():
        parser.error("Output already exists; use a new path to preserve the earlier measurement.")
    sources = _source_hashes()
    runner_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    env = dict(os.environ, **THREAD_ENV)
    results = []
    for name in CASES:
        worker = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker", name],
            check=True,
            capture_output=True,
            text=True,
            env=env,
            timeout=180,
        )
        record = json.loads(worker.stdout)
        results.append(record)
        print(
            f"{name}: {record['warm_median_seconds'] * 1000:.3f} ms; "
            f"traced peak {record['traced_peak_bytes']:,} bytes",
            flush=True,
        )
    if (
        sources != _source_hashes()
        or runner_hash != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    ):
        raise ValueError("Source changed while benchmarking; repeat from stable source.")
    versions = {record["package_version"] for record in results}
    if len(versions) != 1:
        raise ValueError("Package version changed while benchmarking.")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "package_version": versions.pop(),
        "git_head_at_run": head,
        "candidate_source_sha256": sources,
        "benchmark_source_sha256": runner_hash,
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy": results[0]["numpy_version"],
        "repeats": REPEATS,
        "warmup_calls": 1,
        "scope": "Serial local engineering measurements, not calibration or universal performance.",
        "timing_scope": "Imports, input generation, fingerprints and the existing comparison backtest are excluded.",
        "memory_scope": "One separate warmed tracemalloc call; includes newly allocated output, excludes imports and pre-existing input.",
        "blas_thread_environment": THREAD_ENV,
        "cases": results,
        "streaming_memory_comparison": {
            "short_n_obs": 1000,
            "long_n_obs": 100_000,
            "short_peak_bytes": results[-1]["traced_peak_bytes"],
            "long_peak_bytes": results[-2]["traced_peak_bytes"],
            "scope": "Two measured lengths supplement inspection of the single-tracker implementation; they do not prove an asymptotic memory law.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
