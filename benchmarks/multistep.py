"""Reproduce local multi-step interval timings and incremental allocation peaks.

Run only after the source and package version are stable:
    python benchmarks/multistep.py --output benchmarks/results/multistep-0.7.json

Each case runs in a separate serial process with BLAS thread settings fixed to
one. Imports, seeded AR data, forecasts, wrapper/stream consistency checks and
fingerprints are outside timing. These are engineering measurements, not
statistical calibration experiments or universal performance guarantees.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
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
PHI = 0.65
LEAD_TIMES = (1, 6, 12, 24)
THREAD_ENV = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
}
CASES = (
    "batch_12000x4",
    "pooled_stream_50000x4",
    "pooled_stream_1000x4",
    "interlaced_stream_50000x4",
)
SEEDS = {12_000: 20261230, 50_000: 20261231, 1000: 20261232}


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
    """Hash array bytes without expanding or serializing a trajectory."""
    if isinstance(value, np.ndarray):
        return _array_record(value)
    if isinstance(value, np.generic):
        return _encode(value.item(), np)
    if is_dataclass(value):
        return {field.name: _encode(getattr(value, field.name), np) for field in fields(value)}
    if isinstance(value, Mapping):
        return {key: _encode(item, np) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_encode(item, np) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        # Batch arrays may contain unevaluated NaNs, while pending dataclasses
        # may contain empty/full boundaries. Their bytes or symbols are hashed;
        # no nonstandard JSON number is emitted.
        return {"nonfinite_float": "nan" if math.isnan(value) else "inf" if value > 0 else "-inf"}
    return value


def _fingerprint(value, np):
    encoded = json.dumps(_encode(value, np), sort_keys=True, allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _ar_inputs(np, n):
    """Stationary AR(.65), innovation variance one, known-DGP point forecasts."""
    rng = np.random.default_rng(SEEDS[n])
    innovations = rng.normal(size=n)
    actual = np.empty(n)
    actual[0] = innovations[0] / math.sqrt(1 - PHI**2)
    for t in range(1, n):
        actual[t] = PHI * actual[t - 1] + innovations[t]
    powers = PHI ** np.asarray(LEAD_TIMES)
    predicted = actual[:, None] * powers[None, :]
    origins = np.arange(n, dtype=np.int64)
    initial_scales = np.asarray([
        math.sqrt(math.fsum(PHI ** (2 * i) for i in range(lead)))
        for lead in LEAD_TIMES
    ])
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("The fixed seeded benchmark input must be finite.")
    return actual, predicted, origins, initial_scales


def _stream(tracker_class, actual, predicted, options):
    tracker = tracker_class(LEAD_TIMES, **options)
    for t, (label, path) in enumerate(zip(actual, predicted, strict=True)):
        tracker.observe(t, float(label))
        tracker.predict(path)
    # No labels or interval trajectory are retained here, and future pending
    # forecasts are deliberately not flushed. Export happens outside timing.
    return tracker


def _terminal_state(value, *, batch):
    return value.diagnostics if batch else value.to_dict()


def _check_equivalence(batch_result, tracker, np, n):
    """Compare both public interfaces on the same input, outside measurement."""
    batch_state, stream_state = batch_result.diagnostics, tracker.to_dict()
    if batch_state != stream_state or batch_result.pending != tracker.pending:
        raise ValueError("The same-input batch and stream terminal states differ.")
    counts = tuple(int(value) for value in np.count_nonzero(batch_result.evaluated, axis=0))
    if counts != tuple(n - lead for lead in LEAD_TIMES) or counts != tracker.n_updates:
        raise ValueError("Mature feedback counts or the unflushed tail differ.")
    misses = tuple(int(value) for value in np.count_nonzero(
        batch_result.evaluated & batch_result.misses, axis=0
    ))
    lane_misses = tuple(sum(
        state["misses"] for state in stream_state["states"] if state["lead_time"] == lead
    ) for lead in LEAD_TIMES)
    if misses != lane_misses:
        raise ValueError("Batch miss arrays and streaming lane miss counts differ.")
    return {
        "same_input_batch_stream_terminal_state": True,
        "fields_checked": [
            "lane_quantiles", "lane_updates", "lane_misses", "per_lead_evaluated_counts",
            "array_miss_counts", "current_scales", "pending_intervals", "complete_terminal_record",
        ],
        "batch_result_sha256": _fingerprint(batch_result, np),
        "terminal_state_sha256": _fingerprint(stream_state, np),
        "n_evaluated": list(counts),
        "misses": list(misses),
        "n_pending": len(tracker.pending),
    }


def _case(name, si, np):
    n = 12_000 if name == "batch_12000x4" else 1000 if name == "pooled_stream_1000x4" else 50_000
    batch = name.startswith("batch_")
    strategy = "interlaced" if name.startswith("interlaced_") else "pooled"
    actual, predicted, origins, initial_scales = _ar_inputs(np, n)
    options = dict(
        alpha=0.1, step_size=0.1, decay=0.2, scale=initial_scales,
        initial_quantile=0.65, strategy=strategy, scale_decay=0.97,
        scale_source="shortest", scale_floor=1e-8,
    )
    tracker_class, wrapper = si.MultiStepConformal, si.multistep_intervals

    def stream():
        return _stream(tracker_class, actual, predicted, options)

    def trajectory():
        return wrapper(actual, predicted, origins=origins, lead_times=LEAD_TIMES, **options)

    # The untimed reference also validates the four-lead feedback clock. It
    # does not compare fingerprints across lengths or across feedback schemes.
    reference_batch, reference_tracker = trajectory(), stream()
    equivalence = _check_equivalence(reference_batch, reference_tracker, np, n)
    del reference_batch, reference_tracker
    settings = {
        "n_obs": n,
        "n_origins": n,
        "n_leads": len(LEAD_TIMES),
        "lead_times": list(LEAD_TIMES),
        "seed": SEEDS[n],
        "dgp": {
            "model": "Stationary Gaussian AR(1)",
            "phi": PHI,
            "innovation_variance": 1.0,
            "initial_variance": 1 / (1 - PHI**2),
        },
        "forecast": "phi**lead * current_observation; supplied points, no model fitting.",
        "initial_scale_formula": "sqrt(sum(phi**(2*i) for i in range(lead)))",
        "initial_scale_scope": "Known-DGP formula for this engineering input, not an estimated calibration claim.",
        "options": _encode(options, np) | {"scale": initial_scales.tolist()},
        "inputs": {
            "actual": _array_record(actual), "predicted": _array_record(predicted),
            "origins": _array_record(origins),
        },
        "feedback_order": "At each t: observe(t, actual[t]), then predict(predicted[t]).",
        "future_pending_flushed": False,
        "consistency_check_excluded_from_timing": equivalence,
        "retained_output": "Arrays for every origin and lead plus terminal pending forecasts."
        if batch else "One tracker, active lanes and pending forecasts; no interval trajectory.",
        "purpose": "Memory-scaling probe." if n == 1000 else "Local timing and incremental allocation.",
    }
    return trajectory if batch else stream, settings, batch


def _worker(name):
    sys.path.insert(0, str(ROOT / "src"))
    import numpy as np

    import strategy_inference as si

    if Path(si.__file__).resolve().parent != (ROOT / "src/strategy_inference").resolve():
        raise ValueError("Benchmarking requires this checkout's library.")
    function, settings, batch = _case(name, si, np)

    def fingerprint(value):
        return _fingerprint(value if batch else _terminal_state(value, batch=False), np)

    warmup = function()  # One explicit warmup, after untimed interface checks.
    result_hash = fingerprint(warmup)
    del warmup
    samples = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        result = function()
        samples.append(time.perf_counter() - start)
        if fingerprint(result) != result_hash:
            raise ValueError(f"Repeated calls changed the result in {name}.")
        del result
    tracemalloc.start()
    result = function()
    current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    if fingerprint(result) != result_hash:
        raise ValueError(f"The allocation measurement changed the result in {name}.")
    terminal = _terminal_state(result, batch=batch)
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
        "result_sha256": result_hash,
        "terminal_state": terminal,
        "terminal_state_sha256": _fingerprint(terminal, np),
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
    if args.output.exists() or args.output.is_symlink():
        parser.error("Output already exists; use a new path to preserve the earlier measurement.")
    sources = _source_hashes()
    runner_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    env = dict(os.environ, **THREAD_ENV)
    results = []
    for name in CASES:
        worker = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker", name],
            check=True, capture_output=True, text=True, env=env, timeout=600,
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
    by_name = {record["case"]: record for record in results}
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
        "statistical_study_scope": "These performance tasks use scalar step_size=0.1. The separate statistical study uses per-lead 0.1/sqrt(lead); runtime measurements do not establish scale-sharing efficiency or statistical validity.",
        "timing_scope": "Imports, input generation, known-DGP forecasts/scales, same-input wrapper/stream checks, post-call state export and fingerprints are excluded. API-internal output construction is included.",
        "memory_scope": "One separate warmed tracemalloc call; includes newly allocated output, excludes imports, fingerprints and pre-existing input.",
        "blas_thread_environment": THREAD_ENV,
        "cases": results,
        "streaming_memory_comparison": {
            "short_case": "pooled_stream_1000x4",
            "long_case": "pooled_stream_50000x4",
            "short_n_obs": 1000,
            "long_n_obs": 50_000,
            "short_peak_bytes": by_name["pooled_stream_1000x4"]["traced_peak_bytes"],
            "long_peak_bytes": by_name["pooled_stream_50000x4"]["traced_peak_bytes"],
            "scope": "Different seeded inputs with independent fingerprints. Two measured lengths supplement implementation inspection; they do not prove an asymptotic memory law.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
