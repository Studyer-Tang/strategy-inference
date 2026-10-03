"""Compare a local library with a Git revision in isolated, serial processes.

Usage: python benchmarks/run.py --baseline v0.4.0 --output benchmarks/results/library-0.5.json
Requires a source checkout. Benchmarks measure computation, not calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
import tracemalloc
from dataclasses import fields
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "hac_512x20": ("hac", 512, 20, 0),
    "bootstrap_fixed_512x20": ("fixed", 512, 20, 1999),
    "bootstrap_resampled_512x20": ("resampled", 512, 20, 1999),
    "bootstrap_resampled_1024x100": ("resampled", 1024, 100, 499),
    "gaussian_ar_512x20": ("gaussian_ar", 512, 20, 0),
    "gaussian_ar_2048x100": ("gaussian_ar", 2048, 100, 0),
    "multistep_summary_12000x4": ("multistep_summary", 12000, 4, 0),
    "multistep_stream_12000x4": ("multistep_stream", 12000, 4, 0),
    "bootstrap_fixed_8192x100": ("fixed", 8192, 100, 39),
}


def _source_hashes(package: Path):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(package.glob("*.py"))}


def _worker(case: str, package_root: Path, repeats: int):
    sys.path.insert(0, str(package_root))
    import numpy as np

    mode, t, k, b = CASES[case]
    rng = np.random.default_rng(20261201 + list(CASES).index(case))
    rho, phi = 0.35, 0.9
    x = np.sqrt(rho) * rng.normal(size=(t, 1)) + np.sqrt(1 - rho) * rng.normal(size=(t, k))
    for i in range(1, t):
        x[i] = phi * x[i - 1] + np.sqrt(1 - phi**2) * x[i]
    x[:, 0] += 3 * np.sqrt((1 + phi) / ((1 - phi) * t))
    started = time.perf_counter()
    si = importlib.import_module("strategy_inference")
    import_ms = 1000 * (time.perf_counter() - started)
    assert Path(si.__file__).resolve().is_relative_to(package_root.resolve())
    started = time.perf_counter()
    setup_ms, settings = 0.0, None
    if mode.startswith("multistep_"):
        tracker_class, batch = si.MultiStepConformal, si.multistep_intervals
    elif mode == "gaussian_ar":
        function, options = si.wilks_uncertainty_test, {}
    elif mode == "hac":
        function, options = si.long_run_variance, {}
    else:
        function, options = si.audit_returns, dict(n_resamples=b, seed=41, studentization=mode)
    api_load_ms = 1000 * (time.perf_counter() - started)
    if mode.startswith("multistep_"):
        started = time.perf_counter()
        leads = np.array([1, 6, 12, 24])
        actual = x[:, 0].copy()
        origins = np.arange(t - int(leads[-1]), dtype=np.int64)
        points = phi ** leads * actual[origins, None]
        options = dict(scale=np.sqrt((1 - phi ** (2 * leads)) / (1 - phi**2)),
            scale_decay=0.97, scale_source="blended", scale_share_weight=0.5,
            initial_quantile=0.65, step_size=0.1, decay=0.2)
        settings = dict(lead_times=leads.tolist(), options={
            name: value.tolist() if isinstance(value, np.ndarray) else value
            for name, value in options.items()},
            actual_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),
            predicted_sha256=hashlib.sha256(points.tobytes()).hexdigest(),
            scope="Known-coefficient AR workload; input and summary preparation excluded from timing.")
        tracker_options = options
        if mode == "multistep_summary":
            prepared = batch(actual, points, origins=origins, lead_times=leads, **options)
            def function(_):
                return prepared.summary()
        else:
            def function(_):
                tracker = tracker_class(leads, **tracker_options)
                for i, value in enumerate(actual):
                    tracker.observe(i, float(value))
                    if i < len(points):
                        tracker.predict(points[i])
                return tracker
        options = {}
        setup_ms = 1000 * (time.perf_counter() - started)
    started = time.perf_counter()
    result = function(x, **options)
    cold_ms = 1000 * (time.perf_counter() - started)
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        function(x, **options)
        samples.append(1000 * (time.perf_counter() - started))
    if mode.startswith("multistep_"):
        record = result if mode == "multistep_summary" else result.to_dict()
        fingerprint = {"output_sha256": hashlib.sha256(
            json.dumps(record, sort_keys=True, allow_nan=False).encode()).hexdigest(),
            "inputs": settings}
    elif mode == "hac":
        fingerprint = {"numeric": {"long_run_variance": result.tolist()}}
    elif mode == "gaussian_ar":
        evidence = {f.name: getattr(result, f.name) for f in fields(result)}

        def encode(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, Fraction):
                return str(value)
            if hasattr(value, "__dataclass_fields__"):
                return {f.name: getattr(value, f.name) for f in fields(value)}
            raise TypeError(type(value).__name__)

        fingerprint = {"exact_evidence_sha256": hashlib.sha256(
            json.dumps(evidence, default=encode, sort_keys=True).encode()).hexdigest()}
    else:
        fingerprint = {"decisions": (result.adjusted_pvalue <= result.alpha).tolist(),
            "adjusted_pvalue": result.adjusted_pvalue.tolist(), "global_pvalue": result.global_pvalue,
            "numeric": {name: getattr(result, name).tolist() for name in (
                "mean", "standard_error", "statistic", "simultaneous_ci_low", "simultaneous_ci_high",
            )}}
    wrapper_ms = None
    if mode in ("fixed", "resampled", "gaussian_ar") and hasattr(si, "test_returns"):
        kwargs = dict(method="gaussian_ar") if mode == "gaussian_ar" else dict(method="bootstrap", **options)
        si.test_returns(x, **kwargs)
        wrapper_samples = []
        for _ in range(repeats):
            started = time.perf_counter()
            si.test_returns(x, **kwargs)
            wrapper_samples.append(1000 * (time.perf_counter() - started))
        wrapper_ms = statistics.median(wrapper_samples)
    tracemalloc.start()
    function(x, **options)
    _, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return dict(case=case, mode=mode, n_obs=t, k=k, n_resamples=b, code_version=si.__version__,
        import_ms=import_ms, api_load_ms=api_load_ms, preparation_ms=setup_ms, cold_call_ms=cold_ms,
        first_use_ms=import_ms + api_load_ms + cold_ms, warm_call_ms=samples,
        warm_median_ms=statistics.median(samples), unified_api_warm_median_ms=wrapper_ms,
        process_peak_rss_mib=peak / (2**20 if sys.platform == "darwin" else 1024),
        traced_call_peak_mib=traced_peak / 2**20,
        fingerprint=fingerprint)


def _run(case: str, package_root: Path, repeats: int):
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1", BLIS_NUM_THREADS="1")
    completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", case,
        "--package-root", str(package_root), "--repeats", str(repeats)],
        check=True, capture_output=True, text=True, env=env, timeout=120)
    return json.loads(completed.stdout)


def _compare(before, after):
    import numpy as np

    a, b = before["fingerprint"], after["fingerprint"]
    for key in ("decisions", "adjusted_pvalue", "global_pvalue", "exact_evidence_sha256", "output_sha256", "inputs"):
        if key in a:
            if a[key] != b[key]:
                raise ValueError(f"Result differs for {before['case']}: {key}")
    errors = {}
    for name, values in a.get("numeric", {}).items():
        np.testing.assert_allclose(b["numeric"][name], values, rtol=1e-11, atol=1e-12)
        errors[name] = float(np.max(np.abs(np.asarray(b["numeric"][name]) - values)))
    return dict(status="passed", warm_speedup=before["warm_median_ms"] / after["warm_median_ms"],
        maximum_numeric_errors=errors,
        exact_decisions_and_pvalues=True if "decisions" in a else None,
        exact_certificates=True if "exact_evidence_sha256" in a else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="v0.4.0")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", choices=tuple(CASES), help=argparse.SUPPRESS)
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES), default=list(CASES)[:6])
    parser.add_argument("--package-root", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.worker:
        print(json.dumps(_worker(args.worker, args.package_root, args.repeats)))
        return
    if args.output is None:
        parser.error("--output is required")
    revision = subprocess.check_output(["git", "rev-parse", "--verify", "--end-of-options",
        args.baseline + "^{commit}"], cwd=ROOT, text=True).strip()
    archived = subprocess.check_output(["git", "archive", "--format=tar", revision,
        "src/strategy_inference"], cwd=ROOT)
    before, after, comparisons = [], [], []
    with tempfile.TemporaryDirectory(prefix="strategy-inference-baseline-") as directory:
        destination = Path(directory).resolve()
        with tarfile.open(fileobj=io.BytesIO(archived)) as archive:
            for member in archive:
                if member.isdir():
                    continue
                target = destination / member.name
                if not member.isfile() or not target.resolve().is_relative_to(destination):
                    raise ValueError("The baseline archive contains an unsafe entry.")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.extractfile(member).read())
        baseline_root = destination / "src"
        baseline_hashes = _source_hashes(baseline_root / "strategy_inference")
        candidate_hashes = _source_hashes(ROOT / "src/strategy_inference")
        for case in args.cases:
            a, b = _run(case, baseline_root, args.repeats), _run(case, ROOT / "src", args.repeats)
            before.append(a)
            after.append(b)
            comparisons.append(_compare(a, b))
            print(f"{case}: {a['warm_median_ms']:.2f} -> {b['warm_median_ms']:.2f} ms", flush=True)
        if candidate_hashes != _source_hashes(ROOT / "src/strategy_inference"):
            raise ValueError("Library source changed during benchmarking; repeat from stable source.")
    report = dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
        baseline_revision=revision, candidate_source_sha256=candidate_hashes,
        baseline_source_sha256=baseline_hashes, benchmark_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        python=sys.version, platform=platform.platform(), processor=platform.processor(),
        numpy=importlib.import_module("numpy").__version__, scipy=importlib.import_module("scipy").__version__,
        repeats=args.repeats, blas_threads=1, scope="Serial local timings; not statistical calibration or a universal speed guarantee.",
        memory_scope="Whole-worker peak RSS includes interpreter, imports, data and output; not incremental allocation.",
        traced_memory_scope="Peak traced allocations for one warmed core call, including its output; excludes pre-existing input and imports.",
        baseline=before, candidate=after, comparisons=comparisons)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
