"""Evaluate paired-calendar scale transfer with causal fitted forecasts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_sha(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _sources():
    return {p.name: _sha(p) for p in sorted((ROOT / "src/strategy_inference").glob("*.py"))}


def generate(protocol, settings, scenario, seed):
    rng = np.random.default_rng(seed)
    n, burn = settings["n_obs"], protocol["burn_in"]
    total = n + burn
    if scenario["innovation"] == "student":
        df = scenario["degrees_of_freedom"]
        shocks = rng.standard_t(df, size=total) * np.sqrt((df - 2) / df)
    else:
        shocks = rng.normal(size=total)
    scale_shocks = rng.normal(size=total)
    sigma, phi = np.ones(total), np.full(total, scenario["phi"])
    rho, sd = scenario["rho"], scenario["log_scale_sd"]
    if sd:
        log_scale = np.empty(total)
        log_scale[0] = sd / np.sqrt(1 - rho**2) * scale_shocks[0]
        for t in range(1, total):
            log_scale[t] = rho * log_scale[t - 1] + sd * scale_shocks[t]
        sigma = np.exp(log_scale)
    a, b = (burn + int(n * f) for f in protocol["change_fractions"])
    if "middle_scale" in scenario:
        sigma[a:b] = scenario["middle_scale"]
    if "phi_middle" in scenario:
        phi[a:b], phi[b:] = scenario["phi_middle"], scenario["phi_last"]
    actual = np.empty(total)
    actual[0] = sigma[0] * shocks[0] / np.sqrt(1 - phi[0] ** 2)
    for t in range(1, total):
        actual[t] = phi[t] * actual[t - 1] + sigma[t] * shocks[t]
    return actual[burn:], sigma[burn:], phi[burn:]


def forecasts(actual, protocol, scenario):
    """Rolling AR(1); prefix values for t depend only on observations <= t."""
    leads = np.asarray(protocol["lead_times"])
    config = protocol["forecaster"]
    origins = np.arange(config["first_origin"], len(actual) - max(leads), dtype=np.int64)
    x, z = actual[:-1], actual[1:]
    prefix = [np.r_[0.0, np.cumsum(v)] for v in (x, z, x * x, x * z)]
    left = np.maximum(0, origins - config["window"] + 1)
    count = origins - left
    sx, sz, sxx, sxz = (v[origins] - v[left] for v in prefix)
    variance = sxx - sx * sx / count
    if np.any(variance <= 0) or not np.isfinite(variance).all():
        raise ValueError("Rolling forecast inputs have degenerate or unsupported variation.")
    coefficient = np.clip(
        (sxz - sx * sz / count) / variance, -config["coefficient_clip"], config["coefficient_clip"]
    )
    intercept = (sz - coefficient * sx) / count
    powers = coefficient[:, None] ** leads
    points = powers * actual[origins, None] + intercept[:, None] * (1 - powers) / (
        1 - coefficient[:, None]
    )
    if "middle_bias" in scenario:
        a, b = (int(len(actual) * f) for f in protocol["change_fractions"])
        contaminated = (origins >= a) & (origins < b)
        points[contaminated] += (
            scenario["middle_bias"] * ((leads - leads[0]) / (leads[-1] - leads[0])) ** 2
        )
    if not np.isfinite(points).all():
        raise ValueError("Fitted point forecasts are nonfinite.")
    return origins, points


def initial_scales(actual, origins, points, leads, train):
    # All h use the same target times wholly inside the training prefix.
    targets = np.arange(int(origins[0]) + max(leads), train)
    if not len(targets):
        raise ValueError("Training prefix is too short for matched residuals.")
    rows = targets[:, None] - np.asarray(leads)[None, :] - int(origins[0])
    residuals = actual[targets, None] - points[rows, np.arange(len(leads))]
    scales = np.sqrt(np.mean(residuals**2, axis=0))
    if np.any(scales <= 0) or not np.isfinite(scales).all():
        raise ValueError("Training residual scales must be positive and finite.")
    return scales


def replay(actual, points, origins, scales, protocol, method):
    from scale_transfer_prototype import MatureScaleTransfer

    from strategy_inference import MultiStepConformal

    leads = protocol["lead_times"]
    tracker = MultiStepConformal(
        leads,
        start_time=int(origins[0]),
        alpha=protocol["alpha"],
        step_size=protocol["step_size"] / np.sqrt(leads),
        decay=protocol["decay"],
        initial_quantile=protocol["initial_quantile"],
        scale=scales,
        scale_decay=protocol["scale_decay"] if method in ("horizon", "shortest") else None,
        scale_source="shortest" if method == "shortest" else "horizon",
        scale_floor=protocol["scale_floor"],
    )
    transfer = (
        MatureScaleTransfer(
            scales,
            lead_times=leads,
            scale_decay=protocol["scale_decay"],
            ratio_decay=protocol["scale_decay"]
            if method == "paired_equal_rate"
            else protocol["ratio_decay"],
            floor=protocol["scale_floor"],
        )
        if method.startswith("paired")
        else None
    )
    shape = points.shape
    lower, upper, issued_scales, quantiles = (np.empty(shape) for _ in range(4))
    values = np.full(shape, np.nan)
    evaluated, misses, empty, unbounded = (np.zeros(shape, dtype=bool) for _ in range(4))
    lead_index = {h: i for i, h in enumerate(leads)}
    first = int(origins[0])
    paired_trace = []
    for t in range(first, len(actual)):
        updates = tracker.observe(t, float(actual[t]))
        if transfer is not None:
            transfer.update({u.lead_time: abs(u.actual - u.prediction) for u in updates})
        for update in updates:
            f, h = update.origin - first, lead_index[update.lead_time]
            values[f, h], evaluated[f, h], misses[f, h] = update.actual, True, update.miss
        f = t - first
        if f < len(origins):
            for h, interval in enumerate(
                tracker.predict(points[f], scale=transfer.scales if transfer else None)
            ):
                lower[f, h], upper[f, h] = interval.lower, interval.upper
                issued_scales[f, h], quantiles[f, h] = interval.scale, interval.quantile
                empty[f, h], unbounded[f, h] = (
                    interval.kind == "empty",
                    interval.kind == "unbounded",
                )
            if transfer:
                paired_trace.append(transfer.to_dict())
    return SimpleNamespace(
        origins=origins,
        actual=values,
        lower=lower,
        upper=upper,
        evaluated=evaluated,
        misses=misses,
        empty=empty,
        unbounded=unbounded,
        scales=issued_scales,
        quantiles=quantiles,
        paired_trace=paired_trace,
        terminal=tracker.to_dict(),
    )


def _ci(values):
    data = np.asarray(values, dtype=float)
    if not len(data) or not np.isfinite(data).all():
        return dict(mean=None, mcse=None, lower=None, upper=None, n=len(data))
    mean = float(np.mean(data))
    se = float(np.std(data, ddof=1) / np.sqrt(len(data))) if len(data) > 1 else None
    radius = float(student_t.ppf(0.975, len(data) - 1) * se) if se is not None else None
    return dict(
        mean=mean,
        mcse=se,
        lower=mean - radius if radius is not None else None,
        upper=mean + radius if radius is not None else None,
        n=len(data),
    )


def summarize(records, protocol):
    summaries, contrasts = [], []
    for scenario in protocol["scenarios"]:
        name = scenario["name"]
        for lead in protocol["lead_times"]:
            by_method = {
                method: [
                    r
                    for r in records
                    if r["scenario"] == name and r["lead_time"] == lead and r["method"] == method
                ]
                for method in protocol["methods"]
            }
            for method, group in by_method.items():
                valid = all(r["mean_interval_score"] is not None for r in group)
                summaries.append(
                    dict(
                        scenario=name,
                        method=method,
                        lead_time=lead,
                        n_runs=len(group),
                        coverage=_ci([r["coverage"] for r in group]),
                        mean_interval_score=_ci([r["mean_interval_score"] for r in group])
                        if valid
                        else _ci([]),
                        worst_local_coverage_error=_ci(
                            [r["worst_local_coverage_error"] for r in group]
                        ),
                        invalid_score_runs=sum(r["mean_interval_score"] is None for r in group),
                    )
                )
            for baseline in ("horizon", "shortest", "paired_equal_rate"):
                pairs = list(zip(by_method["paired"], by_method[baseline], strict=True))
                if any(a["replicate"] != b["replicate"] for a, b in pairs):
                    raise ValueError("Paired paths must have the same replicate identity.")
                invalid = sum(
                    a["mean_interval_score"] is None or b["mean_interval_score"] is None
                    for a, b in pairs
                )
                differences = (
                    []
                    if invalid
                    else [a["mean_interval_score"] - b["mean_interval_score"] for a, b in pairs]
                )
                contrasts.append(
                    dict(
                        scenario=name,
                        lead_time=lead,
                        baseline=baseline,
                        score_difference=_ci(differences),
                        invalid_score_pairs=invalid,
                        coverage_difference=_ci([a["coverage"] - b["coverage"] for a, b in pairs]),
                    )
                )
    return summaries, contrasts


def run(profile, output):
    from strategy_inference import __version__

    protocol_path = ROOT / "experiments/scale-transfer-protocol.json"
    helper = ROOT / "scripts/scale_transfer_prototype.py"
    protocol = json.loads(protocol_path.read_text())
    settings = protocol["profiles"][profile]
    locked = dict(
        source=_sources(),
        runner=_sha(Path(__file__)),
        helper=_sha(helper),
        protocol=_sha(protocol_path),
        metrics=_sha(ROOT / "scripts/reproduce_multistep.py"),
    )
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new or empty output directory.")
    output.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location(
        "multistep_metrics", ROOT / "scripts/reproduce_multistep.py"
    )
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    records, fingerprints = [], []
    for s, scenario in enumerate(protocol["scenarios"]):
        for rep in range(settings["repetitions"]):
            seed = protocol["seed"] + settings["seed_offset"] + s * 10000 + rep
            actual, _, _ = generate(protocol, settings, scenario, seed)
            all_origins, all_points = forecasts(actual, protocol, scenario)
            scales = initial_scales(
                actual,
                all_origins,
                all_points,
                protocol["lead_times"],
                settings["initial_train_size"],
            )
            keep = all_origins >= settings["initial_train_size"] - 1
            origins, points = all_origins[keep], all_points[keep]
            fingerprints.append(
                dict(
                    scenario=scenario["name"],
                    replicate=rep,
                    seed=seed,
                    actual_sha256=_array_sha(actual),
                    predicted_sha256=_array_sha(points),
                    initial_scales=scales.tolist(),
                )
            )
            for method in protocol["methods"]:
                result = replay(actual, points, origins, scales, protocol, method)
                records.extend(
                    dict(scenario=scenario["name"], replicate=rep, seed=seed, method=method, **row)
                    for row in old.metrics(result, protocol, settings["initial_train_size"])
                )
            print(f"{scenario['name']}: {rep + 1}/{settings['repetitions']}", flush=True)
    aggregate, contrasts = summarize(records, protocol)
    if locked != dict(
        source=_sources(),
        runner=_sha(Path(__file__)),
        helper=_sha(helper),
        protocol=_sha(protocol_path),
        metrics=_sha(ROOT / "scripts/reproduce_multistep.py"),
    ):
        raise RuntimeError("Source or protocol changed during the study.")
    report = dict(
        schema_version=1,
        study=protocol["study"],
        profile=profile,
        package_version=__version__,
        created_utc=datetime.now(timezone.utc).isoformat(),
        protocol=protocol,
        profile_settings=settings,
        candidate_source_sha256=locked["source"],
        runner_sha256=locked["runner"],
        helper_sha256=locked["helper"],
        protocol_sha256=locked["protocol"],
        metrics_runner_sha256=locked["metrics"],
        python=sys.version,
        numpy=np.__version__,
        platform=platform.platform(),
        records=records,
        aggregate=aggregate,
        paired_contrasts=contrasts,
        input_fingerprints=fingerprints,
    )
    (output / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.profile, args.output)


if __name__ == "__main__":
    main()
