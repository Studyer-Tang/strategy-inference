"""Shared data generation, path metrics and validation for scale studies.

Research utilities only; the installed package does not import this module.
Simulation and saved-ledger checks share explicit path and lead conventions.
"""

from __future__ import annotations

import hashlib
import math
from numbers import Integral, Real

import numpy as np
from scipy.stats import t as student_t

LEAD = 24
LABELS = {"fixed": "Fixed", "horizon": "Own lead", "shortest": "Shortest"}
SCENARIOS = {
    "constant": "Constant",
    "persistent_scale": "Persistent scale",
    "heavy_tail": "Heavy tails",
    "persistence_shift": "Persistence shift",
    "scale_shift": "Scale shift",
    "lead_bias": "Lead-specific bias",
}


def _array_sha(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


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


def metrics(result, protocol, train_size):
    from strategy_inference import interval_score

    keep = result.origins >= train_size - 1 + protocol["evaluation_warmup"]
    rows = []
    for col, h in enumerate(protocol["lead_times"]):
        valid = keep & result.evaluated[:, col]
        empty = result.empty[valid, col]
        unbounded = result.unbounded[valid, col]
        finite = ~empty & ~unbounded
        lo, hi = result.lower[valid, col], result.upper[valid, col]
        score = None
        status = (
            "empty_intervals"
            if empty.any()
            else "unbounded_intervals"
            if unbounded.any()
            else "finite"
        )
        if status == "finite":
            values = interval_score(result.actual[valid, col], lo, hi, alpha=protocol["alpha"])
            if np.isfinite(values).all():
                score = float(np.mean(values))
            else:
                status = "numeric_overflow"
        covered = (~result.misses[valid, col]).astype(float)
        local_window = protocol["local_window"]
        local = np.convolve(covered, np.ones(local_window) / local_window, mode="valid")
        rows.append(
            {
                "lead_time": h,
                "n_evaluated": int(valid.sum()),
                "coverage": float(np.mean(covered)),
                "worst_local_coverage_error": float(
                    np.max(np.abs(local - (1 - protocol["alpha"])))
                ),
                "mean_interval_score": score,
                "interval_score_status": status,
                "mean_finite_width": float(np.mean(hi[finite] - lo[finite]))
                if finite.any()
                else None,
                "empty_count": int(empty.sum()),
                "unbounded_count": int(unbounded.sum()),
            }
        )
    return rows


def _integer(value, name, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _real(value, name, *, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number.")
    result = float(value)
    if (
        not math.isfinite(result)
        or (minimum is not None and result < minimum)
        or (maximum is not None and result > maximum)
    ):
        raise ValueError(f"{name} is outside its finite supported range.")
    return result


def _names(values, name):
    if (
        not isinstance(values, list)
        or not values
        or any(not isinstance(value, str) or not value for value in values)
    ):
        raise ValueError(f"{name} must be a nonempty string list.")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must be unique.")
    return tuple(values)


def validate_report(report):
    """Validate the complete independent-path ledger without filtering outcomes."""
    if (
        not isinstance(report, dict)
        or type(report.get("schema_version")) is not int
        or report["schema_version"] != 1
    ):
        raise ValueError("Expected a schema_version=1 scale-study report.")
    protocol = report["protocol"]
    settings = report["profile_settings"]
    if settings != protocol["profiles"][report["profile"]]:
        raise ValueError("Profile settings do not match the recorded protocol.")
    scenarios = protocol["scenarios"]
    names = _names([scenario["name"] for scenario in scenarios], "scenarios")
    methods = _names(protocol["methods"], "methods")
    if "shortest" not in methods or len(methods) < 2:
        raise ValueError("The shortest baseline and a comparison method are required.")
    leads = tuple(_integer(h, "lead_time", minimum=1) for h in protocol["lead_times"])
    if (
        not leads
        or any(b <= a for a, b in zip(leads, leads[1:], strict=False))
        or LEAD not in leads
    ):
        raise ValueError("Strictly increasing leads must include h=24.")
    repetitions = _integer(settings["repetitions"], "repetitions", minimum=2)
    n_obs = _integer(settings["n_obs"], "n_obs", minimum=1)
    train = _integer(settings["initial_train_size"], "initial_train_size", minimum=1)
    warmup = _integer(protocol["evaluation_warmup"], "evaluation_warmup")
    window = _integer(protocol["local_window"], "local_window", minimum=1)
    alpha = _real(protocol["alpha"], "alpha", minimum=0, maximum=1)
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between zero and one.")
    expected_n = n_obs - max(leads) - (train - 1 + warmup)
    if expected_n < window:
        raise ValueError("The evaluation suffix is shorter than the local window.")
    base = _integer(protocol["seed"], "seed") + _integer(settings["seed_offset"], "seed_offset")
    seeds = {
        (name, rep): base + s * 10000 + rep
        for s, name in enumerate(names)
        for rep in range(repetitions)
    }
    expected = {
        (name, rep, method, h)
        for name in names
        for rep in range(repetitions)
        for method in methods
        for h in leads
    }
    rows = {}
    for row in report["records"]:
        rep = _integer(row["replicate"], "replicate")
        h = _integer(row["lead_time"], "lead_time", minimum=1)
        key = row["scenario"], rep, row["method"], h
        if key not in expected or key in rows:
            raise ValueError("Unknown or duplicate path record.")
        if _integer(row["seed"], "record seed") != seeds[key[:2]]:
            raise ValueError("Paired path seed does not match its scenario and replicate.")
        count = _integer(row["n_evaluated"], "n_evaluated", minimum=1)
        if count != expected_n:
            raise ValueError("Evaluated count does not match the common evaluation suffix.")
        _real(row["coverage"], "coverage", minimum=0, maximum=1)
        _real(
            row["worst_local_coverage_error"],
            "worst local coverage error",
            minimum=0,
            maximum=max(alpha, 1 - alpha),
        )
        empty = _integer(row["empty_count"], "empty_count")
        full = _integer(row["unbounded_count"], "unbounded_count")
        if empty + full > count:
            raise ValueError("Empty and unbounded counts exceed evaluated intervals.")
        score, status = row["mean_interval_score"], row["interval_score_status"]
        expected_status = (
            "empty_intervals"
            if empty
            else "unbounded_intervals"
            if full
            else "numeric_overflow"
            if score is None
            else "finite"
        )
        if status != expected_status or ((score is None) != (status != "finite")):
            raise ValueError("Score status and invalid interval counts are inconsistent.")
        if score is not None:
            _real(score, "mean interval score", minimum=0)
        width = row["mean_finite_width"]
        if width is None:
            if empty + full != count:
                raise ValueError("Finite intervals require a finite mean width.")
        else:
            width = _real(width, "mean finite width", minimum=0)
            if empty + full == count:
                raise ValueError("Mean finite width cannot summarize an empty finite subset.")
            if score is not None and score + 1e-12 * max(1, width) < width:
                raise ValueError("Interval score cannot be smaller than interval width.")
        rows[key] = row
    if set(rows) != expected:
        raise ValueError("The complete scenario/path/method/lead ledger is required.")
    if "input_fingerprints" in report:
        seen = set()
        for entry in report["input_fingerprints"]:
            key = entry["scenario"], _integer(entry["replicate"], "fingerprint replicate")
            if (
                key in seen
                or key not in seeds
                or _integer(entry["seed"], "fingerprint seed") != seeds[key]
            ):
                raise ValueError("Input fingerprint path identity is inconsistent.")
            seen.add(key)
        if seen != set(seeds):
            raise ValueError("Input fingerprint paths are incomplete.")
    return protocol, settings, names, methods, leads, rows, {}


def mean_ci(values):
    """Student-t CI over independent paths, with no within-path IID assumption."""
    from scipy.stats import t

    values = tuple(_real(value, "path statistic") for value in values)
    n = len(values)
    if n < 2:
        raise ValueError("At least two independent paths are required for a t interval.")
    mean = math.fsum(value / n for value in values)
    se = math.hypot(*((value - mean) / math.sqrt(n * (n - 1)) for value in values))
    radius = float(t.ppf(0.975, n - 1)) * se
    result = dict(mean=mean, mcse=se, lower=mean - radius, upper=mean + radius, n=n)
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("Monte Carlo statistics exceed the supported finite range.")
    return result


def _panels(plt, names):
    columns = min(3, len(names))
    rows = math.ceil(len(names) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(11.4, 2.6 * rows + 1.0), squeeze=False)
    for axis in axes.flat[len(names) :]:
        axis.set_visible(False)
    return figure, tuple(axes.flat[: len(names)])


def _style(axis, title):
    axis.set_title(SCENARIOS.get(title, title.replace("_", " ")), loc="left", fontsize=11)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=9)
    axis.grid(axis="x", color="0.90", linewidth=0.5)
    axis.set_axisbelow(True)


def _dot(axis, y, estimate):
    axis.errorbar(
        estimate["mean"],
        y,
        xerr=[[estimate["mean"] - estimate["lower"]], [estimate["upper"] - estimate["mean"]]],
        fmt="o",
        color="0.12",
        markersize=4,
        elinewidth=0.85,
        capsize=2,
    )
