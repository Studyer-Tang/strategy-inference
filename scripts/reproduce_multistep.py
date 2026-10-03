"""Reproduce a fixed simulation study of delayed feedback and scale transfer.

python scripts/reproduce_multistep.py --profile full --output results/research/multistep/full
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes():
    return {p.name: _sha(p) for p in sorted((ROOT / "src/strategy_inference").glob("*.py"))}


def _array_sha(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def generate(protocol, settings, scenario, seed):
    """Observed AR process; future volatility is used only by the data generator."""
    rng = np.random.default_rng(seed)
    n, burn = settings["n_obs"], protocol["burn_in"]
    total = n + burn
    shocks = rng.normal(size=total)
    scale_shocks = rng.normal(size=total)
    log_scale = np.zeros(total)
    rho, sd = scenario["rho"], scenario["log_scale_sd"]
    if sd:
        log_scale[0] = sd / np.sqrt(1 - rho**2) * scale_shocks[0]
        for i in range(1, total):
            log_scale[i] = rho * log_scale[i - 1] + sd * scale_shocks[i]
    sigma = np.exp(log_scale)
    if scenario["name"] == "scale_shift":
        sigma[burn + int(n * 0.45) : burn + int(n * 0.72)] = 3.0
    y = np.empty(total)
    phi = protocol["phi"]
    y[0] = sigma[0] * shocks[0] / np.sqrt(1 - phi**2)
    for i in range(1, total):
        y[i] = phi * y[i - 1] + sigma[i] * shocks[i]
    return y[burn:], sigma[burn:]


def _scales(y, train_size, leads, phi):
    # Every residual target lies within the initial training prefix.
    return np.array(
        [np.sqrt(np.mean((y[h:train_size] - phi**h * y[: train_size - h]) ** 2)) for h in leads]
    )


def _mean_ci(values):
    data = np.asarray(values, dtype=float)
    if not len(data) or not np.isfinite(data).all():
        return {"mean": None, "mcse": None, "lower": None, "upper": None, "n": len(data)}
    mean = float(np.mean(data))
    if len(data) < 2:
        return {"mean": mean, "mcse": None, "lower": None, "upper": None, "n": len(data)}
    se = float(np.std(data, ddof=1) / np.sqrt(len(data)))
    radius = float(student_t.ppf(0.975, len(data) - 1) * se)
    return {
        "mean": mean,
        "mcse": se,
        "lower": mean - radius,
        "upper": mean + radius,
        "n": len(data),
    }


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


def aggregate(records, protocol):
    rows, paired = [], []
    for scenario in protocol["scenarios"]:
        name = scenario["name"]
        selected = [r for r in records if r["scenario"] == name]
        for method in protocol["methods"]:
            for h in protocol["lead_times"]:
                group = [
                    r for r in selected if r["method"] == method["name"] and r["lead_time"] == h
                ]
                good = all(r["mean_interval_score"] is not None for r in group)
                rows.append(
                    {
                        "scenario": name,
                        "method": method["name"],
                        "lead_time": h,
                        "coverage": _mean_ci([r["coverage"] for r in group]),
                        "mean_interval_score": _mean_ci([r["mean_interval_score"] for r in group])
                        if good
                        else _mean_ci([]),
                        "worst_local_coverage_error": _mean_ci(
                            [r["worst_local_coverage_error"] for r in group]
                        ),
                        "invalid_score_runs": sum(r["mean_interval_score"] is None for r in group),
                        "empty_count": sum(r["empty_count"] for r in group),
                        "unbounded_count": sum(r["unbounded_count"] for r in group),
                        "n_runs": len(group),
                    }
                )
        for h in protocol["lead_times"]:
            own = {
                r["replicate"]: r
                for r in selected
                if r["method"] == "pooled_horizon" and r["lead_time"] == h
            }
            shared = {
                r["replicate"]: r
                for r in selected
                if r["method"] == "pooled_shortest" and r["lead_time"] == h
            }
            invalid = sum(
                own[i]["mean_interval_score"] is None or shared[i]["mean_interval_score"] is None
                for i in own
            )
            differences = (
                []
                if invalid
                else [shared[i]["mean_interval_score"] - own[i]["mean_interval_score"] for i in own]
            )
            paired.append(
                {
                    "scenario": name,
                    "lead_time": h,
                    "contrast": "pooled_shortest minus pooled_horizon; negative means lower interval score",
                    "score_difference": _mean_ci(differences),
                    "invalid_score_pairs": invalid,
                    "coverage_difference": _mean_ci(
                        [shared[i]["coverage"] - own[i]["coverage"] for i in own]
                    ),
                }
            )
    return rows, paired


COLORS = {
    "pooled_fixed": "#77736c",
    "pooled_horizon": "#395f78",
    "pooled_shortest": "#a56732",
    "interlaced_horizon": "#5c7960",
}
LABELS = {
    "pooled_fixed": "Pooled / fixed scale",
    "pooled_horizon": "Pooled / own-lead EWMA",
    "pooled_shortest": "Pooled / shortest-lead EWMA",
    "interlaced_horizon": "Interlaced / own-lead EWMA",
}
TITLES = {
    "constant": "Constant variance",
    "persistent_scale": "Persistent volatility",
    "fast_scale": "Fast volatility (stress test)",
    "scale_shift": "Abrupt variance shifts",
}


def figures(output, report, representative, protocol):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "svg.fonttype": "none",
        }
    )

    def save(fig, name):
        for ext in ("svg", "png", "pdf"):
            fig.savefig(output / f"{name}.{ext}", dpi=180, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.8), constrained_layout=True)
    for ax, scenario in zip(axes.flat, protocol["scenarios"], strict=True):
        name = scenario["name"]
        group = [r for r in report["paired_contrasts"] if r["scenario"] == name]
        for row in group:
            value = row["score_difference"]
            if value["mean"] is not None:
                errors = [[value["mean"] - value["lower"]], [value["upper"] - value["mean"]]]
                ax.errorbar(
                    row["lead_time"],
                    value["mean"],
                    yerr=errors,
                    fmt="o",
                    color=COLORS["pooled_shortest"],
                    capsize=3,
                )
            else:
                ax.text(
                    row["lead_time"],
                    0.5,
                    f"{row['invalid_score_pairs']} invalid",
                    transform=ax.get_xaxis_transform(),
                    rotation=90,
                    ha="center",
                )
        ax.axhline(0, color="#aaa59d", lw=1)
        ax.set(
            title=TITLES[name],
            xlabel="Forecast lead",
            ylabel="Interval-score difference\nshortest EWMA minus own-lead EWMA",
            xticks=protocol["lead_times"],
        )
    fig.suptitle("Scale transfer: paired differences with pointwise 95% Monte Carlo intervals")
    save(fig, "01-efficiency")
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.8), constrained_layout=True)
    for ax, scenario in zip(axes.flat, protocol["scenarios"], strict=True):
        for method in protocol["methods"]:
            rows = [
                r
                for r in report["aggregate"]
                if r["scenario"] == scenario["name"] and r["method"] == method["name"]
            ]
            xs, ys = [r["lead_time"] for r in rows], [r["coverage"]["mean"] for r in rows]
            ax.plot(xs, ys, "o-", ms=4, color=COLORS[method["name"]], label=LABELS[method["name"]])
            ax.fill_between(
                xs,
                [r["coverage"]["lower"] for r in rows],
                [r["coverage"]["upper"] for r in rows],
                color=COLORS[method["name"]],
                alpha=0.12,
            )
        ax.axhline(1 - protocol["alpha"], color="#aaa59d", ls="--", lw=1)
        ax.set(
            title=TITLES[scenario["name"]],
            xlabel="Forecast lead",
            ylabel="Realized coverage",
            xticks=protocol["lead_times"],
        )
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle(
        "Coverage across independent simulated paths; shaded bands are Monte Carlo intervals"
    )
    save(fig, "02-coverage")
    data = representative["scale_shift"]
    h = protocol["lead_times"][-1]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)
    axes[0].plot(np.arange(len(data["sigma"])), data["sigma"], color="#77736c")
    axes[0].set(
        ylabel="DGP innovation scale",
        xlabel="Observation time",
        title=f"Prespecified first path: variance shifts, lead {h}",
    )
    for method in ("pooled_fixed", "pooled_horizon", "pooled_shortest"):
        path = data[method]
        axes[1].plot(
            path["origins"], path["scales"][:, -1], color=COLORS[method], label=LABELS[method]
        )
        coverage = 1 - path["misses"][:, -1].astype(float)
        w = protocol["local_window"]
        local = np.convolve(coverage, np.ones(w) / w, mode="valid")
        axes[2].plot(path["targets"][w - 1 :, -1], local, color=COLORS[method])
    axes[1].set(ylabel="Scale frozen at issue", xlabel="Forecast origin")
    axes[1].legend(frameon=False, ncol=2)
    axes[2].axhline(1 - protocol["alpha"], color="#aaa59d", ls="--", lw=1)
    axes[2].set(
        ylabel=f"Trailing {protocol['local_window']}-target coverage", xlabel="Matured target time"
    )
    save(fig, "03-adaptation")


def run(profile, output):
    import strategy_inference as si
    from strategy_inference.multistep import multistep_intervals

    protocol_path = ROOT / "experiments/multistep-protocol.json"
    protocol = json.loads(protocol_path.read_text())
    settings = protocol["profiles"][profile]
    source_hashes = _source_hashes()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a new output directory; saved study evidence is not overwritten.")
    output.mkdir(parents=True, exist_ok=True)
    records, representative, fingerprints = [], {}, []
    leads = np.asarray(protocol["lead_times"], dtype=np.int64)
    for scenario_index, scenario in enumerate(protocol["scenarios"]):
        name = scenario["name"]
        for replicate in range(settings["repetitions"]):
            seed = protocol["seed"] + settings["seed_offset"] + scenario_index * 10000 + replicate
            y, sigma = generate(protocol, settings, scenario, seed)
            origins = np.arange(
                settings["initial_train_size"] - 1, len(y) - int(leads[-1]), dtype=np.int64
            )
            predicted = y[origins, None] * protocol["phi"] ** leads[None, :]
            initial_scales = _scales(y, settings["initial_train_size"], leads, protocol["phi"])
            fingerprints.append(
                {
                    "scenario": name,
                    "replicate": replicate,
                    "seed": seed,
                    "actual_sha256": _array_sha(y),
                    "predicted_sha256": _array_sha(predicted),
                    "initial_scales": initial_scales.tolist(),
                }
            )
            if replicate == 0:
                representative[name] = {"sigma": sigma}
            for method in protocol["methods"]:
                result = multistep_intervals(
                    y,
                    predicted,
                    origins=origins,
                    lead_times=leads,
                    scale=initial_scales,
                    alpha=protocol["alpha"],
                    step_size=protocol["step_size"] / np.sqrt(leads),
                    decay=protocol["decay"],
                    initial_quantile=protocol["initial_quantile"],
                    strategy=method["strategy"],
                    scale_decay=protocol["scale_decay"] if method["adaptive_scale"] else None,
                    scale_source=method["scale_source"],
                    scale_floor=protocol["scale_floor"],
                )
                for row in metrics(result, protocol, settings["initial_train_size"]):
                    records.append(
                        {
                            "scenario": name,
                            "replicate": replicate,
                            "seed": seed,
                            "method": method["name"],
                            **row,
                        }
                    )
                if replicate == 0:
                    representative[name][method["name"]] = {
                        "origins": result.origins,
                        "targets": result.target_indices,
                        "scales": result.scales,
                        "misses": result.misses,
                        "lower": result.lower,
                        "upper": result.upper,
                    }
            if (replicate + 1) % 10 == 0 or replicate + 1 == settings["repetitions"]:
                print(
                    f"{name}: {replicate + 1}/{settings['repetitions']} independent paths",
                    flush=True,
                )
    aggregate_rows, paired = aggregate(records, protocol)
    report = {
        "schema_version": 1,
        "study": protocol["study"],
        "profile": profile,
        "package_version": si.__version__,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": _sha(protocol_path),
        "runner_sha256": _sha(Path(__file__)),
        "candidate_source_sha256": source_hashes,
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "protocol": protocol,
        "profile_settings": settings,
        "records": records,
        "aggregate": aggregate_rows,
        "paired_contrasts": paired,
        "input_fingerprints": fingerprints,
        "inference_scope": "Pointwise Monte Carlo t intervals across independent paths. No within-path IID standard errors or simultaneous CI claim. Undefined interval scores invalidate the entire method/lead aggregate and paired contrast, rather than dropping failed paths.",
    }
    figures(output, report, representative, protocol)
    report["figure_sha256"] = {
        p.name: _sha(p) for p in sorted(output.iterdir()) if p.suffix in {".svg", ".png", ".pdf"}
    }
    if _source_hashes() != source_hashes:
        raise RuntimeError(
            "Source changed during the study; choose a new directory and rerun after review."
        )
    (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved three figures and raw records: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("quick", "full"), default="quick")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.profile, args.output)


if __name__ == "__main__":
    main()
