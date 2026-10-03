"""Plot saved scale studies without rerunning forecasts or filtering failed paths.

Usage: python scripts/plot_scale_study.py --input results.json --output figures
The interval-score comparison uses absolute differences on paired independent
paths. Confidence intervals are pointwise Monte Carlo Student-t intervals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path

LEAD = 24
LABELS = {
    "fixed": "Fixed",
    "horizon": "Own lead",
    "shortest": "Shortest",
    "half": "Half mix",
    "mixture": "Learned mix",
    "paired": "Paired transfer",
    "paired_equal_rate": "Equal rate",
}
SCENARIOS = {
    "constant": "Constant",
    "persistent_scale": "Persistent scale",
    "heavy_tail": "Heavy tails",
    "persistence_shift": "Persistence shift",
    "scale_shift": "Scale shift",
    "lead_bias": "Lead-specific bias",
}
STEMS = ("score-difference", "local-coverage-error", "weight-paths")


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
    """Validate the complete path ledger and preset representative trajectories."""
    import numpy as np

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
    paths = {}
    for path in report.get("representative_paths", []):
        name = path["scenario"]
        if name not in names or name in paths or _integer(path["replicate"], "path replicate") != 0:
            raise ValueError("Each representative path must be the unique preset replicate 0.")
        origins = tuple(_integer(t, "origin") for t in path["origins"])
        if origins != tuple(range(train - 1, n_obs - max(leads))):
            raise ValueError("Representative origins must match the full issuance clock.")
        arrays = {}
        for field in ("weights", "issued_scales"):
            values = path[field]
            # Check booleans before conversion: float coercion would otherwise conceal them.
            if len(values) != len(origins) or any(len(row) != len(leads) for row in values):
                raise ValueError(f"Representative {field} have the wrong shape.")
            for row in values:
                for value in row:
                    number = _real(
                        value, field, minimum=0, maximum=1 if field == "weights" else None
                    )
                    if field == "issued_scales" and number == 0:
                        raise ValueError("Issued scales must be positive.")
            arrays[field] = np.asarray(values, dtype=float)
        paths[name] = {"origins": np.asarray(origins), **arrays}
    if "mixture" in methods and set(paths) != set(names):
        raise ValueError("Mixture studies require one preset weight path per scenario.")
    return protocol, settings, names, methods, leads, rows, paths


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


def statistics(report):
    protocol, settings, names, methods, leads, rows, paths = validate_report(report)
    differences, coverage = [], []
    repetitions = settings["repetitions"]
    for name in names:
        baseline = [rows[name, rep, "shortest", LEAD] for rep in range(repetitions)]
        for method in methods:
            group = [rows[name, rep, method, LEAD] for rep in range(repetitions)]
            coverage.append(
                dict(
                    scenario=name,
                    method=method,
                    lead_time=LEAD,
                    estimate=mean_ci(row["worst_local_coverage_error"] for row in group),
                )
            )
            if method == "shortest":
                continue
            invalid = sum(
                a["mean_interval_score"] is None or b["mean_interval_score"] is None
                for a, b in zip(group, baseline, strict=True)
            )
            estimate = (
                None
                if invalid
                else mean_ci(
                    a["mean_interval_score"] - b["mean_interval_score"]
                    for a, b in zip(group, baseline, strict=True)
                )
            )
            differences.append(
                dict(
                    scenario=name,
                    method=method,
                    baseline="shortest",
                    lead_time=LEAD,
                    n_pairs=repetitions,
                    invalid_score_pairs=invalid,
                    estimate=estimate,
                )
            )
    return dict(score_differences=differences, local_coverage_error=coverage), (
        protocol,
        settings,
        names,
        methods,
        leads,
        paths,
    )


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


def make_figures(report):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    computed, info = statistics(report)
    protocol, settings, names, methods, leads, paths = info
    note = (
        f"{report['profile'].capitalize()} · {settings['repetitions']} independent paths per scenario"
        " · pointwise 95% Monte Carlo t intervals; no simultaneous claim."
    )
    figures = []
    for field, title, xlabel, displayed in (
        (
            "score_differences",
            "Interval score difference versus shortest (h = 24)",
            "Mean score difference (score units)",
            tuple(m for m in methods if m != "shortest"),
        ),
        (
            "local_coverage_error",
            f"Worst {protocol['local_window']}-point coverage error (h = 24)",
            "Mean worst-window absolute coverage error",
            methods,
        ),
    ):
        figure, axes = _panels(plt, names)
        for name, axis in zip(names, axes, strict=True):
            _style(axis, name)
            selected = {row["method"]: row for row in computed[field] if row["scenario"] == name}
            for y, method in enumerate(displayed):
                row = selected[method]
                if row["estimate"] is None:
                    axis.text(
                        0.02,
                        y,
                        f"Unavailable ({row['invalid_score_pairs']} invalid pairs)",
                        transform=axis.get_yaxis_transform(),
                        va="center",
                        fontsize=8,
                    )
                else:
                    _dot(axis, y, row["estimate"])
            axis.set_yticks(
                range(len(displayed)), [LABELS.get(m, m.replace("_", " ")) for m in displayed]
            )
            axis.set_ylim(len(displayed) - 0.5, -0.5)
            axis.xaxis.set_major_locator(MaxNLocator(4))
            axis.set_xlabel(xlabel, fontsize=9)
            if field == "score_differences":
                axis.axvline(0, color="0.45", linewidth=0.8, linestyle="--")
        figure.suptitle(title, x=0.035, ha="left", fontsize=14)
        tail = (
            "Differences are method − shortest in score units; negative is better. "
            "Any invalid score makes its entire comparison unavailable."
            if field == "score_differences"
            else f"Within each path: max |window coverage − {1 - protocol['alpha']:g}| over "
            "the evaluation suffix; then average across paths. No window-coverage guarantee."
        )
        figure.text(0.035, 0.075, note, fontsize=8.5)
        figure.text(0.035, 0.035, tail, fontsize=8.5)
        figure.tight_layout(rect=(0.02, 0.13, 1, 0.94), h_pad=1.8, w_pad=2.2)
        figures.append(figure)
    figure, axes = _panels(plt, names)
    col = leads.index(LEAD)
    warmup_end = settings["initial_train_size"] - 1 + protocol["evaluation_warmup"]
    for scenario, axis in zip(protocol["scenarios"], axes, strict=True):
        name = scenario["name"]
        _style(axis, name)
        if name in paths:
            path = paths[name]
            axis.plot(path["origins"], path["weights"][:, col], color="0.12", linewidth=1.1)
            axis.axvspan(path["origins"][0], warmup_end, color="0.94", zorder=-1)
            axis.set_xlim(path["origins"][0], path["origins"][-1])
            process_shift = any(
                key in scenario for key in ("phi_middle", "phi_last", "middle_scale")
            )
            bias_shift = "middle_bias" in scenario
            if process_shift or bias_shift:
                for fraction in protocol["change_fractions"]:
                    time = int(
                        settings["n_obs"] * _real(fraction, "change fraction", minimum=0, maximum=1)
                    )
                    axis.axvline(
                        time, color="0.50", linewidth=0.8, linestyle=":" if bias_shift else "--"
                    )
        else:
            axis.text(
                0.5,
                0.5,
                "No mixture weights recorded",
                transform=axis.transAxes,
                ha="center",
                fontsize=9,
            )
        axis.set_ylim(-0.025, 1.025)
        axis.set_yticks([0, 0.5, 1])
        axis.set_xlabel("Forecast origin", fontsize=9)
        axis.set_ylabel("Short-source weight", fontsize=9)
        axis.xaxis.set_major_locator(MaxNLocator(4, integer=True))
    figure.suptitle(
        "Short-source mixture weight (h = 24): illustrative replicate 0"
        if paths
        else "Mixture weight trajectories unavailable in this saved study",
        x=0.035,
        ha="left",
        fontsize=14,
    )
    figure.text(
        0.035,
        0.075,
        "Preset path, including startup; no average or trajectory confidence band. Shading marks excluded evaluation warmup."
        if paths
        else "No mixture-weight trajectories were recorded; none are reconstructed or selected from other studies.",
        fontsize=8.5,
    )
    figure.text(
        0.035,
        0.035,
        "Dashed lines: process changes on the time axis. Dotted lines: origin-based forecast bias. These are not feedback-arrival markers.",
        fontsize=8.5,
    )
    figure.tight_layout(rect=(0.02, 0.13, 1, 0.94), h_pad=1.8, w_pad=2.2)
    figures.append(figure)
    return figures, computed


def plot_study(input_path, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = input_path.read_bytes()
    report = json.loads(
        data,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"Nonfinite JSON number: {value}")
        ),
    )
    targets = [
        output / f"{stem}.{extension}" for stem in STEMS for extension in ("svg", "png", "pdf")
    ]
    targets.append(output / "figure-data.json")
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        raise ValueError("Output must be an ordinary directory.")
    for target in targets:
        if input_path.resolve() == target.resolve() or (
            target.exists() and input_path.samefile(target)
        ):
            raise ValueError("Input report must not overlap any managed output target.")
        if target.is_symlink() or (
            target.exists() and (not target.is_file() or target.stat().st_nlink != 1)
        ):
            raise ValueError("Managed figure targets must be ordinary, unlinked files.")
    with matplotlib.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.linewidth": 0.65,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.hashsalt": "strategy-inference-scale-study-v1",
            "pdf.fonttype": 42,
        }
    ):
        figures, computed = make_figures(report)
        try:
            output.mkdir(parents=True, exist_ok=True)
            for stem, figure in zip(STEMS, figures, strict=True):
                for extension in ("svg", "png", "pdf"):
                    metadata = (
                        {"Date": None, "Creator": "strategy-inference scale study"}
                        if extension == "svg"
                        else {
                            "CreationDate": None,
                            "ModDate": None,
                            "Creator": "strategy-inference scale study",
                        }
                        if extension == "pdf"
                        else {"Software": "strategy-inference scale study"}
                    )
                    figure.savefig(
                        output / f"{stem}.{extension}",
                        dpi=180,
                        bbox_inches="tight",
                        metadata=metadata,
                    )
            manifest = dict(
                schema_version=1,
                input_sha256=hashlib.sha256(data).hexdigest(),
                plotter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                matplotlib=matplotlib.__version__,
                study=report["study"],
                profile=report["profile"],
                lead_time=LEAD,
                protocol=report["protocol"],
                statistics=computed,
                interpretation="Pointwise path-level Monte Carlo intervals; signed differences in score units; no invalid-path filtering. Weight paths are preset replicate 0, including startup.",
                figure_sha256={
                    target.name: hashlib.sha256(target.read_bytes()).hexdigest()
                    for target in targets[:-1]
                },
            )
            targets[-1].write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        finally:
            for figure in figures:
                plt.close(figure)
    return tuple(targets)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        for target in plot_study(args.input, args.output):
            print(target)
    except (KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
