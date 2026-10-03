"""Plot the prespecified fixed-weight blended-scale study from its saved ledger.

No forecasts, scale updates or simulations are run. Statistics are recomputed
from all independent path records, retaining every invalid-score path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import zipfile
from pathlib import Path

import plot_scale_study as common

COMMON_SHA256 = "2c129a90e5c448bc34201277d0cca2c04f7137d6be996b12187c4c0daf90fd5c"
PRIMARY = ("fixed", "horizon", "shortest", "blend_50")
WEIGHT_METHODS = ("horizon", "blend_25", "blend_50", "blend_75", "shortest")
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
LABELS = {**common.LABELS, "blend_50": "Half blend"}
STEMS = ("score-difference", "local-coverage-error", "weight-sensitivity")
# Frozen inventory of the study runner's source_hashes(), independent of future
# checkout additions. Hashes are recorded evidence, not a claim of a fresh rerun.
PACKAGE_MODULES = (
    "__init__",
    "__main__",
    "_validation",
    "audit",
    "bootstrap",
    "calibration",
    "calibration_plotting",
    "calibration_report",
    "cli",
    "conformal",
    "evaluation",
    "experiments",
    "inference",
    "io",
    "model_selection",
    "multistep",
    "parametric",
    "plotting",
    "quadratic_reference",
    "reference",
    "report",
    "simulations",
    "tail",
    "testing",
    "uncertainty",
    "wilks",
)
REQUIRED_SOURCES = frozenset(
    {f"src/strategy_inference/{name}.py" for name in PACKAGE_MODULES}
    | {
        "pyproject.toml",
        "scripts/reproduce_blended_scales.py",
        "scripts/reproduce_scale_transfer.py",
        "scripts/reproduce_multistep.py",
        "experiments/blended-scale-protocol.json",
    }
)


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def verify_archive(input_path, report):
    """Check an adjacent frozen source snapshot without extracting or importing it."""
    archive = input_path.parent / "source.zip"
    manifest = input_path.parent / "source-manifest.json"
    if not archive.exists() and not manifest.exists():
        return None
    if not archive.is_file() or not manifest.is_file():
        raise ValueError("The frozen source archive and manifest must both be present.")
    source = report["source_sha256"]
    saved = json.loads(manifest.read_bytes())
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if (
        saved.get("schema_version") != 1
        or saved.get("source_sha256") != source
        or saved.get("archive_sha256") != digest
    ):
        raise ValueError("Source snapshot manifest does not match the saved study.")
    try:
        with zipfile.ZipFile(archive) as packed:
            entries = packed.infolist()
            if len(entries) != len(source) or {entry.filename for entry in entries} != set(source):
                raise ValueError("Frozen source archive inventory is incomplete or duplicated.")
            for entry in entries:
                kind = stat.S_IFMT(entry.external_attr >> 16)
                if entry.is_dir() or kind not in (0, stat.S_IFREG):
                    raise ValueError("Frozen source archive entries must be ordinary files.")
                content = packed.read(entry)
                if hashlib.sha256(content).hexdigest() != source[entry.filename]:
                    raise ValueError("Frozen source bytes do not match the recorded source hash.")
                if (
                    entry.filename == "experiments/blended-scale-protocol.json"
                    and json.loads(content) != report["protocol"]
                ):
                    raise ValueError("Embedded protocol does not match the frozen protocol bytes.")
    except (zipfile.BadZipFile, OSError) as exc:
        raise ValueError("Frozen source archive is damaged or unreadable.") from exc
    return dict(sha256=digest, n_files=len(source))


def statistics(report):
    protocol, settings, names, methods, leads, rows, _ = common.validate_report(report)
    if (
        report["study"] != "fixed-cross-lead-scale-blending"
        or protocol["study"] != report["study"]
        or set(methods) != {"fixed", *WEIGHT_METHODS}
    ):
        raise ValueError("Expected the complete prespecified six-method blended study.")
    parameters = protocol["method_parameters"]
    if set(parameters) != set(methods):
        raise ValueError("Every method requires its declared scale parameters.")
    for method, weight in zip(WEIGHT_METHODS, WEIGHTS, strict=True):
        config = parameters[method]
        expected_source = "horizon" if weight == 0 else "shortest" if weight == 1 else "blended"
        if (
            config["scale_source"] != expected_source
            or config["scale_decay"] != protocol["scale_decay"]
        ):
            raise ValueError("Sensitivity methods must use the declared common RMS clock.")
        if (
            0 < weight < 1
            and common._real(config["scale_share_weight"], "scale share weight") != weight
        ):
            raise ValueError("Sensitivity labels do not match the declared sharing weights.")
    if parameters["fixed"] != {"scale_source": "horizon", "scale_decay": None}:
        raise ValueError("The fixed-scale reference must disable scale adaptation.")
    source = report.get("source_sha256")
    if (
        not isinstance(source, dict)
        or set(source) != REQUIRED_SOURCES
        or not all(_digest(digest) for digest in source.values())
    ):
        raise ValueError("A complete, valid frozen study source-hash inventory is required.")
    fingerprints = report.get("input_fingerprints")
    if not isinstance(fingerprints, list):
        raise ValueError("Complete path input fingerprints are required.")
    for entry in fingerprints:
        if not _digest(entry.get("actual_sha256")) or not _digest(entry.get("predicted_sha256")):
            raise ValueError("Input array fingerprints must be SHA-256 digests.")
        scales = entry["initial_scales"]
        if len(scales) != len(leads) or any(
            common._real(v, "initial scale", minimum=0) == 0 for v in scales
        ):
            raise ValueError("Initial scales must be positive and match the lead vector.")
    score_differences, local_errors, sensitivity = [], [], []
    n = settings["repetitions"]
    for name in names:
        groups = {
            method: [rows[name, rep, method, common.LEAD] for rep in range(n)] for method in methods
        }
        for method in PRIMARY:
            group = groups[method]
            local_errors.append(
                dict(
                    scenario=name,
                    method=method,
                    lead_time=common.LEAD,
                    estimate=common.mean_ci(row["worst_local_coverage_error"] for row in group),
                )
            )
            if method == "shortest":
                continue
            pairs = tuple(zip(group, groups["shortest"], strict=True))
            invalid = sum(
                a["mean_interval_score"] is None or b["mean_interval_score"] is None
                for a, b in pairs
            )
            score_differences.append(
                dict(
                    scenario=name,
                    method=method,
                    baseline="shortest",
                    lead_time=common.LEAD,
                    n_pairs=n,
                    invalid_score_pairs=invalid,
                    estimate=None
                    if invalid
                    else common.mean_ci(
                        a["mean_interval_score"] - b["mean_interval_score"] for a, b in pairs
                    ),
                )
            )
        for method, weight in zip(WEIGHT_METHODS, WEIGHTS, strict=True):
            group = groups[method]
            invalid = sum(row["mean_interval_score"] is None for row in group)
            sensitivity.append(
                dict(
                    scenario=name,
                    method=method,
                    short_source_weight=weight,
                    lead_time=common.LEAD,
                    n_paths=n,
                    invalid_score_runs=invalid,
                    estimate=None
                    if invalid
                    else common.mean_ci(row["mean_interval_score"] for row in group),
                )
            )
    return dict(
        score_differences=score_differences,
        local_coverage_error=local_errors,
        weight_sensitivity=sensitivity,
    ), (protocol, settings, names)


def make_figures(report):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    computed, (protocol, settings, names) = statistics(report)
    note = f"{report['profile'].capitalize()} · {settings['repetitions']} independent paths per scenario · nominal pointwise 95% MC t summaries; no simultaneous claim."
    figures = []
    for field, title, xlabel, methods in (
        (
            "score_differences",
            "Interval score difference versus shortest (h = 24)",
            "Mean score difference (score units)",
            tuple(m for m in PRIMARY if m != "shortest"),
        ),
        (
            "local_coverage_error",
            f"Worst {protocol['local_window']}-point coverage error (h = 24)",
            "Mean worst-window absolute coverage error",
            PRIMARY,
        ),
    ):
        figure, axes = common._panels(plt, names)
        for name, axis in zip(names, axes, strict=True):
            common._style(axis, name)
            selected = {row["method"]: row for row in computed[field] if row["scenario"] == name}
            for y, method in enumerate(methods):
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
                    common._dot(axis, y, row["estimate"])
            axis.set_yticks(range(len(methods)), [LABELS[m] for m in methods])
            axis.set_ylim(len(methods) - 0.5, -0.5)
            axis.xaxis.set_major_locator(MaxNLocator(4))
            axis.set_xlabel(xlabel, fontsize=9)
            if field == "score_differences":
                axis.axvline(0, color="0.45", linewidth=0.8, linestyle="--")
        figure.suptitle(title, x=0.035, ha="left", fontsize=14)
        tail = (
            "Method − shortest in score units; negative is better. Any invalid score makes its entire comparison unavailable."
            if field == "score_differences"
            else f"Path statistic: max |window coverage − {1 - protocol['alpha']:g}| on the evaluation suffix. No window-coverage guarantee."
        )
        figure.text(0.035, 0.075, note, fontsize=8.5)
        figure.text(0.035, 0.035, tail, fontsize=8.5)
        figure.tight_layout(rect=(0.02, 0.13, 1, 0.94), h_pad=1.8, w_pad=2.2)
        figures.append(figure)
    figure, axes = common._panels(plt, names)
    for name, axis in zip(names, axes, strict=True):
        common._style(axis, name)
        selected = [row for row in computed["weight_sensitivity"] if row["scenario"] == name]
        for row in selected:
            weight, estimate = row["short_source_weight"], row["estimate"]
            if estimate is None:
                axis.text(
                    weight,
                    0.03,
                    "N/A",
                    transform=axis.get_xaxis_transform(),
                    ha="center",
                    fontsize=8,
                )
            else:
                axis.errorbar(
                    weight,
                    estimate["mean"],
                    yerr=[
                        [estimate["mean"] - estimate["lower"]],
                        [estimate["upper"] - estimate["mean"]],
                    ],
                    fmt="o",
                    color="0.12",
                    markersize=4,
                    elinewidth=0.85,
                    capsize=2,
                )
        axis.set_xticks(WEIGHTS, ["0", ".25", ".50", ".75", "1"])
        axis.set_xlim(-0.055, 1.055)
        axis.yaxis.set_major_locator(MaxNLocator(4))
        axis.set_xlabel("Fixed short-source weight", fontsize=9)
        axis.set_ylabel("Mean interval score", fontsize=9)
    figure.suptitle(
        "Prespecified sharing-weight sensitivity (h = 24)", x=0.035, ha="left", fontsize=14
    )
    figure.text(0.035, 0.075, note, fontsize=8.5)
    figure.text(
        0.035,
        0.035,
        "All five weights shown in original score units. Endpoints: own lead (0), shortest (1). Descriptive sensitivity; no optimal-weight or simultaneous claim.",
        fontsize=8.5,
    )
    figure.tight_layout(rect=(0.02, 0.13, 1, 0.94), h_pad=1.8, w_pad=2.2)
    figures.append(figure)
    return figures, computed


def plot_study(input_path, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    common_path = Path(common.__file__)
    if hashlib.sha256(common_path.read_bytes()).hexdigest() != COMMON_SHA256:
        raise ValueError("The independently tested common plotter source has changed.")
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
    archive = verify_archive(input_path, report)
    with matplotlib.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.linewidth": 0.65,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.hashsalt": "strategy-inference-blended-study-v1",
            "pdf.fonttype": 42,
        }
    ):
        figures, computed = make_figures(report)
        try:
            output.mkdir(parents=True, exist_ok=True)
            for stem, figure in zip(STEMS, figures, strict=True):
                for extension in ("svg", "png", "pdf"):
                    metadata = (
                        {"Date": None, "Creator": "strategy-inference blended study"}
                        if extension == "svg"
                        else {
                            "CreationDate": None,
                            "ModDate": None,
                            "Creator": "strategy-inference blended study",
                        }
                        if extension == "pdf"
                        else {"Software": "strategy-inference blended study"}
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
                plotter_sha256={
                    "scripts/plot_blended_study.py": hashlib.sha256(
                        Path(__file__).read_bytes()
                    ).hexdigest(),
                    "scripts/plot_scale_study.py": COMMON_SHA256,
                },
                matplotlib=matplotlib.__version__,
                study=report["study"],
                profile=report["profile"],
                lead_time=common.LEAD,
                protocol=report["protocol"],
                study_source_sha256=report["source_sha256"],
                source_archive=archive,
                input_fingerprints=report["input_fingerprints"],
                statistics=computed,
                interpretation="Nominal pointwise path-level MC t summaries, not certified population-risk intervals; signed score differences in original units; no invalid-path filtering. All prespecified sensitivity weights shown without selecting an optimum. Finite sampled scores do not establish finite unconditional expected interval score.",
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
    except (KeyError, TypeError, ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
