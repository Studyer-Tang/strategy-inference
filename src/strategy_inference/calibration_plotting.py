"""Scientific figures for the separately frozen calibration comparison."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import MultipleLocator, PercentFormatter, ScalarFormatter  # noqa: E402

from .plotting import (  # noqa: E402
    GRAY,
    NAVY,
    PROCESS_NAMES,
    RUST,
    STYLE,
    _axes,
    _intervals,
    _rate_limit,
    _save,
    _title,
)

Rows = list[dict[str, Any]]
METHODS = {
    "fixed": ("Fixed original HAC scale", NAVY, "o"),
    "resampled": ("Resampled HAC scale", RUST, "s"),
    "oracle": ("Known-covariance Gaussian", GRAY, "^"),
    "iid": ("IID t: same HAC winner", NAVY, "o"),
    "hac": ("HAC z: same HAC winner", RUST, "s"),
}
INTEGER_COLUMNS = {
    "n_obs",
    "n_strategies",
    "delta",
    "hac_lags",
    "n_mc",
    "n_bootstrap",
    "reject_count",
}
FLOAT_COLUMNS = {
    "phi",
    "cross_corr",
    "block_length",
    "rate",
    "ci_low",
    "ci_high",
    "alpha",
    "size_upper_simultaneous",
}


def _read(path: Path) -> Rows:
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"The calibration table is empty: {path.name}")
    converted = []
    for row in rows:
        values: dict[str, Any] = dict(row)
        for name in INTEGER_COLUMNS & row.keys():
            values[name] = int(row[name])
        for name in FLOAT_COLUMNS & row.keys():
            if row[name] != "":
                values[name] = float(row[name])
        converted.append(values)
    return converted


def _curve(ax: Any, rows: Rows, method: str, coordinate: str) -> None:
    selected = sorted((row for row in rows if row["method"] == method), key=lambda r: r[coordinate])
    if not selected:
        raise ValueError(f"Missing calibration curve: {method}")
    points = [row[coordinate] for row in selected]
    if len(set(points)) != len(points):
        raise ValueError(f"Duplicate calibration curve points: {method}")
    label, color, marker = METHODS[method]
    ax.errorbar(
        points,
        [row["rate"] for row in selected],
        yerr=_intervals(selected),
        color=color,
        marker=marker,
        markersize=4,
        linewidth=1.15,
        elinewidth=0.75,
        capsize=2,
        label=label,
        zorder=3,
    )


def _frame(ax: Any, alpha: float) -> None:
    _axes(ax, alpha)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_ylabel("Null rejection rate")


def _percent_ticks(ax: Any, step: float) -> None:
    # Fix the locator as well as the formatter: rounded 2.5% ticks otherwise
    # appear as 2%, 8%, 12%, ... when labels have zero decimal places.
    ax.yaxis.set_major_locator(MultipleLocator(step))


def _legend(ax: Any, columns: int = 1) -> None:
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.02),
        ncol=columns,
        fontsize=7.7,
        handlelength=2,
        columnspacing=1.0,
        borderaxespad=0,
    )


def _subtitle(rows: Rows) -> str:
    fields = ("n_obs", "n_mc", "n_bootstrap", "hac_lags", "block_length")
    if any(len({row[field] for row in rows}) != 1 for field in fields):
        raise ValueError("A main figure must have a single sample size, MC count and bootstrap rule")
    row = rows[0]
    return (
        f"T = {row['n_obs']}, MC = {row['n_mc']}, B = {row['n_bootstrap']}, "
        f"L = {row['block_length']:g}, HAC lags = {row['hac_lags']}"
    )


def _dependence(rows: Rows, output: Path, alpha: float) -> list[Path]:
    low = [row for row in rows if row["method"] in ("fixed", "resampled", "oracle")]
    naive = [row for row in rows if row["method"] in ("iid", "hac")]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), layout="constrained")
    points = sorted({row["phi"] for row in rows})
    for ax in axes:
        _frame(ax, alpha)
        ax.set_xticks(points)
        ax.set_xlabel(r"Autocorrelation $\phi$")
        ax.set_xlim(points[0] - 0.04, points[-1] + 0.04)
    for method in ("fixed", "resampled", "oracle"):
        _curve(axes[0], low, method, "phi")
    for method in ("iid", "hac"):
        _curve(axes[1], naive, method, "phi")
    axes[0].set_ylim(0, _rate_limit(low))
    axes[1].set_ylim(0, _rate_limit(naive))
    _percent_ticks(axes[0], 0.05)
    _percent_ticks(axes[1], 0.1)
    axes[0].set_title("(a) Bootstrap and known-covariance reference", pad=34)
    axes[1].set_title("(b) Pointwise IID and HAC references", pad=34)
    _legend(axes[0], columns=2)
    _legend(axes[1], columns=2)
    _title(
        fig,
        "Autocorrelation and rejection under a zero mean",
        f"Gaussian AR(1), K = 1  |  {_subtitle(rows)}\n"
        "95% pointwise Wilson MC intervals; panels have separate vertical ranges",
    )
    return _save(fig, output / "figure-1-calibration-dependence")


def _selection(rows: Rows, output: Path, alpha: float) -> list[Path]:
    low = [row for row in rows if row["method"] in ("fixed", "resampled", "oracle")]
    naive = [row for row in rows if row["method"] in ("iid", "hac")]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), layout="constrained")
    points = sorted({row["n_strategies"] for row in rows})
    for ax in axes:
        _frame(ax, alpha)
        ax.set_xscale("log")
        ax.set_xticks(points)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.minorticks_off()
        ax.set_xlabel("Searched strategies K (log scale)")
        ax.set_xlim(points[0] / 1.18, points[-1] * 1.18)
    for method in ("fixed", "resampled", "oracle"):
        _curve(axes[0], low, method, "n_strategies")
    for method in ("iid", "hac"):
        _curve(axes[1], naive, method, "n_strategies")
    axes[0].set_ylim(0, _rate_limit(low))
    axes[1].set_ylim(0, 1)
    _percent_ticks(axes[0], 0.05)
    _percent_ticks(axes[1], 0.2)
    axes[0].set_title("(a) Family maximum tests", pad=34)
    axes[1].set_title("(b) Pointwise tests of the selected HAC winner", pad=34)
    _legend(axes[0], columns=2)
    _legend(axes[1], columns=2)
    _title(
        fig,
        "Strategy search and rejection under a zero mean",
        f"Nested Gaussian AR(1), phi = 0.5, shock correlation = 0.35  |  {_subtitle(rows)}\n"
        "95% pointwise Wilson MC intervals; panels have separate vertical ranges",
    )
    return _save(fig, output / "figure-2-calibration-selection")


def _processes(size_rows: Rows, power_rows: Rows, output: Path, alpha: float) -> list[Path]:
    processes = ("gaussian_ar", "student_ar", "garch")
    sizes = [row for row in size_rows if row["method"] in ("fixed", "resampled")]
    powers = [row for row in power_rows if row["method"] == "resampled"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 4.8), layout="constrained")
    _frame(left, alpha)
    _frame(right, alpha)
    positions = np.arange(len(processes))
    for offset, method in zip((-0.08, 0.08), ("fixed", "resampled"), strict=True):
        selected = []
        for process in processes:
            matches = [row for row in sizes if row["process"] == process]
            matches = [row for row in matches if row["method"] == method and row["delta"] == 0]
            if len(matches) != 1:
                raise ValueError(f"Expected one zero-mean size row for {process}/{method}")
            selected.append(matches[0])
        label, color, marker = METHODS[method]
        left.errorbar(
            positions + offset,
            [row["rate"] for row in selected],
            yerr=_intervals(selected),
            linestyle="none",
            color=color,
            marker=marker,
            markersize=4.7,
            elinewidth=0.9,
            capsize=3,
            label=label,
            zorder=3,
        )
    left.set_xticks(
        positions, ["Gaussian\nAR(1)", "Heavy-tailed\nAR(1)", "Heavy-tailed\nGARCH(1,1)"]
    )
    left.set(xlim=(-0.35, 2.35), ylim=(0, _rate_limit(sizes)))
    _percent_ticks(left, 0.05)
    left.set_title("(a) All strategy means are zero", pad=34)
    _legend(left, columns=2)
    for process, color, marker in zip(processes, (NAVY, RUST, GRAY), ("o", "s", "^"), strict=True):
        selected = sorted(
            (row for row in powers if row["process"] == process), key=lambda row: row["delta"]
        )
        if not selected:
            raise ValueError(f"Missing resampled-HAC power curve: {process}")
        right.errorbar(
            [row["delta"] for row in selected],
            [row["rate"] for row in selected],
            yerr=_intervals(selected),
            color=color,
            marker=marker,
            markersize=4,
            linewidth=1.15,
            elinewidth=0.75,
            capsize=2,
            label=PROCESS_NAMES[process],
            zorder=3,
        )
    points = sorted({row["delta"] for row in powers})
    right.set_xticks(points)
    right.set(
        xlabel=r"Signal $\delta$ (population long-run standard errors)",
        ylabel="Rejection probability",
        xlim=(points[0] - 0.12, points[-1] + 0.12),
        ylim=(0, 1),
    )
    _percent_ticks(right, 0.2)
    right.set_title("(b) Resampled-HAC max test: one positive mean", pad=34)
    _legend(right, columns=2)
    _title(
        fig,
        "Null rejection and local power under three processes",
        f"K = 20, AR phi = 0.5, GARCH phi = 0  |  {_subtitle(size_rows)}\n"
        "95% pointwise Wilson MC intervals; signal in the first strategy only",
    )
    return _save(fig, output / "figure-3-calibration-processes")


def plot_calibration(output: str | Path, metadata: Mapping[str, Any]) -> list[Path]:
    """Read the four main CSVs and save PDF/SVG/PNG copies of three figures."""
    output = Path(output).resolve()
    rows = {
        name: _read(output / name)
        for name in (
            "figure-1-dependence.csv",
            "figure-2-selection.csv",
            "figure-3-size.csv",
            "figure-3-power.csv",
        )
    }
    alpha = float(metadata["resolved_protocol"]["alpha"])
    style = {**STYLE, "svg.hashsalt": "strategy-inference-calibration-study-1"}
    with plt.rc_context(style):
        paths = _dependence(rows["figure-1-dependence.csv"], output, alpha)
        paths += _selection(rows["figure-2-selection.csv"], output, alpha)
        paths += _processes(rows["figure-3-size.csv"], rows["figure-3-power.csv"], output, alpha)
    return paths
