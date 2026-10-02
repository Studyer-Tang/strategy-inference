"""Three restrained, exportable scientific figures; no result-dependent tuning."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import ScalarFormatter  # noqa: E402

Rows = Iterable[Mapping[str, Any]]
NAVY = "#344e64"
RUST = "#a7654d"
GRAY = "#75818b"
INK = "#29343b"
COLORS = (NAVY, RUST, GRAY)
MARKERS = ("o", "s", "^")
PROCESS_NAMES = {
    "gaussian_ar": "Gaussian AR(1)",
    "student_ar": "Heavy-tailed AR(1)",
    "garch": "Heavy-tailed GARCH(1,1)",
}
STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.edgecolor": "#9aa2a7",
    "axes.linewidth": 0.7,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "svg.hashsalt": "strategy-inference-frozen-protocol",
}


def _axes(ax: Any, alpha: float = 0.05) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e2e6e9", linewidth=0.65)
    ax.set_axisbelow(True)
    ax.axhline(alpha, color="#92999e", linewidth=1.0, linestyle=(0, (4, 3)),
               label="Nominal 5%", zorder=1)


def _intervals(rows: list[Mapping[str, Any]]) -> np.ndarray:
    rates = np.array([r["rate"] for r in rows], dtype=float)
    low = np.array([r["ci_low"] for r in rows], dtype=float)
    high = np.array([r["ci_high"] for r in rows], dtype=float)
    return np.maximum(np.vstack((rates - low, high - rates)), 0.0)


def _rate_limit(rows: list[Mapping[str, Any]], extra: float = 0.0) -> float:
    upper = max([0.1, extra] + [float(row["ci_high"]) for row in rows])
    return min(1.0, np.ceil((upper + 0.025) * 10) / 10)


def _title(fig: Any, title: str, subtitle: str) -> None:
    fig.suptitle(title + "\n" + subtitle, fontsize=10, linespacing=1.65, weight="normal")


def _save(fig: Any, base: Path) -> list[Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in ("pdf", "svg", "png"):
        path = base.with_suffix("." + suffix)
        if suffix == "pdf":
            metadata = {"Creator": "strategy-inference", "CreationDate": None, "ModDate": None}
        elif suffix == "svg":
            metadata = {"Creator": "strategy-inference", "Date": None}
        else:
            metadata = {"Software": "strategy-inference"}
        fig.savefig(path, dpi=240, metadata=metadata, facecolor="white")
        outputs.append(path)
    plt.close(fig)
    return outputs


def plot_autocorrelation(
    rows: Rows, output: str | Path, *, n_obs: int, n_mc: int, n_bootstrap: int,
) -> list[Path]:
    records = list(rows)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6.5, 4.4), layout="constrained")
        _axes(ax)
        methods = ("iid_t", "hac_z", "stationary_bootstrap")
        labels = ("IID t", "HAC z", "Stationary bootstrap")
        for method, label, color, marker in zip(methods, labels, COLORS, MARKERS, strict=True):
            selected = sorted((r for r in records if r["method"] == method), key=lambda r: r["phi"])
            ax.errorbar([r["phi"] for r in selected], [r["rate"] for r in selected],
                        yerr=_intervals(selected), color=color, marker=marker,
                        markersize=4.3, linewidth=1.25, elinewidth=0.8, capsize=2,
                        label=label, zorder=3)
        unique = {float(r["phi"]): float(r["iid_limit_reference"]) for r in records}
        x = sorted(unique)
        ax.plot(x, [unique[v] for v in x], color=INK, linewidth=1.05,
                linestyle=(0, (1.5, 2)), label="IID normal-limit reference")
        ax.set(xlabel=r"Autocorrelation $\phi$", ylabel="False positive rate",
               xlim=(-0.035, 0.835), ylim=(0, _rate_limit(records, max(unique.values()))))
        ax.set_xticks(x)
        ax.legend(loc="upper left", handlelength=2.5)
        _title(fig, "Ignoring dependence changes the size of a mean test",
               f"Gaussian AR(1), K = 1  |  T = {n_obs}, MC = {n_mc}, B = {n_bootstrap}\n"
               "One-sided tests; 95% pointwise Wilson Monte Carlo intervals")
        return _save(fig, Path(output) / "figure-1-autocorrelation")


def plot_selection(
    rows: Rows, output: str | Path, *, n_obs: int, n_mc: int, n_bootstrap: int,
) -> list[Path]:
    records = list(rows)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6.5, 4.4), layout="constrained")
        _axes(ax)
        methods = ("iid_winner", "hac_winner", "bootstrap_max")
        labels = ("IID t: same HAC winner", "HAC z: same HAC winner", "Stationary bootstrap max")
        for method, label, color, marker in zip(methods, labels, COLORS, MARKERS, strict=True):
            selected = sorted((r for r in records if r["method"] == method), key=lambda r: r["n_strategies"])
            ax.errorbar([r["n_strategies"] for r in selected], [r["rate"] for r in selected],
                        yerr=_intervals(selected), color=color, marker=marker,
                        markersize=4.3, linewidth=1.25, elinewidth=0.8, capsize=2,
                        label=label, zorder=3)
        counts = sorted({int(r["n_strategies"]) for r in records})
        ax.set_xscale("log")
        ax.set_xticks(counts)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.minorticks_off()
        ax.set(xlabel="Number of searched strategies K (log scale)",
               ylabel="False positive rate", xlim=(0.85, 60),
               ylim=(0, _rate_limit(records)))
        ax.legend(loc="upper left", handlelength=2.5)
        _title(fig, "Single-strategy p-values do not account for selecting a winner",
               f"Nested search, Gaussian AR(1), phi = 0.5, shock correlation = 0.35\n"
               f"T = {n_obs}, MC = {n_mc}, B = {n_bootstrap}  |  95% pointwise Wilson intervals")
        return _save(fig, Path(output) / "figure-2-selection")


def plot_robustness(
    size_rows: Rows, power_rows: Rows, output: str | Path, *,
    n_obs: int, n_mc: int, n_bootstrap: int,
) -> list[Path]:
    sizes, powers = list(size_rows), list(power_rows)
    processes = ("gaussian_ar", "student_ar", "garch")
    with plt.rc_context(STYLE):
        fig, (left, right) = plt.subplots(1, 2, figsize=(10.0, 5.2), layout="constrained")
        _axes(left)
        _axes(right)
        positions = np.arange(len(processes))
        methods = ("iid_winner", "hac_winner", "bootstrap_max")
        labels = ("IID t: HAC winner", "HAC z: HAC winner", "Stationary bootstrap max")
        for offset, method, label, color in zip((-0.24, 0.0, 0.24), methods, labels, COLORS, strict=True):
            selected = [next(r for r in sizes if r["process"] == process and r["method"] == method)
                        for process in processes]
            left.bar(positions + offset, [r["rate"] for r in selected], width=0.22,
                     yerr=_intervals(selected), color=color, linewidth=0,
                     capsize=2, error_kw={"ecolor": INK, "elinewidth": 0.75}, label=label, zorder=3)
        left.set_xticks(positions, ["Gaussian\nAR(1)", "Heavy-tailed\nAR(1)", "Heavy-tailed\nGARCH(1,1)"])
        left.set(ylabel="False positive rate", ylim=(0, _rate_limit(sizes)))
        left.set_title("(a) All strategy means are zero", pad=43)
        left.legend(loc="lower left", bbox_to_anchor=(0, 1.025), ncol=2, handlelength=1.7,
                    columnspacing=1.0)

        for process, color, marker in zip(processes, COLORS, MARKERS, strict=True):
            selected = sorted((r for r in powers if r["process"] == process), key=lambda r: r["delta"])
            right.errorbar([r["delta"] for r in selected], [r["rate"] for r in selected],
                           yerr=_intervals(selected), color=color, marker=marker,
                           markersize=4.3, linewidth=1.25, elinewidth=0.8, capsize=2,
                           label=PROCESS_NAMES[process], zorder=3)
        right.set(xlabel=r"Signal $\delta$ (long-run standard errors)",
                  ylabel="Rejection probability", xlim=(-0.12, 4.12), ylim=(0, 1))
        right.set_title("(b) Max-test power: one strategy has a positive mean", pad=43)
        right.set_xticks([0, 1, 2, 3, 4])
        right.legend(loc="lower left", bbox_to_anchor=(0, 1.025), ncol=2, handlelength=2.0,
                     columnspacing=1.0)
        _title(fig, "Finite-sample size and local power under three shock processes",
               f"K = 20, T = {n_obs}, MC = {n_mc}, B = {n_bootstrap}  |  95% pointwise Wilson intervals\n"
               "AR phi = 0.5; GARCH phi = 0; correlated strategies; fixed original HAC scale")
        return _save(fig, Path(output) / "figure-3-robustness")
