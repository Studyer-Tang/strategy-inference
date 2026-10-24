"""Evaluate saved NumPy predictions without refitting models in the evaluator.

This seeded synthetic example illustrates the interface, not model superiority
or statistical calibration. Parameters and model definitions are fixed below.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from strategy_inference import (
    BacktestResult,
    compare_forecasts,
    evaluate_forecasts,
    rolling_splits,
)


def saved_predictions():
    """Stand in for predictions produced earlier by an external model pipeline."""
    rng = np.random.default_rng(17)
    y = np.empty(320)
    y[0] = rng.normal()
    for t in range(1, len(y)):
        y[t] = 0.7 * y[t - 1] + rng.normal()
    splits = rolling_splits(
        len(y), initial_train_size=64, window=48, horizon=3, gap=1, step=4,
    )
    predictions = {name: [] for name in ("naive", "numpy_ar1")}
    for split in splits:
        train = y[split.train_start:split.train_stop]
        design = np.column_stack((np.ones(len(train) - 1), train[:-1]))
        intercept, coefficient = np.linalg.lstsq(design, train[1:], rcond=None)[0]
        path, value = [], train[-1]
        for _ in range(split.test_stop - split.origin - 1):
            value = intercept + coefficient * value
            path.append(value)
        gap = split.test_start - split.train_stop
        predictions["naive"].append(np.repeat(train[-1], 3))
        predictions["numpy_ar1"].append(path[gap:])
    return y, {name: np.asarray(points) for name, points in predictions.items()}, splits


def run():
    y, predictions, splits = saved_predictions()
    # No model callable is passed: only saved predictions and declared ranges.
    panel = BacktestResult.from_predictions(y, predictions, splits=splits)
    scores = evaluate_forecasts(panel, loss="squared")
    comparison = compare_forecasts(
        panel, baseline="naive", lead_time=2, loss="squared",
        n_resamples=999, seed=17, search_complete=True,
    )
    record = dict(
        scope="Synthetic API demonstration; not a model benchmark or calibration study.",
        data=dict(process="Y[t] = 0.7 Y[t-1] + N(0,1)", seed=17, n_observations=len(y)),
        schedule=dict(initial_train_size=64, window=48, horizon=3, gap=1, step=4),
        predictions=panel.to_dict(),
        evaluation=scores.to_dict(),
        comparison=comparison.to_dict(),
    )
    return panel, scores, record


def plot(panel, scores, destination):
    """Optional descriptive figure; importing the library never loads plotting."""
    import matplotlib.pyplot as plt

    colors = ("#8995a3", "#245b78")
    labels = {"naive": "Naive", "numpy_ar1": "NumPy AR(1)"}
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none", "svg.hashsalt": "saved-forecasts"}):
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), layout="constrained")
        x = panel.target_indices[:, 0]
        axes[0].plot(x, panel.actuals[:, 0], color="#202832", lw=1.3, label="Observed")
        for m, name in enumerate(panel.names):
            axes[0].plot(x, panel.forecasts[:, 0, m], color=colors[m], lw=1, label=labels[name])
            axes[1].bar(np.arange(3) + (m - .5) * .32, scores.mean_loss[:, m],
                        width=.3, color=colors[m], label=labels[name])
        axes[0].set(title="Two-step predictions", xlabel="Target position", ylabel="Value")
        axes[1].set_ylim(0, float(scores.mean_loss.max()) * 1.25)
        axes[1].set(title="Error by physical lead", xlabel="Lead from forecast origin",
                    ylabel="Mean squared error", xticks=np.arange(3), xticklabels=scores.lead_times)
        for ax in axes:
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=.15)
            ax.set_axisbelow(True)
            ax.legend(frameon=False, fontsize=8)
        fig.suptitle("Saved forecasts, explicit time alignment", fontsize=13, fontweight="normal")
        fig.supxlabel("Synthetic AR(1), seed 17. Descriptive errors; no calibration claim.", fontsize=9)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, metadata={"Date": None} if destination.suffix.lower() == ".svg" else None)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Save all predictions, indices and results as JSON")
    parser.add_argument("--figure", type=Path, help="Optional SVG or PNG figure; requires matplotlib")
    args = parser.parse_args()
    panel, scores, record = run()
    print(f"{panel.n_folds} forecast origins; physical leads {scores.lead_times}; squared loss")
    print(f"{'Model':<16}{'Lead':>6}{'Mean loss':>14}")
    for row in scores.records():
        print(f"{row['model']:<16}{row['lead_time']:>6}{row['mean_loss']:>14.4f}")
    for row in record["comparison"]["candidates"]:
        print(f"Lead 2, {row['model']} vs naive: improvement={row['mean_improvement']:.4f}, "
              f"adjusted p={row['adjusted_pvalue']:.4f}")
    print(record["scope"])
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if args.figure:
        plot(panel, scores, args.figure)


if __name__ == "__main__":
    main()
