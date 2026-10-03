"""A simulated rolling forecast workflow; this is not a validity experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from strategy_inference import (
    SeasonalNaive,
    adaptive_intervals,
    backtest,
    compare_forecasts,
    drift_forecast,
    evaluate_forecasts,
    naive_forecast,
    sequential_compare_forecasts,
)


def run():
    # Fix the models, lead and interval model before generating the data.
    models = {"naive": naive_forecast, "seasonal": SeasonalNaive(12), "drift": drift_forecast}
    rng = np.random.default_rng(17)
    y = 2 * np.sin(2 * np.pi * np.arange(512) / 12) + rng.normal(size=512)
    rolling = backtest(y, models, initial_train_size=128, window=128, horizon=3)
    scores = evaluate_forecasts(rolling, loss="squared")
    comparison = compare_forecasts(
        rolling, baseline="naive", lead_time=1, n_resamples=999, seed=17, search_complete=True
    )
    sequential = sequential_compare_forecasts(rolling, lead_time=1, loss="absolute")
    # Adjacent one-step forecasts have immediate feedback before the next origin.
    # Scale is fixed from training data, with no use of evaluation labels.
    scale = float(y[:128].std())
    intervals = adaptive_intervals(
        rolling.actuals[:, 0], rolling.forecasts[:, 0, 1], alpha=0.1, scale=scale
    )
    return dict(
        data="Simulated seasonal series, seed 17; an API demonstration.",
        evaluation=scores.to_dict(),
        comparison=comparison.to_dict(),
        sequential=sequential.to_dict(),
        online=dict(
            model="seasonal",
            lead_time=1,
            target_coverage=0.9,
            realized_coverage=intervals.coverage,
            empty_count=int(intervals.empty.sum()),
            unbounded_count=int(intervals.unbounded.sum()),
            scale=scale,
            interpretation="Realized one-step coverage; not per-time validity.",
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    record = run()
    contents = json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(contents, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(contents)
        print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
