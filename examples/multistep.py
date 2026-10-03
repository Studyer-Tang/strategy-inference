"""Multi-step mature-feedback intervals on a simulated random walk.

Run after installing this checkout:
    python examples/multistep.py

This demonstrates alignment and exports, not a coverage or efficiency study.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from strategy_inference import (
    MultiStepConformal,
    backtest,
    multistep_intervals,
    naive_forecast,
)


def run():
    rng = np.random.default_rng(17)
    y = np.cumsum(rng.normal(size=384))
    rolling = backtest(
        y, {"naive": naive_forecast}, initial_train_size=128, window=128, horizon=12,
    )
    origins = np.asarray([split.origin for split in rolling.splits], dtype=np.int64)
    leads = rolling.target_indices[0] - origins[0]
    if not np.array_equal(
        rolling.target_indices - origins[:, None],
        np.broadcast_to(leads, rolling.target_indices.shape),
    ):
        raise ValueError("Every row must have the same physical lead times.")
    # These sqrt(h) units follow from this simulation's unit-variance random
    # walk innovations. They are not a general data-driven scale estimator.
    initial_scales = np.sqrt(leads)
    rates = 0.1 / np.sqrt(leads)  # A damping example, not an optimal-rate claim.
    intervals = multistep_intervals(
        y, rolling.forecasts[:, :, 0], origins=origins, lead_times=leads,
        alpha=0.1, step_size=rates, decay=0.2, scale=initial_scales,
        initial_quantile=0.65, strategy="pooled", scale_decay=0.97,
        scale_source="blended", scale_share_weight=0.5, scale_floor=1e-8,
    )

    # The class observes each current label before issuing its future path.
    # Ending now leaves future targets pending; no future labels are invented.
    tiny_leads = np.asarray([1, 3])
    tracker = MultiStepConformal(
        tiny_leads, scale=np.sqrt(tiny_leads), step_size=0.1 / np.sqrt(tiny_leads),
        decay=0.2, initial_quantile=0.65, scale_decay=0.97, scale_source="blended",
        scale_share_weight=[0.5, 0.75],
    )
    for t, label in enumerate(y[:4]):
        tracker.observe(t, float(label))
        tracker.predict(np.full(len(tiny_leads), label))
    return {
        "data": "Unit-innovation random walk, seed 17; an API demonstration.",
        "batch": {
            "n_origins": intervals.n_origins,
            "lead_times": list(intervals.lead_times),
            "initial_scale": initial_scales.tolist(),
            "step_size": rates.tolist(),
            "summary": intervals.summary(),
            "n_pending": len(intervals.pending),
        },
        "tiny_stream": {
            "observed_through": tracker.last_time,
            "summary": tracker.summary(),
            "current_scales": list(tracker.current_scales),
            "scale_share_weight": list(tracker.current_scale_weights),
            "pending": [interval.to_dict() for interval in tracker.pending],
        },
        "interpretation": "Per-lead realized mature coverage; no per-time, conditional or simultaneous-path guarantee. Fixed blend weights are a prespecified compromise between own-lead and shortest-lead scales, not an optimal-weight claim.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    contents = json.dumps(run(), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(contents, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            handle.write(contents)
        print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
