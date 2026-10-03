"""Compare delayed robust mixing of own and short-lead residual scales."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from reproduce_multistep import metrics  # noqa: E402
from reproduce_scale_transfer import (  # noqa: E402
    _array_sha,
    _ci,
    forecasts,
    generate,
    initial_scales,
)
from reproduce_scale_transfer import (  # noqa: E402
    replay as baseline_replay,
)


def source_hashes():
    paths = list((ROOT / "src/strategy_inference").glob("*.py")) + [
        Path(__file__),
        ROOT / "scripts/scale_mixture_prototype.py",
        ROOT / "scripts/reproduce_scale_transfer.py",
        ROOT / "scripts/scale_transfer_prototype.py",
        ROOT / "scripts/reproduce_multistep.py",
        ROOT / "experiments/scale-mixture-protocol.json",
    ]
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }


def replay(actual, points, origins, scales, protocol):
    from scale_mixture_prototype import TwoScaleMixture

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
    )
    learner = TwoScaleMixture(
        leads, learning_rate=protocol["learning_rate"], df=protocol["student_df"]
    )
    own = [float(v) for v in scales]
    ratios = [v / own[0] for v in own]
    short = own[0]
    a, b = math.sqrt(protocol["scale_decay"]), math.sqrt(1 - protocol["scale_decay"])
    floor = protocol["scale_floor"]
    shape = points.shape
    lower, upper, issued_scales, quantiles, weights = (np.empty(shape) for _ in range(5))
    values = np.full(shape, np.nan)
    evaluated, misses, empty, unbounded = (np.zeros(shape, dtype=bool) for _ in range(4))
    lead_index = {h: i for i, h in enumerate(leads)}
    first, pending, max_pending = int(origins[0]), {}, 0
    for t in range(first, len(actual)):
        updates = tracker.observe(t, float(actual[t]))
        for update in updates:
            f, col = update.origin - first, lead_index[update.lead_time]
            error = abs(update.actual - update.prediction)
            # Contexts use the scales that were issued, not today's updated scales.
            learner.update(update.lead_time, error, pending.pop((update.origin, update.lead_time)))
            own[col] = max(floor, math.hypot(a * own[col], b * error))
            if col == 0:
                short = own[0]
            values[f, col], evaluated[f, col], misses[f, col] = (
                update.actual,
                True,
                update.miss,
            )
        f = t - first
        if f < len(origins):
            contexts = [
                learner.issue(h, short * ratios[col], own[col]) for col, h in enumerate(leads)
            ]
            for col, interval in enumerate(
                tracker.predict(points[f], scale=[context.scale for context in contexts])
            ):
                lower[f, col], upper[f, col] = interval.lower, interval.upper
                issued_scales[f, col], quantiles[f, col] = interval.scale, interval.quantile
                weights[f, col] = contexts[col].weight
                empty[f, col], unbounded[f, col] = (
                    interval.kind == "empty",
                    interval.kind == "unbounded",
                )
                pending[(t, interval.lead_time)] = contexts[col]
            max_pending = max(max_pending, len(pending))
    if pending or tracker.pending:
        raise RuntimeError("Every issued forecast must mature within the study path.")
    if max_pending > sum(leads):
        raise RuntimeError("Pending contexts exceeded the bounded streaming design.")
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
        weights=weights,
        max_pending=max_pending,
        terminal=tracker.to_dict(),
        learner=learner.to_dict(),
    )


def summarize(records, protocol):
    aggregate, contrasts = [], []
    for scenario in protocol["scenarios"]:
        for lead in protocol["lead_times"]:
            groups = {
                method: sorted(
                    (
                        r
                        for r in records
                        if r["scenario"] == scenario["name"]
                        and r["lead_time"] == lead
                        and r["method"] == method
                    ),
                    key=lambda r: r["replicate"],
                )
                for method in protocol["methods"]
            }
            for method, group in groups.items():
                if not group or len({r["replicate"] for r in group}) != len(group):
                    raise ValueError("Each method needs unique independent path records.")
                invalid = sum(r["mean_interval_score"] is None for r in group)
                aggregate.append(
                    dict(
                        scenario=scenario["name"],
                        lead_time=lead,
                        method=method,
                        n_runs=len(group),
                        coverage=_ci([r["coverage"] for r in group]),
                        mean_interval_score=_ci(
                            [] if invalid else [r["mean_interval_score"] for r in group]
                        ),
                        worst_local_coverage_error=_ci(
                            [r["worst_local_coverage_error"] for r in group]
                        ),
                        invalid_score_runs=invalid,
                    )
                )
            for baseline in ("horizon", "shortest"):
                pairs = list(zip(groups["mixture"], groups[baseline], strict=True))
                if any(
                    (a["replicate"], a["seed"]) != (b["replicate"], b["seed"]) for a, b in pairs
                ):
                    raise ValueError("Paired paths must share replicate and seed identity.")
                invalid = sum(
                    a["mean_interval_score"] is None or b["mean_interval_score"] is None
                    for a, b in pairs
                )
                contrasts.append(
                    dict(
                        scenario=scenario["name"],
                        lead_time=lead,
                        baseline=baseline,
                        invalid_score_pairs=invalid,
                        score_difference=_ci(
                            []
                            if invalid
                            else [
                                a["mean_interval_score"] - b["mean_interval_score"]
                                for a, b in pairs
                            ]
                        ),
                        coverage_difference=_ci([a["coverage"] - b["coverage"] for a, b in pairs]),
                    )
                )
    return aggregate, contrasts


def run(profile, output):
    from strategy_inference import __version__

    protocol = json.loads((ROOT / "experiments/scale-mixture-protocol.json").read_text())
    settings = protocol["profiles"][profile]
    locked = source_hashes()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new or empty output directory.")
    output.mkdir(parents=True, exist_ok=True)
    records, fingerprints, trajectories = [], [], []
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
                result = (
                    replay(actual, points, origins, scales, protocol)
                    if method == "mixture"
                    else baseline_replay(actual, points, origins, scales, protocol, method)
                )
                records.extend(
                    dict(scenario=scenario["name"], replicate=rep, seed=seed, method=method, **row)
                    for row in metrics(result, protocol, settings["initial_train_size"])
                )
                if method == "mixture" and rep == 0:
                    trajectories.append(
                        dict(
                            scenario=scenario["name"],
                            replicate=rep,
                            origins=origins.tolist(),
                            weights=result.weights.tolist(),
                            issued_scales=result.scales.tolist(),
                            max_pending=result.max_pending,
                            terminal=result.learner,
                        )
                    )
            print(f"{scenario['name']}: {rep + 1}/{settings['repetitions']}", flush=True)
    aggregate, contrasts = summarize(records, protocol)
    if locked != source_hashes():
        raise RuntimeError("Source or protocol changed during the study.")
    report = dict(
        schema_version=1,
        study=protocol["study"],
        profile=profile,
        package_version=__version__,
        created_utc=datetime.now(timezone.utc).isoformat(),
        protocol=protocol,
        profile_settings=settings,
        source_sha256=locked,
        python=sys.version,
        numpy=np.__version__,
        platform=platform.platform(),
        records=records,
        aggregate=aggregate,
        paired_contrasts=contrasts,
        representative_paths=trajectories,
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
