"""Evaluate fixed cross-lead blending using the public interval API."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

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


def source_hashes():
    paths = list((ROOT / "src/strategy_inference").glob("*.py")) + [
        Path(__file__),
        ROOT / "pyproject.toml",
        ROOT / "scripts/reproduce_scale_transfer.py",
        ROOT / "scripts/reproduce_multistep.py",
        ROOT / "experiments/blended-scale-protocol.json",
    ]
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }


def replay(actual, points, origins, scales, protocol, method):
    from strategy_inference import multistep_intervals

    return multistep_intervals(
        actual,
        points,
        origins=origins,
        lead_times=protocol["lead_times"],
        alpha=protocol["alpha"],
        step_size=protocol["step_size"] / np.sqrt(protocol["lead_times"]),
        decay=protocol["decay"],
        initial_quantile=protocol["initial_quantile"],
        scale=scales,
        scale_floor=protocol["scale_floor"],
        **protocol["method_parameters"][method],
    )


def summarize(records, protocol, repetitions):
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
                if [r["replicate"] for r in group] != list(range(repetitions)):
                    raise ValueError("Every declared independent replicate must be retained.")
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
                        empty_count=sum(r["empty_count"] for r in group),
                        unbounded_count=sum(r["unbounded_count"] for r in group),
                    )
                )
            for baseline in ("horizon", "shortest"):
                pairs = list(zip(groups["blend_50"], groups[baseline], strict=True))
                if any(a["seed"] != b["seed"] for a, b in pairs):
                    raise ValueError("Paired paths must share seed identity.")
                invalid = sum(
                    a["mean_interval_score"] is None or b["mean_interval_score"] is None
                    for a, b in pairs
                )
                contrasts.append(
                    dict(
                        scenario=scenario["name"],
                        lead_time=lead,
                        baseline=baseline,
                        candidate="blend_50",
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

    protocol = json.loads((ROOT / "experiments/blended-scale-protocol.json").read_text())
    settings, locked = protocol["profiles"][profile], source_hashes()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new or empty output directory.")
    output.mkdir(parents=True, exist_ok=True)
    records, fingerprints = [], []
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
                result = replay(actual, points, origins, scales, protocol, method)
                if result.pending or not result.evaluated.all():
                    raise RuntimeError("Every issued study forecast must mature.")
                records.extend(
                    dict(scenario=scenario["name"], replicate=rep, seed=seed, method=method, **row)
                    for row in metrics(result, protocol, settings["initial_train_size"])
                )
            print(f"{scenario['name']}: {rep + 1}/{settings['repetitions']}", flush=True)
    aggregate, contrasts = summarize(records, protocol, settings["repetitions"])
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
        input_fingerprints=fingerprints,
    )
    (output / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.profile, args.output)


if __name__ == "__main__":
    main()
