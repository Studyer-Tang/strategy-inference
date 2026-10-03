"""Holdout completeness and independent path-pairing for public scale blending."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "blended_scale_study", ROOT / "scripts/reproduce_blended_scales.py"
)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)
PROTOCOL = json.loads((ROOT / "experiments/blended-scale-protocol.json").read_text())


def small_protocol():
    return {**PROTOCOL, "scenarios": [PROTOCOL["scenarios"][0]], "lead_times": [1]}


def records():
    return [
        dict(
            scenario="constant",
            lead_time=1,
            method=method,
            replicate=rep,
            seed=100 + rep,
            coverage=0.9,
            worst_local_coverage_error=0.1,
            mean_interval_score=score,
            empty_count=0,
            unbounded_count=0,
        )
        for method in PROTOCOL["methods"]
        for rep, score in enumerate((1.0, 3.0))
    ]


def test_aggregate_keeps_declared_paths_and_propagates_invalid_scores():
    rows = records()
    bad = next(r for r in rows if r["method"] == "blend_50" and r["replicate"] == 1)
    bad["mean_interval_score"] = None
    bad["empty_count"] = 1
    aggregate, contrasts = study.summarize(rows, small_protocol(), 2)
    candidate = next(r for r in aggregate if r["method"] == "blend_50")
    assert candidate["n_runs"] == 2 and candidate["empty_count"] == 1
    assert candidate["invalid_score_runs"] == 1
    assert candidate["mean_interval_score"]["n"] == 0
    assert candidate["mean_interval_score"]["mean"] is None
    assert all(r["invalid_score_pairs"] == 1 for r in contrasts)
    assert all(r["score_difference"]["mean"] is None for r in contrasts)


@pytest.mark.parametrize("change", ["drop", "duplicate", "seed"])
def test_bad_pairing_or_missing_paths_raise(change):
    rows = records()
    if change == "drop":
        rows.pop()
    elif change == "duplicate":
        rows[-1]["replicate"] = 0
    else:
        next(r for r in rows if r["method"] == "blend_50")["seed"] = 777
    with pytest.raises(ValueError):
        study.summarize(rows, small_protocol(), 2)


def test_interval_score_ci_uses_paired_paths_and_handles_record_order():
    rows = records()
    for row in rows:
        if row["method"] == "blend_50":
            row["mean_interval_score"] = (1.5, 2.0)[row["replicate"]]
    _, contrasts = study.summarize(list(reversed(rows)), small_protocol(), 2)
    for row in contrasts:
        assert row["score_difference"]["n"] == 2
        assert row["score_difference"]["mean"] == -0.25
        assert row["score_difference"]["mcse"] == pytest.approx(0.75)


def test_full_seeds_equal_prespecified_untouched_holdout_and_avoid_all_pilots():
    third = json.loads((ROOT / "experiments/interval-scale-mixture-protocol.json").read_text())

    def seeds(protocol, profile):
        settings = protocol["profiles"][profile]
        return {
            protocol["seed"] + settings["seed_offset"] + s * 10000 + rep
            for s in range(len(protocol["scenarios"]))
            for rep in range(settings["repetitions"])
        }

    full = seeds(PROTOCOL, "full")
    assert full == seeds(third, "full")
    assert full.isdisjoint(seeds(PROTOCOL, "smoke"))
    for path in ("scale-transfer", "scale-mixture", "interval-scale-mixture"):
        pilot = json.loads((ROOT / f"experiments/{path}-protocol.json").read_text())
        assert full.isdisjoint(seeds(pilot, "pilot"))


def test_public_methods_have_identical_shortest_lead_and_complete_maturity():
    p = {**PROTOCOL, "lead_times": [1, 3, 7]}
    y = np.random.default_rng(109).normal(size=700).cumsum()
    origins, points = study.forecasts(y, p, p["scenarios"][0])
    scales = study.initial_scales(y, origins, points, p["lead_times"], 300)
    keep = origins >= 299
    origins, points = origins[keep], points[keep]
    results = {
        method: study.replay(y, points, origins, scales, p, method) for method in p["methods"]
    }
    for method in p["methods"][1:]:
        result = results[method]
        assert result.evaluated.all() and not result.pending
        for field in ("lower", "upper", "scales", "quantiles", "misses"):
            np.testing.assert_array_equal(
                getattr(result, field)[:, 0], getattr(results["horizon"], field)[:, 0]
            )
