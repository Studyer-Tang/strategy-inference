"""Holdout completeness and independent path-pairing for public scale blending."""

import ast
import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import t as student_t

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


@pytest.mark.parametrize("change", ["drop", "duplicate", "seed", "extra", "sensitivity_seed"])
def test_bad_pairing_or_missing_paths_raise(change):
    rows = records()
    if change == "drop":
        rows.pop()
    elif change == "duplicate":
        rows[-1]["replicate"] = 0
    elif change == "seed":
        next(r for r in rows if r["method"] == "blend_50")["seed"] = 777
    elif change == "extra":
        rows.append({**rows[-1], "method": "undeclared"})
    else:
        next(r for r in rows if r["method"] == "blend_75")["seed"] = 777
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
    third = json.loads((ROOT / "results/research/interval-scale-mixture/pilot-03/results.json").read_text())["protocol"]

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
    for path, number in (("scale-transfer", 1), ("scale-mixture", 2), ("interval-scale-mixture", 3)):
        pilot = json.loads((ROOT / f"results/research/{path}/pilot-0{number}/results.json").read_text())["protocol"]
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


@pytest.fixture(scope="module")
def frozen_generator():
    commit = "7488a96283890a7884d0cb58346f973bfd4a245b"
    result = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{commit}:scripts/reproduce_scale_transfer.py"],
        check=True, capture_output=True, text=True,
    )
    names = {"generate", "forecasts", "initial_scales", "_ci"}
    nodes = [node for node in ast.parse(result.stdout).body
        if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == names
    namespace = {"np": np, "student_t": student_t}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "v0.8.0-generator", "exec"), namespace)
    return namespace


@pytest.mark.parametrize("scenario", PROTOCOL["scenarios"], ids=lambda s: s["name"])
def test_shared_generation_matches_frozen_inputs_bitwise(frozen_generator, scenario):
    settings = {"n_obs": 1000}
    inputs = study.generate(PROTOCOL, settings, scenario, 1928831)
    reference = frozen_generator["generate"](PROTOCOL, settings, scenario, 1928831)
    for actual, expected in zip(inputs, reference, strict=True):
        np.testing.assert_array_equal(actual, expected)
    origins, points = study.forecasts(inputs[0], PROTOCOL, scenario)
    old_origins, old_points = frozen_generator["forecasts"](inputs[0], PROTOCOL, scenario)
    np.testing.assert_array_equal(origins, old_origins)
    np.testing.assert_array_equal(points, old_points)
    np.testing.assert_array_equal(
        study.initial_scales(inputs[0], origins, points, PROTOCOL["lead_times"], 600),
        frozen_generator["initial_scales"](inputs[0], old_origins, old_points, PROTOCOL["lead_times"], 600),
    )


def test_indexed_aggregate_matches_frozen_study_statistics():
    saved = json.loads((ROOT / "results/research/blended-scales/full/results.json").read_text())
    aggregate, contrasts = study.summarize(saved["records"], saved["protocol"], 40)
    for actual, expected in zip(aggregate + contrasts, saved["aggregate"] + saved["paired_contrasts"], strict=True):
        assert actual.keys() == expected.keys()
        for key, value in actual.items():
            assert value == pytest.approx(expected[key], abs=1e-12) if isinstance(value, dict) else value == expected[key]
