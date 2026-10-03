"""Independent causal and score-bookkeeping checks for the direct interval-score mixture pilot."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "scale_mixture_study", ROOT / "scripts/reproduce_interval_scale_mixture.py"
)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)
PROTOCOL = json.loads((ROOT / "experiments/interval-scale-mixture-protocol.json").read_text())


def prepared(actual, protocol):
    origins, points = study.forecasts(actual, protocol, protocol["scenarios"][0])
    scales = study.initial_scales(actual, origins, points, protocol["lead_times"], 300)
    keep = origins >= 299
    return origins[keep], points[keep], scales


def test_future_observations_leave_issued_mixture_intervals_unchanged():
    protocol = {**PROTOCOL, "lead_times": [1, 3, 7]}
    y = np.random.default_rng(92).normal(size=700).cumsum()
    changed = y.copy()
    changed[450:] += np.random.default_rng(93).normal(size=250) * 100
    o, p, s = prepared(y, protocol)
    oo, pp, ss = prepared(changed, protocol)
    np.testing.assert_array_equal(s, ss)
    a, b = study.replay(y, p, o, s, protocol), study.replay(changed, pp, oo, ss, protocol)
    keep = o < 450
    for name in ("lower", "upper", "scales", "quantiles", "weights"):
        np.testing.assert_array_equal(getattr(a, name)[keep], getattr(b, name)[keep])
    assert a.max_pending == sum(protocol["lead_times"])


def test_shortest_component_matches_public_ewma_controller():
    protocol = {**PROTOCOL, "lead_times": [1, 3, 7]}
    y = np.random.default_rng(94).standard_t(3.5, size=700)
    o, p, s = prepared(y, protocol)
    mixed = study.replay(y, p, o, s, protocol)
    own = study.baseline_replay(y, p, o, s, protocol, "horizon")
    for name in ("lower", "upper", "scales", "quantiles", "misses"):
        np.testing.assert_array_equal(getattr(mixed, name)[:, 0], getattr(own, name)[:, 0])
    np.testing.assert_array_equal(mixed.weights[:, 0], np.full(len(o), 0.5))
    assert mixed.terminal["pending"] == []


def test_unlearned_half_ablation_matches_two_public_residual_sources():
    protocol = {**PROTOCOL, "lead_times": [1, 3, 7]}
    y = np.random.default_rng(96).standard_t(3.5, size=700)
    o, p, s = prepared(y, protocol)
    half = study.replay(y, p, o, s, protocol, learn=False)
    own = study.baseline_replay(y, p, o, s, protocol, "horizon")
    short = study.baseline_replay(y, p, o, s, protocol, "shortest")
    np.testing.assert_allclose(half.scales, 0.5 * (own.scales + short.scales), rtol=3e-15)
    np.testing.assert_array_equal(half.weights, np.full(p.shape, 0.5))
    assert all(row["n_evaluated"] == len(o) for row in half.terminal["summary"])


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
        )
        for method in PROTOCOL["methods"]
        for rep, score in enumerate((1.0, 3.0))
    ]


def small_protocol():
    return {**PROTOCOL, "scenarios": [PROTOCOL["scenarios"][0]], "lead_times": [1]}


def test_one_invalid_score_invalidates_whole_paired_group():
    rows = records()
    rows[-1]["mean_interval_score"] = None
    aggregate, contrasts = study.summarize(rows, small_protocol())
    candidate = next(r for r in aggregate if r["method"] == "mixture")
    assert candidate["n_runs"] == 2
    assert candidate["mean_interval_score"]["mean"] is None
    assert candidate["mean_interval_score"]["n"] == 0
    assert candidate["invalid_score_runs"] == 1
    assert all(r["invalid_score_pairs"] == 1 for r in contrasts)
    assert all(r["score_difference"]["mean"] is None for r in contrasts)


def test_pairing_uses_seed_and_replicate_identity():
    rows = records()
    rows[-1]["seed"] = 777
    with pytest.raises(ValueError, match="identity"):
        study.summarize(rows, small_protocol())


def test_duplicate_paths_are_rejected():
    rows = records()
    rows[-1]["replicate"] = 0
    with pytest.raises(ValueError, match="unique"):
        study.summarize(rows, small_protocol())


def test_paired_differences_use_independent_paths_not_individual_intervals():
    rows = records()
    rows[-2]["mean_interval_score"], rows[-1]["mean_interval_score"] = 1.5, 2.0
    _, contrasts = study.summarize(list(reversed(rows)), small_protocol())
    for row in contrasts:
        ci = row["score_difference"]
        assert ci["n"] == 2
        assert ci["mean"] == -0.25
        assert ci["mcse"] == pytest.approx(0.75)


def test_full_and_pilot_seeds_are_disjoint_from_first_pilot():
    first = json.loads((ROOT / "experiments/scale-transfer-protocol.json").read_text())

    def seeds(protocol, profile):
        settings = protocol["profiles"][profile]
        return {
            protocol["seed"] + settings["seed_offset"] + i * 10000 + rep
            for i in range(len(protocol["scenarios"]))
            for rep in range(settings["repetitions"])
        }

    assert seeds(PROTOCOL, "pilot").isdisjoint(seeds(first, "pilot"))
    assert seeds(PROTOCOL, "pilot").isdisjoint(seeds(PROTOCOL, "full"))
