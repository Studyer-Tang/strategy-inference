"""Check the simulation's information budget and independent-path contrasts."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def study():
    spec = importlib.util.spec_from_file_location(
        "multistep_study", ROOT / "scripts/reproduce_multistep.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_initial_scales_use_only_training_targets(study):
    y = np.random.default_rng(1).normal(size=200)
    leads = np.array([1, 7, 12])
    expected = [np.linalg.norm(y[h:100] - 0.65**h * y[: 100 - h]) / np.sqrt(100 - h) for h in leads]
    before = study._scales(y, 100, leads, 0.65)
    np.testing.assert_allclose(before, expected)
    y[100:] = 1e50
    np.testing.assert_array_equal(study._scales(y, 100, leads, 0.65), before)


def test_monte_carlo_intervals_count_paths_not_observations(study):
    values = [1.0, 3.0, 5.0, 7.0]
    result = study._mean_ci(values)
    assert result["n"] == 4
    assert result["mean"] == 4.0
    assert result["mcse"] == pytest.approx(np.sqrt(20 / 3) / 2)
    assert result["lower"] < 4 < result["upper"]
    assert study._mean_ci([2.0])["mcse"] is None


def _row(method, replicate, score):
    return {
        "scenario": "test",
        "method": method,
        "lead_time": 1,
        "replicate": replicate,
        "coverage": 0.9,
        "mean_interval_score": score,
        "worst_local_coverage_error": 0.1,
        "empty_count": int(score is None),
        "unbounded_count": 0,
    }


def test_paired_contrast_matches_same_simulation_paths(study):
    protocol = {
        "scenarios": [{"name": "test"}],
        "methods": [{"name": "pooled_horizon"}, {"name": "pooled_shortest"}],
        "lead_times": [1],
    }
    rows = [
        _row("pooled_horizon", 0, 100),
        _row("pooled_horizon", 1, 1),
        _row("pooled_shortest", 1, 2),
        _row("pooled_shortest", 0, 99),
    ]
    _, contrast = study.aggregate(rows, protocol)
    assert contrast[0]["score_difference"]["mean"] == 0
    assert contrast[0]["score_difference"]["mcse"] == pytest.approx(1.0)


def test_invalid_interval_scores_invalidate_aggregate_without_dropping_paths(study):
    protocol = {
        "scenarios": [{"name": "test"}],
        "methods": [{"name": "pooled_horizon"}, {"name": "pooled_shortest"}],
        "lead_times": [1],
    }
    rows = [
        _row("pooled_horizon", 0, 10),
        _row("pooled_horizon", 1, 11),
        _row("pooled_shortest", 0, 8),
        _row("pooled_shortest", 1, None),
    ]
    groups, contrast = study.aggregate(rows, protocol)
    shared = next(r for r in groups if r["method"] == "pooled_shortest")
    assert shared["n_runs"] == 2 and shared["invalid_score_runs"] == 1
    assert shared["coverage"]["n"] == 2 and shared["mean_interval_score"]["mean"] is None
    assert (
        contrast[0]["score_difference"]["mean"] is None and contrast[0]["invalid_score_pairs"] == 1
    )


def test_scoring_keeps_whole_line_or_empty_failures_visible(study):
    result = SimpleNamespace(
        origins=np.arange(8),
        evaluated=np.ones((8, 1), dtype=bool),
        empty=np.zeros((8, 1), dtype=bool),
        unbounded=np.zeros((8, 1), dtype=bool),
        lower=np.full((8, 1), -1.0),
        upper=np.full((8, 1), 1.0),
        actual=np.zeros((8, 1)),
        misses=np.zeros((8, 1), dtype=bool),
    )
    protocol = {"lead_times": [1], "evaluation_warmup": 0, "local_window": 2, "alpha": 0.1}
    finite = study.metrics(result, protocol, 1)[0]
    assert finite["mean_interval_score"] == 2.0 and finite["coverage"] == 1.0
    result.unbounded[3, 0] = True
    result.lower[3, 0], result.upper[3, 0] = -np.inf, np.inf
    whole = study.metrics(result, protocol, 1)[0]
    assert whole["mean_interval_score"] is None and whole["unbounded_count"] == 1
    assert whole["mean_finite_width"] == 2.0
    result.empty[4, 0], result.misses[4, 0] = True, True
    result.lower[4, 0], result.upper[4, 0] = np.inf, -np.inf
    empty = study.metrics(result, protocol, 1)[0]
    assert empty["interval_score_status"] == "empty_intervals" and empty["empty_count"] == 1
    assert empty["coverage"] == 0.875
