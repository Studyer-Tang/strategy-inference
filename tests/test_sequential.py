"""Independent rational, subset and finite-tree checks for strong SMCSs."""

import itertools
import json
import math
import os
import subprocess
import sys
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

import strategy_inference.sequential as sequential
from strategy_inference.model_selection import backtest
from strategy_inference.sequential import (
    SequentialModelConfidenceSet,
    sequential_compare_forecasts,
)


def _fraction(value):
    return Fraction.from_float(float(value))


def _loss(actual, point, quantile):
    error = actual - point
    if quantile is None:
        return abs(error)
    return quantile * error if error >= 0 else (quantile - 1) * error


def _rational_differences(points, actual, quantile):
    points, actual = list(map(_fraction, points)), _fraction(actual)
    q = None if quantile is None else _fraction(quantile)
    weight = 1 if q is None else max(q, 1 - q)
    losses = [_loss(actual, point, q) for point in points]
    return [
        [
            (losses[i] - losses[j]) / (abs(points[i] - points[j]) * weight)
            if points[i] != points[j]
            else Fraction(0)
            for j in range(len(points))
        ]
        for i in range(len(points))
    ]


def _brute_adjust(values):
    """Literal closure over every nonempty subset, never the sorted algorithm."""
    m = len(values)
    return [
        min(
            sum(values[j] for j in range(m) if mask & (1 << j)) / mask.bit_count()
            for mask in range(1, 1 << m)
            if mask & (1 << i)
        )
        for i in range(m)
    ]


@pytest.mark.parametrize("m", [2, 4, 6])
@pytest.mark.parametrize("quantile", [None, 0.25, 0.5, 0.75])
def test_stream_evidence_and_closure_match_independent_fraction_subset_calculation(m, quantile):
    names = tuple(f"model_{i}" for i in range(m))
    options = {} if quantile is None else dict(loss="pinball", quantile=quantile)
    tracker = SequentialModelConfidenceSet(names, alpha=0.07, **options)
    pair = [[Fraction(1) for _ in names] for _ in names]
    retained = set(names)
    for t, actual in enumerate([0, 3, -2, 1, 6, -5, 2, 2, 0]):
        points = [i - 2 + (t % 3) / 4 for i in range(m)]
        if t == 3:
            points[-1] = points[0]  # Equal forecasts contribute a neutral factor.
        normalized = _rational_differences(points, actual, quantile)
        for i, j in itertools.product(range(m), repeat=2):
            pair[i][j] *= 1 + Fraction(1, 4) * normalized[i][j]
        row = [sum(pair[i][j] for j in range(m) if i != j) / (m - 1) for i in range(m)]
        adjusted = _brute_adjust(row)
        retained &= {
            name for name, value in zip(names, adjusted, strict=True) if float(value) < 1 / 0.07
        }
        tracker.predict(points)
        assert tracker.update(actual) == tuple(name for name in names if name in retained)
        np.testing.assert_allclose(
            tracker.log_evalues, [math.log(float(v)) for v in row], atol=3e-14, rtol=3e-14
        )
        np.testing.assert_allclose(
            tracker.log_adjusted_evalues,
            [math.log(float(v)) for v in adjusted],
            atol=3e-14,
            rtol=3e-14,
        )
        assert tracker.n_updates == t + 1 and tracker.pending is None


@pytest.mark.parametrize(
    "logs", [[0, 0, 0], [-5, 2, -1, 4], [1000, -1000, 0, 1000], [1e300, -1e300, 1e300]]
)
def test_log_domain_closure_matches_brute_subsets_without_exponentiating_large_logs(logs):
    # A max-shifted log mean for each subset is independent of Appendix H's scan.
    expected = []
    for i in range(len(logs)):
        means = []
        for mask in range(1, 1 << len(logs)):
            if mask & (1 << i):
                subset = [v for j, v in enumerate(logs) if mask & (1 << j)]
                largest = max(subset)
                means.append(
                    largest
                    + math.log(math.fsum(math.exp(v - largest) for v in subset))
                    - math.log(len(subset))
                )
        expected.append(min(means))
    np.testing.assert_allclose(
        sequential._closed_log_evalues(np.array(logs, dtype=float)),
        expected,
        atol=3e-13,
        rtol=3e-15,
    )


@pytest.mark.parametrize("quantile", [None, 0.25, 0.5, 0.75])
def test_stable_loss_differences_match_exact_rational_values_at_float_limits(quantile):
    largest, tiny = np.finfo(float).max, np.nextafter(0.0, 1.0)
    points = np.array([-largest, -tiny, 0, tiny, largest])
    for actual in (-largest, -1, 0, 1, largest):
        expected = np.array(_rational_differences(points, actual, quantile), dtype=float)
        observed = sequential._normalized_differences(points, actual, quantile)
        np.testing.assert_allclose(observed, expected, atol=3e-15, rtol=3e-15)
        assert np.isfinite(observed).all() and np.max(np.abs(observed)) <= 1
        np.testing.assert_array_equal(np.diag(observed), 0)


@pytest.mark.parametrize("quantile", [None, 0.25])
def test_huge_labels_and_subnormal_forecast_gaps_do_not_lose_informative_feedback(quantile):
    options = {} if quantile is None else dict(loss="pinball", quantile=quantile)
    tiny = np.nextafter(0.0, 1.0)
    for points, actual in [
        ([1, 2], 1e308),
        ([0, tiny], 0),
        ([-np.finfo(float).max, np.finfo(float).max], np.finfo(float).max),
    ]:
        tracker = SequentialModelConfidenceSet(["a", "b"], **options)
        tracker.predict(points)
        tracker.update(actual)
        ratios = _rational_differences(points, actual, quantile)
        expected = [math.log1p(0.25 * float(ratios[0][1])), math.log1p(0.25 * float(ratios[1][0]))]
        np.testing.assert_allclose(tracker.log_evalues, expected, atol=2e-15, rtol=2e-15)
        assert tracker.log_evalues[0] != 0


def test_enumerated_strong_null_bernoulli_tree_controls_anytime_family_error():
    # All three forecasts have conditional absolute risk 1/2 at every node.
    names, forecasts, n, alpha = ("zero", "half", "one"), [0, 0.5, 1], 8, 0.2
    excluded_paths = 0
    for path in itertools.product((0, 1), repeat=n):
        tracker = SequentialModelConfidenceSet(names, alpha=alpha, bet_fraction=0.5)
        ever_excluded = False
        for label in path:
            tracker.predict(forecasts)
            tracker.update(label)
            ever_excluded |= len(tracker.confidence_set) != len(names)
        excluded_paths += ever_excluded
    probability = Fraction(excluded_paths, 2**n)
    assert probability == Fraction(1, 32)  # Exhaustive finite tree, not a stochastic estimate.
    assert 0 < probability <= _fraction(alpha)


def test_persistent_inferiority_is_excluded_and_log_exports_do_not_overflow():
    tracker = SequentialModelConfidenceSet(["good", "bad"], bet_fraction=0.5)
    for _ in range(2000):
        tracker.predict([0, 1])
        tracker.update(0)
    assert tracker.confidence_set == ("good",)
    assert tracker.log_evalues[1] > math.log(np.finfo(float).max)
    record = tracker.to_dict()
    assert record["confidence_set"] == ["good"]
    assert all(math.isfinite(v) for v in record["log_evalues"] + record["log_adjusted_evalues"])
    json.dumps(record, allow_nan=False)
    before = tracker.to_dict()
    with pytest.raises(ValueError):
        tracker.predict([0])  # Still supply the original family after exclusion.
    assert tracker.to_dict() == before
    for _ in range(100):
        tracker.predict([0, 1])
        tracker.update(1)
    assert "bad" not in tracker.confidence_set  # Evidence can fall; intersection stays.


def test_prediction_and_diagnostic_copies_are_detached_and_readonly():
    tracker = SequentialModelConfidenceSet(["a", "b"])
    forecasts = np.array([0.0, 1.0])
    tracker.predict(forecasts)
    forecasts[:] = 99
    pending = tracker.pending
    assert not pending.flags.writeable and pending.base is None
    pending.flags.writeable = True
    pending[:] = -99
    np.testing.assert_array_equal(tracker.pending, [0, 1])
    tracker.update(0)
    for property_name in ("log_evalues", "log_adjusted_evalues"):
        observed = getattr(tracker, property_name)
        expected = observed.copy()
        assert not observed.flags.writeable and observed.base is None
        observed.flags.writeable = True
        observed[:] = 88
        np.testing.assert_array_equal(getattr(tracker, property_name), expected)
    record = tracker.to_dict()
    record["names"][0] = "changed"
    record["log_evalues"][0] = 88
    assert tracker.names == ("a", "b") and tracker.log_evalues[0] != 88


def test_all_equal_forecasts_are_exactly_neutral_and_still_validate_the_label():
    tracker = SequentialModelConfidenceSet(["a", "b", "c"], alpha=0.2, bet_fraction=0.5)
    for _ in range(8):
        tracker.predict([0, 1, 2])
        tracker.update(0)
    logs, adjusted, retained = (
        tracker.log_evalues,
        tracker.log_adjusted_evalues,
        tracker.confidence_set,
    )
    tracker.predict([np.finfo(float).max] * 3)
    pending_state = tracker.to_dict()
    with pytest.raises(ValueError):
        tracker.update(np.inf)
    assert tracker.to_dict() == pending_state
    tracker.update(-np.finfo(float).max)
    np.testing.assert_array_equal(tracker.log_evalues, logs)
    np.testing.assert_array_equal(tracker.log_adjusted_evalues, adjusted)
    assert tracker.confidence_set == retained and tracker.n_updates == 9
    assert tracker.pending is None


@pytest.mark.parametrize(
    "bad",
    [
        [0],
        [[0, 1]],
        [True, False],
        [0, np.nan],
        [0, np.inf],
        [0, 1j],
        ["0", "1"],
        np.ma.array([0, 1], mask=[False, True]),
    ],
)
def test_invalid_prediction_is_atomic(bad):
    tracker = SequentialModelConfidenceSet(["a", "b"])
    before = tracker.to_dict()
    with pytest.raises(ValueError):
        tracker.predict(bad)
    assert tracker.to_dict() == before


@pytest.mark.parametrize("bad", [True, np.bool_(False), np.nan, np.inf, -np.inf, 1j, "1"])
def test_invalid_label_preserves_pending_and_all_evidence(bad):
    tracker = SequentialModelConfidenceSet(["a", "b"])
    tracker.predict([0, 1])
    before = tracker.to_dict()
    with pytest.raises(ValueError):
        tracker.update(bad)
    assert tracker.to_dict() == before
    tracker.update(0)
    assert tracker.n_updates == 1


def test_protocol_and_late_computation_failures_do_not_commit(monkeypatch):
    tracker = SequentialModelConfidenceSet(["a", "b"])
    with pytest.raises(RuntimeError):
        tracker.update(0)
    tracker.predict([0, 1])
    before = tracker.to_dict()
    with pytest.raises(RuntimeError):
        tracker.predict([0, 1])
    assert tracker.to_dict() == before
    with monkeypatch.context() as patch:
        patch.setattr(
            sequential, "_closed_log_evalues", lambda values: np.full_like(values, np.nan)
        )
        with pytest.raises(ValueError):
            tracker.update(0)
    assert tracker.to_dict() == before
    tracker.update(0)
    with pytest.raises(RuntimeError):
        tracker.update(0)
    assert tracker.n_updates == 1


@pytest.mark.parametrize("names", [[], ["a"], ["a", "a"], ["a", " "], ["a", 2], "ab", b"ab", None])
def test_original_family_requires_at_least_two_distinct_nonempty_names(names):
    with pytest.raises(ValueError):
        SequentialModelConfidenceSet(names)


@pytest.mark.parametrize(
    "options",
    [
        dict(alpha=0),
        dict(alpha=1),
        dict(alpha=True),
        dict(loss="squared"),
        dict(loss="pinball"),
        dict(loss="pinball", quantile=0),
        dict(loss="pinball", quantile=1),
        dict(loss="pinball", quantile=True),
        dict(quantile=0.5),
        dict(bet_fraction=0),
        dict(bet_fraction=0.5001),
        dict(bet_fraction=True),
        dict(bet_fraction=np.nan),
    ],
)
def test_parameters_fix_the_test_and_loss_before_any_labels(options):
    with pytest.raises(ValueError):
        SequentialModelConfidenceSet(["a", "b"], **options)


def _panel(horizon=1, gap=0, step=1):
    return backtest(
        np.arange(60, dtype=float) % 3,
        {
            name: (lambda train, leads, value=value: np.full(len(leads), value))
            for name, value in [("zero", 0), ("half", 0.5), ("one", 1)]
        },
        initial_train_size=5,
        horizon=horizon,
        gap=gap,
        step=step,
    )


def test_nonoverlapping_batch_matches_stream_and_can_continue_after_replay():
    run = _panel(horizon=2, gap=3, step=5)
    selected = [0, 2, 5, 9]
    run = replace(
        run,
        splits=tuple(run.splits[i] for i in selected),
        forecasts=run.forecasts[selected],
        actuals=run.actuals[selected],
        target_indices=run.target_indices[selected],
    )
    options = dict(loss="pinball", quantile=0.25, bet_fraction=0.5)
    batch = sequential_compare_forecasts(run, lead_time=5, **options)
    stream = SequentialModelConfidenceSet(run.names, **options)
    for points, label in zip(run.forecasts[:, 1], run.actuals[:, 1], strict=True):
        stream.predict(points)
        stream.update(label)
    assert batch.to_dict() == stream.to_dict()
    batch.predict([0, 0.5, 1])
    batch.update(0)
    assert batch.n_updates == len(selected) + 1


def test_batch_requires_one_physical_lead_and_rejects_overlapping_feedback():
    with pytest.raises(ValueError, match="lead_time"):
        sequential_compare_forecasts(_panel(horizon=2))
    with pytest.raises(ValueError, match="nonoverlapping"):
        sequential_compare_forecasts(_panel(horizon=1, gap=3, step=3))
    with pytest.raises(ValueError, match="nonoverlapping"):
        sequential_compare_forecasts(_panel(horizon=2, step=1), lead_time=2)
    for lead in (0, True, 7):
        with pytest.raises(ValueError):
            sequential_compare_forecasts(_panel(), lead_time=lead)
    assert sequential_compare_forecasts(_panel()).n_updates > 0


def test_public_exports_preserve_old_names_and_new_core_stays_lazy():
    code = """
import sys
import strategy_inference as si
assert 'numpy' not in sys.modules and 'scipy' not in sys.modules
assert {'backtest','compare_forecasts','infer_mean','test_returns','multistep_intervals'} <= set(si.__all__)
assert 'sequential_compare_forecasts' in si.__all__
cls = si.SequentialModelConfidenceSet
assert cls is si.SequentialModelConfidenceSet
tracker = cls(['a','b'])
tracker.predict([0,1]); tracker.update(0)
assert 'scipy' not in sys.modules and 'pandas' not in sys.modules
"""
    subprocess.run([sys.executable, "-B", "-c", code], check=True, env=os.environ.copy())
