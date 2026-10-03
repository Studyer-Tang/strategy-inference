"""Prequential ordering, independent recursion and online coverage boundaries."""

import json
from dataclasses import FrozenInstanceError
from fractions import Fraction

import numpy as np
import pytest

from strategy_inference.conformal import AdaptiveConformal, adaptive_intervals


def test_stream_and_batch_match_independent_rational_recursion():
    actual = [0, 4, -2, 0, 8, -1, 0, 12, 0, -8, 0, 2]
    predicted = [0, 1, 0, -1, 2, 0, 1, 3, 0, -2, 0, 0]
    alpha, step, scale, q = Fraction(1, 4), Fraction(1, 8), Fraction(2), Fraction(1, 2)
    tracker = AdaptiveConformal(
        alpha=float(alpha), step_size=float(step), decay=0, scale=float(scale)
    )
    result = adaptive_intervals(
        actual, predicted, alpha=float(alpha), step_size=float(step), decay=0, scale=float(scale)
    )
    for i, (label, point) in enumerate(zip(actual, predicted, strict=True)):
        radius = scale * q / (1 - q)
        residual = abs(Fraction(label) - point)
        miss = residual > radius
        interval = tracker.predict(point)
        assert interval.quantile == result.quantiles[i] == float(q)
        assert interval.lower == pytest.approx(float(point - radius))
        assert interval.upper == pytest.approx(float(point + radius))
        update = tracker.update(label)
        assert update.score == pytest.approx(float(residual / (scale + residual)))
        assert update.miss == result.misses[i] == miss
        assert update.previous_quantile == float(q)
        q += step * (int(miss) - alpha)
        assert update.quantile == float(q)
        assert update.step_size == result.step_sizes[i] == float(step)
    assert result.next_quantile == tracker.quantile == float(q)
    assert tracker.n_updates == result.n_obs == len(actual)
    assert tracker.coverage == result.coverage


def test_future_labels_cannot_change_already_issued_intervals():
    rng = np.random.default_rng(49)
    actual, prediction = rng.normal(size=200), rng.normal(size=200)
    changed = actual.copy()
    changed[73:] += 100
    original = adaptive_intervals(actual, prediction)
    alternative = adaptive_intervals(changed, prediction)
    # Interval 73 is issued before that label; its feedback can affect interval 74.
    for name in ("lower", "upper", "quantiles", "empty", "unbounded"):
        np.testing.assert_array_equal(getattr(original, name)[:74], getattr(alternative, name)[:74])
    assert original.misses[73] != alternative.misses[73]


def test_predict_update_protocol_rejects_repeated_and_missing_feedback():
    tracker = AdaptiveConformal()
    assert tracker.coverage is None and tracker.pending is None
    with pytest.raises(RuntimeError, match="predict before update"):
        tracker.update(1)
    interval = tracker.predict(0)
    with pytest.raises(RuntimeError, match="pending"):
        tracker.predict(2)
    assert tracker.pending is interval and tracker.n_updates == 0
    tracker.update(0)
    assert tracker.pending is None and tracker.n_updates == 1
    with pytest.raises(RuntimeError, match="predict before update"):
        tracker.update(0)
    assert tracker.predict(1).index == 2


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, True, "1", 1j, [1], np.array(1)])
def test_invalid_feedback_does_not_modify_pending_forecast_or_state(bad):
    tracker = AdaptiveConformal()
    interval = tracker.predict(3)
    with pytest.raises(ValueError, match="finite real"):
        tracker.update(bad)
    assert tracker.pending is interval
    assert tracker.quantile == 0.5 and tracker.n_updates == tracker.misses == 0
    assert tracker.update(3).miss is False


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, True, "1", 1j, [1], np.array(1)])
def test_invalid_prediction_does_not_modify_state(bad):
    tracker = AdaptiveConformal()
    with pytest.raises(ValueError, match="finite real"):
        tracker.predict(bad)
    assert tracker.pending is None and tracker.quantile == 0.5 and tracker.n_updates == 0


def test_residual_overflow_is_a_late_transactional_failure():
    largest = np.finfo(float).max
    tracker = AdaptiveConformal(initial_quantile=1)
    interval = tracker.predict(largest)
    with pytest.raises(ValueError, match="Residual overflow"):
        tracker.update(-largest)
    assert tracker.pending is interval and tracker.n_updates == tracker.misses == 0
    assert tracker.quantile == 1
    assert not tracker.update(largest).miss


@pytest.mark.parametrize(
    "scale,quantile,point",
    [(np.finfo(float).max, 0.75, 0), (np.finfo(float).max / 2, 0.5, np.finfo(float).max)],
)
def test_inverse_or_boundary_overflow_rejects_without_issuing_a_forecast(scale, quantile, point):
    tracker = AdaptiveConformal(scale=scale, initial_quantile=quantile)
    with pytest.raises(ValueError, match="boundary overflow"):
        tracker.predict(point)
    assert tracker.pending is None and tracker.n_updates == 0 and tracker.quantile == quantile


def test_score_denominator_overflow_is_avoided_without_losing_feedback():
    largest = np.finfo(float).max
    tracker = AdaptiveConformal(scale=largest, initial_quantile=1)
    tracker.predict(0)
    update = tracker.update(largest)
    assert update.score == 0.5 and not update.miss
    assert update.quantile == 0.99


def test_underflowed_learning_rate_is_a_late_transactional_failure():
    tracker = AdaptiveConformal(step_size=np.nextafter(0.0, 1.0), decay=0.9)
    for _ in range(2):
        tracker.predict(0)
        tracker.update(0)
    interval = tracker.predict(0)
    before = tracker.quantile, tracker.n_updates, tracker.misses
    with pytest.raises(ValueError, match="underflowed"):
        tracker.update(0)
    assert tracker.pending is interval and before == (
        tracker.quantile,
        tracker.n_updates,
        tracker.misses,
    )


def test_quantiles_are_not_clipped_and_empty_and_unbounded_sets_are_explicit():
    tracker = AdaptiveConformal(alpha=0.25, step_size=2, decay=0, initial_quantile=0)
    finite = tracker.predict(0)
    assert finite.kind == "finite" and finite.lower == finite.upper == 0
    assert not tracker.update(0).miss and tracker.quantile == -0.5
    empty = tracker.predict(0)
    assert empty.kind == "empty" and empty.lower == np.inf and empty.upper == -np.inf
    assert tracker.update(0).miss and tracker.quantile == 1
    full = tracker.predict(0)
    assert full.kind == "unbounded" and full.lower == -np.inf and full.upper == np.inf
    assert not tracker.update(1e100).miss
    for interval in (finite, empty, full):
        record = interval.to_dict()
        assert json.loads(json.dumps(record, allow_nan=False)) == record
    assert empty.to_dict()["lower"] is None and full.to_dict()["upper"] is None


@pytest.mark.parametrize("boundary", ["lower", "upper"])
def test_finite_boundaries_are_included_and_adjacent_outside_float_is_a_miss(boundary):
    tracker = AdaptiveConformal(initial_quantile=0.8, scale=2)
    interval = tracker.predict(3)
    label = getattr(interval, boundary)
    update = tracker.update(label)
    assert update.miss is False
    assert json.loads(json.dumps(update.to_dict(), allow_nan=False)) == update.to_dict()
    other = AdaptiveConformal(initial_quantile=0.8, scale=2)
    other.predict(3)
    outside = np.nextafter(label, -np.inf if boundary == "lower" else np.inf)
    assert other.update(outside).miss is True


def test_rounded_boundaries_define_feedback_even_when_diagnostic_score_differs():
    tracker = AdaptiveConformal(initial_quantile=0.55)
    interval = tracker.predict(1e16)
    assert interval.upper == 1e16 + 2
    update = tracker.update(interval.upper)
    assert update.score > interval.quantile
    assert update.miss is False  # The actual returned closed interval contains the label.


@pytest.mark.parametrize("scale", [0.125, 1, 8])
def test_bounded_score_and_inverse_event_agree_away_from_rounding_ties(scale):
    rng = np.random.default_rng(31)
    for q in (0, 0.1, 0.5, 0.9, 1):
        for residual in rng.uniform(-100, 100, size=20):
            tracker = AdaptiveConformal(initial_quantile=q, scale=scale)
            tracker.predict(0)
            update = tracker.update(residual)
            score = abs(residual) / (scale + abs(residual))
            assert update.score == pytest.approx(score)
            assert update.miss == (score > q)


@pytest.mark.parametrize("decay", [0, 0.6, 0.9])
def test_retrospective_bound_for_a_sequence_that_requires_empty_sets(decay):
    # Clipping negative q to zero would always cover here and fail the bound.
    n, alpha, step = 4096, 0.25, 0.4
    result = adaptive_intervals(np.zeros(n), np.zeros(n), alpha=alpha, step_size=step, decay=decay)
    count = np.arange(1, n + 1)
    average_error = np.cumsum(result.misses) / count
    bound = (1 + step) / (count * result.step_sizes)
    assert np.all(np.abs(average_error - alpha) <= bound + 1e-12)
    assert result.empty.any() and abs(result.coverage - (1 - alpha)) < 0.01


def test_constant_step_coverage_balance_matches_independent_telescoping_identity():
    alpha, step = Fraction(1, 4), Fraction(1, 8)
    actual = np.tile([0, 10, 1e100, 0, -1e100, 1], 50)
    result = adaptive_intervals(
        actual, np.zeros_like(actual), alpha=float(alpha), step_size=float(step), decay=0
    )
    total_error = Fraction(int(result.misses.sum())) - len(actual) * alpha
    assert total_error == (Fraction(result.next_quantile) - Fraction(1, 2)) / step


def test_power_of_two_unit_change_preserves_quantile_path_and_decisions():
    rng = np.random.default_rng(78)
    actual, predicted = rng.normal(size=100), rng.normal(size=100)
    original = adaptive_intervals(actual, predicted, scale=2)
    changed = adaptive_intervals(actual * 16, predicted * 16, scale=32)
    for name in ("quantiles", "misses", "empty", "unbounded", "step_sizes"):
        np.testing.assert_array_equal(getattr(original, name), getattr(changed, name))
    np.testing.assert_array_equal(original.lower * 16, changed.lower)
    np.testing.assert_array_equal(original.upper * 16, changed.upper)


def test_batch_json_is_strict_arrays_are_readonly_and_input_mutation_is_detached():
    actual, predicted = np.zeros(4), np.zeros(4)
    result = adaptive_intervals(
        actual, predicted, alpha=0.25, step_size=2, decay=0, initial_quantile=0
    )
    record = result.to_dict()
    assert json.loads(json.dumps(record, allow_nan=False)) == record
    assert [row["kind"] for row in record["intervals"]][:3] == ["finite", "empty", "unbounded"]
    actual[:] = predicted[:] = 100
    assert np.all(result.actual == 0) and np.all(result.predicted == 0)
    for name in (
        "actual",
        "predicted",
        "lower",
        "upper",
        "empty",
        "unbounded",
        "misses",
        "quantiles",
        "step_sizes",
    ):
        with pytest.raises(ValueError):
            getattr(result, name)[0] = 0
    record["intervals"][0]["quantile"] = 999
    assert result.quantiles[0] == 0
    with pytest.raises(FrozenInstanceError):
        result.next_quantile = 0


@pytest.mark.parametrize(
    "name,bad",
    [
        ("alpha", 0),
        ("alpha", 1),
        ("step_size", 0),
        ("step_size", -1),
        ("scale", 0),
        ("scale", -1),
        ("decay", -0.1),
        ("decay", 1),
        ("initial_quantile", -0.1),
        ("initial_quantile", 1.1),
        ("alpha", np.nan),
        ("scale", np.inf),
        ("step_size", True),
        ("decay", "0.6"),
        ("initial_quantile", 1j),
    ],
)
def test_invalid_parameters_are_rejected(name, bad):
    with pytest.raises(ValueError, match=name if name != "step_size" else "step_size"):
        AdaptiveConformal(**{name: bad})


@pytest.mark.parametrize(
    "actual,predicted",
    [
        ([], []),
        ([[1, 2]], [1, 2]),
        ([1], [1, 2]),
        ([1, np.nan], [1, 2]),
        ([1, 2], [1, np.inf]),
        ([True], [0]),
        (["1"], [0]),
        ([1j], [0]),
        (None, [0]),
    ],
)
def test_bad_batch_inputs_are_rejected_without_silent_row_drops(actual, predicted):
    with pytest.raises(ValueError):
        adaptive_intervals(actual, predicted)


@pytest.mark.parametrize("name", ["actual", "predicted"])
def test_masked_observations_are_not_silently_treated_as_available(name):
    series = {"actual": np.array([0.0, 1.0]), "predicted": np.array([0.0, 0.0])}
    series[name] = np.ma.array(series[name], mask=[False, True])
    with pytest.raises(ValueError, match="masked observations"):
        adaptive_intervals(**series)


def test_all_false_mask_does_not_remove_available_observations():
    actual = np.ma.array([0.0, 1.0], mask=False)
    predicted = np.ma.array([0.0, 0.0], mask=False)
    result = adaptive_intervals(actual, predicted)
    assert result.n_obs == 2
    np.testing.assert_array_equal(result.actual, actual.data)


def test_parameters_and_issued_intervals_are_immutable():
    tracker = AdaptiveConformal()
    with pytest.raises(AttributeError):
        tracker.scale = 2
    interval = tracker.predict(0)
    with pytest.raises(FrozenInstanceError):
        interval.quantile = 0
