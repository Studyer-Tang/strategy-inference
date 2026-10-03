"""Independent delayed-feedback references, causal replay and atomic failures."""

import builtins
import json
import subprocess
import sys
from dataclasses import FrozenInstanceError
from fractions import Fraction
from math import hypot, sqrt

import numpy as np
import pytest

from strategy_inference.conformal import AdaptiveConformal, adaptive_intervals
from strategy_inference.multistep import MultiStepConformal, multistep_intervals


def _snapshot(tracker):
    return json.dumps(tracker.to_dict(), allow_nan=False, sort_keys=True)


@pytest.mark.parametrize("strategy", ["pooled", "interlaced"])
def test_stream_matches_independent_rational_calendar(strategy):
    actual = [0, 4, -2, 0, 8, -1, 0, 12, 0, -8, 0, 2]
    leads = (1, 2, 4)
    predictions = [[(t + h) % 3 - 1 for h in leads] for t in range(len(actual))]
    alpha, scale, initial = Fraction(1, 4), Fraction(2), Fraction(1, 2)
    rates = [Fraction(1, 8), Fraction(1, 16), Fraction(1, 32)]
    tracker = MultiStepConformal(leads, alpha=float(alpha), step_size=list(map(float, rates)),
        scale=float(scale), decay=0, strategy=strategy)
    q, counts, queue = {}, {}, {}
    for t, label in enumerate(actual):
        expected = sorted(queue.pop(t, []), key=lambda row: row[1])
        updates = tracker.observe(t, label)
        assert len(updates) == len(expected)
        for update, (origin, h, point, issued, radius) in zip(updates, expected, strict=True):
            lane = origin % h if strategy == "interlaced" else 0
            key, j = (h, lane), leads.index(h)
            previous = q.get(key, initial)
            miss = issued < 0 or (issued < 1 and abs(Fraction(label) - point) > radius)
            q[key] = previous + rates[j] * (int(miss) - alpha)
            counts[key] = counts.get(key, 0) + 1
            assert (update.origin, update.target, update.lead_time) == (origin, t, h)
            assert update.issued_quantile == float(issued)
            assert update.previous_quantile == float(previous)
            assert update.quantile == float(q[key])
            assert update.miss == miss
            assert update.eta == update.step_size == float(rates[j])
            assert update.n_updates == counts[key]
            residual = abs(Fraction(label) - point)
            assert update.score == pytest.approx(float(residual / (scale + residual)))
        intervals = tracker.predict(predictions[t])
        for h, point, interval in zip(leads, predictions[t], intervals, strict=True):
            lane = t % h if strategy == "interlaced" else 0
            threshold = q.get((h, lane), initial)
            radius = scale * threshold / (1 - threshold) if 0 <= threshold < 1 else None
            assert interval.quantile == float(threshold)
            if threshold < 0:
                assert interval.kind == "empty"
            elif threshold >= 1:
                assert interval.kind == "unbounded"
            else:
                assert interval.kind == "finite"
                assert interval.lower == pytest.approx(float(point - radius))
                assert interval.upper == pytest.approx(float(point + radius))
            queue.setdefault(t + h, []).append((t, h, Fraction(point), threshold, radius))
        assert len(tracker.pending) <= sum(leads)
    assert tracker.n_updates == tuple(sum(value for (lead, _), value in counts.items() if lead == h)
        for h in leads)


@pytest.mark.parametrize("strategy", ["pooled", "interlaced"])
@pytest.mark.parametrize("rate", [0.1, [0.1]])
def test_one_step_exactly_matches_legacy_batch(strategy, rate):
    rng = np.random.default_rng(416)
    actual, points = rng.normal(size=81), rng.normal(size=80)
    kwargs = dict(alpha=0.2, decay=0.7, scale=2.5, initial_quantile=0.43)
    old = adaptive_intervals(actual[1:], points, step_size=0.1, **kwargs)
    new = multistep_intervals(actual, points[:, None], origins=np.arange(80), lead_times=[1],
        strategy=strategy, step_size=rate, **kwargs)
    for name in ("actual", "predicted", "lower", "upper", "quantiles", "misses", "empty", "unbounded", "step_sizes"):
        np.testing.assert_array_equal(getattr(new, name)[:, 0], getattr(old, name))
    assert new.coverage == (old.coverage,)
    assert new.pending == ()
    assert new.diagnostics["states"][0]["quantile"] == old.next_quantile


def test_interlaced_matches_independent_old_trackers_per_lane_and_differs_from_pooled():
    leads, rates = (2, 3), (0.25, 0.0625)
    labels = [0, 0, 8, -8, 0, 8, 0, 0, 8, -8, 0, 0, 0]
    delayed = MultiStepConformal(leads, step_size=rates, decay=0.2, strategy="interlaced")
    references = {(h, lane): AdaptiveConformal(step_size=rate, decay=0.2)
        for h, rate in zip(leads, rates, strict=True) for lane in range(h)}
    points = np.zeros((len(labels), len(leads)))
    interlaced = multistep_intervals(labels, points, origins=np.arange(len(labels)),
        lead_times=leads, step_size=rates, decay=0.2, strategy="interlaced")
    pooled = multistep_intervals(labels, points, origins=np.arange(len(labels)),
        lead_times=leads, step_size=rates, decay=0.2)
    for time, label in enumerate(labels):
        for update in delayed.observe(time, label):
            old = references[(update.lead_time, update.origin % update.lead_time)].update(label)
            assert update.quantile == old.quantile
            assert update.step_size == old.step_size
            assert update.miss == old.miss
        for interval in delayed.predict([0, 0]):
            old = references[(interval.lead_time, time % interval.lead_time)].predict(0)
            assert (interval.lower, interval.upper, interval.quantile, interval.kind) == (
                old.lower, old.upper, old.quantile, old.kind)
    assert np.any(interlaced.quantiles != pooled.quantiles)


def test_pooled_feedback_changes_current_quantile_not_the_stale_issued_quantile():
    tracker = MultiStepConformal([2], alpha=0.25, step_size=0.125, decay=0)
    tracker.observe(0, 0)
    first = tracker.predict([0])[0]
    tracker.observe(1, 0)
    second = tracker.predict([0])[0]
    assert first.quantile == second.quantile == 0.5
    first_update = tracker.observe(2, 4)[0]
    tracker.predict([0])
    second_update = tracker.observe(3, 0)[0]
    assert second_update.issued_quantile == 0.5
    assert second_update.previous_quantile == first_update.quantile == 0.59375
    assert second_update.quantile == 0.5625


def test_integer_time_protocol_and_no_forecast_times():
    tracker = MultiStepConformal([1, 3], start_time=np.int64(5))
    assert tracker.last_time is None and tracker.next_time == 5
    assert tracker.coverage == tracker.coverage_bound == (None, None)
    before = _snapshot(tracker)
    with pytest.raises(RuntimeError, match="Observe"):
        tracker.predict([0, 0])
    assert _snapshot(tracker) == before
    assert tracker.observe(5, 9) == ()
    issued = tracker.predict([9, 9])
    with pytest.raises(RuntimeError, match="already issued"):
        tracker.predict([8, 8])
    for time in (4, 5, 7, True, 6.0):
        before = _snapshot(tracker)
        with pytest.raises(ValueError):
            tracker.observe(time, 9)
        assert _snapshot(tracker) == before
    assert tracker.observe(6, 9)[0].origin == 5
    assert tracker.observe(7, 9) == ()  # Optional issuance can be skipped.
    assert tracker.observe(8, 9)[0].lead_time == 3
    assert tracker.pending == () and tracker.n_updates == (1, 1)
    assert tracker.last_time == 8 and tracker.next_time == 9
    with pytest.raises(FrozenInstanceError):
        issued[0].target = 99


def test_same_target_retains_distinct_origin_and_lead_associations():
    result = multistep_intervals([0, 2, 4, 6, 8], [[0, 1], [2, 3]], origins=[0, 1], lead_times=[1, 2])
    np.testing.assert_array_equal(result.target_indices, [[1, 2], [2, 3]])
    np.testing.assert_array_equal(result.actual, [[2, 4], [4, 6]])
    records = [row for row in result.records() if row["target"] == 2]
    assert [(row["origin"], row["lead_time"], row["prediction"]) for row in records] == [(0, 2, 1.0), (1, 1, 2.0)]


@pytest.mark.parametrize("strategy", ["pooled", "interlaced"])
@pytest.mark.parametrize("source", ["horizon", "shortest"])
def test_future_label_changes_cannot_change_any_issued_prefix(strategy, source):
    rng = np.random.default_rng(512)
    actual = rng.normal(size=40)
    changed = actual.copy()
    changed[21:] += 30
    leads, origins = [1, 3, 7], np.arange(0, 40, 2)
    points = rng.normal(size=(len(origins), len(leads)))
    kwargs = dict(origins=origins, lead_times=leads, strategy=strategy, scale_decay=0.7,
        scale_source=source, scale=[1, 2, 4], step_size=[0.1, 0.05, 0.025])
    first = multistep_intervals(actual, points, **kwargs)
    other = multistep_intervals(changed, points, **kwargs)
    for name in ("lower", "upper", "quantiles", "scales", "empty", "unbounded"):
        np.testing.assert_array_equal(getattr(first, name)[origins <= 20], getattr(other, name)[origins <= 20])
    assert np.any(first.scales[origins > 20] != other.scales[origins > 20])


def test_tail_pending_is_explicit_and_does_not_get_fabricated_feedback():
    result = multistep_intervals([0, 1, 2, 3], [[0, 0], [2, 2], [3, 3]],
        origins=[0, 2, 3], lead_times=[1, 4])
    np.testing.assert_array_equal(result.target_indices, [[1, 4], [3, 6], [4, 7]])
    np.testing.assert_array_equal(result.evaluated, [[True, False], [True, False], [False, False]])
    assert np.isnan(result.actual[~result.evaluated]).all()
    assert np.isnan(result.step_sizes[~result.evaluated]).all()
    assert [(row["n_issued"], row["n_evaluated"], row["n_pending"]) for row in result.summary()] == [(3, 2, 1), (3, 0, 3)]
    assert result.summary()[1]["coverage"] is None
    assert len(result.pending) == 4
    for row in result.records():
        if not row["evaluated"]:
            assert row["actual"] is row["miss"] is row["step_size"] is None
    assert json.loads(json.dumps(result.to_dict(), allow_nan=False)) == result.to_dict()


@pytest.mark.parametrize("bad", [True, np.bool_(True), np.nan, np.inf, -np.inf, "1", 1j, [1], np.array(1)])
def test_bad_labels_are_atomic_even_without_due_forecasts(bad):
    tracker = MultiStepConformal([1, 2])
    before = _snapshot(tracker)
    with pytest.raises(ValueError):
        tracker.observe(0, bad)
    assert _snapshot(tracker) == before
    tracker.observe(0, 0)
    tracker.predict([0, 0])
    before = _snapshot(tracker)
    with pytest.raises(ValueError):
        tracker.observe(1, bad)
    assert _snapshot(tracker) == before


@pytest.mark.parametrize("bad", [[], [0], [[0, 0]], [True, 0.0], [0, np.nan], [0, np.inf],
    [0, 1j], np.ma.array([0, 0], mask=[False, True]), np.array([True, False])])
def test_bad_predictions_are_atomic_across_all_leads(bad):
    tracker = MultiStepConformal([1, 2])
    tracker.observe(0, 0)
    before = _snapshot(tracker)
    with pytest.raises(ValueError):
        tracker.predict(bad)
    assert _snapshot(tracker) == before
    assert len(tracker.predict([0, 0])) == 2


def test_late_second_lead_radius_overflow_cannot_partially_issue():
    tracker = MultiStepConformal([1, 2], scale=[1, np.finfo(float).max], initial_quantile=0.75)
    tracker.observe(0, 0)
    before = _snapshot(tracker)
    with pytest.raises(ValueError, match="overflow"):
        tracker.predict([0, 0])
    assert _snapshot(tracker) == before
    assert len(tracker.predict([0, 0], scale=1)) == 2


@pytest.mark.parametrize("bad", [0, True, [1], [1, 0], [1, True], [1, np.inf],
    np.ma.array([1, 1], mask=[0, 1])])
def test_bad_issuance_scale_overrides_leave_all_state_unchanged(bad):
    tracker = MultiStepConformal([1, 3], scale_decay=0.5)
    tracker.observe(0, 0)
    before = _snapshot(tracker)
    with pytest.raises(ValueError):
        tracker.predict([0, 0], scale=bad)
    assert _snapshot(tracker) == before
    tracker.predict([0, 0])


def test_late_second_lead_residual_overflow_cannot_partially_observe():
    largest = float(np.finfo(float).max)
    tracker = MultiStepConformal([1, 2], initial_quantile=1, scale_decay=0.5)
    tracker.observe(0, 0)
    tracker.predict([0, -largest])
    tracker.observe(1, 0)
    tracker.predict([largest, 0])
    before = _snapshot(tracker)
    with pytest.raises(ValueError, match="Residual overflow"):
        tracker.observe(2, largest)
    assert _snapshot(tracker) == before
    assert [update.lead_time for update in tracker.observe(2, 0)] == [1, 2]


def test_delayed_pooled_quantile_overflow_is_atomic_and_can_be_retried():
    largest = float(np.finfo(float).max)
    tracker = MultiStepConformal([3], step_size=largest, decay=0)
    for t in range(3):
        tracker.observe(t, 0)
        tracker.predict([0])
    tracker.observe(3, largest)
    tracker.predict([0])
    before = _snapshot(tracker)
    with pytest.raises(ValueError, match="Quantile update overflow"):
        tracker.observe(4, largest)
    assert _snapshot(tracker) == before
    assert tracker.observe(4, 0)[0].quantile < largest


def test_learning_rate_underflow_in_one_lead_rejects_whole_due_group():
    tracker = MultiStepConformal([1, 2], step_size=[0.1, np.nextafter(0.0, 1.0)], decay=0.9)
    for time in range(4):
        tracker.observe(time, 0)
        tracker.predict([0, 0])
    before = _snapshot(tracker)
    with pytest.raises(ValueError, match="underflowed"):
        tracker.observe(4, 0)
    assert _snapshot(tracker) == before
    assert tracker.n_updates == (3, 2)


def test_horizon_rms_uses_mature_residuals_and_preserves_issued_scales():
    tracker = MultiStepConformal([1, 2], scale=[2, 4], scale_decay=0.25)
    tracker.observe(0, 0)
    first = tracker.predict([0, 0])
    first_update = tracker.observe(1, 3)[0]
    assert tracker.current_scales == (hypot(1, sqrt(0.75) * 3), 4)
    assert first_update.scale == first[0].scale == 2
    second = tracker.predict([1, 1])
    mature = tracker.observe(2, 5)
    assert mature[1].origin == 0 and mature[1].scale == first[1].scale == 4
    assert mature[1].score == pytest.approx(5 / 9)
    assert second[0].scale == hypot(1, sqrt(0.75) * 3)
    assert tracker.current_scales[1] == hypot(2, sqrt(0.75) * 5)


def test_shortest_source_updates_long_leads_before_their_first_feedback_and_override_is_local():
    tracker = MultiStepConformal([1, 3], scale=[1, 3], scale_decay=0, scale_source="shortest")
    tracker.observe(0, 0)
    first = tracker.predict([0, 0])
    tracker.observe(1, 4)
    assert tracker.n_updates == (1, 0) and tracker.current_scales == (4, 12)
    overridden = tracker.predict([1, 1], scale=[7, 8])
    assert tuple(interval.scale for interval in overridden) == (7, 8)
    assert tracker.current_scales == (4, 12)
    tracker.observe(2, 2)
    assert tracker.current_scales == (1, 3)
    tracker.predict([2, 2])
    updates = tracker.observe(3, 4)
    assert updates[1].scale == first[1].scale == 3
    assert updates[1].score == pytest.approx(4 / 7)
    assert tracker.current_scales == (2, 6)


def test_scale_floor_applies_to_updates_not_fixed_initial_or_override_values():
    fixed = MultiStepConformal([1, 2], scale=[1, 2], scale_floor=[3, 4])
    fixed.observe(0, 0)
    assert tuple(interval.scale for interval in fixed.predict([0, 0])) == (1, 2)
    fixed.observe(1, 0)
    assert fixed.current_scales == (1, 2)
    adaptive = MultiStepConformal([1, 2], scale=[1, 2], scale_floor=[3, 4], scale_decay=0)
    adaptive.observe(0, 0)
    adaptive.predict([0, 0])
    adaptive.observe(1, 0)
    assert adaptive.current_scales == (3, 2)
    assert tuple(interval.scale for interval in adaptive.predict([0, 0], scale=0.25)) == (0.25, 0.25)
    adaptive.observe(2, 0)
    assert adaptive.current_scales == (3, 4)


def test_shortest_rescaling_overflow_is_transactional_even_after_valid_quantile_updates():
    largest = float(np.finfo(float).max)
    tracker = MultiStepConformal([1, 2], scale=[1, largest / 2], scale_decay=0,
        scale_source="shortest", initial_quantile=0)
    tracker.observe(0, 0)
    tracker.predict([0, 0])
    before = _snapshot(tracker)
    with pytest.raises(ValueError, match="scale update overflow"):
        tracker.observe(1, 3)
    assert _snapshot(tracker) == before
    tracker.observe(1, 1)
    assert tracker.current_scales == (1, largest / 2)


def test_shortest_ratios_that_cannot_be_represented_are_rejected():
    with pytest.raises(ValueError, match="scale ratios"):
        MultiStepConformal([1, 2], scale=[np.nextafter(0.0, 1.0), 1],
            scale_decay=0.5, scale_source="shortest")


@pytest.mark.parametrize("side", ["lower", "upper"])
def test_returned_finite_boundaries_are_inclusive(side):
    tracker = MultiStepConformal([1], initial_quantile=0.8, scale=2)
    tracker.observe(0, 0)
    interval = tracker.predict([3])[0]
    assert not tracker.observe(1, getattr(interval, side))[0].miss
    outside = MultiStepConformal([1], initial_quantile=0.8, scale=2)
    outside.observe(0, 0)
    interval = outside.predict([3])[0]
    actual = np.nextafter(getattr(interval, side), -np.inf if side == "lower" else np.inf)
    assert outside.observe(1, actual)[0].miss


def test_feedback_uses_returned_rounded_interval_not_score_threshold():
    tracker = MultiStepConformal([1], initial_quantile=0.55)
    tracker.observe(0, 0)
    interval = tracker.predict([1e16])[0]
    update = tracker.observe(1, interval.upper)[0]
    assert update.score > update.issued_quantile and update.miss is False


@pytest.mark.parametrize("strategy", ["pooled", "interlaced"])
def test_scalar_and_broadcast_vector_learning_rates_have_identical_outputs(strategy):
    actual = np.sin(np.arange(50))
    kwargs = dict(origins=np.arange(50), lead_times=[1, 3, 8], strategy=strategy)
    scalar = multistep_intervals(actual, np.zeros((50, 3)), step_size=0.1, **kwargs)
    vector = multistep_intervals(actual, np.zeros((50, 3)), step_size=[0.1] * 3, **kwargs)
    assert scalar.to_dict() == vector.to_dict()


def test_each_lead_uses_its_own_learning_rate_and_bound():
    tracker = MultiStepConformal([1, 2], step_size=[1, 0.5], decay=0)
    for time in range(102):
        updates = tracker.observe(time, 0)
        assert all(update.step_size == (1 if update.lead_time == 1 else 0.5) for update in updates)
        tracker.predict([0, 0])
    assert tracker.step_size == (1.0, 0.5)
    assert tracker.coverage_bound == pytest.approx((2 / 101, 2 / 50))
    assert tracker.to_dict()["step_size"] == [1.0, 0.5]


@pytest.mark.parametrize("strategy", ["pooled", "interlaced"])
def test_coverage_bound_uses_per_lead_or_per_lane_feedback_clocks(strategy):
    tracker = MultiStepConformal([2, 3], step_size=[0.3, 0.2], decay=0.6, strategy=strategy)
    for t in range(200):
        tracker.observe(t, (t % 7) * 2)
        if t % 5 != 0:
            tracker.predict([0, 0])
    expected = []
    state = tracker.to_dict()
    for h, rate, n in zip(tracker.lead_times, tracker.step_size, tracker.n_updates, strict=True):
        if strategy == "pooled":
            bound = (1 + sum(rate * k ** -0.6 for k in range(1, min(h, n) + 1))) / (n * rate * n ** -0.6)
        else:
            bound = sum((1 + rate) / (rate * row["n_updates"] ** -0.6)
                for row in state["states"] if row["lead_time"] == h and row["n_updates"]) / n
        expected.append(min(1, bound))
    assert tracker.coverage_bound == pytest.approx(expected)
    assert all(abs((1 - coverage) - 0.1) <= bound + 1e-14
        for coverage, bound in zip(tracker.coverage, tracker.coverage_bound, strict=True))


def test_lazy_interlaced_lanes_support_large_physical_leads_without_huge_allocations():
    lead = 2**50
    tracker = MultiStepConformal([1, lead], strategy="interlaced")
    tracker.observe(0, 0)
    tracker.predict([0, 0])
    state = tracker.to_dict()
    assert state["lane_counts"] == [1, lead] and len(state["states"]) == 2
    before = _snapshot(tracker)
    unused = tracker.lane_state(lead, lead - 1)
    assert unused == {"quantile": 0.5, "n_updates": 0, "misses": 0}
    unused["quantile"] = 99
    assert _snapshot(tracker) == before
    for h, lane in [(2, 0), (1, 1), (lead, lead), (lead, True)]:
        with pytest.raises(ValueError):
            tracker.lane_state(h, lane)


@pytest.mark.parametrize("bad", [[], [0], [-1], [1, 1], [2, 1], [1.0], [True], [1, False],
    [np.bool_(True), 2], np.array([1, 2], dtype=float), np.ma.array([1, 2], mask=[0, 1]),
    [[1, 2]], [2**63], [np.nan], [1j]])
def test_leads_require_nonboolean_unmasked_strict_positive_integers(bad):
    with pytest.raises(ValueError):
        MultiStepConformal(bad)


@pytest.mark.parametrize("name,bad", [
    ("start_time", True), ("start_time", -1), ("start_time", 1.0), ("start_time", 2**63),
    ("alpha", 0), ("alpha", 1), ("alpha", True), ("alpha", np.nan),
    ("decay", -0.1), ("decay", 1), ("decay", False),
    ("initial_quantile", -0.1), ("initial_quantile", 1.1), ("initial_quantile", True),
    ("strategy", "other"), ("strategy", 1), ("scale_source", "other"),
    ("scale_decay", -0.1), ("scale_decay", 1), ("scale_decay", False), ("scale_decay", np.inf),
])
def test_invalid_scalar_parameters_are_rejected(name, bad):
    with pytest.raises(ValueError):
        MultiStepConformal([1, 2], **{name: bad})


@pytest.mark.parametrize("name", ["scale", "scale_floor", "step_size"])
@pytest.mark.parametrize("bad", [0, -1, True, np.nan, np.inf, [1], [1, 0], [1, True],
    [1, np.nan], [[1, 1]], np.ma.array([1, 1], mask=[False, True])])
def test_positive_scalar_or_h_vector_parameters_are_strict(name, bad):
    with pytest.raises(ValueError):
        MultiStepConformal([1, 2], **{name: bad})


def test_shortest_requires_an_adaptive_scale_rule():
    with pytest.raises(ValueError, match="requires scale_decay"):
        MultiStepConformal([1], scale_source="shortest")


def test_target_int64_boundary_is_checked_before_mutating_prediction_state():
    largest = int(np.iinfo(np.int64).max)
    with pytest.raises(ValueError, match="target"):
        MultiStepConformal([2], start_time=largest - 1)
    tracker = MultiStepConformal([1], start_time=largest - 1)
    tracker.observe(largest - 1, 0)
    assert tracker.predict([0])[0].target == largest
    tracker.observe(largest, 0)
    before = _snapshot(tracker)
    with pytest.raises(ValueError, match="target"):
        tracker.predict([0])
    assert _snapshot(tracker) == before


@pytest.mark.parametrize("field,bad", [
    ("actual", []), ("actual", [[0, 0]]), ("actual", [0, True]),
    ("actual", [0, np.nan]), ("actual", np.ma.array([0, 0], mask=[0, 1])),
    ("predicted", [0]), ("predicted", [[0, 1]]), ("predicted", [[True]]),
    ("origins", []), ("origins", [-1]), ("origins", [2]), ("origins", [0, 0]),
    ("origins", [1, 0]), ("origins", [0.0]), ("origins", [True]),
    ("origins", [[0]]), ("origins", np.ma.array([0], mask=[True])),
])
def test_batch_shapes_and_origins_do_not_silently_skip_bad_rows(field, bad):
    args = dict(actual=[0, 0], predicted=[[0]], origins=[0], lead_times=[1])
    args[field] = bad
    with pytest.raises(ValueError):
        multistep_intervals(**args)


def test_batch_target_overflow_is_rejected_before_numpy_integer_addition():
    with pytest.raises(ValueError, match="target"):
        multistep_intervals([0, 0], [[0]], origins=[1], lead_times=[np.iinfo(np.int64).max])


def test_batch_arrays_are_owned_readonly_and_detached_from_input_and_diagnostics():
    actual, predicted, origins = np.arange(5.0), np.zeros((2, 2)), np.array([0, 4])
    result = multistep_intervals(actual, predicted, origins=origins, lead_times=[1, 2])
    before = result.to_dict()
    actual[:] = 100
    predicted[:] = 100
    origins[:] = 3
    assert result.to_dict() == before
    for name in ("origins", "target_indices", "actual", "predicted", "lower", "upper", "quantiles", "scales",
                 "evaluated", "empty", "unbounded", "misses", "step_sizes"):
        array = getattr(result, name)
        assert not array.flags.writeable and array.flags.owndata
        with pytest.raises(ValueError):
            array.flat[0] = 1
    assert result.feedback_steps is result.step_sizes
    diagnostic = result.diagnostics
    diagnostic["states"][0]["quantile"] = 999
    assert result.to_dict() == before
    with pytest.raises(FrozenInstanceError):
        result.lead_times = (9,)


def test_summary_uses_proper_interval_score_and_only_evaluated_finite_widths():
    result = multistep_intervals([0, 0, 4], [[0], [0], [0]], origins=[0, 1, 2],
        lead_times=[1], alpha=0.25, step_size=0.125, decay=0, scale=2)
    summary = result.summary()[0]
    widths = result.upper[:2, 0] - result.lower[:2, 0]
    scores = widths + (2 / 0.25) * np.maximum(result.actual[:2, 0] - result.upper[:2, 0], 0)
    assert summary["n_evaluated"] == summary["finite_width_count"] == 2
    assert summary["n_pending"] == 1
    assert summary["mean_finite_width"] == pytest.approx(float(widths.mean()))
    assert summary["mean_interval_score"] == pytest.approx(float(scores.mean()))
    assert summary["interval_score_status"] == summary["finite_width_status"] == "finite"


def test_empty_and_unbounded_evaluations_are_retained_and_not_scored_as_finite():
    result = multistep_intervals([0, 0, 0, 0], np.zeros((4, 1)), origins=range(4), lead_times=[1],
        alpha=0.25, step_size=2, decay=0, initial_quantile=0)
    assert [row["kind"] for row in result.records()[:3]] == ["finite", "empty", "unbounded"]
    assert result.records()[1]["miss"] is True and result.records()[2]["miss"] is False
    summary = result.summary()[0]
    assert summary["coverage"] == pytest.approx(2 / 3)
    assert summary["empty_rate"] == summary["unbounded_rate"] == pytest.approx(1 / 3)
    assert summary["finite_width_count"] == 1 and summary["mean_finite_width"] == 0
    assert summary["mean_interval_score"] is None
    assert summary["interval_score_status"] == "empty_or_unbounded"
    for row in result.records()[1:3]:
        assert row["lower"] is row["upper"] is None
    json.dumps(result.to_dict(), allow_nan=False)


def test_summary_width_or_score_overflow_is_explicit_in_strict_json():
    largest = float(np.finfo(float).max)
    result = multistep_intervals([0, largest], [[0]], origins=[0], lead_times=[1],
        scale=largest * 0.6)
    summary = result.summary()[0]
    assert summary["mean_finite_width"] is summary["mean_interval_score"] is None
    assert summary["finite_width_status"] == summary["interval_score_status"] == "overflow"
    json.dumps(result.to_dict(), allow_nan=False)


def test_stable_rms_and_bounded_score_do_not_overflow_at_largest_finite_value():
    largest = float(np.finfo(float).max)
    tracker = MultiStepConformal([1], scale=largest, scale_decay=0.5, initial_quantile=1)
    tracker.observe(0, 0)
    tracker.predict([0])
    update = tracker.observe(1, largest)[0]
    assert update.score == 0.5 and not update.miss
    assert tracker.current_scales == (largest,)
    json.dumps(tracker.to_dict(), allow_nan=False)


def test_all_pending_unbounded_forecasts_have_no_evaluated_score_or_rate():
    result = multistep_intervals([4, 5, 6], [[6]], origins=[2], lead_times=[3], initial_quantile=1)
    summary = result.summary()[0]
    assert summary["n_issued"] == summary["n_pending"] == 1
    assert summary["n_evaluated"] == summary["finite_width_count"] == 0
    for field in ("coverage", "coverage_bound", "empty_rate", "unbounded_rate", "mean_interval_score"):
        assert summary[field] is None
    assert summary["interval_score_status"] == "no_evaluated_intervals"
    assert result.records()[0]["actual"] is result.records()[0]["miss"] is None
    json.dumps(result.to_dict(), allow_nan=False)


def test_core_import_and_replay_do_not_load_scipy_or_pandas():
    completed = subprocess.run([sys.executable, "-c", """
import sys
from strategy_inference import MultiStepConformal, multistep_intervals
MultiStepConformal([1, 3])
multistep_intervals([0, 0], [[0]], origins=[0], lead_times=[1])
assert 'scipy' not in sys.modules
assert 'pandas' not in sys.modules
"""], capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr


def test_optional_frame_import_has_actionable_error_without_core_dependency(monkeypatch):
    result = multistep_intervals([0, 0], [[0]], origins=[0], lead_times=[1])
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "pandas":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(ImportError, match="pandas.*extra"):
        result.to_frame()
