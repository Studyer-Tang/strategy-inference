"""Fixed scale fusion: independent energies, old-mode endpoints and transactions."""

import json
import math
from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np
import pytest

from strategy_inference.multistep import MultiStepConformal, _blend_scale, multistep_intervals


@pytest.mark.parametrize("strategy", ["pooled", "interlaced"])
@pytest.mark.parametrize("leads", [[1, 3, 6], [3, 4, 9]])
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("weight,source", [(0, "horizon"), (1, "shortest")])
def test_endpoint_stream_is_bitwise_equal_to_old_mode_on_common_finite_domain(
    strategy, leads, sparse, weight, source
):
    initial = [0.1012345678901234, np.nextafter(2.3, np.inf), np.nextafter(3.7, 0)]
    options = dict(
        strategy=strategy,
        scale=initial,
        scale_decay=0.73,
        initial_quantile=0.4,
        step_size=[0.015, 0.009, 0.005],
        decay=0.3,
        scale_floor=[0.1, 0.3, 0.4],
    )
    old = MultiStepConformal(leads, scale_source=source, **options)
    blend = MultiStepConformal(leads, scale_source="blended", scale_share_weight=weight, **options)
    assert blend.current_scales == tuple(initial) == old.current_scales
    for time in range(90):
        actual = math.sin(time * 0.7) + (time % 3) * 0.125
        assert blend.observe(time, actual) == old.observe(time, actual)
        assert blend.current_scales == old.current_scales
        if time < 65 and (not sparse or time % 7 in (0, 3)):
            points = [actual * 0.5**lead for lead in leads]
            override = [0.91, 2.01, 4.3] if time % 11 == 0 else None
            assert blend.predict(points, scale=override) == old.predict(points, scale=override)
        assert blend.pending == old.pending
        assert blend.n_updates == old.n_updates
        assert blend.summary() == old.summary()
        assert blend.to_dict()["states"] == old.to_dict()["states"]
    json.dumps(blend.to_dict(), allow_nan=False)


@pytest.mark.parametrize("weights", [[0, 0.25, 1], [0.5, 0.5, 0.5]])
def test_sources_and_blend_follow_independent_fraction_energy_recursion(weights):
    leads, initial, floors = [2, 5, 9], [2, 4, 8], [0.5, 1, 2]
    beta = Fraction(1, 4)
    tracker = MultiStepConformal(
        leads,
        scale=initial,
        scale_floor=floors,
        scale_decay=float(beta),
        scale_source="blended",
        scale_share_weight=weights,
        initial_quantile=0.1,
    )
    own = [Fraction(v * v) for v in initial]
    source = Fraction(initial[0] ** 2)
    shared = own.copy()
    due = {}
    for time in range(40):
        actual = Fraction((time % 7) - 3, 2)
        residuals = {i: abs(actual - point) for i, point in due.pop(time, [])}
        for i, residual in residuals.items():
            own[i] = max(Fraction(floors[i]) ** 2, beta * own[i] + (1 - beta) * residual**2)
        if 0 in residuals:
            source = max(Fraction(floors[0]) ** 2, beta * source + (1 - beta) * residuals[0] ** 2)
            shared = [
                max(Fraction(floor) ** 2, source * Fraction(v, initial[0]) ** 2)
                for floor, v in zip(floors, initial, strict=True)
            ]
        tracker.observe(time, float(actual))
        state = tracker.to_dict()
        with localcontext() as ctx:
            ctx.prec = 100

            def decimal_energy(value):
                return (Decimal(value.numerator) / Decimal(value.denominator)).sqrt()

            expected = [
                float(
                    (1 - Decimal.from_float(float(w))) * decimal_energy(a)
                    + Decimal.from_float(float(w)) * decimal_energy(b)
                )
                for a, b, w in zip(own, shared, weights, strict=True)
            ]
        np.testing.assert_allclose(state["own_scales"], [math.sqrt(v) for v in own], rtol=3e-15)
        np.testing.assert_allclose(
            state["shared_scales"], [math.sqrt(v) for v in shared], rtol=3e-15
        )
        np.testing.assert_allclose(tracker.current_scales, expected, rtol=3e-15)
        if time in (0, 2, 3, 7, 10, 11, 16):
            points = [Fraction(5, 4), Fraction(-3, 4), Fraction(2)]
            tracker.predict(list(map(float, points)))
            for i, (lead, point) in enumerate(zip(leads, points, strict=True)):
                due.setdefault(time + lead, []).append((i, point))


def test_initial_scales_bypass_floors_and_ratio_multiplication_rounding():
    initial = [0.1, 3.3]
    assert initial[0] * (initial[1] / initial[0]) != initial[1]
    tracker = MultiStepConformal(
        [2, 7], scale=initial, scale_floor=[9, 10], scale_decay=0.5, scale_source="blended"
    )
    state = tracker.to_dict()
    assert tracker.current_scales == tuple(initial)
    assert state["own_scales"] == state["shared_scales"] == initial
    assert tracker.current_scale_weights == (0.5, 0.5)
    tracker.observe(0, 0)
    assert tracker.current_scales == tuple(initial)


def test_scalar_default_vector_weights_are_equivalent_and_detached():
    kwargs = dict(scale_source="blended", scale_decay=0.97)
    supplied = np.array([0.5, 0.5])
    trackers = [
        MultiStepConformal([2, 5], **kwargs),
        MultiStepConformal([2, 5], scale_share_weight=0.5, **kwargs),
        MultiStepConformal([2, 5], scale_share_weight=supplied, **kwargs),
    ]
    supplied[:] = 1
    for tracker in trackers:
        assert tracker.current_scale_weights == (0.5, 0.5)
        with pytest.raises(AttributeError):
            tracker.current_scale_weights = (0, 0)
        state = tracker.to_dict()
        state["scale_share_weight"][0] = 9
        state["own_scales"][0] = 9
        state["shared_scales"][0] = 9
        assert tracker.current_scale_weights == (0.5, 0.5)
        assert tracker.current_scales == (1, 1)
        assert tracker.to_dict()["own_scales"] == tracker.to_dict()["shared_scales"] == [1, 1]
    for time in range(25):
        updates = [tracker.observe(time, math.cos(time)) for tracker in trackers]
        assert updates[0] == updates[1] == updates[2]
        if time % 3 == 0:
            intervals = [tracker.predict([0, 0]) for tracker in trackers]
            assert intervals[0] == intervals[1] == intervals[2]


def test_sparse_long_feedback_changes_only_own_and_short_feedback_changes_all_shared():
    tracker = MultiStepConformal(
        [2, 5],
        scale=[1, 3],
        scale_source="blended",
        scale_decay=0,
        scale_share_weight=0.5,
        initial_quantile=0,
    )
    tracker.observe(0, 0)
    tracker.predict([0, 0])
    tracker.observe(1, 0)
    tracker.observe(2, 2)
    assert tracker.to_dict()["own_scales"] == [2, 3]
    assert tracker.to_dict()["shared_scales"] == [2, 6]
    assert tracker.current_scales == (2, 4.5)
    tracker.observe(3, 0)
    tracker.observe(4, 0)
    tracker.observe(5, 8)
    assert tracker.to_dict()["own_scales"] == [2, 8]
    assert tracker.to_dict()["shared_scales"] == [2, 6]
    assert tracker.current_scales == (2, 7)


@pytest.mark.parametrize("weights", [0, 0.17, 0.5, 1, [1, 0, 0.37]])
def test_smallest_physical_lead_keeps_same_source_bitwise(weights):
    tracker = MultiStepConformal(
        [3, 5, 8],
        scale=[1.234, 2, 3],
        scale_source="blended",
        scale_decay=0.97,
        scale_share_weight=weights,
        initial_quantile=0.3,
    )
    own = MultiStepConformal([3, 5, 8], scale=[1.234, 2, 3], scale_decay=0.97, initial_quantile=0.3)
    for time in range(30):
        tracker.observe(time, math.cos(time))
        own.observe(time, math.cos(time))
        state = tracker.to_dict()
        assert state["own_scales"][0] == state["shared_scales"][0] == tracker.current_scales[0]
        assert tracker.current_scales[0] == own.current_scales[0]
        if time % 4 == 0:
            tracker.predict([0, 0, 0])
            own.predict([0, 0, 0])


def test_scale_override_freezes_issuance_and_never_changes_blended_sources():
    tracker = MultiStepConformal(
        [2, 5],
        scale=[1, 3],
        scale_decay=0.5,
        scale_source="blended",
        scale_share_weight=[0.2, 0.8],
    )
    tracker.observe(0, 0)
    before = tracker.to_dict()
    intervals = tracker.predict([0, 0], scale=[100, 200])
    assert tuple(i.scale for i in intervals) == (100, 200)
    assert tracker.current_scales == (1, 3)
    for key in ("own_scales", "shared_scales", "scale_share_weight", "initial_scale"):
        assert tracker.to_dict()[key] == before[key]
    tracker.observe(1, 0)
    feedback = tracker.observe(2, 4)
    assert feedback[0].scale == 100
    assert tracker.to_dict()["shared_scales"][1] == tracker.to_dict()["shared_scales"][0] * 3
    assert tracker.current_scale_weights == (0.2, 0.8)


def test_batch_and_stream_match_and_preserve_pending_tail_and_detached_diagnostics():
    actual = np.sin(np.arange(33) * 0.3)
    origins, leads = np.array([0, 2, 3, 8, 16, 29, 32]), [2, 7]
    points = np.column_stack([actual[origins] * 0.5, actual[origins] * 0.1])
    options = dict(
        scale=[1, 4], scale_source="blended", scale_decay=0.97, scale_share_weight=[0.1, 0.9]
    )
    result = multistep_intervals(actual, points, origins=origins, lead_times=leads, **options)
    tracker = MultiStepConformal(leads, **options)
    issued = []
    row_for = dict(zip(origins, range(len(origins)), strict=True))
    for time, value in enumerate(actual):
        tracker.observe(time, value)
        if time in row_for:
            issued.extend(tracker.predict(points[row_for[time]]))
    assert result.diagnostics == tracker.to_dict()
    assert result.pending == tracker.pending and len(result.pending) == 3
    for row, interval in zip(result.records(), issued, strict=True):
        assert row["scale"] == interval.scale and row["quantile"] == interval.quantile
    assert not result.scales.flags.writeable
    state = result.diagnostics
    state["own_scales"][0] = 900
    assert result.diagnostics["own_scales"][0] != 900
    json.dumps(result.to_dict(), allow_nan=False)


def test_unit_rescaling_preserves_weights_misses_and_quantiles():
    labels = np.sin(np.arange(100) * 0.2)
    origins = np.arange(70)
    points = np.zeros((70, 2))
    kwargs = dict(
        origins=origins,
        lead_times=[2, 7],
        scale_source="blended",
        scale_decay=0.8,
        scale_share_weight=[0.25, 0.75],
        scale=[1, 4],
        scale_floor=[0.01, 0.04],
    )
    result = multistep_intervals(labels, points, **kwargs)
    scaled = multistep_intervals(
        labels * 16, points * 16, **{**kwargs, "scale": [16, 64], "scale_floor": [0.16, 0.64]}
    )
    np.testing.assert_array_equal(scaled.scales, 16 * result.scales)
    np.testing.assert_array_equal(scaled.quantiles, result.quantiles)
    np.testing.assert_array_equal(scaled.misses, result.misses)


def test_future_labels_cannot_change_previously_issued_blended_intervals():
    actual = np.sin(np.arange(50) * 0.2)
    changed = actual.copy()
    changed[25:] += 100
    options = dict(
        origins=np.arange(40),
        lead_times=[2, 7],
        scale_source="blended",
        scale_decay=0.8,
        scale_share_weight=[0.25, 0.75],
        scale=[1, 4],
    )
    points = np.zeros((40, 2))
    first = multistep_intervals(actual, points, **options)
    second = multistep_intervals(changed, points, **options)
    for name in ("scales", "quantiles", "lower", "upper", "empty", "unbounded"):
        np.testing.assert_array_equal(getattr(first, name)[:25], getattr(second, name)[:25])


@pytest.mark.parametrize("weight", [0, 0.5, 1, np.nextafter(1.0, 0.0)])
@pytest.mark.parametrize("reverse", [False, True])
def test_stable_blend_accepts_finite_maximum_and_subnormal_candidates(weight, reverse):
    candidates = (np.finfo(float).max, np.nextafter(0.0, 1.0))
    own, shared = candidates[::-1] if reverse else candidates
    result = _blend_scale(float(own), float(shared), float(weight))
    assert math.isfinite(result) and min(candidates) <= result <= max(candidates)
    if weight in (0, 1):
        assert result == (own if weight == 0 else shared)


@pytest.mark.parametrize("scale", [np.finfo(float).max, np.nextafter(0.0, 1.0)])
def test_extreme_equal_sources_work_through_real_feedback(scale):
    tracker = MultiStepConformal(
        [2, 5],
        scale=scale,
        scale_floor=scale,
        scale_source="blended",
        scale_decay=0,
        initial_quantile=0,
    )
    tracker.observe(0, 0)
    tracker.predict([0, 0])
    tracker.observe(1, 0)
    tracker.observe(2, scale)
    assert tracker.current_scales == (scale, scale)
    json.dumps(tracker.to_dict(), allow_nan=False)


@pytest.mark.parametrize("weight", [0, 0.5, 1])
def test_shared_overflow_is_atomic_even_when_shared_weight_is_zero(weight):
    tracker = MultiStepConformal(
        [1, 2],
        scale=[1, np.finfo(float).max / 2],
        scale_decay=0,
        scale_source="blended",
        scale_share_weight=weight,
        initial_quantile=0,
    )
    tracker.observe(0, 0)
    tracker.predict([0, 0])
    before = tracker.to_dict()
    with pytest.raises(ValueError, match="Adaptive scale update overflow"):
        tracker.observe(1, 4)
    assert tracker.to_dict() == before
    assert tracker.next_time == 1


def test_later_residual_failure_keeps_all_scale_sources_feedback_and_queues_atomic():
    largest = np.finfo(float).max
    tracker = MultiStepConformal(
        [1, 2],
        scale_decay=0.5,
        scale_source="blended",
        initial_quantile=1,
    )
    tracker.observe(0, 0)
    tracker.predict([0, -largest])
    tracker.observe(1, 0)
    tracker.predict([0, 0])
    before = tracker.to_dict()
    with pytest.raises(ValueError, match="Residual overflow"):
        tracker.observe(2, largest)
    assert tracker.to_dict() == before


def test_ratio_overflow_is_rejected_even_for_zero_shared_weight():
    with pytest.raises(ValueError, match="Initial scale ratios"):
        MultiStepConformal(
            [2, 5],
            scale=[np.nextafter(0.0, 1.0), 1],
            scale_source="blended",
            scale_decay=0.5,
            scale_share_weight=0,
        )


@pytest.mark.parametrize("source", ["horizon", "shortest"])
def test_nonblended_rejects_explicit_weights_and_preserves_old_json_keys(source):
    with pytest.raises(ValueError, match="requires scale_source='blended'"):
        MultiStepConformal([1, 3], scale_source=source, scale_decay=0.5, scale_share_weight=0)
    tracker = MultiStepConformal([1, 3], scale_source=source, scale_decay=0.5)
    assert tracker.current_scale_weights is None
    assert not {"scale_share_weight", "own_scales", "shared_scales"} & tracker.to_dict().keys()


@pytest.mark.parametrize(
    "bad",
    [
        -0.1,
        1.1,
        np.nan,
        np.inf,
        True,
        np.bool_(False),
        "0.5",
        [],
        [0.5],
        [0.5, True],
        np.array([True, False]),
        np.ma.array([0.1, 0.9], mask=[False, True]),
        [[0.1, 0.9]],
        [0.5, 1j],
    ],
)
def test_weight_validation_rejects_outside_probability_bool_mask_and_wrong_shape(bad):
    with pytest.raises(ValueError):
        MultiStepConformal([1, 3], scale_source="blended", scale_decay=0.5, scale_share_weight=bad)


def test_blended_requires_adaptive_sources_and_wrapper_validates_weights():
    with pytest.raises(ValueError, match="requires scale_decay"):
        MultiStepConformal([1, 3], scale_source="blended")
    with pytest.raises(ValueError, match="scale_share_weight"):
        multistep_intervals(
            [0, 0],
            [[0, 0]],
            origins=[0],
            lead_times=[1, 3],
            scale_source="blended",
            scale_decay=0.5,
            scale_share_weight=True,
        )
