"""Independent moment references, target pairing and transactional numerics."""

import importlib.util
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def transfer():
    spec = importlib.util.spec_from_file_location(
        "scale_transfer_prototype", ROOT / "scripts/scale_transfer_prototype.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MatureScaleTransfer


def test_updates_match_independent_exact_energy_recursion(transfer):
    leads, initial = (1, 6, 24), (Fraction(2), Fraction(3), Fraction(5))
    fast_decay, slow_decay, floor = Fraction(3, 4), Fraction(7, 8), Fraction(1, 100)
    tracker = transfer(initial, leads, scale_decay=fast_decay, ratio_decay=slow_decay, floor=floor)
    fast = initial[0] ** 2
    numerator, reference = [value**2 for value in initial], [initial[0] ** 2] * 3
    counts, short_count = [0] * 3, 0
    calls = (
        {1: 1, 6: 4, 24: 0},
        {6: 99},
        {1: 0},
        {1: 2, 24: 8},
        {1: 7, 6: 9},
        {24: 1},
        {1: 3, 6: 2, 24: 4},
        {},
    )
    for residuals in calls:
        before = tracker.to_dict()
        output = tracker.update(residuals)
        if 1 not in residuals:
            assert tracker.to_dict() == before
            continue
        short_count += 1
        fast = max(floor**2, fast_decay * fast + (1 - fast_decay) * residuals[1] ** 2)
        for i, lead in enumerate(leads[1:], 1):
            if lead in residuals:
                numerator[i] = max(
                    floor**2, slow_decay * numerator[i] + (1 - slow_decay) * residuals[lead] ** 2
                )
                reference[i] = max(
                    floor**2, slow_decay * reference[i] + (1 - slow_decay) * residuals[1] ** 2
                )
                counts[i] += 1
        expected = [max(float(floor), math.sqrt(float(fast)))] + [
            max(float(floor), math.sqrt(float(fast * numerator[i] / reference[i]))) for i in (1, 2)
        ]
        np.testing.assert_allclose(output, expected, rtol=1e-14)
        np.testing.assert_allclose(
            tracker.ratios,
            [1, *[math.sqrt(float(numerator[i] / reference[i])) for i in (1, 2)]],
            rtol=1e-14,
        )
        state = tracker.to_dict()
        np.testing.assert_allclose(
            state["slow_numerator"], [math.sqrt(float(x)) for x in numerator], rtol=1e-14
        )
        np.testing.assert_allclose(
            state["slow_reference"], [math.sqrt(float(x)) for x in reference], rtol=1e-14
        )
        assert tracker.paired_counts == tuple(counts)
        assert state["shortest_feedback_count"] == short_count
        json.dumps(state, allow_nan=False)


def test_pairs_are_not_carried_across_target_calls_or_leads(transfer):
    helper = transfer([2, 3, 5], [1, 6, 24])
    initial = helper.to_dict()
    helper.update({6: 99})
    assert helper.to_dict() == initial
    helper.update({1: 7})
    assert helper.paired_counts == (0, 0, 0)
    assert helper.ratios == (1, 1.5, 2.5)
    before = helper.to_dict()
    helper.update({24: 9})
    assert helper.to_dict() == before
    helper.update({1: 4, 24: 6})
    assert helper.paired_counts == (0, 0, 1)
    assert helper.to_dict()["slow_numerator"][1] == initial["slow_numerator"][1]
    assert helper.to_dict()["slow_reference"][1] == initial["slow_reference"][1]
    helper.update({1: 2, 6: 8})
    assert helper.paired_counts == (0, 1, 1)


def test_equal_speeds_with_identical_pairing_reduce_to_own_lead_rms(transfer):
    helper = transfer([2, 4], [1, 6], scale_decay=0.75, ratio_decay=0.75)
    own_short, own_long = 2.0, 4.0
    for short, long in ((1, 5), (3, 9), (0, 2), (2, 1)):
        own_short = math.sqrt(0.75 * own_short**2 + 0.25 * short**2)
        own_long = math.sqrt(0.75 * own_long**2 + 0.25 * long**2)
        helper.update({1: short, 6: long})
        np.testing.assert_allclose(helper.scales, [own_short, own_long], rtol=1e-14)
        assert helper.to_dict()["shortest_fast_scale"] == helper.to_dict()["slow_reference"][1]


def test_rescaling_units_rescales_floors_and_output_but_not_ratios(transfer):
    factor = 2.0**20
    first = transfer([2, 4, 8], [3, 12, 24], floor=0.01)
    second = transfer([factor * x for x in [2, 4, 8]], [3, 12, 24], floor=factor * 0.01)
    for values in ({3: 1, 12: 9}, {3: 0, 24: 5}, {24: 2}, {3: 4, 12: 0, 24: 9}):
        first.update(values)
        second.update({lead: factor * value for lead, value in values.items()})
        np.testing.assert_allclose(second.scales, np.array(first.scales) * factor, rtol=1e-14)
        np.testing.assert_allclose(second.ratios, first.ratios, rtol=1e-14)
        assert second.paired_counts == first.paired_counts


def test_shortest_alone_matches_existing_ewma_exactly(transfer):
    from strategy_inference.multistep import MultiStepConformal

    helper = transfer([2], [1], scale_decay=0.75)
    tracker = MultiStepConformal([1], scale=2, scale_decay=0.75)
    for time, actual in enumerate([0, 3, -2, 0, 1]):
        updates = tracker.observe(time, actual)
        if updates:
            update = updates[0]
            helper.update({1: abs(update.actual - update.prediction)})
        assert helper.scales == tracker.current_scales
        assert helper.ratios == (1,)
        assert helper.paired_counts == (0,)
        tracker.predict([actual])


def test_floor_initial_components_and_detached_diagnostics_are_explicit(transfer):
    supplied = np.array([1e-10, 2e-10])
    helper = transfer(supplied, [1, 6], scale_decay=0, ratio_decay=0, floor=1e-8)
    supplied[:] = 99
    assert helper.scales == (1e-8, 1e-8)
    assert helper.ratios == (1, 2)
    state = helper.to_dict()
    assert state["initial_scales"] == [1e-10, 2e-10]
    assert state["slow_numerator"] == [1e-10, 2e-10]
    state["slow_numerator"][1] = 99
    assert helper.to_dict()["slow_numerator"][1] == 2e-10
    helper.update({1: 0, 6: 0})
    assert helper.scales == (1e-8, 1e-8) and helper.ratios == (1, 1)
    json.dumps(helper.to_dict(), allow_nan=False)


@pytest.mark.parametrize("bad", [0, -1, np.nan, np.inf, True, np.bool_(True), 1j, "1"])
def test_initial_scales_must_be_positive_finite_and_nonboolean(transfer, bad):
    with pytest.raises(ValueError):
        transfer([1, bad], [1, 6])


@pytest.mark.parametrize("leads", [[], [0], [-1], [True], [np.bool_(True)], [1.0], [1, 1], [6, 1]])
def test_leads_must_be_a_strict_positive_integer_vector(transfer, leads):
    with pytest.raises(ValueError):
        transfer([1] * len(leads), leads)


@pytest.mark.parametrize("name", ["scale_decay", "ratio_decay"])
@pytest.mark.parametrize("bad", [-0.1, 1, np.nan, np.inf, True, ".9"])
def test_decay_validation(transfer, name, bad):
    with pytest.raises(ValueError):
        transfer([1], [1], **{name: bad})


@pytest.mark.parametrize("bad", [0, -1, np.nan, np.inf, True])
def test_floor_validation(transfer, bad):
    with pytest.raises(ValueError):
        transfer([1], [1], floor=bad)


@pytest.mark.parametrize(
    "bad",
    [
        None,
        [],
        {999: 1},
        {1.0: 1},
        {True: 1},
        {1: 3, 6: -1},
        {1: 3, 6: np.nan},
        {1: 3, 6: np.inf},
        {1: 3, 6: True},
        {1: 3, 6: np.bool_(True)},
        {1: 3, 6: "1"},
        {1: 3, 6: 1j},
    ],
)
def test_invalid_or_late_invalid_mapping_leaves_all_state_unchanged(transfer, bad):
    helper = transfer([1, 2], [1, 6])
    helper.update({1: 2, 6: 3})
    before = helper.to_dict()
    with pytest.raises(ValueError):
        helper.update(bad)
    assert helper.to_dict() == before


def test_hypot_updates_remain_finite_when_raw_squares_overflow(transfer):
    helper = transfer([1e200, 2e200], [1, 6], scale_decay=0.75, ratio_decay=0.5)
    helper.update({1: 3e200, 6: 4e200})
    assert all(math.isfinite(value) and value > 0 for value in helper.scales)
    json.dumps(helper.to_dict(), allow_nan=False)


def test_scale_overflow_is_a_late_atomic_failure(transfer):
    helper = transfer([1, 2], [1, 6], scale_decay=0)
    before = helper.to_dict()
    with pytest.raises(ValueError, match="scale.*overflow"):
        helper.update({1: np.finfo(float).max})
    assert helper.to_dict() == before


def test_unrepresentable_ratio_is_atomic_even_when_scaled_output_could_be_finite(transfer):
    helper = transfer([1e-100, 1e200], [1, 6], scale_decay=0, ratio_decay=0, floor=1e-300)
    before = helper.to_dict()
    with pytest.raises(ValueError, match="ratios"):
        helper.update({1: 0, 6: 1e100})
    assert helper.to_dict() == before
