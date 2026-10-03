"""High-precision surrogate reference, delayed feedback and numeric bounds."""

import importlib.util
import json
import math
from dataclasses import FrozenInstanceError, replace
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def prototype():
    spec = importlib.util.spec_from_file_location(
        "scale_mixture_prototype", ROOT / "scripts/scale_mixture_prototype.py"
    )
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves field annotations using the module's globals, with
    # no future annotations here; no registration or package imports are needed.
    spec.loader.exec_module(module)
    return module.TwoScaleMixture


def _reference(current, residual, context, rate, df):
    with localcontext() as ctx:
        ctx.prec = 100
        one = Decimal(1)
        r, scale, nu = map(Decimal.from_float, (float(residual), context.scale, float(df)))
        gap = (
            Decimal.from_float(context.short_scale).ln()
            - Decimal.from_float(context.own_scale).ln()
        )
        z = r * r / (nu * scale * scale + r * r)
        gradient = (one - (nu + one) * z) * gap / (one + abs(gap))
        proposed = Decimal.from_float(float(current)) - Decimal.from_float(float(rate)) * gradient
        return float(max(Decimal(0), min(one, proposed)))


@pytest.mark.parametrize("df", [0.25, 1, 3, 30])
@pytest.mark.parametrize(
    "scales,residual", [((1, 4), 0), ((1, 4), 3), ((10, 0.25), 0.1), ((1e-200, 1e200), 1e100)]
)
def test_update_matches_high_precision_logscale_loss_derivative(prototype, df, scales, residual):
    learner = prototype([1, 6], learning_rate=0.05, df=df)
    context = learner.issue(6, *scales)
    before = learner.to_dict()
    expected = _reference(0.5, residual, context, 0.05, df)
    assert learner.to_dict() == before  # Issuance only returns frozen context.
    assert learner.update(6, residual, context) == pytest.approx(expected, abs=2e-15)
    assert learner.n_updates == (0, 1)
    json.dumps(context.to_dict(), allow_nan=False)
    json.dumps(learner.to_dict(), allow_nan=False)


def test_delayed_feedback_uses_old_issued_scale_but_current_weight(prototype):
    learner = prototype([6])
    first, delayed = learner.issue(6, 1, 4), learner.issue(6, 1, 4)
    after_first = learner.update(6, 0, first)
    expected = _reference(after_first, 3, delayed, 0.05, 3)
    wrong_reset = _reference(delayed.weight, 3, delayed, 0.05, 3)
    wrongly_reissued = learner.issue(6, 1, 4)
    wrong_scale = _reference(after_first, 3, wrongly_reissued, 0.05, 3)
    assert expected != pytest.approx(wrong_reset)
    assert expected != pytest.approx(wrong_scale)
    assert learner.update(6, 3, delayed) == pytest.approx(expected, abs=2e-15)
    assert delayed.weight == 0.5 and delayed.scale == 2
    assert learner.n_updates == (2,)


def test_swapping_source_names_reflects_weights_and_preserves_geometric_scales(prototype):
    original, swapped = prototype([6]), prototype([6])
    for short, own, residual in ((1, 4, 0), (1, 4, 3), (2, 8, 100), (3, 1, 0.2)):
        left, right = original.issue(6, short, own), swapped.issue(6, own, short)
        assert left.scale == pytest.approx(right.scale, rel=1e-14)
        original.update(6, residual, left)
        swapped.update(6, residual, right)
        assert original.weights[0] + swapped.weights[0] == pytest.approx(1, abs=2e-15)


def test_units_rescale_issued_scales_without_changing_learning(prototype):
    first, second = prototype([1, 6]), prototype([1, 6])
    factor = 10000.0
    for short, own, residual in ((1, 4, 0), (1, 4, 3), (2, 8, 100), (3, 1, 0.2)):
        left = first.issue(6, short, own)
        right = second.issue(6, factor * short, factor * own)
        assert right.scale == pytest.approx(factor * left.scale, rel=1e-14)
        first.update(6, residual, left)
        second.update(6, factor * residual, right)
        np.testing.assert_allclose(first.weights, second.weights, rtol=1e-14)


def test_equal_scales_leave_weight_unchanged_and_preserve_exact_source_value(prototype):
    learner = prototype([1, 6])
    for scale, residual in (
        (1.0, 0),
        (2.0, 100),
        (np.finfo(float).max, 1),
        (np.nextafter(0.0, 1.0), 3),
    ):
        context = learner.issue(1, scale, scale)
        assert context.scale == scale and context.loggap == 0
        assert learner.update(1, residual, context) == 0.5
    assert learner.weights == (0.5, 0.5)
    assert learner.n_updates == (4, 0)


@pytest.mark.parametrize(
    "scales",
    [
        (np.finfo(float).max, np.nextafter(0.0, 1.0)),
        (np.nextafter(0.0, 1.0), np.finfo(float).max),
        (np.finfo(float).max, np.finfo(float).max / 2),
        (np.nextafter(0.0, 1.0), np.nextafter(0.0, 1.0) * 2),
    ],
)
def test_extreme_positive_scales_and_residuals_never_overflow_logistic_or_interpolation(
    prototype, scales
):
    learner = prototype([6], learning_rate=0.1)
    for residual in (0, np.nextafter(0.0, 1.0), 1, np.finfo(float).max):
        context = learner.issue(6, *scales)
        assert math.isfinite(context.scale) and min(scales) <= context.scale <= max(scales)
        assert math.isfinite(context.loggap)
        updated = learner.update(6, residual, context)
        assert math.isfinite(updated) and 0 <= updated <= 1
    json.dumps(learner.to_dict(), allow_nan=False)


def test_near_equal_huge_scales_keep_small_loggap_instead_of_log_cancellation(prototype):
    own = 1e300
    short = np.nextafter(own, np.inf)
    context = prototype([6]).issue(6, short, own)
    with localcontext() as ctx:
        ctx.prec = 100
        expected = float(Decimal.from_float(float(short)).ln() - Decimal.from_float(own).ln())
    assert context.loggap > 0
    assert context.loggap == pytest.approx(expected, rel=1e-14)


def test_projection_reaches_both_endpoints_and_preserves_exact_endpoint_scale(prototype):
    learner = prototype([6], learning_rate=100)
    learner.update(6, 0, learner.issue(6, 1, 4))
    assert learner.weights == (1,)
    assert learner.issue(6, 1, 4).scale == 1
    learner.update(6, 100, learner.issue(6, 1, 4))
    assert learner.weights == (0,)
    assert learner.issue(6, 1, 4).scale == 4


def test_context_is_frozen_signed_and_specific_to_learner_and_lead(prototype):
    learner, other = prototype([1, 6]), prototype([1, 6])
    context = learner.issue(6, 1, 4)
    with pytest.raises(FrozenInstanceError):
        context.weight = 0.9
    before = learner.to_dict()
    for invalid in (
        replace(context, weight=0.9),
        replace(context, scale=100),
        replace(context, own_scale=3),
        replace(context, _signature=b"invalid"),
        replace(context, _signature="invalid"),
        other.issue(6, 1, 4),
        None,
    ):
        with pytest.raises(ValueError, match="Context"):
            learner.update(6, 1, invalid)
        assert learner.to_dict() == before
    with pytest.raises(ValueError, match="Context"):
        learner.update(1, 1, context)
    assert learner.to_dict() == before


@pytest.mark.parametrize("name", ["learning_rate", "df"])
@pytest.mark.parametrize("bad", [0, -1, np.nan, np.inf, True, np.bool_(True), "3"])
def test_parameters_require_positive_finite_nonboolean_reals(prototype, name, bad):
    with pytest.raises(ValueError):
        prototype([6], **{name: bad})


@pytest.mark.parametrize("leads", [None, [], [0], [-1], [True], [1.0], [6, 1], [1, 1]])
def test_leads_are_strictly_increasing_positive_integers(prototype, leads):
    with pytest.raises(ValueError):
        prototype(leads)


@pytest.mark.parametrize("bad", [0, -1, np.nan, np.inf, True, np.bool_(True), "1", 1j])
def test_invalid_scale_issuance_is_transactional(prototype, bad):
    learner = prototype([6])
    before = learner.to_dict()
    with pytest.raises(ValueError):
        learner.issue(6, 1, bad)
    assert learner.to_dict() == before


@pytest.mark.parametrize("bad", [-1, np.nan, np.inf, True, np.bool_(True), "1", 1j])
def test_invalid_feedback_does_not_change_weights_or_counts(prototype, bad):
    learner = prototype([6])
    context = learner.issue(6, 1, 4)
    before = learner.to_dict()
    with pytest.raises(ValueError):
        learner.update(6, bad, context)
    assert learner.to_dict() == before


@pytest.mark.parametrize("lead", [1, 0, -1, True, 6.0])
def test_unknown_or_noninteger_lead_is_rejected_in_issue_and_update(prototype, lead):
    learner = prototype([6])
    context = learner.issue(6, 1, 4)
    before = learner.to_dict()
    with pytest.raises(ValueError):
        learner.issue(lead, 1, 4)
    with pytest.raises(ValueError):
        learner.update(lead, 1, context)
    assert learner.to_dict() == before


def test_overflow_of_a_huge_learning_rate_is_an_atomic_failure(prototype):
    learner = prototype([6], df=np.finfo(float).max, learning_rate=np.finfo(float).max)
    context = learner.issue(6, 1, 4)
    before = learner.to_dict()
    with pytest.raises(ValueError, match="overflow"):
        learner.update(6, np.finfo(float).max, context)
    assert learner.to_dict() == before
