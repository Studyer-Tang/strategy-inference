"""Independent interval-score derivative, delayed issuance and numerical checks."""

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
        "interval_scale_mixture_prototype", ROOT / "scripts/interval_scale_mixture_prototype.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.IntervalScaleMixture


def _reference(current, miss, context, rate=0.05, alpha=0.1):
    with localcontext() as ctx:
        ctx.prec = 100
        one = Decimal(1)
        scale, short, own, q, a = map(
            Decimal.from_float,
            (context.scale, context.short_scale, context.own_scale, context.quantile, alpha),
        )
        # Full proper interval-score subgradient and its frozen normalizer.
        k = q / (one - q)
        delta = short - own
        derivative = 2 * k * (one - Decimal(int(miss)) / a) * delta
        divisor = 2 * k * (scale + abs(delta))
        proposed = Decimal.from_float(current) - Decimal.from_float(rate) * derivative / divisor
        return float(max(Decimal(0), min(one, proposed)))


@pytest.mark.parametrize("alpha", [0.01, 0.1, 0.5, 0.99])
@pytest.mark.parametrize("quantile", [0.1, 0.65, np.nextafter(1.0, 0.0)])
@pytest.mark.parametrize("miss", [False, True])
def test_update_matches_high_precision_normalized_proper_score_derivative(
    prototype, alpha, quantile, miss
):
    learner = prototype([1, 6], alpha=alpha)
    before = learner.to_dict()
    context = learner.issue(6, 1, 4, quantile)
    assert learner.to_dict() == before
    assert context.scale == 2.5 and context.normalized_delta == pytest.approx(-3 / 5.5)
    assert learner.update(6, miss, context) == pytest.approx(
        _reference(0.5, miss, context, alpha=alpha), abs=2e-15
    )
    assert learner.n_updates == (0, 1) and learner.n_learned == (0, 1)
    json.dumps(context.to_dict(), allow_nan=False)
    json.dumps(learner.to_dict(), allow_nan=False)


def _score(weight, residual, short=1.0, own=4.0, q=0.65, alpha=0.1):
    radius = q / (1 - q) * ((1 - weight) * own + weight * short)
    return 2 * radius + 2 / alpha * max(residual - radius, 0)


def test_closed_boundary_hit_selects_a_valid_interval_score_subgradient(prototype):
    learner = prototype([6])
    context = learner.issue(6, 1, 4, 0.65)
    k = context.quantile / (1 - context.quantile)
    boundary = k * context.scale
    derivative = 2 * k * (context.short_scale - context.own_scale)  # miss=False
    at_issue = _score(context.weight, boundary)
    for candidate in (0.0, 0.1, 0.49, 0.51, 0.9, 1.0):
        assert _score(candidate, boundary) >= (
            at_issue + derivative * (candidate - context.weight) - 1e-12
        )
    assert learner.update(6, False, context) == pytest.approx(_reference(0.5, False, context))


def test_delayed_feedback_uses_issued_normalizer_and_updates_current_weight(prototype):
    learner = prototype([6])
    first = learner.issue(6, 1, 4, 0.65)
    delayed = learner.issue(6, 1, 4, 0.65)
    after_first = learner.update(6, False, first)
    expected = _reference(after_first, True, delayed)
    wrong_reset = _reference(delayed.weight, True, delayed)
    wrong_reissue = _reference(after_first, True, learner.issue(6, 1, 4, 0.65))
    assert expected != pytest.approx(wrong_reset)
    assert expected != pytest.approx(wrong_reissue)
    assert learner.update(6, True, delayed) == pytest.approx(expected, abs=2e-15)
    assert delayed.scale == 2.5 and delayed.weight == 0.5


def test_unit_rescaling_and_source_exchange_preserve_learning(prototype):
    original, rescaled, swapped = prototype([6]), prototype([6]), prototype([6])
    for short, own, miss in ((1, 4, False), (1, 4, True), (2, 8, True), (3, 1, False)):
        left = original.issue(6, short, own, 0.65)
        units = rescaled.issue(6, short * 10000, own * 10000, 0.65)
        right = swapped.issue(6, own, short, 0.65)
        assert units.scale == pytest.approx(10000 * left.scale)
        assert units.normalized_delta == pytest.approx(left.normalized_delta)
        assert right.scale == pytest.approx(left.scale)
        original.update(6, miss, left)
        rescaled.update(6, miss, units)
        swapped.update(6, miss, right)
        assert original.weights == pytest.approx(rescaled.weights, abs=2e-15)
        assert original.weights[0] + swapped.weights[0] == pytest.approx(1, abs=2e-15)


def test_h1_equal_sources_preserve_exact_candidate_and_weight(prototype):
    learner = prototype([1, 6])
    for scale in (1.0, 1.2, np.finfo(float).max, np.nextafter(0.0, 1.0)):
        for miss in (False, True):
            context = learner.issue(1, scale, scale, 0.65)
            assert context.scale == scale and context.normalized_delta == 0
            assert learner.update(1, miss, context) == 0.5
    assert learner.n_updates == learner.n_learned == (8, 0)


def test_equal_sources_have_zero_gradient_even_with_extreme_positive_alpha(prototype):
    learner = prototype([1], alpha=np.nextafter(0.0, 1.0), learning_rate=np.finfo(float).max)
    context = learner.issue(1, np.finfo(float).max, np.finfo(float).max, 0.65)
    assert learner.update(1, True, context) == 0.5
    assert learner.n_updates == learner.n_learned == (1,)


@pytest.mark.parametrize("miss", [False, True])
def test_normalized_step_respects_the_subgradient_magnitude_bound(prototype, miss):
    rate, alpha = 0.01, 0.1
    learner = prototype([6], learning_rate=rate, alpha=alpha)
    context = learner.issue(6, 100, 1, 0.65)
    before = learner.weights[0]
    after = learner.update(6, miss, context)
    bound = rate * max(1, 1 / alpha - 1) * abs(context.normalized_delta)
    assert abs(after - before) <= bound + 1e-15


@pytest.mark.parametrize("quantile", [-10, -0.1, 0, 1, 1.1, 100])
def test_nonpositive_and_unbounded_quantiles_skip_only_weight_learning(prototype, quantile):
    learner = prototype([6])
    context = learner.issue(6, 1, 4, quantile)
    reason = "nonpositive_quantile" if quantile <= 0 else "unbounded_quantile"
    assert context.skip_reason == reason
    assert learner.update(6, True, context) == 0.5
    assert learner.n_updates == (1,) and learner.n_learned == (0,)
    state = learner.to_dict()
    assert state["skipped"][reason] == [1]
    assert sum(v[0] for v in state["skipped"].values()) == 1


@pytest.mark.parametrize(
    "scales",
    [
        (np.finfo(float).max, np.nextafter(0.0, 1.0)),
        (np.nextafter(0.0, 1.0), np.finfo(float).max),
        (np.finfo(float).max, np.finfo(float).max / 2),
        (np.nextafter(0.0, 1.0), np.nextafter(0.0, 1.0) * 2),
        (1e300, np.nextafter(1e300, np.inf)),
    ],
)
def test_extreme_scales_never_overflow_interpolation_or_normalization(prototype, scales):
    learner = prototype([6])
    for miss in (False, True) * 4:
        context = learner.issue(6, *scales, 0.65)
        assert math.isfinite(context.scale) and min(scales) <= context.scale <= max(scales)
        assert math.isfinite(context.normalized_delta) and abs(context.normalized_delta) <= 1
        with localcontext() as ctx:
            ctx.prec = 100
            delta = Decimal.from_float(float(scales[0])) - Decimal.from_float(float(scales[1]))
            expected = float(delta / (Decimal.from_float(context.scale) + abs(delta)))
        assert context.normalized_delta == pytest.approx(expected, abs=2e-15)
        expected_weight = _reference(learner.weights[0], miss, context)
        assert learner.update(6, miss, context) == pytest.approx(expected_weight, abs=2e-15)
    json.dumps(learner.to_dict(), allow_nan=False)


@pytest.mark.parametrize("weight", [np.nextafter(0.0, 1.0), 2**-54, 0.5, np.nextafter(1.0, 0.0)])
@pytest.mark.parametrize("reverse", [False, True])
def test_interpolation_is_bounded_at_every_representative_weight(prototype, weight, reverse):
    learner = prototype([6])
    # Exercise possible binary64 states directly, independently of the gradient's trajectory.
    learner._weights[6] = float(weight)
    large, small = np.finfo(float).max, np.nextafter(0.0, 1.0)
    short, own = (small, large) if reverse else (large, small)
    context = learner.issue(6, short, own, 0.65)
    assert math.isfinite(context.scale) and small <= context.scale <= large
    assert math.isfinite(context.normalized_delta) and abs(context.normalized_delta) <= 1


def test_projection_reaches_endpoints_and_returns_exact_source(prototype):
    learner = prototype([6], learning_rate=100)
    learner.update(6, False, learner.issue(6, 1, 4, 0.65))
    assert learner.weights == (1,) and learner.issue(6, 1, 4, 0.65).scale == 1
    learner.update(6, True, learner.issue(6, 1, 4, 0.65))
    assert learner.weights == (0,) and learner.issue(6, 1, 4, 0.65).scale == 4


def test_context_is_frozen_signed_and_specific_to_learner_and_lead(prototype):
    learner, other = prototype([1, 6]), prototype([1, 6])
    context = learner.issue(6, 1, 4, 0.65)
    with pytest.raises(FrozenInstanceError):
        context.weight = 0.9
    before = learner.to_dict()
    for invalid in (
        replace(context, weight=0.9),
        replace(context, scale=100),
        replace(context, quantile=1),
        replace(context, normalized_delta=1),
        replace(context, skip_reason="unbounded_quantile"),
        replace(context, _signature=b"invalid"),
        replace(context, _signature="invalid"),
        other.issue(6, 1, 4, 0.65),
        None,
    ):
        with pytest.raises(ValueError, match="Context"):
            learner.update(6, True, invalid)
        assert learner.to_dict() == before
    with pytest.raises(ValueError, match="Context"):
        learner.update(1, True, context)
    assert learner.to_dict() == before


@pytest.mark.parametrize("alpha", [0, -1, 1, 2, np.nan, np.inf, True, np.bool_(True), "0.1"])
def test_alpha_requires_a_finite_nonboolean_probability(prototype, alpha):
    with pytest.raises(ValueError):
        prototype([6], alpha=alpha)


@pytest.mark.parametrize("rate", [0, -1, np.nan, np.inf, True, np.bool_(True), "0.1"])
def test_rate_requires_a_positive_finite_nonboolean_real(prototype, rate):
    with pytest.raises(ValueError):
        prototype([6], learning_rate=rate)


@pytest.mark.parametrize("leads", [None, [], [0], [-1], [True], [1.0], [6, 1], [1, 1]])
def test_leads_require_a_nonempty_increasing_integer_vector(prototype, leads):
    with pytest.raises(ValueError):
        prototype(leads)


@pytest.mark.parametrize("bad", [0, -1, np.nan, np.inf, True, np.bool_(True), "1", 1j])
def test_invalid_scale_issuance_is_transactional(prototype, bad):
    learner = prototype([6])
    before = learner.to_dict()
    with pytest.raises(ValueError):
        learner.issue(6, 1, bad, 0.65)
    assert learner.to_dict() == before


@pytest.mark.parametrize("bad", [np.nan, np.inf, True, np.bool_(True), "1", 1j])
def test_invalid_quantile_is_transactional(prototype, bad):
    learner = prototype([6])
    before = learner.to_dict()
    with pytest.raises(ValueError):
        learner.issue(6, 1, 4, bad)
    assert learner.to_dict() == before


@pytest.mark.parametrize("bad", [0, 1, -1, np.nan, np.inf, "False", None, np.array([False])])
def test_feedback_requires_a_boolean_scalar_and_failure_is_atomic(prototype, bad):
    learner = prototype([6])
    context = learner.issue(6, 1, 4, 0.65)
    before = learner.to_dict()
    with pytest.raises(ValueError, match="boolean"):
        learner.update(6, bad, context)
    assert learner.to_dict() == before


def test_numpy_bool_feedback_is_accepted_without_a_numpy_runtime_import(prototype):
    learner = prototype([6])
    assert learner.update(6, np.bool_(False), learner.issue(6, 1, 4, 0.65)) > 0.5
    source = (ROOT / "scripts/interval_scale_mixture_prototype.py").read_text()
    assert "import numpy" not in source


@pytest.mark.parametrize("lead", [1, 0, -1, True, 6.0])
def test_unknown_and_noninteger_lead_is_rejected(prototype, lead):
    learner = prototype([6])
    context = learner.issue(6, 1, 4, 0.65)
    before = learner.to_dict()
    with pytest.raises(ValueError):
        learner.issue(lead, 1, 4, 0.65)
    with pytest.raises(ValueError):
        learner.update(lead, True, context)
    assert learner.to_dict() == before


def test_nonfinite_weight_step_is_rejected_atomically_and_skips_still_count(prototype):
    learner = prototype([6], learning_rate=np.finfo(float).max)
    context = learner.issue(6, 1, 4, 0.65)
    before = learner.to_dict()
    with pytest.raises(ValueError, match="overflow"):
        learner.update(6, True, context)
    assert learner.to_dict() == before
    assert learner.update(6, True, learner.issue(6, 1, 4, 1)) == 0.5
    assert learner.n_updates == (1,) and learner.n_learned == (0,)
