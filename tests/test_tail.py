from decimal import Decimal, localcontext

import numpy as np
import pytest

from strategy_inference.inference import long_run_variance
from strategy_inference.tail import ar1_tail_factor, tail_variance


@pytest.mark.parametrize("phi", [-0.9, -0.1, 0.0, 0.6, 0.99])
@pytest.mark.parametrize("bandwidth", [1, 2, 6, 31])
def test_factor_matches_independent_dense_population_covariance(phi, bandwidth):
    time = np.arange(bandwidth)
    covariance = phi ** np.abs(time[:, None] - time[None, :])
    expected = (1 + phi) / (1 - phi) / (covariance.sum() / bandwidth)
    assert float(ar1_tail_factor(phi, bandwidth)) == pytest.approx(expected, rel=3e-13)


@pytest.mark.parametrize("phi", [np.nextafter(-1.0, 0), np.nextafter(1.0, 0)])
@pytest.mark.parametrize("bandwidth", [2, 7, 64])
def test_boundary_factor_matches_high_precision_population_sum(phi, bandwidth):
    with localcontext() as context:
        context.prec = 80
        p = Decimal.from_float(phi)
        mass = Decimal(1) + 2 * sum(
            (1 - Decimal(h) / bandwidth) * p**h for h in range(1, bandwidth)
        )
        expected = float((1 + p) / (1 - p) / mass)
    assert float(ar1_tail_factor(phi, bandwidth)) == pytest.approx(expected, rel=2e-13)


def test_finite_sample_factor_uses_the_exact_mean_variance_not_long_run_limit():
    phi, size, bandwidth = 0.99, 20, 6
    time = np.arange(size)
    true_target = (phi ** np.abs(time[:, None] - time[None, :])).sum() / size
    time = np.arange(bandwidth)
    mass = (phi ** np.abs(time[:, None] - time[None, :])).sum() / bandwidth
    assert ar1_tail_factor(phi, bandwidth, target="finite_sample", n_obs=size) == pytest.approx(
        true_target / mass
    )
    assert ar1_tail_factor(phi, bandwidth, target="finite_sample", n_obs=size) < ar1_tail_factor(
        phi, bandwidth
    )


def test_estimation_uses_the_papers_centered_divisor_T_autocorrelation():
    data = np.random.default_rng(78).normal(size=(32, 3))
    centered = data - data.mean(axis=0)
    phi = (centered[1:] * centered[:-1]).sum(axis=0) / (centered**2).sum(axis=0)
    result = tail_variance(data, lags=5)
    np.testing.assert_allclose(result.phi, phi, rtol=1e-14)
    np.testing.assert_allclose(result.unadjusted, long_run_variance(data, 5))
    np.testing.assert_allclose(result.variance, result.unadjusted * result.factor)
    assert result.parameter_source == "estimated"
    assert result.lags == 5


def test_translation_and_unit_changes_do_not_change_the_fitted_coefficient():
    data = np.random.default_rng(14).normal(size=(50, 2))
    original = tail_variance(data, lags=6, target="finite_sample")
    changed = tail_variance(3 * data + [4, -2], lags=6, target="finite_sample")
    np.testing.assert_allclose(changed.phi, original.phi)
    np.testing.assert_allclose(changed.variance, 9 * original.variance)


def test_known_parameters_and_per_column_parameters_are_explicit():
    data = np.random.default_rng(29).normal(size=(20, 2))
    result = tail_variance(data, phi=[0.4, 0.8], lags=3)
    np.testing.assert_allclose(result.phi, [0.4, 0.8])
    assert result.parameter_source == "known"
    np.testing.assert_allclose(tail_variance(data, phi=0.4).phi, [0.4, 0.4])
    with pytest.raises(ValueError, match="one entry"):
        tail_variance(data, phi=[0.4])


@pytest.mark.parametrize("phi", [True, "0.5", 1, -1, float("nan"), 0.5 + 0j, [[0.5]]])
def test_invalid_coefficients_are_not_clipped(phi):
    with pytest.raises(ValueError):
        ar1_tail_factor(phi, 6)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bandwidth": 0},
        {"bandwidth": 2.5},
        {"target": "unknown"},
        {"n_obs": 20},
        {"target": "finite_sample"},
        {"target": "finite_sample", "n_obs": 6},
    ],
)
def test_invalid_factor_contracts_raise(kwargs):
    with pytest.raises(ValueError):
        ar1_tail_factor(0.6, **{"bandwidth": 6, **kwargs})


def test_vectorized_factors_match_scalar_evaluation():
    values = [-0.8, 0, 0.9]
    for target in ("long_run", "finite_sample"):
        arguments = {"target": target, "n_obs": 23} if target == "finite_sample" else {}
        np.testing.assert_allclose(
            ar1_tail_factor(values, 7, **arguments),
            [float(ar1_tail_factor(p, 7, **arguments)) for p in values],
        )
