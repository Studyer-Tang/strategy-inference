from decimal import Decimal, localcontext

import numpy as np
import pytest
from scipy.stats import norm

from strategy_inference.reference import (
    equicorrelated_max_cdf,
    equicorrelated_max_quantile,
    equicorrelated_max_tail,
    gaussian_ar_mean_variance,
)


@pytest.mark.parametrize("n_obs", [1, 2, 7, 32, 257])
@pytest.mark.parametrize("phi", [-0.8, -0.001, 0, 0.6, 0.999])
def test_finite_ar_mean_variance_matches_dense_known_covariance(n_obs, phi):
    time = np.arange(n_obs)
    covariance = 0.03**2 * phi ** np.abs(time[:, None] - time[None, :])
    expected = covariance.sum() / n_obs**2
    assert gaussian_ar_mean_variance(n_obs, phi, sigma=0.03) == pytest.approx(expected, rel=2e-12)


@pytest.mark.parametrize("phi", [np.nextafter(-1.0, 0), np.nextafter(1.0, 0)])
@pytest.mark.parametrize("n_obs", [2, 100])
def test_ar_boundary_variance_matches_high_precision_formula(phi, n_obs):
    with localcontext() as context:
        context.prec = 70
        coefficient = Decimal.from_float(phi)
        size = Decimal(n_obs)
        factor = Decimal(1) + 2 * sum(
            (1 - Decimal(lag) / size) * coefficient**lag for lag in range(1, n_obs)
        )
        expected = float(factor / size)
    assert gaussian_ar_mean_variance(n_obs, phi) == pytest.approx(expected, rel=2e-13)


def test_finite_variance_keeps_boundary_weight_instead_of_long_run_limit():
    actual = gaussian_ar_mean_variance(8, 0.9)
    asymptotic = (1 + 0.9) / ((1 - 0.9) * 8)
    assert actual < 1
    assert asymptotic > 1


@pytest.mark.parametrize("value", [-3, 0, 2, 8])
@pytest.mark.parametrize("n_strategies", [1, 2, 32])
def test_independent_maximum_has_closed_form_and_stable_tail(value, n_strategies):
    expected_cdf = norm.cdf(value) ** n_strategies
    expected_tail = -np.expm1(n_strategies * norm.logcdf(value))
    assert equicorrelated_max_cdf(value, n_strategies, 0) == pytest.approx(expected_cdf)
    assert equicorrelated_max_tail(value, n_strategies, 0) == pytest.approx(
        expected_tail, rel=1e-13, abs=0
    )


@pytest.mark.parametrize("cross_corr", [0, 0.35, 0.999999, 1])
def test_one_candidate_does_not_depend_on_cross_correlation(cross_corr):
    assert equicorrelated_max_cdf(2, 1, cross_corr) == norm.cdf(2)
    assert equicorrelated_max_tail(2, 1, cross_corr) == norm.sf(2)
    assert equicorrelated_max_quantile(0.95, 1, cross_corr) == norm.ppf(0.95)


def test_perfect_common_component_reduces_to_single_gaussian():
    assert equicorrelated_max_cdf(1.4, 64, 1) == norm.cdf(1.4)
    assert equicorrelated_max_tail(1.4, 64, 1) == norm.sf(1.4)
    assert equicorrelated_max_quantile(0.99, 64, 1) == norm.ppf(0.99)


def test_two_candidate_zero_threshold_has_arcsine_reference():
    for cross_corr in [0.01, 0.35, 0.9, 0.999999]:
        expected = 0.25 + np.arcsin(cross_corr) / (2 * np.pi)
        assert equicorrelated_max_cdf(0, 2, cross_corr) == pytest.approx(expected, abs=2e-11)


@pytest.mark.parametrize("cross_corr", [0.01, 0.35, 0.9, 1 - 1e-10])
@pytest.mark.parametrize("n_strategies", [2, 32])
def test_integrated_cdf_and_tail_are_complements(cross_corr, n_strategies):
    cdf = equicorrelated_max_cdf(2.5, n_strategies, cross_corr)
    tail = equicorrelated_max_tail(2.5, n_strategies, cross_corr)
    assert cdf + tail == pytest.approx(1, abs=3e-11)
    assert norm.cdf(2.5) ** n_strategies <= cdf <= norm.cdf(2.5)


@pytest.mark.parametrize("probability", [0.01, 0.5, 0.95, 0.999999])
@pytest.mark.parametrize("cross_corr", [0, 0.35, 0.9, 1 - 1e-10, 1])
def test_quantile_inverts_reference_distribution(probability, cross_corr):
    quantile = equicorrelated_max_quantile(probability, 16, cross_corr)
    if probability > 0.5:
        assert equicorrelated_max_tail(quantile, 16, cross_corr) == pytest.approx(
            1 - probability, rel=2e-8, abs=1e-13
        )
    else:
        assert equicorrelated_max_cdf(quantile, 16, cross_corr) == pytest.approx(
            probability, abs=2e-11
        )


def test_gaussian_maximum_quantile_changes_with_family_and_correlation():
    single = equicorrelated_max_quantile(0.95, 1, 0.35)
    small_family = equicorrelated_max_quantile(0.95, 4, 0.35)
    large_family = equicorrelated_max_quantile(0.95, 32, 0.35)
    common_family = equicorrelated_max_quantile(0.95, 32, 0.9)
    assert single < small_family < large_family
    assert common_family < large_family


@pytest.mark.parametrize("cross_corr", [1e-20, 1e-12, np.nextafter(1.0, 0)])
def test_reference_quantile_is_stable_near_correlation_boundaries(cross_corr):
    quantile = equicorrelated_max_quantile(0.95, 32, cross_corr)
    assert equicorrelated_max_tail(quantile, 32, cross_corr) == pytest.approx(0.05, abs=2e-11)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_obs": 0},
        {"n_obs": 2.5},
        {"phi": -1},
        {"phi": 1},
        {"phi": np.nan},
        {"phi": 0.5 + 0j},
        {"sigma": 0},
        {"sigma": np.inf},
        {"sigma": "1"},
    ],
)
def test_invalid_ar_reference_parameters_raise_value_error(kwargs):
    with pytest.raises(ValueError):
        gaussian_ar_mean_variance(**{"n_obs": 8, "phi": 0.5, **kwargs})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"value": np.inf},
        {"value": np.nan},
        {"value": 1j},
        {"value": "2"},
        {"n_strategies": 0},
        {"n_strategies": True},
        {"n_strategies": 2.5},
        {"cross_corr": -0.01},
        {"cross_corr": 1.01},
        {"cross_corr": 0.5 + 0j},
    ],
)
def test_invalid_maximum_reference_parameters_raise_value_error(kwargs):
    parameters = {"value": 2.0, "n_strategies": 8, "cross_corr": 0.35, **kwargs}
    for function in (equicorrelated_max_cdf, equicorrelated_max_tail):
        with pytest.raises(ValueError):
            function(**parameters)


@pytest.mark.parametrize("probability", [0, 1, np.nan, "0.95", 0.95 + 0j])
def test_invalid_reference_quantile_probability_raises_value_error(probability):
    with pytest.raises(ValueError):
        equicorrelated_max_quantile(probability, 8, 0.35)
