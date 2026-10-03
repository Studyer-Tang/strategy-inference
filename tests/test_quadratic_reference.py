"""Independent finite-sample checks for the known Gaussian HAC reference."""

import math

import numpy as np
import pytest

from strategy_inference.inference import default_lags, long_run_variance
from strategy_inference.quadratic_reference import gaussian_ar_hac_moments


def _dense_reference(n_obs, phi, sigma, lags):
    covariance = np.array(
        [[sigma**2 * phi ** abs(row - column) for column in range(n_obs)] for row in range(n_obs)]
    )
    weights = np.array(
        [
            [max(0, 1 - abs(row - column) / (lags + 1)) for column in range(n_obs)]
            for row in range(n_obs)
        ]
    )
    centering = np.eye(n_obs) - np.ones((n_obs, n_obs)) / n_obs
    quadratic = centering @ weights @ centering / n_obs
    product = quadratic @ covariance
    return (
        float(np.trace(product)),
        float(2 * np.trace(product @ product)),
        float(covariance.sum() / n_obs),
        covariance,
        quadratic,
    )


@pytest.mark.parametrize("n_obs", [8, 17, 31])
@pytest.mark.parametrize("phi", [-0.8, 0, 0.6, 0.99])
@pytest.mark.parametrize("lag_choice", [0, 1, "maximum"])
def test_matches_independent_dense_gaussian_quadratic_identity(n_obs, phi, lag_choice):
    lags = n_obs - 2 if lag_choice == "maximum" else lag_choice
    sigma = 1.7
    result = gaussian_ar_hac_moments(n_obs, phi, sigma, lags=lags)
    expectation, variance, target, _, _ = _dense_reference(n_obs, phi, sigma, lags)
    assert result.expectation == pytest.approx(expectation, rel=2e-11, abs=1e-13)
    assert result.variance == pytest.approx(variance, rel=2e-11, abs=1e-13)
    assert result.target == pytest.approx(target, rel=2e-12, abs=1e-13)
    truncated = sigma**2 * math.fsum(
        [1.0] + [2 * (1 - lag / (lags + 1)) * phi**lag for lag in range(1, lags + 1)]
    )
    assert result.population_truncated_lrv == pytest.approx(truncated, rel=2e-12)
    assert result.long_run_limit == pytest.approx(sigma**2 * (1 + phi) / (1 - phi))


@pytest.mark.parametrize("n_obs", [8, 67, 256, 2048])
@pytest.mark.parametrize("sigma", [0.25, 1, 5])
def test_iid_no_lags_is_scaled_chi_square(n_obs, sigma):
    # T*vhat/sigma^2 ~ chi-square(T-1), because the observed mean is removed.
    result = gaussian_ar_hac_moments(n_obs, 0, sigma, lags=0)
    assert result.expectation == pytest.approx(sigma**2 * (n_obs - 1) / n_obs)
    assert result.variance == pytest.approx(2 * sigma**4 * (n_obs - 1) / n_obs**2)
    assert result.target == pytest.approx(sigma**2)
    assert result.population_truncated_lrv == pytest.approx(sigma**2)
    assert result.long_run_limit == pytest.approx(sigma**2)


@pytest.mark.parametrize("n_obs,lags", [(8, 1), (19, 7), (64, 62), (2048, 2046)])
def test_iid_with_lags_matches_projection_trace_closed_form(n_obs, lags):
    # For W symmetric and P=I-11'/T:
    # tr((PWP)^2)=tr(W^2)-2||W1||^2/T+(1'W1)^2/T^2.
    total = n_obs + 2 * math.fsum((n_obs - h) * (1 - h / (lags + 1)) for h in range(1, lags + 1))
    squared = n_obs + 2 * math.fsum(
        (n_obs - h) * (1 - h / (lags + 1)) ** 2 for h in range(1, lags + 1)
    )

    def one_sided_row_sum(length):
        return length - length * (length + 1) / (2 * (lags + 1))

    row_sums = [
        1 + one_sided_row_sum(min(lags, row)) + one_sided_row_sum(min(lags, n_obs - 1 - row))
        for row in range(n_obs)
    ]
    centered_squared = (
        squared - 2 * math.fsum(row**2 for row in row_sums) / n_obs + total**2 / n_obs**2
    )
    result = gaussian_ar_hac_moments(n_obs, 0, lags=lags)
    assert result.expectation == pytest.approx(1 - total / n_obs**2, rel=1e-12)
    assert result.variance == pytest.approx(2 * centered_squared / n_obs**2, rel=1e-12)


def test_finite_target_truncation_and_centering_are_distinct():
    result = gaussian_ar_hac_moments(256, 0.95, lags=12)
    assert 0 < result.expectation < result.population_truncated_lrv
    assert result.population_truncated_lrv < result.target < result.long_run_limit
    assert result.target > 1
    assert result.lags == 12
    assert result.sample_size == 256
    assert result.phi == 0.95
    assert result.sigma == 1


def test_default_lags_and_change_of_units():
    original = gaussian_ar_hac_moments(67, -0.7)
    scaled = gaussian_ar_hac_moments(67, -0.7, sigma=3)
    assert original.lags == default_lags(67)
    for name in ["expectation", "target", "population_truncated_lrv", "long_run_limit"]:
        assert getattr(scaled, name) == pytest.approx(9 * getattr(original, name))
    assert scaled.variance == pytest.approx(81 * original.variance)


@pytest.mark.parametrize("phi", [-0.6, 0.7])
def test_monte_carlo_moments_with_gaussian_quadratic_error_bounds(phi):
    n_obs, n_draws, lags, sigma = 24, 40_000, 4, 1.3
    expected = gaussian_ar_hac_moments(n_obs, phi, sigma, lags=lags)
    rng = np.random.default_rng(7803)
    innovations = rng.normal(size=(n_obs, n_draws))
    sample = np.empty_like(innovations)
    sample[0] = sigma * innovations[0]
    innovation_scale = sigma * math.sqrt(1 - phi**2)
    for row in range(1, n_obs):
        sample[row] = phi * sample[row - 1] + innovation_scale * innovations[row]
    observed = long_run_variance(sample, lags)
    assert observed.mean() == pytest.approx(
        expected.expectation, abs=6 * math.sqrt(expected.variance / n_draws)
    )

    # For a Gaussian quadratic form, mu4 = 3*Var(q)^2 + 48*tr((A C)^4).
    # This gives the actual MC uncertainty of the sample variance, rather than
    # pretending that the quadratic form itself is Gaussian.
    _, variance, _, covariance, quadratic = _dense_reference(n_obs, phi, sigma, lags)
    product = quadratic @ covariance
    fourth_central_moment = 3 * variance**2 + 48 * np.trace(np.linalg.matrix_power(product, 4))
    sample_variance_error = math.sqrt(
        (fourth_central_moment - (n_draws - 3) / (n_draws - 1) * variance**2) / n_draws
    )
    assert observed.var(ddof=1) == pytest.approx(expected.variance, abs=6 * sample_variance_error)
    shifted = long_run_variance(sample + 13, lags)
    np.testing.assert_allclose(shifted, observed, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("phi", [np.nextafter(-1.0, 0.0), np.nextafter(1.0, 0.0)])
@pytest.mark.parametrize("lags", [0, 1, 5, 14])
def test_near_stationarity_boundary_remains_positive_and_finite(phi, lags):
    result = gaussian_ar_hac_moments(16, phi, lags=lags)
    moments = [
        result.expectation,
        result.variance,
        result.target,
        result.population_truncated_lrv,
        result.long_run_limit,
    ]
    assert np.isfinite(moments).all()
    assert np.all(np.array(moments) > 0)


def test_positive_near_unit_root_matches_first_order_centered_covariance():
    # phi=1-delta gives C=11' - delta*D + O(delta^2), D_ij=|i-j|.
    # This separate limiting calculation catches loss of the residual covariance
    # from subtracting numbers close to one during sample centering.
    n_obs, lags = 17, 5
    phi = np.nextafter(1.0, 0.0)
    delta = 1 - phi
    centering = np.eye(n_obs) - np.ones((n_obs, n_obs)) / n_obs
    distances = np.abs(np.arange(n_obs)[:, None] - np.arange(n_obs)[None, :])
    limiting_covariance = -delta * centering @ distances @ centering
    weights = np.maximum(0, 1 - distances / (lags + 1))
    product = weights @ limiting_covariance
    result = gaussian_ar_hac_moments(n_obs, phi, lags=lags)
    assert result.expectation == pytest.approx(np.trace(product) / n_obs, rel=2e-12)
    assert result.variance == pytest.approx(2 * np.trace(product @ product) / n_obs**2, rel=2e-12)


def test_negative_near_unit_root_truncated_even_window_preserves_small_value():
    phi = np.nextafter(-1.0, 0.0)
    result = gaussian_ar_hac_moments(16, phi, lags=1)
    # The two-term population quantity is exactly 1+phi here.
    assert result.population_truncated_lrv == pytest.approx(1 + phi, rel=1e-12, abs=0)


@pytest.mark.parametrize("n_obs", [0, 7, 2049, True, 8.5, "16", [16]])
def test_invalid_sample_size(n_obs):
    with pytest.raises(ValueError, match="n_obs"):
        gaussian_ar_hac_moments(n_obs, 0.5)


@pytest.mark.parametrize("phi", [-1, 1, math.nan, math.inf, True, 0.2j, "0.2", [0.2]])
def test_invalid_coefficient(phi):
    with pytest.raises(ValueError, match="phi"):
        gaussian_ar_hac_moments(16, phi)


@pytest.mark.parametrize("sigma", [0, -1, math.nan, math.inf, True, 1j, "1", [1]])
def test_invalid_scale(sigma):
    with pytest.raises(ValueError, match="sigma"):
        gaussian_ar_hac_moments(16, 0.5, sigma)


@pytest.mark.parametrize("lags", [-1, 15, 2.5, True, "2", [2]])
def test_invalid_lags(lags):
    with pytest.raises(ValueError, match="lags"):
        gaussian_ar_hac_moments(16, 0.5, lags=lags)


@pytest.mark.parametrize("sigma", [1e-200, 1e200])
def test_unrepresentable_moments_are_rejected(sigma):
    with pytest.raises(ValueError, match="positive finite float range"):
        gaussian_ar_hac_moments(16, 0.5, sigma)


def test_small_quadratic_variance_can_be_scaled_without_intermediate_overflow():
    # sigma^4 overflows, but the nearly constant process has a tiny unit-scale
    # quadratic variance; both the result and its comparison remain finite.
    phi = np.nextafter(1.0, 0.0)
    unit = gaussian_ar_hac_moments(16, phi, lags=1)
    scaled = gaussian_ar_hac_moments(16, phi, sigma=1e80, lags=1)
    assert scaled.variance == pytest.approx(
        math.exp(math.log(unit.variance) + 320 * math.log(10)), rel=2e-12
    )
    assert math.isfinite(scaled.long_run_limit)
