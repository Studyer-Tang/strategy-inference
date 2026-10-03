from decimal import Decimal, localcontext

import numpy as np
import pytest

from strategy_inference.parametric import (
    _bartlett_mass,
    fit_equicorrelated_ar1,
    known_phi_gls_t,
    replay_pvalue,
    tail_statistics,
)
from strategy_inference.simulations import simulate_returns
from strategy_inference.tail import tail_variance


@pytest.mark.parametrize("size", [1, 2, 7, 32, 200])
def test_fast_mass_matches_independent_dense_covariance(size):
    phi = np.array([-0.99, -0.7, -0.01, 0, 0.1, 0.7, 0.99, 0.9999])
    time = np.arange(size)
    expected = [np.sum(p ** np.abs(time[:, None] - time[None, :])) / size for p in phi]
    np.testing.assert_allclose(_bartlett_mass(phi, size), expected, rtol=2e-12, atol=1e-14)


@pytest.mark.parametrize("size", [2, 7, 64, 2048])
@pytest.mark.parametrize("phi", [np.nextafter(-1.0, 0), np.nextafter(1.0, 0), -0.999999, 0.999999])
def test_fast_mass_matches_decimal_at_both_ar_boundaries(size, phi):
    with localcontext() as context:
        context.prec = 80
        p = Decimal.from_float(phi)
        mass = Decimal(1) + 2 * sum(
            (1 - Decimal(h) / size) * p**h for h in range(1, size)
        )
    assert _bartlett_mass(np.array([phi]), size)[0] == pytest.approx(float(mass), rel=5e-14)


@pytest.mark.parametrize("size", [8, 200, 2048])
def test_fast_mass_is_continuous_across_series_switch(size):
    boundary = 1 - 0.5 / size
    phi = np.array([np.nextafter(boundary, 0), boundary, np.nextafter(boundary, 1)])
    with localcontext() as context:
        context.prec = 60
        expected = [
            float(1 + 2 * sum(
                (1 - Decimal(h) / size) * Decimal.from_float(p)**h
                for h in range(1, size)
            ))
            for p in phi
        ]
    np.testing.assert_allclose(_bartlett_mass(phi, size), expected, rtol=1e-14)


def test_fit_matches_dense_pearson_and_columnwise_phi():
    rng = np.random.default_rng(81)
    shared = rng.normal(size=(64, 1))
    data = shared + 0.7 * rng.normal(size=(64, 5))
    centered = data - data.mean(axis=0)
    phi = np.mean(np.sum(centered[1:] * centered[:-1], axis=0) / np.sum(centered**2, axis=0))
    correlation = np.corrcoef(data, rowvar=False)
    rho = correlation[np.triu_indices(5, k=1)].mean()
    fit = fit_equicorrelated_ar1(data)
    assert fit.phi == pytest.approx(phi, abs=1e-15)
    assert fit.raw_rho == pytest.approx(rho, abs=1e-15)
    assert fit.rho == fit.raw_rho
    assert not fit.projected
    assert (fit.n_obs, fit.k) == (64, 5)


def test_fit_projection_and_single_column_are_explicit():
    x = np.arange(24.0)
    fit = fit_equicorrelated_ar1(np.column_stack((x, -x)))
    assert fit.raw_rho == pytest.approx(-1)
    assert fit.rho == 0
    assert fit.projected
    single = fit_equicorrelated_ar1(x)
    assert (single.k, single.rho, single.raw_rho, single.projected) == (1, 0, 0, False)


def test_fit_is_invariant_to_translation_and_positive_column_units():
    data = np.random.default_rng(9).normal(size=(40, 3))
    original = fit_equicorrelated_ar1(data)
    changed = fit_equicorrelated_ar1(data * [2, 0.5, 3] + [4, -8, 1])
    assert changed.phi == pytest.approx(original.phi, abs=1e-15)
    assert changed.raw_rho == pytest.approx(original.raw_rho, abs=1e-15)


@pytest.mark.parametrize("lags", [0, 4, 30])
def test_batched_tail_matches_existing_scalar_draw_implementation(lags):
    data = np.random.default_rng(127).normal(size=(5, 32, 4))
    statistics, factors, coefficients = tail_statistics(data, lags)
    for index, draw in enumerate(data):
        reference = tail_variance(draw, lags=lags, target="finite_sample")
        np.testing.assert_allclose(factors[index], reference.factor, rtol=5e-14)
        np.testing.assert_allclose(coefficients[index], reference.phi, atol=1e-15)
        expected = draw.mean(axis=0) / np.sqrt(reference.variance / len(draw))
        np.testing.assert_allclose(statistics[index], expected, rtol=5e-14, atol=1e-15)


@pytest.mark.parametrize("phi", [-0.99, 0.9, 0.99])
def test_persistent_ar_batches_match_existing_scalar_draw_implementation(phi):
    draws = np.array([
        simulate_returns(200, 3, phi=phi, sigma=1, cross_corr=0.35, seed=seed)
        for seed in range(4)
    ])
    statistics, factors, coefficients = tail_statistics(draws, 4)
    for index, draw in enumerate(draws):
        reference = tail_variance(draw, lags=4, target="finite_sample")
        np.testing.assert_allclose(factors[index], reference.factor, rtol=8e-14)
        np.testing.assert_allclose(coefficients[index], reference.phi, atol=1e-15)
        expected = draw.mean(axis=0) / np.sqrt(reference.variance / len(draw))
        np.testing.assert_allclose(statistics[index], expected, rtol=8e-14)


def test_frozen_ablation_fixes_only_correction_factors():
    data = np.random.default_rng(67).normal(size=(3, 40, 2))
    frozen = np.array([2.0, 0.7])
    statistics, factors, coefficients = tail_statistics(data, 5, frozen)
    for i, draw in enumerate(data):
        reference = tail_variance(draw, lags=5, target="finite_sample")
        np.testing.assert_allclose(factors[i], frozen)
        np.testing.assert_allclose(coefficients[i], reference.phi)
        expected = draw.mean(axis=0) / np.sqrt(reference.unadjusted * frozen / len(draw))
        np.testing.assert_allclose(statistics[i], expected, rtol=1e-14)
    factors[0, 0] = 3
    assert frozen[0] == 2


def test_tail_batch_partition_and_units_preserve_statistics():
    data = np.random.default_rng(134).normal(size=(7, 50, 3))
    entire = tail_statistics(data, 6)
    split = [tail_statistics(data[:3], 6), tail_statistics(data[3:], 6)]
    for position in range(3):
        np.testing.assert_array_equal(entire[position], np.concatenate([x[position] for x in split]))
    np.testing.assert_allclose(tail_statistics(data * [0.1, 2, 7], 6)[0], entire[0], rtol=1e-13)


@pytest.mark.parametrize("phi", [-0.9, 0, 0.6, 0.99])
def test_gls_t_matches_independent_dense_gls(phi):
    data = np.random.default_rng(28).normal(size=(24, 3)) + [0.2, -0.5, 0.8]
    time = np.arange(len(data))
    covariance = phi ** np.abs(time[:, None] - time[None, :])
    ones = np.ones(len(data))
    precision_ones = np.linalg.solve(covariance, ones)
    information = ones @ precision_ones
    means = precision_ones @ data / information
    residual = data - means
    scales = np.sum(residual * np.linalg.solve(covariance, residual), axis=0) / (len(data) - 1)
    expected = means * np.sqrt(information / scales)
    np.testing.assert_allclose(known_phi_gls_t(data, phi), expected, rtol=5e-13, atol=1e-13)


def test_gls_zero_phi_is_iid_t_and_positive_unit_invariant():
    data = np.random.default_rng(33).normal(size=(64, 3))
    expected = np.sqrt(len(data)) * data.mean(axis=0) / data.std(axis=0, ddof=1)
    np.testing.assert_allclose(known_phi_gls_t(data, 0), expected, rtol=1e-14)
    for phi in [-0.99, 0.7, np.nextafter(1.0, 0)]:
        np.testing.assert_allclose(
            known_phi_gls_t(data * [0.1, 2, 7], phi), known_phi_gls_t(data, phi), rtol=1e-13
        )


def test_replay_rank_includes_ties_and_observation_and_is_scale_invariant():
    assert replay_pvalue(2, [0, 2, 3]) == 3 / 4
    assert replay_pvalue(4, [0, 2, 3]) == 1 / 4
    assert replay_pvalue(-1, [0, 2, 3]) == 1
    draws = np.random.default_rng(45).normal(size=99)
    assert replay_pvalue(0.7, draws) == replay_pvalue(2.1, 3 * draws)


def test_common_known_tail_factor_cancels_from_complete_replay_rank():
    draws = np.random.default_rng(486).normal(size=(100, 32, 3))
    raw = tail_statistics(draws, 4, frozen_factor=np.ones(3))[0].max(axis=1)
    corrected = tail_statistics(draws, 4, frozen_factor=np.full(3, 19.0))[0].max(axis=1)
    np.testing.assert_allclose(corrected, raw / np.sqrt(19), rtol=5e-15)
    assert replay_pvalue(raw[0], raw[1:]) == replay_pvalue(corrected[0], corrected[1:])


def test_exchangeable_rank_grid_has_exact_size_and_ties_are_conservative():
    values = np.random.default_rng(193).normal(size=20)
    pvalues = [replay_pvalue(x, np.delete(values, i)) for i, x in enumerate(values)]
    np.testing.assert_allclose(sorted(pvalues), np.arange(1, 21) / 20, rtol=0, atol=0)
    assert sum(p <= 0.05 for p in pvalues) == 1
    tied = np.zeros(20)
    assert all(replay_pvalue(x, np.delete(tied, i)) == 1 for i, x in enumerate(tied))


@pytest.mark.parametrize("invalid", [np.ones((8, 2)), [[0, 1]] * 7, [[0, np.nan]] * 8, [[1j]] * 8])
def test_fit_and_gls_invalid_data_are_not_dropped(invalid):
    with pytest.raises(ValueError):
        fit_equicorrelated_ar1(invalid)
    with pytest.raises(ValueError):
        known_phi_gls_t(invalid, 0.5)


@pytest.mark.parametrize("phi", [True, "0.5", 1, -1, np.nan, 0.5j, [0.5]])
def test_invalid_gls_coefficients_fail(phi):
    with pytest.raises(ValueError):
        known_phi_gls_t(np.arange(12.0), phi)


@pytest.mark.parametrize("invalid", [
    np.ones((2, 8, 1)), np.zeros((0, 8, 1)), np.zeros((2, 7, 1)),
    np.zeros((2, 8, 0)), np.ones((8, 1)), np.full((2, 8, 1), np.nan),
    np.full((2, 8, 1), 1j), np.full((2, 8, 1), True),
])
def test_tail_invalid_draws_fail_as_a_whole_batch(invalid):
    with pytest.raises(ValueError):
        tail_statistics(invalid, 2)


@pytest.mark.parametrize("lags", [-1, 7, True, 1.5])
def test_invalid_lags_fail(lags):
    with pytest.raises(ValueError):
        tail_statistics(np.random.default_rng(19).normal(size=(2, 8, 2)), lags)


@pytest.mark.parametrize("factor", [[0, 1], [np.inf, 1], [1], [[1, 1]], [1j, 1], [True, True]])
def test_invalid_frozen_factors_fail(factor):
    with pytest.raises(ValueError):
        tail_statistics(np.random.default_rng(19).normal(size=(2, 8, 2)), 2, factor)


@pytest.mark.parametrize("observed, draws", [
    (np.inf, [0]), (0, []), (0, [np.nan]), (0, [[1]]), (True, [0]),
    (0, [1j]), ("1", [0]), ([1], [0]), (0, [True]),
])
def test_invalid_replay_statistics_fail(observed, draws):
    with pytest.raises(ValueError):
        replay_pvalue(observed, draws)
