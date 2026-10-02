import numpy as np
import pytest

from strategy_inference.simulations import GARCH_ALPHA, GARCH_BETA, STUDENT_DF, simulate_returns


@pytest.mark.parametrize("phi", [-0.8, 0.95])
def test_gaussian_initial_observations_have_stationary_variance(phi):
    # Independent columns act as replications, exposing zero-initialization bias.
    values = simulate_returns(2, 20000, phi=phi, cross_corr=0, sigma=1, seed=172, burnin=0)
    np.testing.assert_allclose(values.var(axis=1), 1, atol=0.04, rtol=0)
    assert np.corrcoef(values)[0, 1] == pytest.approx(phi, abs=0.015)


def test_gaussian_variance_and_dependence_match_declared_process():
    values = simulate_returns(20000, 3, phi=0.6, cross_corr=0.35, sigma=1, seed=542)
    np.testing.assert_allclose(values.var(axis=0), 1, atol=0.08, rtol=0)
    correlations = np.corrcoef(values, rowvar=False)
    np.testing.assert_allclose(correlations[np.triu_indices(3, 1)], 0.35, atol=0.04, rtol=0)
    assert np.corrcoef(values[:-1, 0], values[1:, 0])[0, 1] == pytest.approx(0.6, abs=0.035)


@pytest.mark.parametrize("process, phi", [("gaussian_ar", 0.5), ("student_ar", 0.5), ("garch", 0)])
def test_seed_mean_shift_and_unit_changes_are_reproducible(process, phi):
    kwargs = dict(n_obs=128, n_strategies=3, process=process, phi=phi, seed=532, sigma=0.02)
    noise = simulate_returns(**kwargs)
    np.testing.assert_array_equal(simulate_returns(**kwargs), noise)
    assert not np.array_equal(simulate_returns(**{**kwargs, "seed": 533}), noise)
    means = np.array([0.01, -0.02, 0.03])
    shifted = simulate_returns(**kwargs, mean=means)
    np.testing.assert_allclose(shifted - noise, np.broadcast_to(means, noise.shape), atol=1e-15)
    scaled = simulate_returns(**{**kwargs, "sigma": 2.0})
    np.testing.assert_allclose(scaled, 100 * noise, rtol=1e-13, atol=1e-13)
    assert noise.shape == (128, 3)
    assert np.isfinite(noise).all()


@pytest.mark.parametrize("process, phi", [("student_ar", 0.5), ("garch", 0)])
def test_heavy_tailed_processes_have_unit_stationary_variance(process, phi):
    values = simulate_returns(40000, 3, process=process, phi=phi, sigma=1, seed=721, burnin=1024)
    np.testing.assert_allclose(values.var(axis=0), 1, atol=0.12, rtol=0)
    np.testing.assert_allclose(values.mean(axis=0), 0, atol=0.04, rtol=0)


def test_garch_parameters_admit_a_finite_fourth_moment():
    innovation_fourth_moment = 3 * (STUDENT_DF - 2) / (STUDENT_DF - 4)
    upper_bound = (GARCH_ALPHA**2 * innovation_fourth_moment
                   + 2 * GARCH_ALPHA * GARCH_BETA + GARCH_BETA**2)
    assert GARCH_ALPHA + GARCH_BETA < 1
    assert upper_bound < 1


def test_gaussian_needs_no_burnin_when_initialized_stationarily():
    kwargs = dict(n_obs=48, n_strategies=3, phi=0.95, seed=831)
    np.testing.assert_array_equal(simulate_returns(**kwargs, burnin=0), simulate_returns(**kwargs, burnin=1024))


@pytest.mark.parametrize("process, phi", [("gaussian_ar", 0.5), ("student_ar", 0.5), ("garch", 0)])
def test_perfect_common_shocks_give_identical_candidates(process, phi):
    values = simulate_returns(128, 4, process=process, phi=phi, cross_corr=1, seed=492)
    np.testing.assert_array_equal(values, np.repeat(values[:, :1], 4, axis=1))


def test_garch_cannot_silently_ignore_autoregressive_phi():
    with pytest.raises(ValueError, match="phi=0"):
        simulate_returns(32, 3, process="garch", phi=0.5)


@pytest.mark.parametrize(
    "changed",
    [{"n_obs": 1}, {"n_strategies": 0}, {"burnin": -1}, {"seed": -1},
     {"phi": 1}, {"phi": -1}, {"cross_corr": -0.1}, {"cross_corr": 1.1},
     {"sigma": 0}, {"sigma": np.inf}, {"process": "unrecognized"},
     {"mean": [0, 1]}, {"mean": np.nan}, {"mean": 1j}, {"phi": "0.5"}],
)
def test_invalid_simulation_parameters_are_rejected(changed):
    kwargs = {"n_obs": 32, "n_strategies": 3, **changed}
    with pytest.raises(ValueError):
        simulate_returns(**kwargs)
