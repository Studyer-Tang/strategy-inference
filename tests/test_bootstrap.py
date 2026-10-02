import numpy as np
import pytest

from strategy_inference import stationary_bootstrap_means, stationary_indices
from strategy_inference.bootstrap import stationary_bootstrap_statistics, stationary_mean_variance
from strategy_inference.inference import infer_mean


@pytest.fixture
def returns():
    return np.random.default_rng(29).normal(size=(67, 4))


def test_indices_have_requested_shape_and_stay_in_sample():
    indices = stationary_indices(19, 31, block_length=5, seed=78)
    assert indices.shape == (31, 19)
    assert np.issubdtype(indices.dtype, np.integer)
    assert np.all((indices >= 0) & (indices < 19))


def test_blocks_wrap_at_sample_boundary():
    indices = stationary_indices(11, 1024, block_length=11, seed=48)
    reaches_end = indices[:, :-1] == 10
    # At the end, a continued block must return to the first observation.
    wrapped = indices[:, 1:][reaches_end] == 0
    expected = 1 - 1 / 11 + 1 / 121
    assert abs(wrapped.mean() - expected) < 0.04


@pytest.mark.parametrize("block_length", [1, 4])
def test_restart_probability_matches_geometric_block_model(block_length):
    # A restarted index can, by chance, equal the circular successor.
    size = 32
    indices = stationary_indices(size, 4096, block_length=block_length, seed=719)
    successor = (indices[:, :-1] + 1) % size
    observed = np.mean(indices[:, 1:] == successor)
    expected = 1 - 1 / block_length + 1 / (block_length * size)
    assert abs(observed - expected) < 0.005
    frequencies = np.bincount(indices.ravel(), minlength=size) / indices.size
    np.testing.assert_allclose(frequencies, 1 / size, atol=0.004, rtol=0)


def test_seed_reproduces_indices_and_different_seed_changes_them():
    reference = stationary_indices(24, 17, block_length=4, seed=91)
    repeated = stationary_indices(24, 17, block_length=4, seed=91)
    changed = stationary_indices(24, 17, block_length=4, seed=92)
    np.testing.assert_array_equal(reference, repeated)
    assert not np.array_equal(reference, changed)


def test_resampled_means_match_explicit_indexed_sample(returns):
    indices = stationary_indices(len(returns), 53, block_length=7, seed=83)
    expected = returns[indices].mean(axis=1)
    observed = stationary_bootstrap_means(
        returns, n_resamples=53, block_length=7, seed=83, batch_size=11
    )
    np.testing.assert_allclose(observed, expected, atol=1e-14, rtol=1e-14)


def test_common_rows_preserve_cross_strategy_linear_relation(returns):
    values = np.column_stack([returns[:, 0], returns[:, 0], 3 * returns[:, 0] - 2])
    means = stationary_bootstrap_means(values, n_resamples=79, seed=15)
    np.testing.assert_array_equal(means[:, 0], means[:, 1])
    np.testing.assert_allclose(means[:, 2], 3 * means[:, 0] - 2, atol=1e-14)


def test_centered_resamples_do_not_depend_on_column_location(returns):
    kwargs = dict(n_resamples=91, block_length=5, seed=134, center=True)
    reference = stationary_bootstrap_means(returns, **kwargs)
    changed = stationary_bootstrap_means(returns + np.array([1.5, -3, 0.5, 2]), **kwargs)
    np.testing.assert_allclose(changed, reference, atol=1e-14)


@pytest.mark.parametrize("batch_size", [1, 7, 128])
def test_batch_partition_does_not_change_random_sample(returns, batch_size):
    kwargs = dict(n_resamples=93, block_length=6, seed=341, center=True)
    reference = stationary_bootstrap_means(returns, batch_size=93, **kwargs)
    changed = stationary_bootstrap_means(returns, batch_size=batch_size, **kwargs)
    np.testing.assert_allclose(changed, reference, atol=1e-14, rtol=1e-14)


def test_one_dimensional_input_returns_one_strategy(returns):
    means = stationary_bootstrap_means(returns[:, 0], n_resamples=37, seed=15)
    assert means.shape == (37, 1)


@pytest.mark.parametrize("block_length", ["3", 3 + 0j, object(), [3]])
def test_nonnumeric_block_length_raises_value_error(returns, block_length):
    with pytest.raises(ValueError):
        stationary_bootstrap_means(returns, n_resamples=19, block_length=block_length)


def test_studentized_draws_match_separate_hac_inference_on_each_sample(returns):
    indices = stationary_indices(len(returns), 41, block_length=6, seed=99)
    centered = returns - returns.mean(axis=0)
    expected = np.array(
        [infer_mean(centered[index], method="hac", lags=7).statistic for index in indices]
    )
    observed = stationary_bootstrap_statistics(
        returns,
        n_resamples=41,
        block_length=6,
        lags=7,
        seed=99,
        batch_size=9,
        column_batch_size=2,
    )
    np.testing.assert_allclose(observed, expected, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("batch_size, column_batch_size", [(1, 1), (7, 2), (32, 8)])
def test_studentized_draws_do_not_depend_on_computation_chunks(
    returns, batch_size, column_batch_size
):
    kwargs = dict(n_resamples=47, block_length=6, lags=5, seed=319)
    expected = stationary_bootstrap_statistics(
        returns, batch_size=47, column_batch_size=4, **kwargs
    )
    observed = stationary_bootstrap_statistics(
        returns, batch_size=batch_size, column_batch_size=column_batch_size, **kwargs
    )
    np.testing.assert_allclose(observed, expected, rtol=1e-13, atol=1e-13)


def test_studentized_draws_preserve_shared_rows_units_location_and_column_order(returns):
    kwargs = dict(n_resamples=47, block_length=6, lags=5, seed=319)
    expected = stationary_bootstrap_statistics(returns, **kwargs)
    scales = np.array([0.01, 100, 2, 7])
    locations = np.array([0.5, -2, 7, -0.3])
    order = np.array([2, 0, 3, 1])
    observed = stationary_bootstrap_statistics((returns * scales + locations)[:, order], **kwargs)
    np.testing.assert_allclose(observed, expected[:, order], rtol=1e-12, atol=1e-12)
    duplicates = np.column_stack([returns[:, 0], returns[:, 0], 3 * returns[:, 0] - 2])
    duplicate_draws = stationary_bootstrap_statistics(duplicates, **kwargs)
    np.testing.assert_allclose(
        duplicate_draws, np.repeat(duplicate_draws[:, :1], 3, axis=1), atol=1e-13
    )


def test_degenerate_studentized_draw_raises_instead_of_being_removed(returns, monkeypatch):
    monkeypatch.setattr(
        "strategy_inference.bootstrap._indices",
        lambda n_obs, n_resamples, *_args: np.zeros((n_resamples, n_obs), dtype=int),
    )
    with pytest.raises(ValueError, match="resampled HAC variance"):
        stationary_bootstrap_statistics(returns, n_resamples=19)


@pytest.mark.parametrize("n_obs", [8, 9])
@pytest.mark.parametrize("block_fraction", [0, 0.25, 1])
def test_exact_conditional_mean_variance_matches_markov_chain_pair_expectations(
    n_obs, block_fraction
):
    values = np.random.default_rng(108).normal(size=(n_obs, 3))
    block_length = max(1, block_fraction * n_obs)
    centered = values - values.mean(axis=0)
    continuation = np.zeros((n_obs, n_obs))
    continuation[np.arange(n_obs), (np.arange(n_obs) + 1) % n_obs] = 1
    transition = (1 - 1 / block_length) * continuation + np.ones((n_obs, n_obs)) / (
        block_length * n_obs
    )
    expected = np.sum(centered**2, axis=0) / n_obs**2
    lag_transition = np.eye(n_obs)
    for lag in range(1, n_obs):
        lag_transition = lag_transition @ transition
        covariance = np.sum(centered * (lag_transition @ centered), axis=0) / n_obs
        expected += 2 * (n_obs - lag) * covariance / n_obs**2
    observed = stationary_mean_variance(values, block_length=block_length)
    np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-14)
    assert np.all(observed > 0)


def test_conditional_iid_bootstrap_variance_uses_population_sample_divisor(returns):
    expected = returns.var(axis=0, ddof=0) / len(returns)
    np.testing.assert_allclose(stationary_mean_variance(returns, block_length=1), expected)


def test_conditional_bootstrap_variance_matches_sampler_second_moment():
    values = np.random.default_rng(491).normal(size=(17, 2))
    expected = stationary_mean_variance(values, block_length=4)
    draws = stationary_bootstrap_means(values, n_resamples=40000, block_length=4, seed=72)
    np.testing.assert_allclose(draws.var(axis=0, ddof=1), expected, rtol=0.04, atol=0)


def test_conditional_variance_transforms_with_units_and_location(returns):
    scales = np.array([0.01, 100, 2, 7])
    expected = stationary_mean_variance(returns, block_length=7)
    observed = stationary_mean_variance(returns * scales + 3, block_length=7)
    np.testing.assert_allclose(observed, expected * scales**2, rtol=1e-12)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_resamples": 0},
        {"batch_size": 0},
        {"column_batch_size": 0},
        {"lags": -1},
        {"lags": 67},
        {"lags": 0.5},
        {"block_length": 0},
        {"block_length": "3"},
    ],
)
def test_invalid_studentized_resampling_parameters_raise_value_error(returns, kwargs):
    with pytest.raises(ValueError):
        stationary_bootstrap_statistics(returns, **kwargs)


@pytest.mark.parametrize("block_length", [0, 68, "3", 3 + 0j])
def test_invalid_conditional_variance_parameters_raise_value_error(returns, block_length):
    with pytest.raises(ValueError):
        stationary_mean_variance(returns, block_length=block_length)
