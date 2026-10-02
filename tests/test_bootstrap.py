import numpy as np
import pytest

from strategy_inference import stationary_bootstrap_means, stationary_indices


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
