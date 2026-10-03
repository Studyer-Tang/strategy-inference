"""Optimized chunks retain the v0.4 draw stream and Bartlett definition."""

import numpy as np
import pytest

from strategy_inference import infer_mean, long_run_variance
from strategy_inference.bootstrap import (
    stationary_bootstrap_means,
    stationary_bootstrap_statistics,
    stationary_indices,
)


def _v04_indices(n_obs, n_resamples, block_length, rng):
    """Frozen v0.4 indexing recipe, before temporary-storage changes."""
    draws = rng.random((n_resamples, n_obs, 2))
    restarts = draws[:, :, 0] < 1 / block_length
    restarts[:, 0] = True
    starts = (draws[:, :, 1] * n_obs).astype(np.int64)
    time = np.arange(n_obs, dtype=np.int64)[None, :]
    last_restart = np.maximum.accumulate(np.where(restarts, time, 0), axis=1)
    block_start = np.take_along_axis(starts, last_restart, axis=1)
    return (block_start + time - last_restart) % n_obs


def _v04_statistics(values, indices, lags):
    """Frozen T-first reductions; deliberately unlike the optimized layout."""
    n_obs = len(values)
    sample = (values - values.mean(axis=0))[indices]
    mean = sample.mean(axis=1)
    sample -= mean[:, None, :]
    variance = np.einsum("btk,btk->bk", sample, sample) / n_obs
    for lag in range(1, lags + 1):
        covariance = np.einsum("btk,btk->bk", sample[:, lag:], sample[:, :-lag]) / n_obs
        variance += 2 * (1 - lag / (lags + 1)) * covariance
    return mean / np.sqrt(variance / n_obs)


@pytest.mark.parametrize("n_obs", [8, 9, 16, 67, 128])
@pytest.mark.parametrize("block_fraction", [0, 0.17, 1])
def test_indices_and_generator_position_match_v04(n_obs, block_fraction):
    block_length = max(1, n_obs * block_fraction)
    old_rng = np.random.default_rng(np.random.SeedSequence([71, 903]))
    new_rng = np.random.default_rng(np.random.SeedSequence([71, 903]))
    expected = _v04_indices(n_obs, 37, block_length, old_rng)
    observed = stationary_indices(n_obs, 37, block_length, seed=new_rng)
    np.testing.assert_array_equal(observed, expected)
    np.testing.assert_array_equal(new_rng.random(11), old_rng.random(11))


@pytest.mark.parametrize("lags", [0, 1, 7, 29])
@pytest.mark.parametrize("shape", [(31, 1), (67, 11), (128, 20)])
def test_time_contiguous_statistics_match_v04_and_preserve_generator(shape, lags):
    values = np.random.default_rng(317).normal(size=shape)
    old_rng = np.random.default_rng(7183)
    new_rng = np.random.default_rng(7183)
    indices = _v04_indices(len(values), 41, 7.3, old_rng)
    expected = _v04_statistics(values, indices, lags)
    observed = stationary_bootstrap_statistics(
        values,
        n_resamples=41,
        block_length=7.3,
        lags=lags,
        seed=new_rng,
        batch_size=13,
        column_batch_size=4,
    )
    np.testing.assert_allclose(observed, expected, rtol=2e-13, atol=2e-13)
    np.testing.assert_array_equal(new_rng.random(11), old_rng.random(11))


def test_oversized_requested_chunks_obey_work_budget_and_retain_v04(monkeypatch):
    import strategy_inference.bootstrap as bootstrap

    values = np.random.default_rng(401).normal(size=(128, 48))
    budget = 1024 * 1024
    monkeypatch.setattr(bootstrap, "_MAX_BOOTSTRAP_WORK_BYTES", budget)
    observed_shapes = []
    original_contiguous = np.ascontiguousarray

    def record_sample_shape(array, *args, **kwargs):
        if array.ndim == 3:
            observed_shapes.append(array.shape)
        return original_contiguous(array, *args, **kwargs)

    monkeypatch.setattr(bootstrap.np, "ascontiguousarray", record_sample_shape)
    indices = _v04_indices(128, 101, 7, np.random.default_rng(611))
    expected = _v04_statistics(values, indices, 6)
    observed = stationary_bootstrap_statistics(
        values,
        n_resamples=101,
        block_length=7,
        lags=6,
        seed=611,
        batch_size=10000,
        column_batch_size=10000,
    )
    np.testing.assert_allclose(observed, expected, rtol=2e-13, atol=2e-13)
    assert len(observed_shapes) > 1
    assert all(
        draws * time * (16 * columns + 32) <= budget for draws, columns, time in observed_shapes
    )
    observed_means = stationary_bootstrap_means(
        values, n_resamples=101, block_length=7, seed=611, batch_size=10000, center=True
    )
    np.testing.assert_allclose(
        observed_means,
        (values - values.mean(axis=0))[indices].mean(axis=1),
        rtol=2e-13,
        atol=2e-13,
    )


def test_hac_inference_validates_once_and_matches_independent_scalar_loops(monkeypatch):
    import strategy_inference.inference as inference

    values = np.random.default_rng(701).normal(size=(71, 5)) + 0.13
    expected = []
    lags = 13
    for column in values.T:
        mean = sum(float(value) for value in column) / len(column)
        centered = [float(value) - mean for value in column]
        variance = sum(value * value for value in centered) / len(centered)
        for lag in range(1, lags + 1):
            covariance = sum(
                centered[i] * centered[i - lag] for i in range(lag, len(centered))
            ) / len(centered)
            variance += 2 * (1 - lag / (lags + 1)) * covariance
        expected.append(variance)
    np.testing.assert_allclose(long_run_variance(values, lags), expected, rtol=2e-13)
    calls = []
    original_validator = inference.as_returns

    def record_validation(array):
        calls.append(array.shape)
        return original_validator(array)

    monkeypatch.setattr(inference, "as_returns", record_validation)
    result = infer_mean(values, lags=lags)
    np.testing.assert_allclose(result.standard_error**2, np.array(expected) / len(values))
    assert calls == [values.shape]
