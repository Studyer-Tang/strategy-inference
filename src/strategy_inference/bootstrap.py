"""A joint stationary bootstrap with bounded intermediate memory."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import as_returns, positive_integer
from .inference import default_lags


def default_block_length(n_obs: int) -> int:
    """Prespecified heuristic expected block length, ceil(2 T**(1/3))."""
    n_obs = positive_integer(n_obs, "n_obs", 8)
    return min(int(np.ceil(2 * n_obs ** (1 / 3))), n_obs)


def _block_length(value: float, n_obs: int) -> float:
    if (
        isinstance(value, (bool, np.bool_, str, bytes))
        or not np.isscalar(value)
        or not np.isrealobj(value)
    ):
        raise ValueError("block_length must lie in [1, T].")
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("block_length must lie in [1, T].") from exc
    if not np.isfinite(result) or not 1 <= result <= n_obs:
        raise ValueError("block_length must lie in [1, T].")
    return result


def _indices(
    n_obs: int, n_resamples: int, block_length: float, rng: np.random.Generator
) -> NDArray[np.int64]:
    # Interleaving the two uniforms at each time keeps the random stream the same
    # when batches change. The first observation always starts a new block.
    draws = rng.random((n_resamples, n_obs, 2))
    restarts = draws[:, :, 0] < 1 / block_length
    restarts[:, 0] = True
    starts = (draws[:, :, 1] * n_obs).astype(np.int64)
    time = np.arange(n_obs, dtype=np.int64)[None, :]
    last_restart = np.maximum.accumulate(np.where(restarts, time, 0), axis=1)
    block_start = np.take_along_axis(starts, last_restart, axis=1)
    return (block_start + time - last_restart) % n_obs


def stationary_indices(
    n_obs: int,
    n_resamples: int,
    block_length: float,
    seed: int | np.random.Generator | np.random.SeedSequence = 0,
) -> NDArray[np.int64]:
    """Return circular row indices with geometric block lengths (mean L)."""
    n_obs = positive_integer(n_obs, "n_obs", 8)
    n_resamples = positive_integer(n_resamples, "n_resamples")
    block_length = _block_length(block_length, n_obs)
    return _indices(n_obs, n_resamples, block_length, np.random.default_rng(seed))


def stationary_bootstrap_means(
    returns: ArrayLike,
    *,
    n_resamples: int = 999,
    block_length: float | None = None,
    seed: int | np.random.Generator | np.random.SeedSequence = 0,
    batch_size: int = 128,
    center: bool = False,
) -> NDArray[np.float64]:
    """Return B x K resampled means; all columns share the same sampled rows.

    Row counts multiply the data matrix instead of allocating a B x T x K
    sampled array. Intermediate storage is O(batch_size * T), plus B x K output.
    center=True subtracts each sample column mean before resampling.
    """
    data = as_returns(returns)
    n_obs, n_strategies = data.shape
    n_resamples = positive_integer(n_resamples, "n_resamples")
    batch_size = positive_integer(batch_size, "batch_size")
    block_length = default_block_length(n_obs) if block_length is None else block_length
    block_length = _block_length(block_length, n_obs)
    rng = np.random.default_rng(seed)
    values = data - data.mean(axis=0) if center else data
    output = np.empty((n_resamples, n_strategies), dtype=np.float64)
    for first in range(0, n_resamples, batch_size):
        last = min(first + batch_size, n_resamples)
        indices = _indices(n_obs, last - first, block_length, rng)
        offsets = n_obs * np.arange(last - first, dtype=np.int64)[:, None]
        counts = np.bincount((indices + offsets).ravel(), minlength=(last - first) * n_obs)
        counts = counts.reshape(last - first, n_obs)
        output[first:last] = counts @ values / n_obs
    return output


def stationary_bootstrap_statistics(
    returns: ArrayLike,
    *,
    n_resamples: int = 999,
    block_length: float | None = None,
    lags: int | None = None,
    seed: int | np.random.Generator | np.random.SeedSequence = 0,
    batch_size: int = 32,
    column_batch_size: int = 8,
) -> NDArray[np.float64]:
    """Centered stationary-bootstrap t statistics with a new HAC scale per draw.

    The original and resampled statistics must use the same Bartlett lag count.
    All columns share row indices. Time-series arrays are materialized only for
    a bounded batch of draws and columns, not for the entire B x T x K tensor.
    Degenerate draws raise an error rather than being discarded or regularized.
    This construction is asymptotic, not a finite-sample calibration guarantee.
    """
    data = as_returns(returns)
    n_obs, n_strategies = data.shape
    n_resamples = positive_integer(n_resamples, "n_resamples")
    batch_size = positive_integer(batch_size, "batch_size")
    column_batch_size = positive_integer(column_batch_size, "column_batch_size")
    lags = default_lags(n_obs) if lags is None else positive_integer(lags, "lags", 0)
    if lags > n_obs - 2:
        raise ValueError("lags must be <= T - 2.")
    block_length = default_block_length(n_obs) if block_length is None else block_length
    block_length = _block_length(block_length, n_obs)
    rng = np.random.default_rng(seed)
    values = data - data.mean(axis=0)
    output = np.empty((n_resamples, n_strategies), dtype=np.float64)
    for first in range(0, n_resamples, batch_size):
        last = min(first + batch_size, n_resamples)
        indices = _indices(n_obs, last - first, block_length, rng)
        for column in range(0, n_strategies, column_batch_size):
            end = min(column + column_batch_size, n_strategies)
            sample = values[:, column:end][indices]
            if np.any(sample.max(axis=1) == sample.min(axis=1)):
                raise ValueError("A resampled HAC variance is undefined for a constant column.")
            mean = sample.mean(axis=1)
            sample -= mean[:, None, :]
            variance = np.einsum("btk,btk->bk", sample, sample) / n_obs
            for lag in range(1, lags + 1):
                covariance = np.einsum("btk,btk->bk", sample[:, lag:], sample[:, :-lag]) / n_obs
                variance += 2 * (1 - lag / (lags + 1)) * covariance
            if not np.isfinite(variance).all() or np.any(variance <= 0):
                raise ValueError("A resampled HAC variance is not positive and finite.")
            standard_error = np.sqrt(variance / n_obs)
            if not np.isfinite(standard_error).all() or np.any(standard_error <= 0):
                raise ValueError(
                    "A resampled HAC standard error is outside the positive finite float range."
                )
            statistics = mean / standard_error
            if not np.isfinite(statistics).all():
                raise ValueError("A resampled HAC statistic is outside the finite float range.")
            output[first:last, column:end] = statistics
    return output


def stationary_mean_variance(
    returns: ArrayLike, *, block_length: float | None = None
) -> NDArray[np.float64]:
    """Exact conditional variance of each stationary-bootstrap sample mean.

    Uses circular sample autocovariances and geometric survival probabilities,
    including the finite-T pair-count factor. This diagnoses the bootstrap
    distribution; it is not the unknown population variance of the mean.
    """
    data = as_returns(returns)
    n_obs = len(data)
    block_length = default_block_length(n_obs) if block_length is None else block_length
    block_length = _block_length(block_length, n_obs)
    centered = data - data.mean(axis=0)
    try:
        with np.errstate(over="raise", invalid="raise"):
            spectrum = np.fft.rfft(centered, axis=0)
            circular = np.fft.irfft(spectrum * spectrum.conj(), n=n_obs, axis=0) / n_obs
            lag = np.arange(1, n_obs)
            weights = (1 - lag / n_obs) * (1 - 1 / block_length) ** lag
            variance = (circular[0] + 2 * weights @ circular[1:]) / n_obs
    except FloatingPointError as exc:
        raise ValueError(
            "The conditional bootstrap mean variance is outside the finite float range."
        ) from exc
    if not np.isfinite(variance).all() or np.any(variance <= 0):
        raise ValueError(
            "The conditional bootstrap mean variance is outside the positive finite float range."
        )
    return variance
