"""A joint stationary bootstrap with bounded intermediate memory."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import as_returns, positive_integer
from .inference import _lag_count

# Bound transient draw/index arrays even when callers request very large chunks.
# The input T x K matrix and required B x K output are separate from this budget.
_MAX_BOOTSTRAP_WORK_BYTES = 16 * 1024 * 1024


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
    del draws
    time = np.arange(n_obs, dtype=np.int64)[None, :]
    last_restart = np.where(restarts, time, 0)
    del restarts
    np.maximum.accumulate(last_restart, axis=1, out=last_restart)
    # Store start-time offsets so the final arithmetic needs no extra B x T arrays.
    starts -= time
    indices = starts[np.arange(n_resamples)[:, None], last_restart]
    indices += time
    # Power-of-two circular lengths permit exactly the same wrap without division.
    if n_obs & (n_obs - 1) == 0:
        np.bitwise_and(indices, n_obs - 1, out=indices)
    else:
        np.remainder(indices, n_obs, out=indices)
    return indices


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
    return _stationary_bootstrap_means(
        data,
        n_resamples=n_resamples,
        block_length=block_length,
        seed=seed,
        batch_size=batch_size,
        center=center,
    )


def _stationary_bootstrap_means(
    data: NDArray[np.float64],
    *,
    n_resamples: int,
    block_length: float | None,
    seed: int | np.random.Generator | np.random.SeedSequence,
    batch_size: int,
    center: bool,
) -> NDArray[np.float64]:
    """Resample a matrix already checked by ``as_returns``."""
    n_obs, n_strategies = data.shape
    n_resamples = positive_integer(n_resamples, "n_resamples")
    batch_size = positive_integer(batch_size, "batch_size")
    batch_size = min(batch_size, max(1, _MAX_BOOTSTRAP_WORK_BYTES // (40 * n_obs)))
    block_length = default_block_length(n_obs) if block_length is None else block_length
    block_length = _block_length(block_length, n_obs)
    rng = np.random.default_rng(seed)
    values = data - data.mean(axis=0) if center else data
    output = np.empty((n_resamples, n_strategies), dtype=np.float64)
    for first in range(0, n_resamples, batch_size):
        last = min(first + batch_size, n_resamples)
        indices = _indices(n_obs, last - first, block_length, rng)
        offsets = n_obs * np.arange(last - first, dtype=np.int64)[:, None]
        indices += offsets
        counts = np.bincount(indices.ravel(), minlength=(last - first) * n_obs)
        counts = counts.reshape(last - first, n_obs)
        output[first:last] = counts @ values / n_obs
        del indices, counts
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
    return _stationary_bootstrap_statistics(
        data,
        n_resamples=n_resamples,
        block_length=block_length,
        lags=lags,
        seed=seed,
        batch_size=batch_size,
        column_batch_size=column_batch_size,
    )


def _stationary_bootstrap_statistics(
    data: NDArray[np.float64],
    *,
    n_resamples: int,
    block_length: float | None,
    lags: int | None,
    seed: int | np.random.Generator | np.random.SeedSequence,
    batch_size: int,
    column_batch_size: int = 8,
) -> NDArray[np.float64]:
    """Studentize draws of a matrix already checked by ``as_returns``."""
    n_obs, n_strategies = data.shape
    n_resamples = positive_integer(n_resamples, "n_resamples")
    batch_size = positive_integer(batch_size, "batch_size")
    column_batch_size = positive_integer(column_batch_size, "column_batch_size")
    # The gather and its time-contiguous copy can briefly coexist. Reserve index
    # storage too; a single draw/column is the irreducible minimum for large T.
    column_batch_size = min(
        column_batch_size, n_strategies, max(1, _MAX_BOOTSTRAP_WORK_BYTES // (48 * n_obs))
    )
    batch_size = min(
        batch_size,
        max(1, _MAX_BOOTSTRAP_WORK_BYTES // (n_obs * (16 * column_batch_size + 32))),
    )
    lags = _lag_count(n_obs, lags)
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
            # Gather adjacent columns together, then make time the contiguous
            # axis. Repeated HAC/max/min reductions no longer stride through K.
            sample = np.ascontiguousarray(values[:, column:end][indices].transpose(0, 2, 1))
            if np.any(sample.max(axis=2) == sample.min(axis=2)):
                raise ValueError("A resampled HAC variance is undefined for a constant column.")
            mean = sample.mean(axis=2)
            sample -= mean[:, :, None]
            variance = np.einsum("bkt,bkt->bk", sample, sample) / n_obs
            for lag in range(1, lags + 1):
                covariance = (
                    np.einsum("bkt,bkt->bk", sample[:, :, lag:], sample[:, :, :-lag]) / n_obs
                )
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
            del sample
        del indices
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
