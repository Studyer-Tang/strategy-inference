"""A joint stationary bootstrap with bounded intermediate memory."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import as_returns, positive_integer


def default_block_length(n_obs: int) -> int:
    """Prespecified heuristic expected block length, ceil(2 T**(1/3))."""
    n_obs = positive_integer(n_obs, "n_obs", 8)
    return min(int(np.ceil(2 * n_obs ** (1 / 3))), n_obs)


def _block_length(value: float, n_obs: int) -> float:
    if not np.isscalar(value) or not np.isfinite(value) or not 1 <= value <= n_obs:
        raise ValueError("block_length must lie in [1, T].")
    return float(value)


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
