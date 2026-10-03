"""Pointwise mean inference using an IID t test or a Bartlett HAC estimate."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import as_returns, positive_integer, probability


@dataclass(frozen=True)
class MeanInference:
    sample_size: int
    n_strategies: int
    mean: NDArray[np.float64]
    standard_error: NDArray[np.float64]
    statistic: NDArray[np.float64]
    pvalue: NDArray[np.float64]
    ci_low: NDArray[np.float64]
    ci_high: NDArray[np.float64]
    method: str
    lags: int


def default_lags(n_obs: int) -> int:
    """A sample-size rule, not an estimated optimal bandwidth."""
    n_obs = positive_integer(n_obs, "n_obs", 8)
    return min(int(np.floor(4 * (n_obs / 100) ** (2 / 9))), n_obs - 2)


def _lag_count(n_obs: int, lags: int | None) -> int:
    lags = default_lags(n_obs) if lags is None else positive_integer(lags, "lags", 0)
    if lags > n_obs - 2:
        raise ValueError("lags must be <= T - 2.")
    return lags


def _bartlett_variance(centered: NDArray[np.float64], lags: int) -> NDArray[np.float64]:
    """HAC kernel for already validated, centered T x K data and lag count."""
    n_obs = len(centered)
    variance = np.einsum("ij,ij->j", centered, centered) / n_obs
    for lag in range(1, lags + 1):
        covariance = np.einsum("ij,ij->j", centered[lag:], centered[:-lag]) / n_obs
        variance += 2 * (1 - lag / (lags + 1)) * covariance
    if not np.isfinite(variance).all() or np.any(variance <= 0):
        raise ValueError("The HAC long-run variance is not positive and finite.")
    return variance


def long_run_variance(returns: ArrayLike, lags: int | None = None) -> NDArray[np.float64]:
    """Bartlett HAC: gamma[0] + 2 sum_h (1 - h/(lags+1)) gamma[h].

    Every sample autocovariance has divisor T. No degrees-of-freedom correction
    is used. This estimates the asymptotic variance of sqrt(T) times the mean.
    """
    data = as_returns(returns)
    used_lags = _lag_count(len(data), lags)
    return _bartlett_variance(data - data.mean(axis=0), used_lags)


def infer_mean(
    returns: ArrayLike,
    *,
    method: str = "hac",
    lags: int | None = None,
    confidence: float = 0.95,
) -> MeanInference:
    """Test H0: mean <= 0, column by column; intervals are two-sided and pointwise.

    These p values do not account for choosing a candidate after seeing the data.
    The IID t test is finite-sample exact only for independent Gaussian data.
    """
    data = as_returns(returns)
    return _infer_mean(data, method=method, lags=lags, confidence=confidence)


def _infer_mean(
    data: NDArray[np.float64], *, method: str, lags: int | None, confidence: float
) -> MeanInference:
    """Compute inference for a matrix already checked by ``as_returns``."""
    confidence = probability(confidence, "confidence")
    n_obs, n_strategies = data.shape
    mean = data.mean(axis=0)
    if method == "iid":
        if lags is not None:
            raise ValueError("lags is only used by method='hac'.")
        from scipy.stats import t

        standard_error = data.std(axis=0, ddof=1) / np.sqrt(n_obs)
        distribution = t(n_obs - 1)
        used_lags = 0
    elif method == "hac":
        from scipy.stats import norm

        used_lags = _lag_count(n_obs, lags)
        standard_error = np.sqrt(_bartlett_variance(data - mean, used_lags) / n_obs)
        distribution = norm
    else:
        raise ValueError("method must be 'iid' or 'hac'.")
    if not np.isfinite(standard_error).all() or np.any(standard_error <= 0):
        raise ValueError("The standard error is outside the positive finite float range.")
    statistic = mean / standard_error
    if not np.isfinite(statistic).all():
        raise ValueError("The mean statistic is outside the finite float range.")
    radius = distribution.ppf((1 + confidence) / 2) * standard_error
    return MeanInference(
        sample_size=n_obs,
        n_strategies=n_strategies,
        mean=mean,
        standard_error=standard_error,
        statistic=statistic,
        pvalue=distribution.sf(statistic),
        ci_low=mean - radius,
        ci_high=mean + radius,
        method=method,
        lags=used_lags,
    )
