"""Research helpers for Gaussian AR replay; not a calibrated public audit API.

The fitted generator assumes a common AR coefficient and nonnegative
equicorrelation. Replaying the statistic at estimated parameters does not give
the finite-sample guarantee of simulation at the true parameters.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import as_returns, positive_integer


def _coefficient(value: object) -> float:
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "iuf":
        raise ValueError("phi must be a real scalar strictly between -1 and 1.")
    phi = float(raw)
    if not np.isfinite(phi) or abs(phi) >= 1:
        raise ValueError("phi must lie strictly between -1 and 1.")
    return phi


def _bartlett_mass(phi: NDArray[np.float64], size: int) -> NDArray[np.float64]:
    """Return size * Var(mean) for unit marginal AR variance, in O(phi.size).

    For positive coefficients near one, the closed form subtracts two large
    terms. Its expansion in d = 1 - phi is
    n + (2/n) sum_{r>=1} binom(n+1,r+2) (-d)**r.
    At n*d <= 1/2, the remainder after 18 terms is at most
    2*n*(1/2)**19/(21!*(1-1/44)), below double precision. For negative coefficients the
    closed form is a sum of nonnegative terms, with parity handled explicitly.
    """
    if size == 1:
        return np.ones_like(phi)
    values = np.asarray(phi, dtype=float)
    result = np.ones_like(values)
    positive = values > 0
    near = positive & (size * (1 - values) <= 0.5)
    d = 1 - values[near]
    total = np.full_like(d, float(size))
    term = -(size - 1) * (size + 1) * d / 3
    for order in range(1, 19):
        total += term
        term *= -d * (size - order - 1) / (order + 3)
    result[near] = total
    ordinary = positive & ~near
    p = values[ordinary]
    d = 1 - p
    difference = -np.expm1(size * np.log(p))
    result[ordinary] = (1 + p) / d - 2 * p * difference / (size * d * d)
    negative = values < 0
    q = -values[negative]
    log_power = size * np.log(q)
    difference = -np.expm1(log_power) if size % 2 == 0 else 1 + np.exp(log_power)
    result[negative] = (1 - q) / (1 + q) + 2 * q * difference / (size * (1 + q) ** 2)
    if not np.isfinite(result).all() or np.any(result <= 0):
        raise ValueError("The AR Bartlett mass is not positive and finite.")
    return result


@dataclass(frozen=True)
class ReplayFit:
    phi: float
    rho: float
    raw_rho: float
    projected: bool
    n_obs: int
    k: int


def fit_equicorrelated_ar1(data: ArrayLike) -> ReplayFit:
    """Fit the restricted generator in O(T*K), without a K-by-K matrix.

    Common phi is the mean of the centered, divisor-T lag-one sample
    correlations. Raw rho is the mean off-diagonal Pearson correlation of the
    return columns, projected onto [0, 1] for the common-factor generator.
    This marginal correlation fit is not an innovation-covariance estimator.
    The projection and its unprojected value are returned explicitly.
    """
    values = as_returns(data)
    n_obs, k = values.shape
    centered = values - values.mean(axis=0)
    squares = np.einsum("tk,tk->k", centered, centered)
    phi = _coefficient(np.mean(np.einsum("tk,tk->k", centered[1:], centered[:-1]) / squares))
    if k == 1:
        return ReplayFit(phi, 0.0, 0.0, False, n_obs, k)
    standardized = centered / np.sqrt(squares)
    row_sum = standardized.sum(axis=1)
    raw_rho = float((np.dot(row_sum, row_sum) - k) / (k * (k - 1)))
    if not np.isfinite(raw_rho):
        raise ValueError("The fitted cross-correlation is not finite.")
    rho = float(np.clip(raw_rho, 0.0, 1.0))
    return ReplayFit(phi, rho, raw_rho, rho != raw_rho, n_obs, k)


def tail_statistics(
    draws: ArrayLike,
    lags: int,
    frozen_factor: ArrayLike | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Batched finite-T tail statistics, factors and fitted columnwise phi.

    Input shape is (B, T, K); every draw recomputes its mean, centered Bartlett
    HAC and lag-one coefficient. ``frozen_factor`` fixes only the K correction
    factors, for the frozen-parameter ablation. Invalid draws abort the batch;
    none are dropped, clipped, or assigned a replacement statistic.
    """
    try:
        raw = np.asarray(draws)
        if raw.dtype.kind not in "iuf":
            raise ValueError("Draws must contain real numeric values.")
        values = np.asarray(draws, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Draws must be a rectangular real numeric array.") from exc
    if values.ndim != 3 or values.shape[0] < 1 or values.shape[1] < 8 or values.shape[2] < 1:
        raise ValueError("Draws must have shape (B, T, K), with B >= 1, T >= 8 and K >= 1.")
    if not np.isfinite(values).all():
        raise ValueError("Draws contain non-finite values; no draws are dropped.")
    _, n_obs, k = values.shape
    lags = positive_integer(lags, "lags", 0)
    if lags > n_obs - 2:
        raise ValueError("lags must be <= T - 2.")
    mean = values.mean(axis=1)
    centered = values - mean[:, None, :]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        squares = np.einsum("btk,btk->bk", centered, centered)
        fitted_phi = np.einsum("btk,btk->bk", centered[:, 1:], centered[:, :-1]) / squares
    if (
        not np.isfinite(squares).all()
        or np.any(squares <= 0)
        or not np.isfinite(fitted_phi).all()
        or np.any(np.abs(fitted_phi) >= 1)
    ):
        raise ValueError("Each draw and candidate must have positive finite variance and valid phi.")
    hac = squares / n_obs
    for lag in range(1, lags + 1):
        covariance = np.einsum("btk,btk->bk", centered[:, lag:], centered[:, :-lag]) / n_obs
        hac += 2 * (1 - lag / (lags + 1)) * covariance
    if not np.isfinite(hac).all() or np.any(hac <= 0):
        raise ValueError("The replay HAC variance is not positive and finite.")
    if frozen_factor is None:
        factors = _bartlett_mass(fitted_phi, n_obs) / _bartlett_mass(fitted_phi, lags + 1)
    else:
        raw_factor = np.asarray(frozen_factor)
        if raw_factor.dtype.kind not in "iuf" or raw_factor.shape != (k,):
            raise ValueError("frozen_factor must be a real array with one entry per candidate.")
        factor = np.asarray(frozen_factor, dtype=float)
        if not np.isfinite(factor).all() or np.any(factor <= 0):
            raise ValueError("frozen_factor must be positive and finite.")
        factors = np.broadcast_to(factor, fitted_phi.shape).copy()
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        variance = hac * factors
        statistics = mean / np.sqrt(variance / n_obs)
    if not np.isfinite(variance).all() or np.any(variance <= 0) or not np.isfinite(statistics).all():
        raise ValueError("The replay scale or statistic is outside the finite float range.")
    return statistics, factors, fitted_phi


def known_phi_gls_t(data: ArrayLike, phi: float) -> NDArray[np.float64]:
    """Columnwise exact t_(T-1) under Gaussian AR with known temporal phi.

    Innovation whitening uses the stationary first observation. The whitened
    mean direction is projected out before estimating the column variances.
    Cross-column covariance is unrestricted and need not be estimated. A joint
    max reference still requires calibration; marginal t plus Holm/Bonferroni
    provides strong FWER control under these stated model conditions.
    """
    values = as_returns(data)
    phi = _coefficient(phi)
    n_obs = len(values)
    first = np.sqrt((1 - phi) * (1 + phi))
    innovations = np.empty_like(values)
    innovations[0] = first * values[0]
    innovations[1:] = values[1:] - phi * values[:-1]
    direction = np.full(n_obs, 1 - phi)
    direction[0] = first
    direction /= np.hypot(first, np.sqrt(n_obs - 1) * (1 - phi))
    score = direction @ innovations
    residual = innovations - direction[:, None] * score
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        scale = np.sqrt(np.einsum("tk,tk->k", residual, residual) / (n_obs - 1))
        statistic = score / scale
    if not np.isfinite(scale).all() or np.any(scale <= 0) or not np.isfinite(statistic).all():
        raise ValueError("The GLS residual scale or statistic is outside the finite float range.")
    return statistic


def replay_pvalue(observed_max: float, draw_max: ArrayLike) -> float:
    """Inclusive Monte Carlo rank p-value; plug-in replay is not finite-T exact."""
    raw_observed = np.asarray(observed_max)
    raw_draws = np.asarray(draw_max)
    if raw_observed.ndim != 0 or raw_observed.dtype.kind not in "iuf":
        raise ValueError("observed_max must be a finite real scalar.")
    if raw_draws.ndim != 1 or raw_draws.size < 1 or raw_draws.dtype.kind not in "iuf":
        raise ValueError("draw_max must be a nonempty one-dimensional real array.")
    observed = float(raw_observed)
    draws = np.asarray(draw_max, dtype=float)
    if not np.isfinite(observed) or not np.isfinite(draws).all():
        raise ValueError("All replay statistics must be finite; no draws are dropped.")
    return float((1 + np.count_nonzero(draws >= observed)) / (draws.size + 1))
