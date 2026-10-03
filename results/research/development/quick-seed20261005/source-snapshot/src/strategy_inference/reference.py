"""Known-covariance Gaussian references for simulation checks only.

These functions use the generating parameters, not estimates from a return
sample. Their exact Gaussian reference law does not apply to unknown-covariance
financial data or to the heavy-tailed and GARCH simulation processes.
"""

import math
from functools import lru_cache

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import log_ndtr, ndtri_exp
from scipy.stats import norm

from ._validation import positive_integer
from ._validation import probability as validate_probability


def _finite_real(value: float, name: str) -> float:
    if (
        isinstance(value, (bool, np.bool_, str, bytes))
        or not np.isscalar(value)
        or not np.isrealobj(value)
    ):
        raise ValueError(f"{name} must be a finite real number.")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real number.") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number.")
    return value


def gaussian_ar_mean_variance(n_obs: int, phi: float, sigma: float = 1.0) -> float:
    """Finite-T variance of a stationary Gaussian AR(1) sample mean.

    ``sigma`` is the marginal standard deviation, not the innovation scale.
    This equals sigma²/T [1 + 2 sum_(h=1)^(T-1) (1-h/T) phi^h]. The computation
    uses nonnegative innovation-weight squares to avoid cancellation for phi
    close to -1. No asymptotic long-run-variance approximation is substituted.
    """
    n_obs = positive_integer(n_obs, "n_obs")
    phi = _finite_real(phi, "phi")
    sigma = _finite_real(sigma, "sigma")
    if not -1 < phi < 1:
        raise ValueError("phi must lie strictly between -1 and 1.")
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    if phi == 0:
        factor = float(n_obs)
    else:
        periods = np.arange(1, n_obs + 1)
        log_power = periods * math.log(abs(phi))
        numerators = -np.expm1(log_power)
        if phi < 0:
            odd = periods % 2 == 1
            numerators[odd] = 1 + np.exp(log_power[odd])
        weights = numerators / (1 - phi)
        initial_weight = phi * weights[-1]
        factor = initial_weight**2 + (1 - phi) * (1 + phi) * math.fsum(weights**2)
    standard_error = sigma * (math.sqrt(factor) / n_obs)
    variance = standard_error * standard_error
    if not math.isfinite(variance) or variance <= 0:
        raise ValueError("The mean variance is outside the positive finite float range.")
    return variance


def _maximum_parameters(n_strategies: int, cross_corr: float) -> tuple[int, float]:
    n_strategies = positive_integer(n_strategies, "n_strategies")
    cross_corr = _finite_real(cross_corr, "cross_corr")
    if not 0 <= cross_corr <= 1:
        raise ValueError("cross_corr must lie in [0, 1].")
    return n_strategies, cross_corr


def _maximum_probability(
    value: float, n_strategies: int, cross_corr: float, *, upper: bool, tolerance: float
) -> float:
    if n_strategies == 1 or cross_corr == 1:
        return float(norm.sf(value) if upper else norm.cdf(value))
    if cross_corr == 0:
        log_probability = n_strategies * log_ndtr(value)
        return float(-np.expm1(log_probability) if upper else np.exp(log_probability))

    common_scale = math.sqrt(cross_corr)
    residual_scale = math.sqrt(1 - cross_corr)

    def integrand(common):
        log_probability = n_strategies * log_ndtr((value - common_scale * common) / residual_scale)
        conditional = -math.expm1(log_probability) if upper else math.exp(log_probability)
        return conditional * math.exp(-common * common / 2) / math.sqrt(2 * math.pi)

    # Splitting the common-factor transition preserves accuracy as rho -> 1.
    # Normal-density landmarks avoid a very wide interval when rho -> 0.
    transition = value / common_scale
    width = 12 * residual_scale / common_scale
    points = [-math.inf, -8.0, 0.0, 8.0, math.inf]
    points.extend(
        point for point in (transition - width, transition, transition + width) if -8 < point < 8
    )
    points = sorted(set(points))
    areas = [
        quad(integrand, left, right, epsabs=tolerance / len(points), epsrel=1e-10, limit=200)[0]
        for left, right in zip(points[:-1], points[1:], strict=True)
    ]
    return min(1.0, max(0.0, math.fsum(areas)))


def equicorrelated_max_cdf(value: float, n_strategies: int, cross_corr: float) -> float:
    """CDF of max_j Z_j for standard Gaussian Z with common correlation rho.

    For 0 < rho < 1, Z_j = sqrt(rho) U + sqrt(1-rho) epsilon_j and the CDF is
    the one-dimensional integral E[Phi((value-sqrt(rho)U)/sqrt(1-rho))**K].
    Numerical quadrature has an absolute tolerance of 1e-12. Tiny probabilities
    do not have guaranteed relative accuracy. This is a model reference, not
    an estimated test.
    """
    value = _finite_real(value, "value")
    n_strategies, cross_corr = _maximum_parameters(n_strategies, cross_corr)
    return _maximum_probability(value, n_strategies, cross_corr, upper=False, tolerance=1e-12)


def equicorrelated_max_tail(value: float, n_strategies: int, cross_corr: float) -> float:
    """One-sided Gaussian maximum tail, without subtracting a CDF from 1.

    Quadrature is intended for ordinary test levels; relative accuracy in
    extremely small tails is not guaranteed.
    """
    value = _finite_real(value, "value")
    n_strategies, cross_corr = _maximum_parameters(n_strategies, cross_corr)
    union_bound = min(1.0, n_strategies * norm.sf(value))
    tolerance = max(np.finfo(float).tiny, min(1e-12, union_bound * 1e-9))
    return _maximum_probability(value, n_strategies, cross_corr, upper=True, tolerance=tolerance)


@lru_cache(maxsize=256)
def _maximum_quantile(probability: float, n_strategies: int, cross_corr: float) -> float:
    lower = float(norm.ppf(probability))
    upper = float(ndtri_exp(math.log(probability) / n_strategies))
    if n_strategies == 1 or cross_corr == 1:
        return lower
    if cross_corr == 0:
        return upper
    # Jensen's inequality brackets this quantile between the common and
    # independent Gaussian limits. Invert the smaller probability for stability.
    use_tail = probability > 0.5
    target = 1 - probability if use_tail else probability
    tolerance = max(np.finfo(float).tiny, min(1e-12, target * 1e-10))

    def residual(value):
        return (
            _maximum_probability(
                value, n_strategies, cross_corr, upper=use_tail, tolerance=tolerance
            )
            - target
        )

    return float(brentq(residual, lower, upper, xtol=1e-11, rtol=1e-12))


def equicorrelated_max_quantile(probability: float, n_strategies: int, cross_corr: float) -> float:
    """Quantile of the known-correlation Gaussian maximum; inputs are validated.

    At level alpha, use probability=1-alpha. Repeated simulation settings
    reuse the cached deterministic critical value. This function does not
    estimate covariance or choose an inferential method from observed returns.
    The numerical common-factor case supports probabilities in [1e-12, 1-1e-12].
    More extreme requests are rejected because unscaled quadrature can miss a
    distant density peak. Analytic K=1 and rho=0/1 cases have no such restriction.
    """
    probability = validate_probability(probability, "probability")
    n_strategies, cross_corr = _maximum_parameters(n_strategies, cross_corr)
    if n_strategies > 1 and 0 < cross_corr < 1 and not 1e-12 <= probability <= 1 - 1e-12:
        raise ValueError(
            "Numerical Gaussian maximum quantiles require probability in [1e-12, 1-1e-12]."
        )
    return _maximum_quantile(probability, n_strategies, cross_corr)
