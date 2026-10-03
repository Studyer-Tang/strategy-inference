"""Known-model finite-sample moments of the centered Bartlett HAC estimate.

These are Gaussian AR(1) references for simulation and estimator diagnostics.
They require the true AR coefficient and marginal standard deviation; they do
not supply an inferential correction when those quantities are unknown.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.linalg import toeplitz

from ._validation import positive_integer
from .inference import default_lags
from .reference import gaussian_ar_mean_variance

MAX_OBSERVATIONS = 2048


@dataclass(frozen=True)
class GaussianARHACMoments:
    """All variance levels use the units of the squared observations.

    ``variance`` is Var[vhat] and therefore has fourth-power units.
    ``target`` is T Var[mean], not Var[mean].
    """

    sample_size: int
    lags: int
    phi: float
    sigma: float
    expectation: float
    variance: float
    target: float
    population_truncated_lrv: float
    long_run_limit: float


def _finite_real(value: object, name: str) -> float:
    if (
        isinstance(value, (bool, np.bool_, str, bytes))
        or not np.isscalar(value)
        or not np.isrealobj(value)
    ):
        raise ValueError(f"{name} must be a finite real number.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite real number.") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number.")
    return result


def _rescale(value: float, sigma: float, power: int, name: str) -> float:
    """Avoid an overflowing sigma**power when the final moment is representable."""
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} is outside the positive finite float range.")
    if sigma == 1:
        return value
    try:
        result = value * sigma**power
    except OverflowError:
        result = math.inf
    if not math.isfinite(result) or result == 0:
        try:
            result = math.exp(math.log(value) + power * math.log(sigma))
        except OverflowError:
            result = math.inf
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} is outside the positive finite float range.")
    return result


def _truncated_factor(phi: float, lags: int) -> float:
    if lags == 0:
        return 1.0
    width = lags + 1
    if phi >= 0:
        return math.fsum([1.0] + [2 * (1 - lag / width) * phi**lag for lag in range(1, width)])
    # The two terms are nonnegative, unlike the alternating autocovariance sum.
    log_power = width * math.log(-phi)
    one_minus_power = -math.expm1(log_power) if width % 2 == 0 else 1 + math.exp(log_power)
    denominator = 1 - phi
    return (1 + phi) / denominator + ((-2 * phi / width) * one_minus_power / denominator**2)


def _centered_covariance(n_obs: int, phi: float) -> np.ndarray:
    distances = np.arange(n_obs)
    if phi > 0.5:
        # P 11' P = 0: subtract the common level before double centering. This
        # preserves the small residual covariance when phi is close to one.
        first_column = np.expm1(distances * math.log(phi))
    else:
        first_column = np.power(phi, distances, dtype=np.float64)
    covariance = toeplitz(first_column)
    row_mean = covariance.mean(axis=1)
    covariance -= row_mean[:, None]
    covariance -= row_mean[None, :]
    covariance += row_mean.mean()
    return covariance


def _quadratic_moments(covariance: np.ndarray, lags: int) -> tuple[float, float]:
    n_obs = len(covariance)
    trace = math.fsum(
        [float(np.trace(covariance))]
        + [
            2 * (1 - lag / (lags + 1)) * float(np.diagonal(covariance, lag).sum())
            for lag in range(1, lags + 1)
        ]
    )
    expectation = trace / n_obs
    if lags == 0:
        squared_norm = float(np.einsum("ij,ij->", covariance, covariance))
        return expectation, 2 * squared_norm / n_obs**2

    # W = R' R / (lags + 1), where R takes all zero-padded moving sums.
    # For H = P C P, tr((W H)^2) = ||R H R' / (lags + 1)||_F^2.
    # Rectangle sums form this Gram covariance without a cubic matrix product.
    prefix = np.zeros((n_obs + 1, n_obs + 1))
    np.cumsum(covariance, axis=0, out=prefix[1:, 1:])
    np.cumsum(prefix[1:, 1:], axis=1, out=prefix[1:, 1:])
    indices = np.arange(n_obs + lags)
    starts = np.maximum(0, indices - lags)
    ends = np.minimum(n_obs, indices + 1)
    squared_norms = []
    for first in range(0, len(indices), 32):
        last = min(first + 32, len(indices))
        row_starts = starts[first:last, None]
        row_ends = ends[first:last, None]
        block = prefix[row_ends, ends[None, :]]
        block -= prefix[row_starts, ends[None, :]]
        block -= prefix[row_ends, starts[None, :]]
        block += prefix[row_starts, starts[None, :]]
        block /= lags + 1
        squared_norms.append(float(np.einsum("ij,ij->", block, block)))
    return expectation, 2 * math.fsum(squared_norms) / n_obs**2


def gaussian_ar_hac_moments(
    n_obs: int,
    phi: float,
    sigma: float = 1.0,
    *,
    lags: int | None = None,
) -> GaussianARHACMoments:
    """Exact Gaussian quadratic-form identities, evaluated in float64.

    Assume a stationary Gaussian AR(1) with Cov[X_i, X_j] =
    sigma**2 * phi**abs(i-j), where sigma is the marginal standard deviation.
    The constant mean is arbitrary. The estimate matches ``long_run_variance``:
    sample-mean centering, autocovariance divisor T, Bartlett weights
    1 - h/(lags + 1), and no degrees-of-freedom correction.

    Writing P = I - 11'/T and W_ij = (1 - abs(i-j)/(lags+1))_+, the
    estimate is X' P W P X / T. Its mean is tr(W P C P)/T and its variance
    is 2 tr((W P C P)**2)/T**2, for the true covariance C. The population
    truncated LRV excludes finite-sample edge and sample-centering effects.
    The target T Var[mean] remains distinct from the infinite-horizon limit.

    Supported sizes are 8 <= T <= 2048 and 0 <= lags <= T-2. Work is
    O(T**2 + (T+lags)**2), and memory is O(T**2), with row blocks used for
    the moving-sum covariance. Identities are exact; floating-point results
    are approximate. No uniform relative-accuracy claim is made as phi
    approaches +/-1 or a requested moment approaches floating-point limits.
    Every returned moment must be positive and finite.
    """
    n_obs = positive_integer(n_obs, "n_obs", 8)
    if n_obs > MAX_OBSERVATIONS:
        raise ValueError(f"n_obs must be <= {MAX_OBSERVATIONS} for this reference.")
    phi = _finite_real(phi, "phi")
    if not -1 < phi < 1:
        raise ValueError("phi must satisfy -1 < phi < 1.")
    sigma = _finite_real(sigma, "sigma")
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    lags = default_lags(n_obs) if lags is None else positive_integer(lags, "lags", 0)
    if lags > n_obs - 2:
        raise ValueError("lags must be <= T - 2.")

    covariance = _centered_covariance(n_obs, phi)
    expectation, variance = _quadratic_moments(covariance, lags)
    target = n_obs * gaussian_ar_mean_variance(n_obs, phi)
    return GaussianARHACMoments(
        sample_size=n_obs,
        lags=lags,
        phi=phi,
        sigma=sigma,
        expectation=_rescale(expectation, sigma, 2, "expectation"),
        variance=_rescale(variance, sigma, 4, "variance"),
        target=_rescale(target, sigma, 2, "target"),
        population_truncated_lrv=_rescale(
            _truncated_factor(phi, lags), sigma, 2, "population_truncated_lrv"
        ),
        long_run_limit=_rescale((1 + phi) / (1 - phi), sigma, 2, "long_run_limit"),
    )
