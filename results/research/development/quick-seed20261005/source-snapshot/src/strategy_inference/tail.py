"""Research implementation of Liu--Chan AR(1) Bartlett tail postcoloring.

The formulas are independently implemented from Definition 3.1, Example 3.1
and Remark 3.2 of doi:10.1080/01621459.2026.2676715. The authors' GPL R source
is not incorporated here. These estimates are not a calibrated joint test and
are deliberately not integrated into ``audit_returns``.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import as_returns, positive_integer
from .inference import default_lags, long_run_variance


def _coefficients(phi: ArrayLike) -> NDArray[np.float64]:
    original = np.asarray(phi)
    if original.dtype.kind not in "iuf" or original.ndim > 1:
        raise ValueError("phi must be a real scalar or one-dimensional array.")
    values = np.asarray(phi, dtype=float)
    if not np.isfinite(values).all() or np.any(np.abs(values) >= 1):
        raise ValueError("phi must lie strictly between -1 and 1.")
    return values


def _ar_bartlett_mass(phi: NDArray[np.float64], bandwidth: int) -> NDArray[np.float64]:
    """Population kernel mass for marginal variance one, in O(ell * K).

    The innovation-weight-square representation avoids cancellation in the
    closed-form denominator near either AR boundary. Temporary memory is O(K).
    This equals ell * Var(mean of ell stationary observations).
    """
    weight = np.zeros_like(phi)
    squares = np.zeros_like(phi)
    for _ in range(bandwidth):
        weight = 1 + phi * weight
        squares += weight * weight
    return (phi * phi * weight * weight + (1 - phi) * (1 + phi) * squares) / bandwidth


def ar1_tail_factor(
    phi: ArrayLike,
    bandwidth: int,
    *,
    target: str = "long_run",
    n_obs: int | None = None,
) -> NDArray[np.float64]:
    """AR(1) population target divided by population Bartlett kernel mass.

    ``bandwidth`` is the paper's ell, equal to this package's ``lags + 1``.
    ``target='finite_sample'`` returns [T Var(mean)] / M_ell, as in Remark 3.2;
    it does not remove finite-sample centering bias. Array input is vectorized.
    No estimated coefficient is clipped or silently regularized.
    """
    values = _coefficients(phi)
    bandwidth = positive_integer(bandwidth, "bandwidth")
    denominator = _ar_bartlett_mass(values, bandwidth)
    if target == "long_run":
        if n_obs is not None:
            raise ValueError("n_obs is only used for target='finite_sample'.")
        numerator = (1 + values) / (1 - values)
    elif target == "finite_sample":
        n_obs = positive_integer(n_obs, "n_obs", 2)
        if bandwidth >= n_obs:
            raise ValueError("bandwidth must be < n_obs.")
        numerator = _ar_bartlett_mass(values, n_obs)
    else:
        raise ValueError("target must be 'long_run' or 'finite_sample'.")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        result = numerator / denominator
    if not np.isfinite(result).all() or np.any(result <= 0):
        raise ValueError("The tail factor is outside the positive finite float range.")
    return np.asarray(result)


@dataclass(frozen=True)
class TailVariance:
    variance: NDArray[np.float64]
    unadjusted: NDArray[np.float64]
    phi: NDArray[np.float64]
    factor: NDArray[np.float64]
    lags: int
    target: str
    parameter_source: str


def tail_variance(
    returns: ArrayLike,
    *,
    lags: int | None = None,
    phi: ArrayLike | None = None,
    target: str = "long_run",
) -> TailVariance:
    """Columnwise Bartlett tail estimates, scaled for sqrt(T) times the mean.

    By default, phi_hat is gamma_hat[1] / gamma_hat[0], both centered and with
    divisor T, matching the paper. Supplying phi uses known model parameters;
    that option is an oracle diagnostic, not a parameter estimate. Constant
    columns and invalid scales fail explicitly through the existing validators.
    """
    data = as_returns(returns)
    n_obs, n_strategies = data.shape
    lags = default_lags(n_obs) if lags is None else positive_integer(lags, "lags", 0)
    unadjusted = long_run_variance(data, lags)
    if phi is None:
        centered = data - data.mean(axis=0)
        denominator = np.einsum("ij,ij->j", centered, centered)
        numerator = np.einsum("ij,ij->j", centered[1:], centered[:-1])
        coefficients = _coefficients(numerator / denominator)
        source = "estimated"
    else:
        values = _coefficients(phi)
        if values.ndim == 0:
            coefficients = np.full(n_strategies, float(values))
        elif values.shape == (n_strategies,):
            coefficients = values.copy()
        else:
            raise ValueError("phi must be scalar or have one entry per strategy.")
        source = "known"
    factor = ar1_tail_factor(
        coefficients,
        lags + 1,
        target=target,
        n_obs=n_obs if target == "finite_sample" else None,
    )
    with np.errstate(over="ignore", invalid="ignore"):
        adjusted = factor * unadjusted
    if not np.isfinite(adjusted).all() or np.any(adjusted <= 0):
        raise ValueError("The adjusted variance is outside the positive finite float range.")
    return TailVariance(adjusted, unadjusted, coefficients, factor, lags, target, source)
