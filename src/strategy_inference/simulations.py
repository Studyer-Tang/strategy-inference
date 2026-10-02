"""Predeclared, unit-variance shock processes for inference experiments.

The t5 component shocks are scaled by their theoretical variance, rather than
rescaled using the realized sample. Their common/idiosyncratic mixture and AR
returns are not themselves Student-t distributions.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from numpy.typing import NDArray

PROCESSES = ("gaussian_ar", "student_ar", "garch")
STUDENT_DF = 5
GARCH_ALPHA = 0.06
GARCH_BETA = 0.90


def _integer(value: int, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _finite_number(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_, str, bytes)) or not np.isscalar(value):
        raise ValueError(f"{name} must be a finite real number")
    if not np.isrealobj(value):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _means(value: float | NDArray[np.float64], size: int) -> NDArray[np.float64]:
    raw = np.asarray(value)
    if not np.isrealobj(raw) or raw.dtype.kind not in "iuf":
        raise ValueError("mean must be a finite real scalar or a length-K vector")
    values = np.asarray(value, dtype=np.float64)
    if values.ndim == 0:
        values = np.full(size, float(values), dtype=np.float64)
    elif values.ndim != 1 or values.shape[0] != size:
        raise ValueError("mean must be a finite real scalar or a length-K vector")
    if not np.all(np.isfinite(values)):
        raise ValueError("mean must be finite")
    return values


def _generator(seed: int | np.random.SeedSequence) -> np.random.Generator:
    if isinstance(seed, np.random.SeedSequence):
        return np.random.default_rng(seed)
    return np.random.default_rng(_integer(seed, "seed", 0))


def simulate_returns(
    n_obs: int,
    n_strategies: int,
    process: Literal["gaussian_ar", "student_ar", "garch"] = "gaussian_ar",
    phi: float = 0.5,
    cross_corr: float = 0.35,
    sigma: float = 0.01,
    mean: float | NDArray[np.float64] = 0.0,
    seed: int | np.random.SeedSequence = 0,
    burnin: int = 512,
) -> NDArray[np.float64]:
    """Return a T-by-K matrix of synthetic excess returns.

    Gaussian AR(1) starts from its exact joint stationary Gaussian law. Heavy-
    tailed AR(1) starts at zero and GARCH starts at its unconditional variance;
    both discard ``burnin`` observations and are only approximately stationary.
    In GARCH ``phi`` must be zero. Its innovation correlation is ``cross_corr``;
    random, strategy-specific volatility need not preserve that exact return
    correlation. All processes have theoretical marginal standard deviation
    ``sigma`` in stationarity. Mean shifts are added after simulating the noise.
    """
    n_obs = _integer(n_obs, "n_obs", 2)
    n_strategies = _integer(n_strategies, "n_strategies", 1)
    burnin = _integer(burnin, "burnin", 0)
    if not isinstance(process, str) or process not in PROCESSES:
        raise ValueError(f"process must be one of {PROCESSES}")
    phi = _finite_number(phi, "phi")
    cross_corr = _finite_number(cross_corr, "cross_corr")
    sigma = _finite_number(sigma, "sigma")
    if not -1.0 < phi < 1.0:
        raise ValueError("phi must lie strictly between -1 and 1")
    if not 0.0 <= cross_corr <= 1.0:
        raise ValueError("cross_corr must lie between 0 and 1")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if process == "garch" and phi != 0.0:
        raise ValueError("GARCH uses phi=0; pass phi=0 explicitly")
    shifts = _means(mean, n_strategies)
    rng = _generator(seed)

    n_steps = n_obs if process == "gaussian_ar" else n_obs + burnin
    # Column zero is common; the remaining K columns are idiosyncratic.
    if process == "gaussian_ar":
        raw = rng.standard_normal((n_steps + 1, n_strategies + 1))
    else:
        raw = rng.standard_t(STUDENT_DF, (n_steps, n_strategies + 1))
        raw *= math.sqrt((STUDENT_DF - 2) / STUDENT_DF)
    shocks = math.sqrt(cross_corr) * raw[:, :1] + math.sqrt(1 - cross_corr) * raw[:, 1:]
    values = np.empty((n_steps, n_strategies), dtype=np.float64)

    if process == "garch":
        variance = np.full(n_strategies, sigma**2, dtype=np.float64)
        omega = (1 - GARCH_ALPHA - GARCH_BETA) * sigma**2
        for index in range(n_steps):
            innovation = np.sqrt(variance) * shocks[index]
            values[index] = innovation
            variance = omega + GARCH_ALPHA * innovation**2 + GARCH_BETA * variance
    else:
        if process == "gaussian_ar":
            previous = sigma * shocks[0]
            innovations = shocks[1:]
        else:
            previous = np.zeros(n_strategies, dtype=np.float64)
            innovations = shocks
        innovation_scale = sigma * math.sqrt(1 - phi**2)
        for index in range(n_steps):
            previous = phi * previous + innovation_scale * innovations[index]
            values[index] = previous
    start = 0 if process == "gaussian_ar" else burnin
    result = values[start:] + shifts
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("simulation overflowed; reduce the scale or extreme parameters")
    return result
