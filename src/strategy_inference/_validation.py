"""Shared input checks; observations are rows, candidate strategies are columns."""

from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray


def positive_integer(value: int, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def probability(value: float, name: str) -> float:
    if not np.isscalar(value) or not np.isfinite(value) or not 0 < value < 1:
        raise ValueError(f"{name} must lie strictly between 0 and 1.")
    return float(value)


def as_returns(values: ArrayLike) -> NDArray[np.float64]:
    try:
        data = np.asarray(values, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError("Returns must be a rectangular numeric array.") from exc
    if data.ndim == 1:
        data = data[:, None]
    if data.ndim != 2 or data.shape[0] < 8 or data.shape[1] < 1:
        raise ValueError("Returns must have shape (T, K), with T >= 8 and K >= 1.")
    if not np.isfinite(data).all():
        raise ValueError("Returns contain missing or non-finite values; no rows are dropped.")
    centered = data - data.mean(axis=0)
    variance = np.mean(centered * centered, axis=0)
    if not np.isfinite(variance).all() or np.any(variance <= 0):
        raise ValueError("Each candidate must have positive, finite sample variance.")
    return np.ascontiguousarray(data)
