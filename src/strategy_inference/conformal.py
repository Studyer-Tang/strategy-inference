"""One-step online prediction intervals with a decaying quantile tracker.

The update follows Angelopoulos, Barber and Bates (ICML 2024):
https://proceedings.mlr.press/v235/angelopoulos24a.html
The fixed-scale bounded residual transform is an implementation choice, not
a new conformal algorithm. In ideal arithmetic its scores lie in [0, 1] and
the update controls retrospective average coverage for arbitrary sequences.
It does not ensure coverage at each time, conditional coverage, or validity
with delayed feedback. Additional IID/score assumptions are needed for the
paper's quantile convergence theorem. This implementation uses ordinary
binary64 arithmetic; it does not certify floating-point rounding error.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import inf, isfinite
from numbers import Real
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

IntervalKind = Literal["finite", "empty", "unbounded"]


def _finite(value: Real, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar.")
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real scalar.") from exc
    if not isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar.")
    return result


def _bounded_score(residual: float, scale: float) -> float:
    """residual/(scale+residual), without overflow in the denominator."""
    if residual > scale:
        return 1 / (1 + scale / residual)
    ratio = residual / scale
    return ratio / (1 + ratio)


@dataclass(frozen=True)
class ConformalInterval:
    """An issued interval; negative quantiles give an explicit empty set.

    Empty intervals store lower=+inf and upper=-inf. Unbounded intervals
    store lower=-inf and upper=+inf. ``to_dict`` uses null boundaries and
    ``kind`` to distinguish both cases in strict JSON.
    """

    index: int
    prediction: float
    lower: float
    upper: float
    quantile: float
    kind: IntervalKind

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if self.kind != "finite":
            result["lower"] = result["upper"] = None
        return result


@dataclass(frozen=True)
class ConformalUpdate:
    """Feedback evaluated against the interval issued before seeing the label.

    ``score`` records the bounded residual. ``miss`` uses the returned closed
    interval; binary64 boundary rounding can make it differ from score>quantile.
    """

    index: int
    actual: float
    prediction: float
    score: float
    miss: bool
    step_size: float
    previous_quantile: float
    quantile: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AdaptiveConformal:
    """Fixed-scale one-step intervals, with exactly one pending prediction.

    The score is |actual-prediction|/(scale+|actual-prediction|). With
    0<=q<1, its ideal inverse gives radius scale*q/(1-q). Quantiles are
    never clipped: q<0 gives an empty set, and q>=1 gives all real values.
    Each feedback updates q by eta_t*(miss-alpha), eta_t=step_size*t**(-decay).

    For 0<=decay<1, bounded scores and an initial quantile in [0,1], the
    ideal recursion has average miscoverage error at most
    (1+step_size)/(T*eta_T). This is a retrospective bound. Quantile/coverage
    convergence with decay in (1/2,1) requires the paper's additional IID,
    fixed-score, unique-quantile and continuity assumptions.

    ``predict`` must precede ``update``. Only ordered immediate feedback is
    supported; callers must align each label with its pending forecast. A
    second pending prediction is rejected. Invalid inputs raise without changing
    state. Finite interval boundaries are inclusive, and feedback uses the
    actual returned interval, including binary64 boundary rounding.
    """

    __slots__ = (
        "_alpha",
        "_step_size",
        "_decay",
        "_scale",
        "_initial_quantile",
        "_quantile",
        "_n_updates",
        "_misses",
        "_pending",
    )

    def __init__(
        self,
        *,
        alpha: float = 0.1,
        step_size: float = 0.1,
        decay: float = 0.6,
        scale: float = 1.0,
        initial_quantile: float = 0.5,
    ) -> None:
        alpha = _finite(alpha, "alpha")
        step_size = _finite(step_size, "step_size")
        decay = _finite(decay, "decay")
        scale = _finite(scale, "scale")
        initial_quantile = _finite(initial_quantile, "initial_quantile")
        if not 0 < alpha < 1:
            raise ValueError("alpha must lie strictly between 0 and 1.")
        if step_size <= 0 or scale <= 0:
            raise ValueError("step_size and scale must be positive.")
        if not 0 <= decay < 1:
            raise ValueError("decay must lie in [0, 1).")
        if not 0 <= initial_quantile <= 1:
            raise ValueError("initial_quantile must lie in [0, 1].")
        self._alpha, self._step_size, self._decay = alpha, step_size, decay
        self._scale, self._initial_quantile = scale, initial_quantile
        self._quantile = initial_quantile
        self._n_updates = self._misses = 0
        self._pending: ConformalInterval | None = None

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def step_size(self) -> float:
        return self._step_size

    @property
    def decay(self) -> float:
        return self._decay

    @property
    def scale(self) -> float:
        return self._scale

    @property
    def initial_quantile(self) -> float:
        return self._initial_quantile

    @property
    def quantile(self) -> float:
        """The next threshold; pending forecasts retain their issued threshold."""
        return self._quantile

    @property
    def n_updates(self) -> int:
        return self._n_updates

    @property
    def misses(self) -> int:
        return self._misses

    @property
    def coverage(self) -> float | None:
        """Realized coverage among evaluated forecasts; None before feedback."""
        return 1 - self._misses / self._n_updates if self._n_updates else None

    @property
    def pending(self) -> ConformalInterval | None:
        return self._pending

    def predict(self, point: float) -> ConformalInterval:
        """Issue a prediction without observing its label; boundary overflow raises."""
        if self._pending is not None:
            raise RuntimeError("A prediction is pending; update its label before predicting again.")
        point = _finite(point, "point")
        q = self._quantile
        if q < 0:
            lower, upper, kind = inf, -inf, "empty"
        elif q >= 1:
            lower, upper, kind = -inf, inf, "unbounded"
        else:
            radius = self._scale * (q / (1 - q))
            lower, upper, kind = point - radius, point + radius, "finite"
            if not isfinite(lower) or not isfinite(upper):
                raise ValueError(
                    "Interval boundary overflow; choose smaller numeric units or scale."
                )
        interval = ConformalInterval(self._n_updates + 1, point, lower, upper, q, kind)
        self._pending = interval
        return interval

    def update(self, actual: float) -> ConformalUpdate:
        """Evaluate the pending forecast, then update; every validation is atomic."""
        interval = self._pending
        if interval is None:
            raise RuntimeError("No prediction is pending; call predict before update.")
        actual = _finite(actual, "actual")
        residual = abs(actual - interval.prediction)
        if not isfinite(residual):
            raise ValueError("Residual overflow; express actual and prediction in smaller units.")
        score = _bounded_score(residual, self._scale)
        miss = interval.kind == "empty" or not interval.lower <= actual <= interval.upper
        step = self._step_size * (self._n_updates + 1) ** (-self._decay)
        if not isfinite(step) or step <= 0:
            raise ValueError("The step size underflowed; choose a larger step_size.")
        q = interval.quantile + step * (int(miss) - self._alpha)
        if not isfinite(q):
            raise ValueError("Quantile update overflow; choose a smaller step_size.")
        result = ConformalUpdate(
            interval.index, actual, interval.prediction, score, miss, step, interval.quantile, q
        )
        self._quantile, self._n_updates = q, self._n_updates + 1
        self._misses += int(miss)
        self._pending = None
        return result


@dataclass(frozen=True)
class ConformalResult:
    """Prequential one-step intervals and observed coverage, in input order.

    ``quantiles`` are the thresholds used when issuing the intervals;
    ``step_sizes`` are the subsequent feedback learning rates. Prediction
    values must already have been constructed without future information.
    """

    actual: NDArray[np.float64]
    predicted: NDArray[np.float64]
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    empty: NDArray[np.bool_]
    unbounded: NDArray[np.bool_]
    misses: NDArray[np.bool_]
    quantiles: NDArray[np.float64]
    step_sizes: NDArray[np.float64]
    alpha: float
    step_size: float
    decay: float
    scale: float
    initial_quantile: float
    next_quantile: float

    @property
    def n_obs(self) -> int:
        return len(self.actual)

    @property
    def coverage(self) -> float:
        return 1 - float(np.count_nonzero(self.misses)) / self.n_obs

    def records(self) -> list[dict[str, Any]]:
        """Strict-JSON rows, with explicit kind and null nonfinite boundaries."""
        rows = []
        for i in range(self.n_obs):
            kind = "empty" if self.empty[i] else "unbounded" if self.unbounded[i] else "finite"
            rows.append(
                {
                    "index": i + 1,
                    "actual": float(self.actual[i]),
                    "prediction": float(self.predicted[i]),
                    "kind": kind,
                    "lower": float(self.lower[i]) if kind == "finite" else None,
                    "upper": float(self.upper[i]) if kind == "finite" else None,
                    "miss": bool(self.misses[i]),
                    "quantile": float(self.quantiles[i]),
                    "step_size": float(self.step_sizes[i]),
                }
            )
        return rows

    def to_dict(self) -> dict[str, Any]:
        """Export the observations and parameters without JSON Infinity/NaN."""
        return {
            "schema_version": 1,
            "method": "adaptive_conformal",
            "alpha": self.alpha,
            "step_size": self.step_size,
            "decay": self.decay,
            "scale": self.scale,
            "initial_quantile": self.initial_quantile,
            "n_obs": self.n_obs,
            "coverage": self.coverage,
            "next_quantile": self.next_quantile,
            "intervals": self.records(),
        }


def _series(values: ArrayLike, name: str) -> NDArray[np.float64]:
    if np.ma.isMaskedArray(values) and np.any(np.ma.getmaskarray(values)):
        raise ValueError(f"{name} contains masked observations; no rows are dropped.")
    try:
        raw = np.asarray(values)
        if raw.dtype.kind not in "iuf" or raw.ndim != 1 or not len(raw):
            raise ValueError
        with np.errstate(over="ignore", invalid="ignore"):
            result = np.array(raw, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a nonempty one-dimensional real numeric series.") from exc
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values; no rows are dropped.")
    return result


def adaptive_intervals(
    actual: ArrayLike,
    predicted: ArrayLike,
    *,
    alpha: float = 0.1,
    step_size: float = 0.1,
    decay: float = 0.6,
    scale: float = 1.0,
    initial_quantile: float = 0.5,
) -> ConformalResult:
    """Replay strictly ordered predict-then-feedback updates on one-step forecasts.

    The supplied predictions must be made without future labels; this wrapper
    cannot establish that provenance. It does not train or choose a forecaster.
    All labels must be present, finite and aligned, with no delayed feedback.
    """
    observed, forecast = _series(actual, "actual"), _series(predicted, "predicted")
    if observed.shape != forecast.shape:
        raise ValueError("actual and predicted must have the same length.")
    tracker = AdaptiveConformal(
        alpha=alpha,
        step_size=step_size,
        decay=decay,
        scale=scale,
        initial_quantile=initial_quantile,
    )
    n = len(observed)
    lower, upper, quantiles, steps = (np.empty(n) for _ in range(4))
    empty, unbounded, misses = (np.empty(n, dtype=bool) for _ in range(3))
    for i, (label, point) in enumerate(zip(observed, forecast, strict=True)):
        interval = tracker.predict(point)
        update = tracker.update(label)
        lower[i], upper[i], quantiles[i] = interval.lower, interval.upper, interval.quantile
        empty[i], unbounded[i] = interval.kind == "empty", interval.kind == "unbounded"
        misses[i], steps[i] = update.miss, update.step_size
    for array in (observed, forecast, lower, upper, empty, unbounded, misses, quantiles, steps):
        array.flags.writeable = False
    return ConformalResult(
        observed,
        forecast,
        lower,
        upper,
        empty,
        unbounded,
        misses,
        quantiles,
        steps,
        tracker.alpha,
        tracker.step_size,
        tracker.decay,
        tracker.scale,
        tracker.initial_quantile,
        tracker.quantile,
    )
