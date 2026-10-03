"""Strong sequential model confidence sets with predictable forecast bounds.

Arnold et al., JRSSB (2026), Proposition 3.2, equations (6)--(7), Appendix H:
https://doi.org/10.1093/jrsssb/qkag066
This implements the strong conditional-superiority construction, not the
paper's uniformly weak or time-varying average-risk constructions.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import log
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import positive_integer, probability
from .conformal import _finite
from .evaluation import _numeric, _panel, _readonly

if TYPE_CHECKING:
    from .model_selection import BacktestResult


def _closed_log_evalues(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Closed arithmetic-mean testing in O(M log M), without exponentiation."""
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    prefix = np.logaddexp.accumulate(sorted_values)
    adjusted = np.empty_like(values)
    adjusted[order[0]] = sorted_values[0]
    i = 0
    for k in range(1, len(values)):
        best = np.logaddexp(sorted_values[k], prefix[i]) - log(i + 2)
        while i < k - 1:
            candidate = np.logaddexp(sorted_values[k], prefix[i + 1]) - log(i + 3)
            if candidate > best:
                break
            i += 1
            best = candidate
        adjusted[order[k]] = best
    return adjusted


def _normalized_differences(points, actual, quantile):
    """Loss_i-loss_j divided by its predictable bound; avoid huge raw losses."""
    left, right = points[:, None], points[None, :]
    lower, upper = np.minimum(left, right), np.maximum(left, right)
    clipped = np.clip(actual, lower, upper)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        gap = upper - lower
        fraction = np.divide(clipped - lower, gap, out=np.zeros_like(gap), where=gap > 0)
    overflow = ~np.isfinite(gap)
    if overflow.any():
        # Scaling only these pairs also handles opposite near-float-limit points.
        scale = np.maximum(np.abs(lower[overflow]), np.abs(upper[overflow]))
        low = lower[overflow] / scale
        fraction[overflow] = (clipped[overflow] / scale - low) / (upper[overflow] / scale - low)
    fraction = np.clip(fraction, 0, 1)
    direction = (right > left).astype(float) - (right < left)
    if quantile is None:
        return direction * (2 * fraction - 1)
    return direction * (fraction - (1 - quantile)) / max(quantile, 1 - quantile)


class SequentialModelConfidenceSet:
    """Continuously compare a fixed family of one-step or nonoverlapping forecasts.

    Register all forecasts with ``predict`` before receiving their common label,
    then call ``update``. Forecasts may be refitted using past data. The model
    names, loss, quantile and betting fraction must be fixed before evaluation.

    The target is models i satisfying E[loss_i-loss_j | available information
    at issue] <= 0 for every j and every evaluation time. This strong set may
    be empty; it differs from models having the smallest average realized or
    unconditional expected loss. Under that null, the ideal-arithmetic running
    confidence set retains all such models at every time with probability at
    least 1-alpha, without IID, stationarity or a fixed stopping time.

    Absolute and pinball loss differences have predictable bounds B_ij equal
    to |forecast_i-forecast_j| and max(q,1-q)*|forecast_i-forecast_j|. Pairwise
    wealth is multiplied by 1+bet_fraction*(loss_i-loss_j)/B_ij. Equal forecasts
    have factor 1. bet_fraction in (0,0.5] follows the paper's bound; 0.25 is a
    fixed conservative choice, not an optimized bet. Pairwise e-processes are
    averaged per model, then adjusted by closed arithmetic-mean testing.

    Exclusions are permanent (running intersection); keep supplying all original
    models after an exclusion. Only one label can be pending. Invalid inputs
    leave state unchanged. Memory is O(M^2), with no stored observation history;
    updates are O(M^2) plus O(M log M) adjustment. Ordinary float64 arithmetic
    does not provide a formal rounding certificate.
    """

    __slots__ = (
        "_names",
        "_alpha",
        "_loss",
        "_quantile",
        "_bet_fraction",
        "_pending",
        "_pair_logs",
        "_log_evalues",
        "_log_adjusted_evalues",
        "_retained",
        "_n_updates",
    )

    def __init__(
        self,
        names: Sequence[str],
        *,
        alpha: float = 0.05,
        loss: Literal["absolute", "pinball"] = "absolute",
        quantile: float | None = None,
        bet_fraction: float = 0.25,
    ) -> None:
        if isinstance(names, (str, bytes)):
            raise ValueError("names must contain at least two unique nonempty strings.")
        try:
            names = tuple(names)
        except TypeError as exc:
            raise ValueError("names must contain at least two unique nonempty strings.") from exc
        if (
            len(names) < 2
            or any(not isinstance(name, str) or not name.strip() for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("names must contain at least two unique nonempty strings.")
        alpha = probability(alpha, "alpha")
        if loss not in ("absolute", "pinball"):
            raise ValueError("loss must be 'absolute' or 'pinball'.")
        if loss == "pinball":
            quantile = probability(quantile, "quantile")
        elif quantile is not None:
            raise ValueError("quantile is only used with loss='pinball'.")
        bet_fraction = _finite(bet_fraction, "bet_fraction")
        if not 0 < bet_fraction <= 0.5:
            raise ValueError("bet_fraction must lie in (0, 0.5].")
        self._names, self._alpha, self._loss = names, alpha, loss
        self._quantile, self._bet_fraction = quantile, bet_fraction
        self._pending = None
        self._pair_logs = np.zeros((len(names), len(names)))
        np.fill_diagonal(self._pair_logs, -np.inf)
        self._log_evalues = np.zeros(len(names))
        self._log_adjusted_evalues = np.zeros(len(names))
        self._retained = np.ones(len(names), dtype=bool)
        self._n_updates = 0

    @property
    def names(self) -> tuple[str, ...]:
        return self._names

    @property
    def n_updates(self) -> int:
        return self._n_updates

    @property
    def confidence_set(self) -> tuple[str, ...]:
        return tuple(name for name, keep in zip(self._names, self._retained, strict=True) if keep)

    @property
    def pending(self) -> NDArray[np.float64] | None:
        return None if self._pending is None else _readonly(self._pending)

    @property
    def log_evalues(self) -> NDArray[np.float64]:
        """Current model-level log e-values, before the closure adjustment."""
        return _readonly(self._log_evalues)

    @property
    def log_adjusted_evalues(self) -> NDArray[np.float64]:
        """Current adjusted evidence; exclusions also retain earlier crossings."""
        return _readonly(self._log_adjusted_evalues)

    def predict(self, forecasts: ArrayLike) -> None:
        """Freeze a finite vector in names order before its common label arrives."""
        if self._pending is not None:
            raise RuntimeError("Forecasts are pending; update their label before predicting again.")
        values = _numeric(forecasts, "forecasts")
        if values.shape != (len(self._names),):
            raise ValueError("forecasts must be a one-dimensional vector matching names.")
        self._pending = _readonly(values)

    def update(self, actual: float) -> tuple[str, ...]:
        """Apply a mature label and return the running model confidence set."""
        if self._pending is None:
            raise RuntimeError("No forecasts are pending; call predict before update.")
        actual = _finite(actual, "actual")
        if np.all(self._pending == self._pending[0]):
            self._n_updates += 1
            self._pending = None
            return self.confidence_set
        differences = _normalized_differences(self._pending, actual, self._quantile)
        with np.errstate(over="ignore", invalid="ignore"):
            pair_logs = self._pair_logs + np.log1p(self._bet_fraction * differences)
            values = np.logaddexp.reduce(pair_logs, axis=1) - log(len(self._names) - 1)
            adjusted = _closed_log_evalues(values)
        if not np.isfinite(values).all() or not np.isfinite(adjusted).all():
            raise ValueError("Sequential log evidence exceeded the finite float range.")
        retained = self._retained & (adjusted < -log(self._alpha))
        self._pair_logs, self._log_evalues = pair_logs, values
        self._log_adjusted_evalues, self._retained = adjusted, retained
        self._n_updates += 1
        self._pending = None
        return self.confidence_set

    def to_dict(self) -> dict:
        """A detached strict-JSON snapshot; log values avoid wealth overflow."""
        return dict(
            method="sequential_model_confidence_set",
            null="strong conditional superiority",
            names=list(self._names),
            alpha=self._alpha,
            loss=self._loss,
            quantile=self._quantile,
            bet_fraction=self._bet_fraction,
            n_updates=self._n_updates,
            confidence_set=list(self.confidence_set),
            log_evalues=self._log_evalues.tolist(),
            log_adjusted_evalues=self._log_adjusted_evalues.tolist(),
            pending=None if self._pending is None else self._pending.tolist(),
        )


def sequential_compare_forecasts(
    result: BacktestResult,
    *,
    lead_time: int | None = None,
    alpha: float = 0.05,
    loss: Literal["absolute", "pinball"] = "absolute",
    quantile: float | None = None,
    bet_fraction: float = 0.25,
) -> SequentialModelConfidenceSet:
    """Run the same streaming comparison on one preselected backtest lead.

    All original models are compared, without choosing a baseline. Multi-step
    backtests require an explicit physical lead_time. Consecutive origins must
    be at least that far apart, so each label matures before the next forecast.
    Overlapping feedback is rejected. Irregular nonoverlapping origins are
    allowed. The user must ensure forecasts use only information at issue;
    this wrapper does not establish the strong conditional-superiority null.
    The returned tracker can continue with new predict/update pairs.
    """
    origins, leads = _panel(result)
    if len(leads) > 1 and lead_time is None:
        raise ValueError("Specify lead_time for a multi-step backtest; leads are not pooled.")
    lead_time = leads[0] if lead_time is None else positive_integer(lead_time, "lead_time")
    if lead_time not in leads:
        raise ValueError("lead_time must be one of the evaluated lead times.")
    if np.any(np.diff(origins) < lead_time):
        raise ValueError(
            "Sequential comparison requires nonoverlapping feedback: origin step >= lead_time."
        )
    h = leads.index(lead_time)
    forecasts = _numeric(result.forecasts[:, h], "forecasts")
    actual = _numeric(result.actuals[:, h], "actual")
    if forecasts.shape != (len(origins), len(result.names)) or actual.shape != (len(origins),):
        raise ValueError("Backtest forecasts and actuals must align with origins and names.")
    tracker = SequentialModelConfidenceSet(
        result.names,
        alpha=alpha,
        loss=loss,
        quantile=quantile,
        bet_fraction=bet_fraction,
    )
    for points, label in zip(forecasts, actual, strict=True):
        tracker.predict(points)
        tracker.update(label)
    return tracker
