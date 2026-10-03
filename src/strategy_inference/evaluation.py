"""Forecast losses and origin-aligned comparisons, without pooling horizons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import positive_integer, probability

if TYPE_CHECKING:
    from .model_selection import BacktestResult
    from .testing import TestResult

Loss = Literal["squared", "absolute", "pinball"]


def _numeric(values: ArrayLike, name: str) -> NDArray[np.float64]:
    try:
        if np.ma.isMaskedArray(values) and np.any(np.ma.getmaskarray(values)):
            raise ValueError
        raw = np.asarray(values)
        if raw.dtype.kind not in "iuf" or not np.isrealobj(raw):
            raise ValueError
        data = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain real numeric values.") from exc
    if not data.size or not np.isfinite(data).all():
        raise ValueError(f"{name} must be nonempty and finite; no observations are dropped.")
    return data


def _aligned(actual: ArrayLike, predicted: ArrayLike):
    actual = _numeric(actual, "actual")
    predicted = _numeric(predicted, "predicted")
    if predicted.shape == actual.shape:
        return actual, predicted
    if predicted.ndim == actual.ndim + 1 and predicted.shape[:-1] == actual.shape:
        return actual[..., None], predicted
    raise ValueError("predicted must match actual, optionally with a final model axis.")


def forecast_loss(
    actual: ArrayLike,
    predicted: ArrayLike,
    *,
    loss: Loss = "squared",
    quantile: float | None = None,
) -> NDArray[np.float64]:
    """Return observation-level squared, absolute or quantile (pinball) loss.

    Arrays must match, except that predictions may have a trailing model axis.
    No averaging, time sorting, annualization or missing-value removal is done.
    Pinball predictions represent the specified quantile, not an arbitrary mean.
    """
    if loss not in ("squared", "absolute", "pinball"):
        raise ValueError("loss must be 'squared', 'absolute' or 'pinball'.")
    if loss == "pinball":
        quantile = probability(quantile, "quantile")
    elif quantile is not None:
        raise ValueError("quantile is only used with loss='pinball'.")
    actual, predicted = _aligned(actual, predicted)
    try:
        with np.errstate(over="raise", invalid="raise"):
            error = actual - predicted
            if loss == "squared":
                result = error * error
            elif loss == "absolute":
                result = np.abs(error)
            else:
                result = np.maximum(quantile * error, (quantile - 1) * error)
    except FloatingPointError as exc:
        raise ValueError("Forecast loss is outside the finite float range.") from exc
    return result


def interval_score(
    actual: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    *,
    alpha: float = 0.1,
) -> NDArray[np.float64]:
    """Proper central interval score: width plus 2/alpha times each miss distance.

    Finite bounds must be ordered. Whole-real-line intervals score +infinity;
    empty intervals are unsupported. Bounds must have the same shape, matching
    actual or adding a trailing model axis. This score describes intervals and
    does not establish their coverage.
    """
    alpha = probability(alpha, "alpha")
    actual = _numeric(actual, "actual")
    bounds = []
    for values in (lower, upper):
        if np.ma.isMaskedArray(values) and np.any(np.ma.getmaskarray(values)):
            raise ValueError("Interval bounds cannot contain masked observations.")
        raw = np.asarray(values)
        if raw.dtype.kind not in "iuf" or not np.isrealobj(raw):
            raise ValueError("Interval bounds must be real numeric arrays.")
        bound = np.asarray(raw, dtype=np.float64)
        if not bound.size or np.isnan(bound).any():
            raise ValueError("Interval bounds must be nonempty and contain no NaN.")
        bounds.append(bound)
    lower, upper = bounds
    if lower.shape != upper.shape or np.any(lower > upper):
        raise ValueError("Interval bounds must have the same shape and be ordered.")
    if lower.shape != actual.shape:
        if lower.ndim != actual.ndim + 1 or lower.shape[:-1] != actual.shape:
            raise ValueError("Interval bounds must match actual, optionally with a model axis.")
        actual = actual[..., None]
    if np.isposinf(lower).any() or np.isneginf(upper).any():
        raise ValueError("Infinite lower/upper bounds must point outwards.")
    with np.errstate(over="ignore"):
        return (
            upper
            - lower
            + 2 * (np.maximum(lower - actual, 0) / alpha)
            + 2 * (np.maximum(actual - upper, 0) / alpha)
        )


def _readonly(values, dtype=None):
    result = np.array(values, dtype=dtype, copy=True)
    result.flags.writeable = False
    return result


def _nonnegative_mean(values):
    """Scale nonnegative losses before summing, including near-float-limit inputs."""
    scale = values.max(axis=0)
    ratios = np.divide(values, scale, out=np.zeros_like(values), where=scale > 0)
    return ratios.mean(axis=0) * scale


def _panel(result: BacktestResult):
    from .model_selection import BacktestResult

    if not isinstance(result, BacktestResult):
        raise TypeError("Supply a BacktestResult from backtest().")
    origins = np.asarray([split.origin for split in result.splits], dtype=np.int64)
    leads = result.target_indices - origins[:, None]
    if not len(origins) or not np.all(leads == leads[0]) or np.any(leads <= 0):
        raise ValueError("All folds must use the same positive lead times.")
    return origins, tuple(int(lead) for lead in leads[0])


@dataclass(frozen=True)
class ForecastEvaluation:
    """Keep origin x lead x model losses; descriptive means retain each lead."""

    names: tuple[str, ...]
    lead_times: tuple[int, ...]
    origins: NDArray[np.int64]
    target_indices: NDArray[np.int64]
    loss: Loss
    quantile: float | None
    losses: NDArray[np.float64]
    mean_loss: NDArray[np.float64]

    def records(self):
        return [
            dict(
                model=name,
                lead_time=lead,
                n_origins=len(self.origins),
                mean_loss=float(self.mean_loss[h, m]),
            )
            for h, lead in enumerate(self.lead_times)
            for m, name in enumerate(self.names)
        ]

    def to_dict(self):
        return dict(
            schema_version=1,
            loss=self.loss,
            quantile=self.quantile,
            n_origins=len(self.origins),
            scores=self.records(),
        )

    def to_frame(self):
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("to_frame() requires pandas; install the 'pandas' extra.") from exc
        return pd.DataFrame.from_records(self.records()).set_index(["lead_time", "model"])


def evaluate_forecasts(
    result: BacktestResult,
    *,
    loss: Loss = "squared",
    quantile: float | None = None,
) -> ForecastEvaluation:
    """Score every retained forecast, averaging over origins separately by lead."""
    origins, leads = _panel(result)
    if loss == "pinball":
        quantile = probability(quantile, "quantile")
    losses = forecast_loss(result.actuals, result.forecasts, loss=loss, quantile=quantile)
    return ForecastEvaluation(
        result.names,
        leads,
        _readonly(origins),
        _readonly(result.target_indices),
        loss,
        quantile,
        _readonly(losses),
        _readonly(_nonnegative_mean(losses)),
    )


@dataclass(frozen=True)
class ForecastComparison:
    """Bootstrap family comparison against one prespecified baseline and lead."""

    baseline: str
    lead_time: int
    origin_step: int
    loss: Loss
    quantile: float | None
    baseline_loss: float
    candidate_loss: NDArray[np.float64]
    inference: TestResult
    warnings: tuple[str, ...] = ()

    def records(self):
        rows = self.inference.records()
        for i, row in enumerate(rows):
            row["model"] = row.pop("name")
            row["mean_improvement"] = row.pop("mean")
            row["mean_loss"] = float(self.candidate_loss[i])
            row["baseline_loss"] = self.baseline_loss
            row["relative_improvement"] = (
                row["mean_improvement"] / self.baseline_loss if self.baseline_loss > 0 else None
            )
        return rows

    def to_dict(self):
        record = self.inference.to_dict()
        record["null"] = "No supplied candidate has lower expected loss than the baseline."
        record.update(
            baseline=self.baseline,
            lead_time=self.lead_time,
            origin_step=self.origin_step,
            loss=self.loss,
            quantile=self.quantile,
            comparison_warnings=list(self.warnings),
            candidates=self.records(),
        )
        return record

    def to_frame(self):
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("to_frame() requires pandas; install the 'pandas' extra.") from exc
        return pd.DataFrame.from_records(self.records()).set_index("model")


def compare_forecasts(
    result: BacktestResult,
    *,
    baseline: str,
    lead_time: int | None = None,
    loss: Loss = "squared",
    quantile: float | None = None,
    alpha: float = 0.05,
    lags: int | None = None,
    block_length: float | None = None,
    **bootstrap_options,
) -> ForecastComparison:
    """Compare fixed models at one fixed lead with shared-row max bootstrap.

    Positive differences mean baseline loss minus candidate loss. Never flatten
    overlapping horizons into observations. The bootstrap is approximate and
    requires a stationary, weakly dependent loss-difference sequence with suitable
    moments and nondegenerate variance. Adaptive model search, nonstationary loss,
    repeated monitoring and joint inference across leads need separate methods.

    Default lag/block heuristics also consider overlap in units of origin steps;
    this is not a consistently estimated optimal dependence length or a validity
    guarantee. Explicit lags/block lengths are retained as supplied. Constant
    differences raise rather than being silently removed from the tested family.
    """
    from .bootstrap import default_block_length
    from .inference import default_lags
    from .testing import test_returns

    origins, leads = _panel(result)
    if len(leads) > 1 and lead_time is None:
        raise ValueError("Specify lead_time for a multi-step backtest; leads are not pooled.")
    lead_time = leads[0] if lead_time is None else positive_integer(lead_time, "lead_time")
    if lead_time not in leads:
        raise ValueError("lead_time must be one of the evaluated lead times.")
    if baseline not in result.names or len(result.names) < 2:
        raise ValueError("baseline must name one of at least two supplied models.")
    steps = np.diff(origins)
    if not len(steps) or np.any(steps <= 0) or not np.all(steps == steps[0]):
        raise ValueError("Inference requires regularly spaced, increasing forecast origins.")
    step = int(steps[0])
    if loss == "pinball":
        quantile = probability(quantile, "quantile")
    scores = forecast_loss(
        result.actuals[:, leads.index(lead_time)],
        result.forecasts[:, leads.index(lead_time)],
        loss=loss,
        quantile=quantile,
    )
    base = result.names.index(baseline)
    candidates = [i for i in range(len(result.names)) if i != base]
    names = tuple(result.names[i] for i in candidates)
    try:
        with np.errstate(over="raise", invalid="raise"):
            improvement = scores[:, base, None] - scores[:, candidates]
    except FloatingPointError as exc:
        raise ValueError("Loss differences are outside the finite float range.") from exc
    constant = np.max(improvement, axis=0) == np.min(improvement, axis=0)
    if constant.any():
        invalid = ", ".join(name for name, flag in zip(names, constant, strict=True) if flag)
        raise ValueError(f"Constant loss differences have undefined inference: {invalid}.")
    overlap = (lead_time + step - 1) // step
    warnings = ()
    if overlap - 1 > len(origins) - 2:
        warnings = (
            "The forecast lead spans more origin steps than the available HAC lag range; "
            "default bandwidths are capped. Extend the evaluation before relying on inference.",
        )
    lags = (
        min(max(default_lags(len(origins)), overlap - 1), len(origins) - 2)
        if lags is None
        else lags
    )
    block_length = (
        min(max(default_block_length(len(origins)), overlap), len(origins))
        if block_length is None
        else block_length
    )
    inference = test_returns(
        improvement,
        method="bootstrap",
        names=names,
        alpha=alpha,
        lags=lags,
        block_length=block_length,
        **bootstrap_options,
    )
    mean_scores = _nonnegative_mean(scores)
    return ForecastComparison(
        baseline,
        lead_time,
        step,
        loss,
        quantile,
        float(mean_scores[base]),
        _readonly(mean_scores[candidates]),
        inference,
        warnings,
    )
