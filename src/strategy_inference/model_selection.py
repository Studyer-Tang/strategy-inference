"""Rolling-origin forecasts with explicit time indices and isolated training data."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import positive_integer

Forecaster = Callable[[NDArray[np.float64], NDArray[np.int64]], ArrayLike]


@dataclass(frozen=True)
class RollingSplit:
    """Zero-based, half-open training/test ranges and the last observed index."""

    train_start: int
    train_stop: int
    test_start: int
    test_stop: int
    origin: int


def rolling_splits(
    n_obs: int,
    *,
    initial_train_size: int,
    horizon: int = 1,
    step: int = 1,
    gap: int = 0,
    window: int | None = None,
) -> tuple[RollingSplit, ...]:
    """Return all complete rolling-origin folds, without a partial final test.

    The first origin is ``initial_train_size - 1``; later origins advance by
    ``step``. ``gap`` observations follow each origin before its test starts.
    ``window=None`` expands the training history. An integer window caps the
    history length, including when it is smaller than ``initial_train_size``.
    Overlapping test ranges are retained as separate forecast occasions.
    """
    n_obs = positive_integer(n_obs, "n_obs")
    initial_train_size = positive_integer(initial_train_size, "initial_train_size")
    horizon = positive_integer(horizon, "horizon")
    step = positive_integer(step, "step")
    gap = positive_integer(gap, "gap", 0)
    if window is not None:
        window = positive_integer(window, "window")
    if initial_train_size > n_obs:
        raise ValueError("initial_train_size must be <= n_obs.")
    last_train_stop = n_obs - gap - horizon
    if initial_train_size > last_train_stop:
        raise ValueError("The split settings leave no complete test horizon.")
    return tuple(
        RollingSplit(
            0 if window is None else max(0, train_stop - window),
            train_stop,
            train_stop + gap,
            train_stop + gap + horizon,
            train_stop - 1,
        )
        for train_stop in range(initial_train_size, last_train_stop + 1, step)
    )


def _real_vector(values: ArrayLike, name: str) -> NDArray[np.float64]:
    try:
        raw = np.asarray(values)
        if raw.ndim != 1 or raw.size == 0 or raw.dtype.kind not in "iuf":
            raise ValueError
        if np.ma.isMaskedArray(values) and np.any(np.ma.getmaskarray(values)):
            raise ValueError
        with np.errstate(over="ignore", invalid="ignore"):
            data = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a nonempty one-dimensional real numeric series.") from exc
    if not np.isfinite(data).all():
        raise ValueError(f"{name} must contain only finite values; no observations are removed.")
    return data


def _lead_times(values: ArrayLike) -> NDArray[np.int64]:
    try:
        if np.ma.isMaskedArray(values) and np.any(np.ma.getmaskarray(values)):
            raise ValueError
        raw = np.asarray(values)
        if (
            raw.ndim != 1
            or raw.size == 0
            or raw.dtype.kind not in "iu"
            or np.any(raw <= 0)
            or np.any(raw > np.iinfo(np.int64).max)
        ):
            raise ValueError
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "lead_times must be a nonempty one-dimensional array of positive integers."
        ) from exc
    return np.asarray(raw, dtype=np.int64)


def _readonly_copy(values, dtype):
    result = np.array(values, dtype=dtype, copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class BacktestResult:
    """Every fold's forecasts, actuals and positions, in supplied model order.

    ``forecasts`` has shape (folds, horizon, models); ``actuals`` and
    ``target_indices`` have shape (folds, horizon). Overlapping targets remain
    separate rows. These are forecasts and observations, without a statistical
    independence assertion or an automatic choice of model or loss function.
    """

    sample_size: int
    names: tuple[str, ...]
    splits: tuple[RollingSplit, ...]
    forecasts: NDArray[np.float64]
    target_indices: NDArray[np.int64]
    actuals: NDArray[np.float64]

    @property
    def n_obs(self) -> int:
        return self.sample_size

    @property
    def n_folds(self) -> int:
        return len(self.splits)

    @property
    def horizon(self) -> int:
        return self.target_indices.shape[1]

    @property
    def n_models(self) -> int:
        return len(self.names)

    def records(self) -> list[dict[str, Any]]:
        """One row per forecast occasion, lead time and model, in that order."""
        return [
            {
                "fold": fold,
                "origin": split.origin,
                "lead_time": int(target - split.origin),
                "target_index": int(target),
                "model": name,
                "actual": float(self.actuals[fold, lead]),
                "forecast": float(self.forecasts[fold, lead, model]),
            }
            for fold, split in enumerate(self.splits)
            for lead, target in enumerate(self.target_indices[fold])
            for model, name in enumerate(self.names)
        ]

    def to_dict(self) -> dict[str, Any]:
        """A detached JSON-serializable record retaining all folds and forecasts."""
        return {
            "schema_version": 1,
            "sample_size": self.sample_size,
            "n_folds": self.n_folds,
            "horizon": self.horizon,
            "names": list(self.names),
            "splits": [asdict(split) for split in self.splits],
            "target_indices": self.target_indices.tolist(),
            "actuals": self.actuals.tolist(),
            "forecasts": self.forecasts.tolist(),
        }

    def to_frame(self):
        """Return the long forecast table; pandas is an optional dependency."""
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "to_frame() requires pandas; install pandas or the 'pandas' extra."
            ) from exc
        return pd.DataFrame.from_records(self.records())


def backtest(
    y: ArrayLike,
    forecasters: Mapping[str, Forecaster],
    *,
    initial_train_size: int,
    horizon: int = 1,
    step: int = 1,
    gap: int = 0,
    window: int | None = None,
) -> BacktestResult:
    """Forecast every complete fold using only that fold's training history.

    Each callable receives ``(train_1d, lead_times_1d)`` and must return a finite
    real one-dimensional forecast of length ``horizon``. Lead times are measured
    from the last training observation: ``gap + 1, ..., gap + horizon``.
    Model names and output columns retain the mapping's insertion order.

    Each model/fold gets independent read-only copies of its history and lead
    times. Their NumPy bases expose no future observations. This isolates the
    data passed by this function; a callable must also avoid using future data
    through its own captured variables, external state or other data sources.
    Failed calls or invalid forecasts stop the run, rather than omitting a fold.
    """
    values = _readonly_copy(_real_vector(y, "y"), np.float64)
    if not isinstance(forecasters, Mapping) or not forecasters:
        raise ValueError("forecasters must be a nonempty mapping of model names to callables.")
    models = tuple(forecasters.items())
    if any(not isinstance(name, str) or not name.strip() for name, _ in models):
        raise ValueError("forecasters must use nonempty string model names.")
    if any(not callable(callback) for _, callback in models):
        raise ValueError("Every forecaster must be callable.")
    splits = rolling_splits(
        len(values),
        initial_train_size=initial_train_size,
        horizon=horizon,
        step=step,
        gap=gap,
        window=window,
    )
    horizon = splits[0].test_stop - splits[0].test_start
    starts = np.fromiter((split.test_start for split in splits), dtype=np.int64)
    target_indices = starts[:, None] + np.arange(horizon, dtype=np.int64)
    actuals = values[target_indices]
    forecasts = np.empty((len(splits), horizon, len(models)), dtype=np.float64)
    leads = np.arange(gap + 1, gap + horizon + 1, dtype=np.int64)
    for fold, split in enumerate(splits):
        for model, (name, callback) in enumerate(models):
            train = _readonly_copy(values[split.train_start : split.train_stop], np.float64)
            model_leads = _readonly_copy(leads, np.int64)
            context = f"Forecaster {name!r} at fold {fold} (origin {split.origin})"
            try:
                prediction = callback(train, model_leads)
            except Exception as exc:
                raise RuntimeError(f"{context} failed: {exc}") from exc
            prediction = _real_vector(prediction, f"{context} forecasts")
            if len(prediction) != horizon:
                raise ValueError(f"{context} must return exactly {horizon} forecasts.")
            forecasts[fold, :, model] = prediction
    for array in (forecasts, target_indices, actuals):
        array.flags.writeable = False
    return BacktestResult(
        len(values),
        tuple(name for name, _ in models),
        splits,
        forecasts,
        target_indices,
        actuals,
    )


def naive_forecast(train: ArrayLike, lead_times: ArrayLike) -> NDArray[np.float64]:
    """Repeat the last training observation at every requested positive lead."""
    values, leads = _real_vector(train, "train"), _lead_times(lead_times)
    return np.full(len(leads), values[-1], dtype=np.float64)


@dataclass(frozen=True)
class SeasonalNaive:
    """Repeat the final full cycle; ``period`` is specified before the backtest."""

    period: int

    def __post_init__(self):
        object.__setattr__(self, "period", positive_integer(self.period, "period"))

    def __call__(self, train: ArrayLike, lead_times: ArrayLike) -> NDArray[np.float64]:
        values, leads = _real_vector(train, "train"), _lead_times(lead_times)
        if len(values) < self.period:
            raise ValueError("SeasonalNaive requires at least period training observations.")
        indices = len(values) - self.period + (leads - 1) % self.period
        return values[indices]


def drift_forecast(train: ArrayLike, lead_times: ArrayLike) -> NDArray[np.float64]:
    """Extend the average first-to-last training change to each requested lead."""
    values, leads = _real_vector(train, "train"), _lead_times(lead_times)
    if len(values) < 2:
        raise ValueError("drift_forecast requires at least two training observations.")
    with np.errstate(over="ignore", invalid="ignore"):
        forecasts = values[-1] + leads * ((values[-1] - values[0]) / (len(values) - 1))
    return _real_vector(forecasts, "Drift forecasts")


@dataclass(frozen=True)
class Autoregression:
    """Fit a recursive dense or sparse AR with an intercept, using history only.

    Integer ``lags=p`` uses 1, ..., p. A tuple selects distinct positive lags,
    e.g. ``(1, 2, 24, 168)``. At least ``max_lag + n_lags + 1`` training
    observations are required (``2 * p + 1`` for dense AR). Training values are
    centered and divided by their largest absolute deviation before fitting.
    ``ridge`` minimizes normalized residual sum of squares plus
    ``ridge * sum(coefficients**2)``; the intercept is not penalized. Zero ridge
    is ordinary least squares; rank-deficient centered coefficient systems use
    the minimum-norm solution.
    Leads may be unordered or nonconsecutive; intervening values are forecast
    recursively. There is no automatic parameter choice or state across calls.
    """

    lags: int | tuple[int, ...] = 12
    ridge: float = 0.0

    def __post_init__(self):
        if isinstance(self.lags, tuple):
            lags = tuple(positive_integer(lag, "lags") for lag in self.lags)
            if not lags or len(set(lags)) != len(lags):
                raise ValueError("lags must be a nonempty tuple of distinct positive integers.")
            object.__setattr__(self, "lags", tuple(sorted(lags)))
        else:
            object.__setattr__(self, "lags", positive_integer(self.lags, "lags"))
        if np.ma.isMaskedArray(self.ridge) or not np.isscalar(self.ridge):
            raise ValueError("ridge must be a finite nonnegative real scalar.")
        try:
            raw = np.asarray(self.ridge)
            if raw.dtype.kind not in "iuf":
                raise ValueError
            ridge = float(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("ridge must be a finite nonnegative real scalar.") from exc
        if not np.isfinite(ridge) or ridge < 0:
            raise ValueError("ridge must be a finite nonnegative real scalar.")
        object.__setattr__(self, "ridge", ridge)

    def __call__(self, train: ArrayLike, lead_times: ArrayLike) -> NDArray[np.float64]:
        values = _real_vector(train, "train")
        leads = _lead_times(lead_times)
        dense = isinstance(self.lags, int)
        count = self.lags if dense else len(self.lags)
        maximum_lag = self.lags if dense else self.lags[-1]
        if len(values) < maximum_lag + count + 1:
            condition = "2 * lags + 1" if dense else "max(lags) + len(lags) + 1"
            raise ValueError(f"Autoregression requires at least {condition} training observations.")
        # Normalize before centering, so opposite finite extremes do not overflow.
        magnitude = float(np.max(np.abs(values)))
        scaled = values / magnitude if magnitude else np.zeros_like(values)
        center = float(scaled.mean())
        centered = scaled - center
        spread = float(np.max(np.abs(centered)))
        if not spread:
            return np.full(len(leads), values[-1], dtype=np.float64)
        normalized = centered / spread
        rows = np.lib.stride_tricks.sliding_window_view(normalized, maximum_lag + 1)
        indices = slice(None) if dense else np.asarray(self.lags, dtype=np.intp) - 1
        design = rows[:, :-1][:, ::-1] if dense else rows[:, maximum_lag - indices - 1]
        target = rows[:, -1]
        x_mean, y_mean = design.mean(axis=0), float(target.mean())
        design, target = design - x_mean, target - y_mean
        if self.ridge:
            design = np.vstack((design, np.sqrt(self.ridge) * np.eye(count)))
            target = np.concatenate((target, np.zeros(count)))
        coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
        intercept = y_mean - x_mean @ coefficients
        history = normalized[-maximum_lag:][::-1].copy()
        unique_leads, inverse = np.unique(leads, return_inverse=True)
        forecasts = np.empty(len(unique_leads))
        requested = 0
        with np.errstate(over="ignore", invalid="ignore"):
            for lead in range(1, int(unique_leads[-1]) + 1):
                prediction = intercept + history[indices] @ coefficients
                if not np.isfinite(prediction):
                    raise ValueError("Autoregression produced a non-finite recursive forecast.")
                history[1:] = history[:-1]
                history[0] = prediction
                if lead == unique_leads[requested]:
                    forecasts[requested] = (prediction * spread + center) * magnitude
                    requested += 1
        return _real_vector(forecasts[inverse], "Autoregression forecasts")


@dataclass(frozen=True)
class Differenced:
    """Forecast differences with any callback, then restore original levels.

    ``period=1`` uses successive changes; larger periods use seasonal changes.
    Each call differences only its training history, calls the supplied model
    once for every lead from 1 to the maximum requested lead, and recursively
    restores levels. Requested leads may be unordered, repeated or gapped.
    No differencing order, seasonality or model is selected automatically.
    """

    forecaster: Forecaster
    period: int = 1

    def __post_init__(self):
        if not callable(self.forecaster):
            raise ValueError("forecaster must be callable.")
        object.__setattr__(self, "period", positive_integer(self.period, "period"))

    def __call__(self, train: ArrayLike, lead_times: ArrayLike) -> NDArray[np.float64]:
        values, leads = _real_vector(train, "train"), _lead_times(lead_times)
        if len(values) <= self.period:
            raise ValueError("Differenced requires more than period training observations.")
        try:
            with np.errstate(over="raise", invalid="raise"):
                differences = values[self.period:] - values[:-self.period]
        except FloatingPointError as exc:
            raise ValueError("Training differences are outside the finite float range.") from exc
        differences.flags.writeable = False
        all_leads = np.arange(1, int(leads.max()) + 1, dtype=np.int64)
        all_leads.flags.writeable = False
        predictions = _real_vector(self.forecaster(differences, all_leads), "Difference forecasts")
        if len(predictions) != len(all_leads):
            raise ValueError("forecaster must return one forecast for every intervening lead.")
        restored = predictions.copy()
        try:
            with np.errstate(over="raise", invalid="raise"):
                for index in range(len(restored)):
                    anchor = (
                        values[len(values) - self.period + index]
                        if index < self.period else restored[index - self.period]
                    )
                    restored[index] += anchor
        except FloatingPointError as exc:
            raise ValueError("Restored forecasts are outside the finite float range.") from exc
        return restored[leads - 1]
