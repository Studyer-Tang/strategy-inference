"""Integer-time multi-step intervals with transactional, mature feedback.

The quantile recursion uses the existing bounded residual transform. Each lead
has either one delayed pooled state or origin-modulo-lead interlaced states.
Coverage bounds describe retrospective averages in ideal arithmetic; these
ordinary binary64 calculations are not rounding certificates or pathwise bounds.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from math import fsum, hypot, inf, isfinite, sqrt
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import positive_integer
from .conformal import AdaptiveConformal, IntervalKind, _bounded_score, _finite

_MAX_TIME = np.iinfo(np.int64).max


def _contains_bool(values):
    if isinstance(values, (bool, np.bool_)):
        return True
    return isinstance(values, (list, tuple)) and any(_contains_bool(value) for value in values)


def _array(values, name, *, integer=False):
    try:
        if _contains_bool(values) or (
            np.ma.isMaskedArray(values) and np.any(np.ma.getmaskarray(values))
        ):
            raise ValueError
        raw = np.asarray(values)
        if not raw.size or raw.dtype.kind not in ("iu" if integer else "iuf"):
            raise ValueError
        if integer:
            if np.any(raw > _MAX_TIME):
                raise ValueError
            return np.array(raw, dtype=np.int64, copy=True)
        with np.errstate(over="ignore", invalid="ignore"):
            result = np.array(raw, dtype=np.float64, copy=True)
        if not np.isfinite(result).all():
            raise ValueError
        return result
    except (TypeError, ValueError, OverflowError) as exc:
        kind = "integer" if integer else "finite real numeric"
        raise ValueError(f"{name} must contain nonempty, unmasked, nonboolean {kind} values.") from exc


def _leads(values):
    leads = _array(values, "lead_times", integer=True)
    if leads.ndim != 1 or np.any(leads <= 0) or np.any(leads[1:] <= leads[:-1]):
        raise ValueError("lead_times must be a strictly increasing vector of positive integers.")
    return tuple(int(value) for value in leads)


def _time(value, name):
    value = positive_integer(value, name, 0)
    if value > _MAX_TIME:
        raise ValueError(f"{name} must fit in int64.")
    return value


def _scales(values, h, name):
    if np.isscalar(values):
        value = _finite(values, name)
        result = np.full(h, value)
    else:
        result = _array(values, name)
        if result.shape != (h,):
            raise ValueError(f"{name} must be a scalar or a vector with {h} entries.")
    if np.any(result <= 0):
        raise ValueError(f"{name} must be positive and finite.")
    return tuple(float(value) for value in result)


@dataclass(frozen=True, slots=True)
class MultiStepInterval:
    origin: int
    target: int
    lead_time: int
    prediction: float
    lower: float
    upper: float
    quantile: float
    scale: float
    kind: IntervalKind

    def to_dict(self):
        record = asdict(self)
        if self.kind != "finite":
            record["lower"] = record["upper"] = None
        return record


@dataclass(frozen=True, slots=True)
class MultiStepUpdate:
    origin: int
    target: int
    lead_time: int
    actual: float
    prediction: float
    scale: float
    issued_quantile: float
    previous_quantile: float
    quantile: float
    step_size: float
    score: float
    miss: bool
    lane: int
    n_updates: int

    @property
    def new_quantile(self):
        return self.quantile

    @property
    def eta(self):
        return self.step_size

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _Lane:
    quantile: float
    n_updates: int = 0
    misses: int = 0


class MultiStepConformal:
    """Observe each integer time, then optionally issue once at that time.

    ``start_time`` is the first required observation, not an already observed
    origin. Missing prediction times are permitted; missing observation times
    are not. Each forecast keeps its issued threshold and scale. Pooled updates
    add feedback to the current threshold, rather than resetting to an old
    issued threshold. Interlaced states use ``origin % physical_lead``.

    ``step_size`` is a positive scalar or one constant per configured lead;
    scalar values broadcast. Feedback clocks are per lead for pooled states
    and per lane for interlaced states. No threshold is clipped.

    Scale overrides apply only to one issuance. Optional EWMA RMS scales use
    mature residuals; ``shortest`` shares the shortest lead's residual source
    using the original scale ratios, followed by per-lead floors. Floors apply
    to adaptive updates, not to the supplied initial or override scales.
    """

    __slots__ = (
        "_leads", "_lead_index", "_start", "_last", "_last_prediction", "_alpha",
        "_step_size", "_decay", "_initial_quantile", "_strategy", "_initial_scales",
        "_scales", "_floors", "_scale_decay", "_scale_source", "_source_scale",
        "_ratios", "_states", "_pending", "_issued", "_evaluated", "_misses",
    )

    def __init__(
        self, lead_times: ArrayLike, *, start_time: int = 0, alpha: float = 0.1,
        step_size: ArrayLike = 0.1, decay: float = 0.6, scale: ArrayLike = 1.0,
        initial_quantile: float = 0.5, strategy: str = "pooled",
        scale_decay: float | None = None, scale_source: str = "horizon",
        scale_floor: ArrayLike = 1e-8,
    ):
        leads = _leads(lead_times)
        start = _time(start_time, "start_time")
        if start + leads[-1] > _MAX_TIME:
            raise ValueError("A forecast target would exceed int64.")
        base = AdaptiveConformal(alpha=alpha, step_size=1.0, decay=decay,
            initial_quantile=initial_quantile)
        if not isinstance(strategy, str) or strategy not in ("pooled", "interlaced"):
            raise ValueError("strategy must be 'pooled' or 'interlaced'.")
        if not isinstance(scale_source, str) or scale_source not in ("horizon", "shortest"):
            raise ValueError("scale_source must be 'horizon' or 'shortest'.")
        if scale_decay is not None:
            scale_decay = _finite(scale_decay, "scale_decay")
            if not 0 <= scale_decay < 1:
                raise ValueError("scale_decay must lie in [0, 1).")
        if scale_source == "shortest" and scale_decay is None:
            raise ValueError("scale_source='shortest' requires scale_decay.")
        initial = _scales(scale, len(leads), "scale")
        floors = _scales(scale_floor, len(leads), "scale_floor")
        rates = _scales(step_size, len(leads), "step_size")
        ratios = tuple(value / initial[0] for value in initial) if scale_source == "shortest" else ()
        if any(not isfinite(value) or value <= 0 for value in ratios):
            raise ValueError("Initial scale ratios exceed the supported float range.")
        self._leads, self._lead_index = leads, {lead: i for i, lead in enumerate(leads)}
        self._start, self._last, self._last_prediction = start, None, None
        self._alpha, self._step_size, self._decay = base.alpha, rates, base.decay
        self._initial_quantile, self._strategy = base.initial_quantile, strategy
        self._initial_scales, self._scales, self._floors = initial, list(initial), floors
        self._scale_decay, self._scale_source = scale_decay, scale_source
        self._source_scale, self._ratios = initial[0], ratios
        self._states: dict[tuple[int, int], _Lane] = {}
        self._pending: dict[int, list[MultiStepInterval]] = {}
        self._issued, self._evaluated, self._misses = ([0] * len(leads) for _ in range(3))

    @property
    def lead_times(self):
        return self._leads

    @property
    def last_time(self):
        return self._last

    @property
    def next_time(self):
        return self._start if self._last is None else self._last + 1

    @property
    def current_scales(self):
        return tuple(self._scales)

    @property
    def step_size(self):
        """The configured positive learning-rate constant for each lead."""
        return self._step_size

    @property
    def pending(self):
        return tuple(interval for target in sorted(self._pending) for interval in self._pending[target])

    @property
    def n_updates(self):
        return tuple(self._evaluated)

    @property
    def coverage(self):
        return tuple(1 - misses / n if n else None
            for misses, n in zip(self._misses, self._evaluated, strict=True))

    def _lane(self, i, origin):
        return origin % self._leads[i] if self._strategy == "interlaced" else 0

    def _state(self, i, lane):
        return self._states.get((i, lane), _Lane(self._initial_quantile))

    def lane_state(self, lead_time: int, lane: int = 0):
        """Detached state, including not-yet-used lanes without allocating them."""
        lead_time = positive_integer(lead_time, "lead_time")
        if lead_time not in self._lead_index:
            raise ValueError("lead_time must be configured.")
        lane = positive_integer(lane, "lane", 0)
        if lane >= (lead_time if self._strategy == "interlaced" else 1):
            raise ValueError("lane is outside the configured strategy.")
        return asdict(self._state(self._lead_index[lead_time], lane))

    def predict(self, predicted: ArrayLike, *, scale: ArrayLike | None = None):
        """Issue all configured leads atomically at the latest observed time."""
        if self._last is None:
            raise RuntimeError("Observe the current time before predicting.")
        if self._last_prediction == self._last:
            raise RuntimeError("Predictions were already issued at this time.")
        points = _array(predicted, "predicted")
        if points.shape != (len(self._leads),):
            raise ValueError("predicted must have one entry per configured lead.")
        scales = self.current_scales if scale is None else _scales(scale, len(self._leads), "scale")
        if self._last + self._leads[-1] > _MAX_TIME:
            raise ValueError("A forecast target would exceed int64.")
        intervals, keys = [], []
        for i, (lead, point, width_scale) in enumerate(zip(self._leads, points, scales, strict=True)):
            point = float(point)
            key = (i, self._lane(i, self._last))
            q = self._state(*key).quantile
            if q < 0:
                lower, upper, kind = inf, -inf, "empty"
            elif q >= 1:
                lower, upper, kind = -inf, inf, "unbounded"
            else:
                radius = width_scale * (q / (1 - q))
                lower, upper, kind = point - radius, point + radius, "finite"
                if not isfinite(lower) or not isfinite(upper):
                    raise ValueError("Interval radius or boundary overflow.")
            intervals.append(MultiStepInterval(self._last, self._last + lead, lead,
                point, lower, upper, q, width_scale, kind))
            keys.append(key)
        for key, interval in zip(keys, intervals, strict=True):
            self._states.setdefault(key, _Lane(self._initial_quantile))
            self._pending.setdefault(interval.target, []).append(interval)
        self._issued = [n + 1 for n in self._issued]
        self._last_prediction = self._last
        return tuple(intervals)

    def observe(self, time: int, actual: float):
        """Advance exactly one time and commit all mature feedback together."""
        time = _time(time, "time")
        if time != self.next_time:
            raise ValueError(f"Observe time {self.next_time} next; repeated or skipped times are invalid.")
        actual = _finite(actual, "actual")
        due = sorted(self._pending.get(time, ()), key=lambda interval: interval.lead_time)
        updates, states, residuals = [], [], {}
        for interval in due:
            i = self._lead_index[interval.lead_time]
            lane = self._lane(i, interval.origin)
            state = self._state(i, lane)
            residual = abs(actual - interval.prediction)
            if not isfinite(residual):
                raise ValueError("Residual overflow; use smaller numeric units.")
            miss = interval.kind == "empty" or not interval.lower <= actual <= interval.upper
            n = state.n_updates + 1
            eta = self._step_size[i] * n ** (-self._decay)
            if not isfinite(eta) or eta <= 0:
                raise ValueError("The feedback step size underflowed.")
            q = state.quantile + eta * (int(miss) - self._alpha)
            if not isfinite(q):
                raise ValueError("Quantile update overflow.")
            updates.append(MultiStepUpdate(interval.origin, time, interval.lead_time, actual,
                interval.prediction, interval.scale, interval.quantile, state.quantile, q, eta,
                _bounded_score(residual, interval.scale), bool(miss), lane, n))
            states.append(((i, lane), _Lane(q, n, state.misses + int(miss))))
            residuals[i] = residual
        new_scales, source = list(self._scales), self._source_scale
        if self._scale_decay is not None and residuals:
            beta = self._scale_decay
            if self._scale_source == "horizon":
                for i, residual in residuals.items():
                    new_scales[i] = max(self._floors[i],
                        hypot(sqrt(beta) * self._scales[i], sqrt(1 - beta) * residual))
            elif 0 in residuals:
                source = max(self._floors[0],
                    hypot(sqrt(beta) * source, sqrt(1 - beta) * residuals[0]))
                new_scales = [max(floor, source * ratio)
                    for floor, ratio in zip(self._floors, self._ratios, strict=True)]
            if not isfinite(source) or any(not isfinite(value) or value <= 0 for value in new_scales):
                raise ValueError("Adaptive scale update overflow.")
        # Nothing above this point mutates time, queues, scales or any lane.
        for key, state in states:
            self._states[key] = state
        for update in updates:
            i = self._lead_index[update.lead_time]
            self._evaluated[i] += 1
            self._misses[i] += int(update.miss)
        self._scales, self._source_scale = new_scales, source
        self._pending.pop(time, None)
        self._last = time
        return tuple(updates)

    @property
    def coverage_bound(self):
        """Per-lead ideal-arithmetic average-error bounds, capped at one.

        Pooled states use (1 + sum_{j<=min(h,n)} eta_j)/(n*eta_n).
        Interlaced states sum the individual lane bounds, weighted by count.
        No feedback gives None. This property performs the bound calculation;
        issuing and feedback do not sum a lead-length step-size series.
        """
        bounds = []
        for i, (lead, n) in enumerate(zip(self._leads, self._evaluated, strict=True)):
            if not n:
                bounds.append(None)
                continue
            if self._strategy == "pooled":
                eta = self._step_size[i] * n ** (-self._decay)
                step_ratio_sum = fsum(j ** (-self._decay) for j in range(1, min(lead, n) + 1))
                bound = (1 / eta + step_ratio_sum * (self._step_size[i] / eta)) / n
            else:
                terms = []
                for (h, _), state in self._states.items():
                    if h == i and state.n_updates:
                        eta = self._step_size[i] * state.n_updates ** (-self._decay)
                        terms.append(1 / eta + self._step_size[i] / eta)
                bound = sum(terms) / n
            bounds.append(min(1.0, bound))
        return tuple(bounds)

    def summary(self):
        pending = [0] * len(self._leads)
        for group in self._pending.values():
            for interval in group:
                pending[self._lead_index[interval.lead_time]] += 1
        coverage, bounds = self.coverage, self.coverage_bound
        return [dict(lead_time=lead, n_issued=self._issued[i], n_evaluated=self._evaluated[i],
            n_pending=pending[i], coverage=coverage[i], coverage_bound=bounds[i])
            for i, lead in enumerate(self._leads)]

    def to_dict(self):
        """Strict JSON with active lanes; unused lanes retain the stated default."""
        return dict(schema_version=1, method="multistep_conformal", lead_times=list(self._leads),
            start_time=self._start, last_time=self.last_time, next_time=self.next_time,
            alpha=self._alpha, step_size=list(self._step_size), decay=self._decay,
            initial_quantile=self._initial_quantile, strategy=self._strategy,
            initial_scale=list(self._initial_scales), current_scales=list(self._scales),
            scale_decay=self._scale_decay, scale_source=self._scale_source, scale_floor=list(self._floors),
            default_lane_state=asdict(_Lane(self._initial_quantile)),
            lane_counts=[lead if self._strategy == "interlaced" else 1 for lead in self._leads],
            states=[dict(lead_time=self._leads[i], lane=lane, **asdict(state))
                for (i, lane), state in sorted(self._states.items())],
            summary=self.summary(), pending=[interval.to_dict() for interval in self.pending],
            coverage_bound_arithmetic="ordinary binary64; not a rounding certificate")


def _mean_nonnegative(values):
    scale = max(values)
    return (fsum(value / scale for value in values) / len(values)) * scale if scale else 0.0


@dataclass(frozen=True)
class MultiStepResult:
    origins: NDArray[np.int64]
    target_indices: NDArray[np.int64]
    lead_times: tuple[int, ...]
    actual: NDArray[np.float64]
    predicted: NDArray[np.float64]
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    quantiles: NDArray[np.float64]
    scales: NDArray[np.float64]
    evaluated: NDArray[np.bool_]
    empty: NDArray[np.bool_]
    unbounded: NDArray[np.bool_]
    misses: NDArray[np.bool_]
    step_sizes: NDArray[np.float64]
    pending: tuple[MultiStepInterval, ...]
    _state_record: dict[str, Any] = field(repr=False, compare=False)

    @property
    def n_origins(self):
        return len(self.origins)

    @property
    def coverage(self):
        return tuple(1 - int(np.count_nonzero(self.evaluated[:, h] & self.misses[:, h])) / n
            if (n := int(np.count_nonzero(self.evaluated[:, h]))) else None
            for h in range(len(self.lead_times)))

    @property
    def coverage_bound(self):
        return tuple(row["coverage_bound"] for row in self._state_record["summary"])

    @property
    def feedback_steps(self):
        return self.step_sizes

    @property
    def diagnostics(self):
        return deepcopy(self._state_record)

    def records(self):
        rows = []
        for f, origin in enumerate(self.origins):
            for h, lead in enumerate(self.lead_times):
                observed = bool(self.evaluated[f, h])
                finite = not self.empty[f, h] and not self.unbounded[f, h]
                kind = "finite" if finite else "empty" if self.empty[f, h] else "unbounded"
                rows.append(dict(origin=int(origin), target=int(self.target_indices[f, h]),
                    lead_time=lead, prediction=float(self.predicted[f, h]),
                    lower=float(self.lower[f, h]) if finite else None,
                    upper=float(self.upper[f, h]) if finite else None, kind=kind,
                    quantile=float(self.quantiles[f, h]), scale=float(self.scales[f, h]),
                    evaluated=observed, actual=float(self.actual[f, h]) if observed else None,
                    miss=bool(self.misses[f, h]) if observed else None,
                    step_size=float(self.step_sizes[f, h]) if observed else None))
        return rows

    def summary(self):
        rows = []
        bounds = self.coverage_bound
        for h, lead in enumerate(self.lead_times):
            mask = self.evaluated[:, h]
            n = int(mask.sum())
            finite = mask & ~self.empty[:, h] & ~self.unbounded[:, h]
            widths = [float(self.upper[f, h]) - float(self.lower[f, h]) for f in np.flatnonzero(finite)]
            width_ok = bool(widths) and all(isfinite(value) for value in widths)
            empty, full = int(np.count_nonzero(mask & self.empty[:, h])), int(np.count_nonzero(mask & self.unbounded[:, h]))
            status, score = "no_evaluated_intervals", None
            if n and (empty or full):
                status = "empty_or_unbounded"
            elif n:
                alpha = self._state_record["alpha"]
                scores = [float(self.upper[f, h]) - float(self.lower[f, h])
                    + 2 * (max(float(self.lower[f, h]) - float(self.actual[f, h]), 0) / alpha)
                    + 2 * (max(float(self.actual[f, h]) - float(self.upper[f, h]), 0) / alpha)
                    for f in np.flatnonzero(mask)]
                status = "finite" if all(isfinite(value) for value in scores) else "overflow"
                if status == "finite":
                    score = _mean_nonnegative(scores)
            rows.append(dict(lead_time=lead, n_issued=self.n_origins, n_evaluated=n,
                n_pending=self.n_origins - n,
                coverage=1 - int(np.count_nonzero(mask & self.misses[:, h])) / n if n else None,
                coverage_bound=bounds[h], empty_rate=empty / n if n else None,
                unbounded_rate=full / n if n else None, finite_width_count=len(widths),
                mean_finite_width=_mean_nonnegative(widths) if width_ok else None,
                finite_width_status="finite" if width_ok else "overflow" if widths else "no_finite_evaluations",
                mean_interval_score=score, interval_score_status=status))
        return rows

    def to_dict(self):
        state = self.diagnostics
        state.update(n_origins=self.n_origins, observed_through=state["last_time"],
            summary=self.summary(), intervals=self.records(),
            finite_width_scope="finite evaluated intervals only")
        return state

    def to_frame(self):
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("to_frame() requires pandas; install the 'pandas' extra.") from exc
        return pd.DataFrame.from_records(self.records())


def multistep_intervals(
    actual: ArrayLike, predicted: ArrayLike, *, origins: ArrayLike, lead_times: ArrayLike,
    alpha: float = 0.1, step_size: ArrayLike = 0.1, decay: float = 0.6,
    scale: ArrayLike = 1.0, initial_quantile: float = 0.5, strategy: str = "pooled",
    scale_decay: float | None = None, scale_source: str = "horizon", scale_floor: ArrayLike = 1e-8,
) -> MultiStepResult:
    """Replay available labels in integer time; future-tail forecasts stay pending.

    ``actual[t]`` is observed at time t. Forecast origins must be increasing
    positions within that series; a target may exceed its end. Every time from
    the first origin to the final observation is visited, including times without
    forecasts. Points must already have been made without future information.
    This wrapper neither trains a model nor validates its external provenance.
    """
    observed = _array(actual, "actual")
    points = _array(predicted, "predicted")
    origin = _array(origins, "origins", integer=True)
    leads = _leads(lead_times)
    if observed.ndim != 1:
        raise ValueError("actual must be one-dimensional.")
    if origin.ndim != 1 or np.any(origin < 0) or np.any(origin[1:] <= origin[:-1]) or origin[-1] >= len(observed):
        raise ValueError("origins must be strictly increasing positions within actual.")
    if points.shape != (len(origin), len(leads)):
        raise ValueError("predicted must have shape (origins, lead_times).")
    if int(origin[-1]) + leads[-1] > _MAX_TIME:
        raise ValueError("A forecast target would exceed int64.")
    tracker = MultiStepConformal(leads, start_time=int(origin[0]), alpha=alpha,
        step_size=step_size, decay=decay, scale=scale, initial_quantile=initial_quantile,
        strategy=strategy, scale_decay=scale_decay, scale_source=scale_source, scale_floor=scale_floor)
    shape = points.shape
    target = origin[:, None] + np.asarray(leads, dtype=np.int64)
    values, steps = (np.full(shape, np.nan) for _ in range(2))
    lower, upper, quantiles, scales = (np.empty(shape) for _ in range(4))
    evaluated, empty, full, misses = (np.zeros(shape, dtype=bool) for _ in range(4))
    row_by_origin = {int(value): i for i, value in enumerate(origin)}
    lead_index = {lead: h for h, lead in enumerate(leads)}
    for time in range(int(origin[0]), len(observed)):
        for update in tracker.observe(time, float(observed[time])):
            f, h = row_by_origin[update.origin], lead_index[update.lead_time]
            values[f, h], steps[f, h] = update.actual, update.step_size
            evaluated[f, h], misses[f, h] = True, update.miss
        if time in row_by_origin:
            f = row_by_origin[time]
            for h, interval in enumerate(tracker.predict(points[f])):
                lower[f, h], upper[f, h] = interval.lower, interval.upper
                quantiles[f, h], scales[f, h] = interval.quantile, interval.scale
                empty[f, h], full[f, h] = interval.kind == "empty", interval.kind == "unbounded"
    for array in (origin, target, points, values, steps, lower, upper, quantiles, scales,
                  evaluated, empty, full, misses):
        array.flags.writeable = False
    return MultiStepResult(origin, target, leads, values, points, lower, upper, quantiles,
        scales, evaluated, empty, full, misses, steps, tracker.pending, tracker.to_dict())
