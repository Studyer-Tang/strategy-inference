"""Research-only paired scale transfer; no forecasts, labels or trackers stored.

A fast RMS A follows the shortest lead. For each longer lead, paired slow
RMS values B_h and C_h use the same mature target and update mask. The next
issuance scale is max(floor, A * B_h / C_h); the shortest ratio is one.
This is an engineering baseline, not a new conformal guarantee. Fixed-decay
RMS values are filters, not consistent variance estimators. Heavy tails and
forecast bias remain limitations.

Each update mapping must contain absolute residuals for ONE mature target.
The caller is responsible for target identity: residuals from separate calls
are never paired. Scales are read before issuing a forecast and frozen by the
caller. Issuance overrides do not change this helper's state.

The floor applies to output scales immediately and to every updated RMS
component. Initial components retain their supplied physical units. Numeric
failures leave all state unchanged. Ordinary binary64 is used, without a
rounding certificate; unrepresentable ratios or scales are rejected.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import hypot, isfinite, sqrt
from numbers import Integral, Real
from typing import Any


def _real(value: Real, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite nonboolean real scalar.")
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite nonboolean real scalar.") from exc
    if not isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{name} must be {'positive and ' if positive else ''}finite.")
    return result


def _lead(value: Integral) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError("Physical leads must be positive nonboolean integers.")
    return int(value)


def _sequence(values, name):
    try:
        result = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a nonempty vector.") from exc
    if not result:
        raise ValueError(f"{name} must be a nonempty vector.")
    return result


class MatureScaleTransfer:
    """Constant-memory filters over fixed physical leads and mature residuals.

    ``initial_scales`` has one positive value per strictly increasing lead.
    ``scales`` returns a detached tuple. ``update`` atomically consumes one
    same-target residual mapping and returns the next scales. A longer lead
    without the shortest lead in that call cannot update any filter; the
    shortest by itself updates A but leaves all paired ratios unchanged.

    The slow short-reference C_h is separate for each h, so sparse feedback
    never gives numerator and denominator different target histories.
    ``paired_counts[0]`` is zero: the shortest lead does not fit its own ratio.
    """

    __slots__ = (
        "_leads",
        "_index",
        "_initial",
        "_scale_decay",
        "_ratio_decay",
        "_floor",
        "_fast_weights",
        "_slow_weights",
        "_fast",
        "_numerator",
        "_reference",
        "_paired",
        "_shortest_updates",
        "_scales",
        "_ratios",
    )

    def __init__(
        self,
        initial_scales,
        lead_times,
        *,
        scale_decay: float = 0.97,
        ratio_decay: float = 0.995,
        floor: float = 1e-8,
    ) -> None:
        leads = tuple(_lead(value) for value in _sequence(lead_times, "lead_times"))
        if any(right <= left for left, right in zip(leads, leads[1:], strict=False)):
            raise ValueError("lead_times must be strictly increasing.")
        initial = tuple(
            _real(value, "initial_scales", positive=True)
            for value in _sequence(initial_scales, "initial_scales")
        )
        if len(initial) != len(leads):
            raise ValueError("initial_scales must have one value per configured lead.")
        fast_decay = _real(scale_decay, "scale_decay")
        slow_decay = _real(ratio_decay, "ratio_decay")
        if not 0 <= fast_decay < 1 or not 0 <= slow_decay < 1:
            raise ValueError("scale_decay and ratio_decay must lie in [0, 1).")
        floor = _real(floor, "floor", positive=True)
        ratios = tuple(value / initial[0] for value in initial)
        if any(not isfinite(value) or value <= 0 for value in ratios):
            raise ValueError("Initial scale ratios exceed the supported float range.")
        self._leads, self._index = leads, {lead: i for i, lead in enumerate(leads)}
        self._initial, self._floor = initial, floor
        self._scale_decay, self._ratio_decay = fast_decay, slow_decay
        self._fast_weights = sqrt(fast_decay), sqrt(1 - fast_decay)
        self._slow_weights = sqrt(slow_decay), sqrt(1 - slow_decay)
        self._fast = initial[0]
        self._numerator = list(initial)
        self._reference = [initial[0]] * len(leads)
        self._paired, self._shortest_updates = [0] * len(leads), 0
        self._ratios = ratios
        # Preserve each supplied first-issuance scale without divide/multiply rounding.
        self._scales = tuple(max(floor, value) for value in initial)

    @property
    def lead_times(self) -> tuple[int, ...]:
        return self._leads

    @property
    def scales(self) -> tuple[float, ...]:
        return self._scales

    @property
    def ratios(self) -> tuple[float, ...]:
        return self._ratios

    @property
    def paired_counts(self) -> tuple[int, ...]:
        return tuple(self._paired)

    def _rms(self, previous, residual, weights):
        value = max(self._floor, hypot(weights[0] * previous, weights[1] * residual))
        if not isfinite(value):
            raise ValueError("RMS update overflow.")
        return value

    def update(self, residuals: Mapping[Integral, Real]) -> tuple[float, ...]:
        """Consume same-target absolute residuals; empty/unpaired calls are no-ops."""
        if not isinstance(residuals, Mapping):
            raise ValueError("residuals must map physical leads to absolute residuals.")
        values = {}
        for key, raw in residuals.items():
            lead = _lead(key)
            if lead not in self._index:
                raise ValueError(f"Unknown physical lead {lead}.")
            value = _real(raw, "residual")
            if value < 0:
                raise ValueError("Residuals must be nonnegative absolute residuals.")
            values[lead] = value
        shortest = self._leads[0]
        if shortest not in values:
            return self._scales
        short_residual = values[shortest]
        fast = self._rms(self._fast, short_residual, self._fast_weights)
        numerator, reference, counts = (
            list(self._numerator),
            list(self._reference),
            list(self._paired),
        )
        for lead, residual in values.items():
            i = self._index[lead]
            if i:
                numerator[i] = self._rms(numerator[i], residual, self._slow_weights)
                reference[i] = self._rms(reference[i], short_residual, self._slow_weights)
                counts[i] += 1
        ratios = (1.0, *(numerator[i] / reference[i] for i in range(1, len(self._leads))))
        if any(not isfinite(value) or value <= 0 for value in ratios):
            raise ValueError("Updated scale ratios exceed the supported float range.")
        scales = tuple(max(self._floor, fast * ratio) for ratio in ratios)
        if any(not isfinite(value) for value in scales):
            raise ValueError("Transferred scale update overflow.")
        # All validation and prospective computations finish before any state changes.
        self._fast, self._numerator, self._reference = fast, numerator, reference
        self._paired, self._ratios, self._scales = counts, ratios, scales
        self._shortest_updates += 1
        return scales

    def to_dict(self) -> dict[str, Any]:
        """Detached strict-JSON state, in configured physical-lead order."""
        return {
            "schema_version": 1,
            "method": "paired_dual_speed_rms",
            "lead_times": list(self._leads),
            "initial_scales": list(self._initial),
            "scale_decay": self._scale_decay,
            "ratio_decay": self._ratio_decay,
            "floor": self._floor,
            "floor_scope": "output scales and each updated RMS component; initial components unchanged",
            "shortest_fast_scale": self._fast,
            "slow_numerator": list(self._numerator),
            "slow_reference": list(self._reference),
            "ratios": list(self._ratios),
            "scales": list(self._scales),
            "paired_counts": list(self._paired),
            "shortest_feedback_count": self._shortest_updates,
            "pairing_scope": "one mature target per update call; caller supplies target identity",
            "arithmetic": "ordinary binary64; not a rounding certificate",
        }
