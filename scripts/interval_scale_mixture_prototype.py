"""Research-only scale interpolation with an issuance-normalized interval score.

For 0<q<1, radius R=k*c(w), k=q/(1-q), and absolute residual r,
score=2*R+2/alpha*max(r-R,0). A closed-boundary hit selects a valid
subgradient. Divide that derivative by frozen 2*k*(c_issue+abs(delta)).
This is not unweighted interval-score regret or arbitrary-delay regret.
Only weights are projected; the external threshold and its feedback are not.
No history is stored: callers align each context and apply its feedback once.
Signatures check fields and learner identity, not time. Binary64 is not certified.
"""

from dataclasses import dataclass, field
from hashlib import sha256
from hmac import compare_digest
from hmac import new as hmac_new
from math import isfinite
from numbers import Integral, Real
from secrets import token_bytes


def _real(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite nonboolean real scalar.")
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} exceeds the supported float range.") from exc
    if not isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{name} must be {'positive and ' if positive else ''}finite.")
    return result


def _lead(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError("lead must be a positive nonboolean integer.")
    return int(value)


@dataclass(frozen=True, slots=True)
class IntervalScaleContext:
    lead_time: int
    weight: float
    short_scale: float
    own_scale: float
    quantile: float
    scale: float
    normalized_delta: float
    skip_reason: str | None
    _signature: bytes = field(repr=False, compare=False)

    def to_dict(self):
        names = (
            "lead_time",
            "weight",
            "short_scale",
            "own_scale",
            "quantile",
            "scale",
            "normalized_delta",
            "skip_reason",
        )
        return {name: getattr(self, name) for name in names}


class IntervalScaleMixture:
    """One short-source weight per lead, starting at 1/2; delayed contexts are frozen."""

    __slots__ = ("_leads", "_rate", "_alpha", "_weights", "_counts", "_learned", "_skips", "_key")

    def __init__(self, lead_times, *, alpha=0.1, learning_rate=0.05):
        try:
            leads = tuple(_lead(value) for value in lead_times)
        except TypeError as exc:
            raise ValueError("lead_times must be a nonempty integer vector.") from exc
        if not leads or any(b <= a for a, b in zip(leads, leads[1:], strict=False)):
            raise ValueError("lead_times must be strictly increasing and nonempty.")
        self._alpha = _real(alpha, "alpha", positive=True)
        if self._alpha >= 1:
            raise ValueError("alpha must be strictly below one.")
        self._rate = _real(learning_rate, "learning_rate", positive=True)
        self._leads, self._key = leads, token_bytes(32)
        self._weights, self._counts = dict.fromkeys(leads, 0.5), dict.fromkeys(leads, 0)
        self._learned = dict.fromkeys(leads, 0)
        self._skips = {
            reason: dict.fromkeys(leads, 0)
            for reason in ("nonpositive_quantile", "unbounded_quantile")
        }

    @property
    def weights(self):
        return tuple(self._weights[h] for h in self._leads)

    @property
    def n_updates(self):
        return tuple(self._counts[h] for h in self._leads)

    @property
    def n_learned(self):
        return tuple(self._learned[h] for h in self._leads)

    def _configured(self, lead):
        lead = _lead(lead)
        if lead not in self._weights:
            raise ValueError(f"Unknown physical lead {lead}.")
        return lead

    def _sign(self, values):
        return hmac_new(self._key, repr(tuple(values)).encode(), sha256).digest()

    def issue(self, lead, short_scale, own_scale, quantile):
        lead = self._configured(lead)
        short = _real(short_scale, "short_scale", positive=True)
        own = _real(own_scale, "own_scale", positive=True)
        quantile = _real(quantile, "quantile")
        weight = self._weights[lead]
        if weight == 0 or short == own:
            scale = own
        elif weight == 1:
            scale = short
        else:
            largest = max(short, own)
            fraction = (1 - weight) * (own / largest) + weight * (short / largest)
            scale = max(min(short, own), largest * min(1.0, fraction))
        delta = short - own
        norm = max(abs(delta), scale)
        normalized = (delta / norm) / (scale / norm + abs(delta / norm))
        reason = (
            "nonpositive_quantile"
            if quantile <= 0
            else ("unbounded_quantile" if quantile >= 1 else None)
        )
        values = lead, weight, short, own, quantile, scale, normalized, reason
        return IntervalScaleContext(*values, self._sign(values))

    def update(self, lead, miss, context):
        lead = self._configured(lead)
        if (
            not isinstance(context, IntervalScaleContext)
            or context.lead_time != lead
            or not isinstance(context._signature, bytes)
            or not compare_digest(context._signature, self._sign(context.to_dict().values()))
        ):
            raise ValueError("Context must be an unchanged issuance from this learner and lead.")
        if not isinstance(miss, bool) and not (
            type(miss).__module__ == "numpy" and type(miss).__name__ in {"bool", "bool_"}
        ):
            raise ValueError("miss must be a boolean scalar.")
        weight = self._weights[lead]
        if context.skip_reason is None:
            change = (
                0.0
                if context.normalized_delta == 0
                else (self._rate * (1 - int(miss) / self._alpha) * context.normalized_delta)
            )
            if not isfinite(change):
                raise ValueError("Weight update overflow.")
            weight = max(0.0, min(1.0, weight - change))
        self._weights[lead] = weight
        self._counts[lead] += 1
        if context.skip_reason is None:
            self._learned[lead] += 1
        else:
            self._skips[context.skip_reason][lead] += 1
        return weight

    def to_dict(self):
        return {
            "method": "issuance_normalized_interval_score_scale_surrogate",
            "lead_times": list(self._leads),
            "alpha": self._alpha,
            "learning_rate": self._rate,
            "weights": list(self.weights),
            "n_updates": list(self.n_updates),
            "n_learned": list(self.n_learned),
            "skipped": {
                name: [counts[h] for h in self._leads] for name, counts in self._skips.items()
            },
            "arithmetic": "ordinary binary64",
        }
