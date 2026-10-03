"""Research-only geometric scale interpolation with a Student-t surrogate.
For log scale a, loss = a + (df+1)/2 * log(1+r**2/(df*exp(2*a))).
Its derivative lies in [-df,1]. Feedback uses the frozen issued scales and
weight, then updates the current weight: delayed forecasts are not rewritten.
The surrogate has no interval-score, conditional-coverage or delay-regret
guarantee. df is a tuning constant. Binary64 is not rounding-certified.
No pending records are stored: the caller aligns feedback and applies it once.
Signatures check issued fields and learner identity, not time.
"""

from dataclasses import dataclass, field
from hashlib import sha256
from hmac import compare_digest
from hmac import new as hmac_new
from math import exp, isfinite, log, log1p
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
class ScaleContext:
    lead_time: int
    weight: float
    short_scale: float
    own_scale: float
    loggap: float
    scale: float
    _signature: bytes = field(repr=False, compare=False)

    def to_dict(self):
        names = ("lead_time", "weight", "short_scale", "own_scale", "loggap", "scale")
        return {name: getattr(self, name) for name in names}


class TwoScaleMixture:
    """One short-source weight per lead; each starts at 1/2, without history."""

    __slots__ = ("_leads", "_rate", "_df", "_weights", "_counts", "_key")

    def __init__(self, lead_times, *, learning_rate=0.05, df=3.0):
        try:
            leads = tuple(_lead(value) for value in lead_times)
        except TypeError as exc:
            raise ValueError("lead_times must be a nonempty integer vector.") from exc
        if not leads or any(b <= a for a, b in zip(leads, leads[1:], strict=False)):
            raise ValueError("lead_times must be strictly increasing and nonempty.")
        self._leads = leads
        self._rate = _real(learning_rate, "learning_rate", positive=True)
        self._df = _real(df, "df", positive=True)
        self._weights, self._counts = dict.fromkeys(leads, 0.5), dict.fromkeys(leads, 0)
        self._key = token_bytes(32)  # Does not consume the simulation RNG or enter saved JSON.

    @property
    def weights(self):
        return tuple(self._weights[h] for h in self._leads)

    @property
    def n_updates(self):
        return tuple(self._counts[h] for h in self._leads)

    def _configured(self, lead):
        lead = _lead(lead)
        if lead not in self._weights:
            raise ValueError(f"Unknown physical lead {lead}.")
        return lead

    def _sign(self, values):
        return hmac_new(self._key, repr(tuple(values)).encode(), sha256).digest()

    def issue(self, lead, short_scale, own_scale):
        lead = self._configured(lead)
        short = _real(short_scale, "short_scale", positive=True)
        own = _real(own_scale, "own_scale", positive=True)
        weight = self._weights[lead]
        ls, lo = log(short), log(own)
        relative = (short - own) / own
        gap = log1p(relative) if -0.5 <= relative <= 1 else ls - lo
        if weight == 0 or short == own:
            scale = own
        elif weight == 1:
            scale = short
        else:
            logscale = min(max(lo + weight * gap, min(ls, lo)), max(ls, lo))
            scale = min(max(exp(logscale), min(short, own)), max(short, own))
        values = lead, weight, short, own, gap, scale
        return ScaleContext(*values, self._sign(values))

    def update(self, lead, absresidual, context):
        lead = self._configured(lead)
        if (
            not isinstance(context, ScaleContext)
            or context.lead_time != lead
            or not isinstance(context._signature, bytes)
            or not compare_digest(context._signature, self._sign(context.to_dict().values()))
        ):
            raise ValueError("Context must be an unchanged issuance from this learner and lead.")
        residual = _real(absresidual, "absresidual")
        if residual < 0:
            raise ValueError("absresidual must be nonnegative.")
        z = 0.0
        if residual:
            x = 2 * (log(residual) - log(context.scale)) - log(self._df)
            z = 1 / (1 + exp(-x)) if x >= 0 else exp(x) / (1 + exp(x))
        gradient = (1 - (self._df + 1) * z) * (context.loggap / (1 + abs(context.loggap)))
        change = self._rate * gradient
        if not isfinite(change):
            raise ValueError("Weight update overflow.")
        weight = max(0.0, min(1.0, self._weights[lead] - change))
        self._weights[lead] = weight
        self._counts[lead] += 1
        return weight

    def to_dict(self):
        return {
            "method": "student_t_geometric_scale_surrogate",
            "lead_times": list(self._leads),
            "learning_rate": self._rate,
            "df": self._df,
            "weights": list(self.weights),
            "n_updates": list(self.n_updates),
            "arithmetic": "ordinary binary64",
        }
