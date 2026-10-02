"""Fixed-scale max-statistic inference over a supplied candidate family."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm

from ._validation import as_returns, probability
from .bootstrap import default_block_length, stationary_bootstrap_means
from .inference import infer_mean


@dataclass(frozen=True)
class AuditResult:
    sample_size: int
    names: tuple[str, ...]
    selected_index: int
    mean: NDArray[np.float64]
    standard_error: NDArray[np.float64]
    statistic: NDArray[np.float64]
    iid_pvalue: NDArray[np.float64]
    hac_pvalue: NDArray[np.float64]
    marginal_bootstrap_pvalue: NDArray[np.float64]
    adjusted_pvalue: NDArray[np.float64]
    global_pvalue: float
    simultaneous_ci_low: NDArray[np.float64]
    simultaneous_ci_high: NDArray[np.float64]
    one_sided_lower: NDArray[np.float64]
    max_cutoff: float
    max_abs_cutoff: float
    n_resamples: int
    block_length: float
    lags: int
    alpha: float
    search_complete: bool | None
    bootstrap_statistics: NDArray[np.float64] = field(repr=False, compare=False)

    @property
    def selected_name(self) -> str:
        return self.names[self.selected_index]

    @property
    def n_strategies(self) -> int:
        return len(self.names)

    @property
    def bootstrap_tail_interval(self) -> tuple[float, float]:
        """95% Wilson interval for the conditional bootstrap tail probability.

        This concerns simulation error conditional on the observed data, not
        statistical validity or uncertainty about the population mean.
        """
        count = np.count_nonzero(self.bootstrap_statistics.max(axis=1) >= self.statistic[self.selected_index])
        estimate = count / self.n_resamples
        z = norm.ppf(0.975)
        denominator = 1 + z * z / self.n_resamples
        center = (estimate + z * z / (2 * self.n_resamples)) / denominator
        radius = z * np.sqrt(estimate * (1 - estimate) / self.n_resamples + z * z / (4 * self.n_resamples**2)) / denominator
        return float(center - radius), float(center + radius)

    @property
    def warnings(self) -> list[str]:
        messages = [
            "Inference is asymptotic under stationarity, weak dependence and suitable moments.",
            "Data timing, transaction costs and unrecorded trials are not inferred from returns.",
        ]
        if self.search_complete is False:
            messages.append("The search family is known to be incomplete; adjustment covers supplied columns only.")
        elif self.search_complete is None:
            messages.append(
                "The complete search family is unconfirmed; adjustment covers supplied columns only."
            )
        if self.n_resamples * self.alpha < 20:
            messages.append("Few bootstrap draws in the target tail; p values have substantial Monte Carlo error.")
        low, high = self.bootstrap_tail_interval
        if low <= self.alpha <= high:
            messages.append("The conditional bootstrap tail interval straddles alpha; increase draws before a boundary decision.")
        if self.sample_size < 100:
            messages.append("Short time series: asymptotic and block-bootstrap approximations may be poor.")
        return messages

    def to_dict(self) -> dict[str, Any]:
        """A compact, JSON-serializable record, excluding bootstrap draws."""
        candidates = []
        for index, name in enumerate(self.names):
            candidates.append({
                "name": name,
                "mean": float(self.mean[index]),
                "standard_error": float(self.standard_error[index]),
                "statistic": float(self.statistic[index]),
                "iid_pvalue": float(self.iid_pvalue[index]),
                "hac_pvalue": float(self.hac_pvalue[index]),
                "marginal_bootstrap_pvalue": float(self.marginal_bootstrap_pvalue[index]),
                "adjusted_pvalue": float(self.adjusted_pvalue[index]),
                "simultaneous_ci": [float(self.simultaneous_ci_low[index]), float(self.simultaneous_ci_high[index])],
                "one_sided_lower": float(self.one_sided_lower[index]),
            })
        return {
            "schema_version": 1,
            "method": "fixed-scale stationary-bootstrap max test",
            "null": "All supplied candidates have mean benchmark-adjusted return <= 0.",
            "sample_size": self.sample_size,
            "n_strategies": self.n_strategies,
            "selected_name": self.selected_name,
            "selection_rule": "largest HAC mean statistic; ties use first column",
            "global_pvalue": self.global_pvalue,
            "alpha": self.alpha,
            "n_resamples": self.n_resamples,
            "pvalue_resolution": 1 / (self.n_resamples + 1),
            "conditional_bootstrap_tail_interval_95": list(self.bootstrap_tail_interval),
            "block_length": self.block_length,
            "lags": self.lags,
            "search_complete": self.search_complete,
            "max_cutoff": self.max_cutoff,
            "max_abs_cutoff": self.max_abs_cutoff,
            "warnings": self.warnings,
            "candidates": candidates,
        }


def audit_returns(
    returns: ArrayLike,
    *,
    n_resamples: int = 999,
    block_length: float | None = None,
    lags: int | None = None,
    alpha: float = 0.05,
    seed: int | np.random.Generator | np.random.SeedSequence = 0,
    batch_size: int = 128,
    names: list[str] | tuple[str, ...] | None = None,
    search_complete: bool | None = None,
) -> AuditResult:
    """Test the joint null and report single-step max-adjusted p values.

    The complete, prespecified candidate set must be supplied for an adjustment
    covering that search. This is not a leakage detector or an SPA implementation.
    """
    data = as_returns(returns)
    alpha = probability(alpha, "alpha")
    if search_complete is not None and not isinstance(search_complete, bool):
        raise ValueError("search_complete must be True, False or None.")
    if names is None:
        names = tuple(f"strategy_{index + 1}" for index in range(data.shape[1]))
    else:
        if isinstance(names, str):
            raise ValueError("names must contain one distinct label per candidate.")
        names = tuple(names)
        if (len(names) != data.shape[1] or len(set(names)) != len(names)
                or any(not isinstance(name, str) or not name.strip() for name in names)):
            raise ValueError("names must contain one distinct, nonempty label per candidate.")
    iid = infer_mean(data, method="iid", confidence=1 - alpha)
    hac = infer_mean(data, method="hac", lags=lags, confidence=1 - alpha)
    block_length = default_block_length(len(data)) if block_length is None else block_length
    draws = stationary_bootstrap_means(
        data, n_resamples=n_resamples, block_length=block_length,
        seed=seed, batch_size=batch_size, center=True,
    ) / hac.standard_error
    maximum = draws.max(axis=1)
    absolute_maximum = np.abs(draws).max(axis=1)
    marginal = (1 + (draws >= hac.statistic).sum(axis=0)) / (len(draws) + 1)
    adjusted = (1 + (maximum[:, None] >= hac.statistic).sum(axis=0)) / (len(draws) + 1)
    selected = int(np.argmax(hac.statistic))
    max_cutoff = float(np.quantile(maximum, 1 - alpha, method="higher"))
    max_abs_cutoff = float(np.quantile(absolute_maximum, 1 - alpha, method="higher"))
    return AuditResult(
        sample_size=len(data), names=names, selected_index=selected,
        mean=hac.mean, standard_error=hac.standard_error, statistic=hac.statistic,
        iid_pvalue=iid.pvalue, hac_pvalue=hac.pvalue,
        marginal_bootstrap_pvalue=marginal, adjusted_pvalue=adjusted,
        global_pvalue=float(adjusted[selected]),
        simultaneous_ci_low=hac.mean - max_abs_cutoff * hac.standard_error,
        simultaneous_ci_high=hac.mean + max_abs_cutoff * hac.standard_error,
        one_sided_lower=hac.mean - max_cutoff * hac.standard_error,
        max_cutoff=max_cutoff, max_abs_cutoff=max_abs_cutoff,
        n_resamples=len(draws), block_length=float(block_length), lags=hac.lags,
        alpha=alpha, search_complete=search_complete, bootstrap_statistics=draws,
    )
