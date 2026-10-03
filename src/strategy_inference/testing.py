"""A small, consistent interface for simultaneous mean-return tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .io import ReturnTable

if TYPE_CHECKING:
    from .audit import AuditResult
    from .wilks import WilksResult


def _labels(values: Any, names: Sequence[str] | None, k: int) -> tuple[str, ...]:
    if names is None:
        if isinstance(values, ReturnTable):
            names = values.names
        elif hasattr(values, "columns"):
            names = tuple(str(name) for name in values.columns)
        else:
            names = tuple(f"strategy_{i + 1}" for i in range(k))
    if isinstance(names, (str, bytes)):
        raise ValueError("names must contain one distinct, nonempty string per column.")
    try:
        labels = tuple(names)
    except TypeError as exc:
        raise ValueError("names must contain one distinct, nonempty string per column.") from exc
    if (
        len(labels) != k
        or any(not isinstance(name, str) or not name.strip() for name in labels)
        or len(set(labels)) != k
    ):
        raise ValueError("names must contain one distinct, nonempty string per column.")
    return labels


def _readonly(values, dtype):
    result = np.array(values, dtype=dtype, copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class TestResult:
    """Column decisions and a compact export; ``details`` retains full evidence.

    ``mean`` is the ordinary sample mean, in the input's return units.
    Bootstrap adjusted p-values and intervals are approximate. The Gaussian AR
    method returns level-specific certified decisions, without inventing p-values
    or a single fitted-parameter standard error. Arrays preserve column order.
    """

    method: Literal["bootstrap", "gaussian_ar"]
    sample_size: int
    names: tuple[str, ...]
    alpha: float
    mean: NDArray[np.float64]
    decisions: NDArray[np.bool_]
    adjusted_pvalue: NDArray[np.float64] | None
    global_pvalue: float | None
    parameter_intervals: tuple[tuple[float, float], ...] | None
    diagnostics: Mapping[str, Any]
    details: AuditResult | WilksResult = field(repr=False, compare=False)

    @property
    def global_reject(self) -> bool:
        return bool(self.decisions.any())

    @property
    def n_strategies(self) -> int:
        return len(self.names)

    @property
    def n_obs(self) -> int:
        return self.sample_size

    def records(self) -> list[dict[str, Any]]:
        """Return one plain record per strategy, suitable for CSV or a table."""
        rows = []
        for i, name in enumerate(self.names):
            row = {"name": name, "mean": float(self.mean[i]), "reject": bool(self.decisions[i])}
            if self.adjusted_pvalue is not None:
                row["adjusted_pvalue"] = float(self.adjusted_pvalue[i])
            rows.append(row)
        return rows

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable results, excluding bootstrap draws and polynomials."""
        return {
            "schema_version": 1,
            "method": self.method,
            "null": "All supplied candidates have mean input return <= 0.",
            "sample_size": self.sample_size,
            "n_strategies": self.n_strategies,
            "alpha": self.alpha,
            "global_reject": self.global_reject,
            "global_pvalue": self.global_pvalue,
            "parameter_intervals": (
                None if self.parameter_intervals is None
                else [list(interval) for interval in self.parameter_intervals]
            ),
            "diagnostics": {
                key: list(value) if isinstance(value, tuple) else value
                for key, value in self.diagnostics.items()
            },
            "candidates": self.records(),
        }

    def to_frame(self):
        """Return a pandas DataFrame; pandas is an optional dependency."""
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("to_frame() requires pandas; install pandas or the 'pandas' extra.") from exc
        frame = pd.DataFrame.from_records(self.records(), index="name")
        frame.index.name = "strategy"
        return frame


_OPTIONS = {
    "bootstrap": frozenset({
        "n_resamples", "block_length", "lags", "seed", "batch_size", "search_complete",
        "studentization",
    }),
    "gaussian_ar": frozenset({
        "beta", "max_dimension", "block_lengths", "ci_depth", "certificate_depth", "max_nodes",
        "critical_bits",
    }),
}


def test_returns(
    returns: ArrayLike | ReturnTable,
    *,
    method: Literal["bootstrap", "gaussian_ar"] = "bootstrap",
    alpha: float = 0.05,
    names: Sequence[str] | None = None,
    **options,
) -> TestResult:
    """Test nonpositive means across supplied strategies, with one output schema.

    Accept an array (T by K, or one series), a numeric pandas DataFrame or a
    ``ReturnTable`` from ``read_returns_csv``. DataFrame/CSV column labels are
    retained. Values must be finite; missing rows are never silently removed.

    ``bootstrap`` uses a shared-row stationary bootstrap and single-step max
    adjustment; defaults to resampled HAC scales, 999 draws and seed 0. Options:
    n_resamples, block_length, lags, seed, batch_size, studentization and
    search_complete. Its approximation requires stationary, weakly dependent
    series and suitable moments; candidate selection covers supplied columns.

    ``gaussian_ar`` uses the joint multiscale confidence-set method. Options:
    beta, max_dimension, block_lengths, ci_depth, certificate_depth, max_nodes
    and critical_bits. Coverage and strong FWER require the documented common
    stationary Gaussian AR(1), 0<=phi<1, fixed candidates and positive marginal
    variances. Certificates concern supplied floats, excluding input-rounding
    error. It reports no p-values. No method is selected by inspecting data.
    """
    if not isinstance(method, str) or method not in _OPTIONS:
        raise ValueError("method must be 'bootstrap' or 'gaussian_ar'.")
    unknown = options.keys() - _OPTIONS[method]
    if unknown:
        raise TypeError(f"Unsupported options for method={method!r}: {', '.join(sorted(unknown))}.")
    values = returns.values if isinstance(returns, ReturnTable) else returns
    if hasattr(returns, "columns") and hasattr(returns, "dtypes") and all(
        getattr(dtype, "kind", None) in ("i", "u", "f") for dtype in returns.dtypes
    ):
        # Numeric pandas extension dtypes (e.g. Float64) otherwise become object
        # arrays. Preserve missing values as NaN for the ordinary strict check.
        values = returns.to_numpy(dtype=np.float64, na_value=np.nan, copy=False)
    # Resolve labels before the expensive test, without duplicating input scans.
    shape = np.asarray(values).shape
    k = shape[1] if len(shape) == 2 else 1
    labels = _labels(returns, names, k)
    if method == "bootstrap":
        from .audit import audit_returns

        options.setdefault("studentization", "resampled")
        result = audit_returns(values, alpha=alpha, names=labels, **options)
        mean = result.mean
        decisions = result.adjusted_pvalue <= result.alpha
        adjusted = _readonly(result.adjusted_pvalue, np.float64)
        global_pvalue, intervals = result.global_pvalue, None
        n_obs = result.sample_size
        diagnostics = dict(studentization=result.studentization, n_resamples=result.n_resamples,
            block_length=result.block_length, lags=result.lags, search_complete=result.search_complete,
            pvalue_resolution=1 / (result.n_resamples + 1), warnings=tuple(result.warnings),
            conditional_bootstrap_tail_interval_95=result.bootstrap_tail_interval)
        seed = options.get("seed", 0)
        diagnostics["seed"] = int(seed) if isinstance(seed, (int, np.integer)) else None
        diagnostics["random_state_type"] = type(seed).__name__
    else:
        from .wilks import wilks_uncertainty_test

        result = wilks_uncertainty_test(values, alpha=alpha, **options)
        mean = np.asarray(values, dtype=np.float64).reshape(result.n_obs, result.k).mean(axis=0)
        decisions = result.decisions
        adjusted, global_pvalue, intervals = None, None, result.interval_bounds
        n_obs = result.n_obs
        diagnostics = dict(shape_dimension=result.dimension, beta=result.beta,
            block_lengths=result.block_lengths,
            phi1_retained=result.phi1_retained, empty_parameter_set=result.empty_ci,
            singular_fallback=result.singular_fallback,
            ci_unresolved_cells=result.ci_unresolved_cells,
            certificate_unresolved=int(np.count_nonzero(result.certificate_unresolved)))
    return TestResult(method, n_obs, labels, result.alpha, _readonly(mean, np.float64),
        _readonly(decisions, bool), adjusted, global_pvalue, intervals,
        MappingProxyType(diagnostics), result)
