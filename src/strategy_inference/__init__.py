"""Inference on mean returns under time dependence and candidate selection."""

from .audit import AuditResult, audit_returns
from .bootstrap import (
    default_block_length,
    stationary_bootstrap_means,
    stationary_bootstrap_statistics,
    stationary_indices,
    stationary_mean_variance,
)
from .inference import MeanInference, default_lags, infer_mean, long_run_variance

__version__ = "0.2.0"
__all__ = [
    "AuditResult",
    "MeanInference",
    "audit_returns",
    "default_block_length",
    "default_lags",
    "infer_mean",
    "long_run_variance",
    "stationary_bootstrap_means",
    "stationary_bootstrap_statistics",
    "stationary_indices",
    "stationary_mean_variance",
]
