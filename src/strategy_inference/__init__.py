"""Time-series evaluation, online uncertainty and simultaneous mean inference."""

from importlib import import_module

__version__ = "0.6.0"
_EXPORTS = {
    "AuditResult": "audit",
    "audit_returns": "audit",
    "default_block_length": "bootstrap",
    "stationary_bootstrap_means": "bootstrap",
    "stationary_bootstrap_statistics": "bootstrap",
    "stationary_indices": "bootstrap",
    "stationary_mean_variance": "bootstrap",
    "MeanInference": "inference",
    "default_lags": "inference",
    "infer_mean": "inference",
    "long_run_variance": "inference",
    "ReturnTable": "io",
    "read_returns_csv": "io",
    "TestResult": "testing",
    "test_returns": "testing",
    "RollingSplit": "model_selection",
    "rolling_splits": "model_selection",
    "BacktestResult": "model_selection",
    "backtest": "model_selection",
    "naive_forecast": "model_selection",
    "drift_forecast": "model_selection",
    "SeasonalNaive": "model_selection",
    "ForecastEvaluation": "evaluation",
    "ForecastComparison": "evaluation",
    "forecast_loss": "evaluation",
    "interval_score": "evaluation",
    "evaluate_forecasts": "evaluation",
    "compare_forecasts": "evaluation",
    "AdaptiveConformal": "conformal",
    "ConformalInterval": "conformal",
    "ConformalUpdate": "conformal",
    "ConformalResult": "conformal",
    "adaptive_intervals": "conformal",
    "UncertaintyResult": "uncertainty",
    "uncertainty_test": "uncertainty",
    "WilksResult": "wilks",
    "WilksScale": "wilks",
    "wilks_uncertainty_test": "wilks",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    """Load numerical dependencies only when a numerical interface is requested."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
