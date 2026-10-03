# strategy-inference

A Python time-series toolbox for forecast evaluation, online uncertainty and strategy mean inference. Version 0.6.0 provides the first workflow: rolling forecasts, losses by lead, fixed-family comparisons and one-step online intervals. Existing return-mean interfaces remain available.

[中文](README.md) · [Online docs](https://studyer-tang.github.io/strategy-inference/library/) · [Time-series API](docs/time-series.md) · [Mean API](docs/api.md) · [Roadmap](docs/toolbox-roadmap.md) · [Research and reproduction](docs/research.md)

## Install

Requires Python 3.10+, NumPy and SciPy. Install the GitHub release wheel without cloning the repository. The package has not been published to PyPI.

```bash
python -m pip install https://github.com/Studyer-Tang/strategy-inference/releases/download/v0.6.0/strategy_inference-0.6.0-py3-none-any.whl
```

Source installation from the corresponding tag:

```bash
python -m pip install 'git+https://github.com/Studyer-Tang/strategy-inference.git@v0.6.0'
```

For development, run `python -m pip install -e '.[dev]'` in a source checkout. DataFrame exports such as `to_frame()` require optional pandas: `python -m pip install 'pandas>=2'`.

## Forecast, compare and calibrate

This example generates its own data and fixes models and comparison lead in advance:

```python
import numpy as np
from strategy_inference import (
    backtest, evaluate_forecasts, compare_forecasts, adaptive_intervals,
    naive_forecast, drift_forecast, SeasonalNaive,
)

rng = np.random.default_rng(17)
y = np.empty(384)
y[0] = rng.normal()
for t in range(1, len(y)):
    y[t] = 0.6 * y[t - 1] + 0.8 * rng.normal()

models = {
    "naive": naive_forecast,
    "seasonal": SeasonalNaive(period=12),
    "drift": drift_forecast,
}
run = backtest(
    y, models, initial_train_size=120, window=120, horizon=3,
)
scores = evaluate_forecasts(run, loss="squared")
comparison = compare_forecasts(
    run, baseline="naive", lead_time=1, loss="squared",
    n_resamples=1999, seed=17, search_complete=True,
)
# Consecutive one-step forecasts: issue naive's forecast before its feedback.
intervals = adaptive_intervals(
    run.actuals[:, 0], run.forecasts[:, 0, 0], alpha=0.1, scale=1.0,
)

print(run.forecasts.shape)        # (262, 3, 3): origin × lead × model
print(scores.mean_loss)           # (3, 3): mean loss for each lead and model
print(comparison.records())       # Improvement over naive, adjusted p values, decisions
print(intervals.coverage)         # Realized average coverage of evaluated one-step forecasts
```

A positive `mean_improvement` means lower candidate loss than baseline loss. Rejection tests positive expected improvement; choosing the smallest observed loss and then interpreting an unadjusted p value requires separate treatment. This example demonstrates the interfaces. Realized average coverage is not coverage probability at every time.

## Implemented interfaces

| Interface | Inputs and outputs |
| --- | --- |
| `rolling_splits(...)`, `backtest(y, forecasters, ...)` | Finite, equally spaced one-dimensional series; retains training/test positions, forecasts, actuals and target indices. `window=None` expands history; an integer caps rolling history |
| `naive_forecast`, `SeasonalNaive(period)`, `drift_forecast` | Three transparent baselines; alternatively supply `callback(train, lead_times)` returning one forecast per requested lead |
| `forecast_loss(...)`, `evaluate_forecasts(run, ...)` | Squared, absolute and pinball losses; retains origin × lead × model losses and summarizes each lead separately. Pinball forecasts must represent the specified quantile |
| `interval_score(actual, lower, upper, alpha=...)` | Central interval score: width plus missed-distance penalties. Whole-line intervals score infinity; empty intervals are unsupported |
| `compare_forecasts(run, baseline=..., lead_time=...)` | Shared-index max bootstrap against one prespecified baseline at one lead, for a fixed candidate family; returns family/candidate decisions and diagnostics |
| `AdaptiveConformal`, `adaptive_intervals(...)` | One-step decaying quantile tracker with ordered complete feedback; returns intervals, empty/whole-line states and realized coverage |
| `test_returns(...)`, `infer_mean(...)` | Existing simultaneous or marginal inference on aligned return means |

`run.to_dict()` retains full backtest arrays and splits. `scores.losses` stores losses at every origin; `scores.to_dict()` exports summaries. Comparisons and online intervals also offer `to_dict()`. Candidate tables and long forecast tables can be exported with `to_frame()`, requiring pandas. See the [time-series API](docs/time-series.md) for options, shapes and field meanings.

## Methods and scope

Backtests follow input order and use each fold's training history. Callbacks receive separate read-only training and lead arrays. Callers must also prevent access to future information through captured variables, external state or data sources. Missing/nonfinite inputs and invalid predictions raise errors; dates are not automatically sorted, filled or resampled.

Comparisons use baseline loss minus candidate loss. Bootstrap approximation requires stationary, weakly dependent loss differences, suitable moments and nondegenerate variance; asymptotic discussion keeps the family fixed. Overlapping forecasts retain origin order. Multi-step backtests require a `lead_time`; leads are not flattened into independent observations. Default HAC lags and block lengths account for overlap and sample size but remain heuristics. Rolling/expanding splits do not establish statistical assumptions. Constant loss differences raise errors. Prespecify candidates, baseline, loss and lead; joint inference across leads, hidden search and repeated monitoring require separate methods.

Online intervals implement the decaying update of [Angelopoulos–Barber–Bates, ICML 2024](https://proceedings.mlr.press/v235/angelopoulos24a.html), with a fixed-scale bounded residual transform. The ideal recursion targets retrospective time-average coverage. Per-time, conditional and delayed multi-step guarantees are outside this implementation's scope. Empty and whole-line intervals are retained explicitly, without clipping thresholds. The implementation uses ordinary floating-point arithmetic, evaluating feedback against the returned closed interval. Read the [time-series API](docs/time-series.md) for theoretical and numerical limits.

## Return-mean inference

```python
import numpy as np
from strategy_inference import test_returns

rng = np.random.default_rng(17)
returns = rng.normal(0.0, 0.01, size=(512, 3))
result = test_returns(returns, n_resamples=1999, seed=17)
print(result.global_pvalue, result.adjusted_pvalue)
```

`test_returns` accepts NumPy, optional DataFrames and tables from `read_returns_csv`; rows are time and columns are candidates. Require `T >= 8`, equally spaced aligned observations, finite values and positive sample variances. Units remain single-period units; subtract benchmarks and costs before testing adjusted returns.

The default `method="bootstrap"` shares stationary-bootstrap time indices and recomputes HAC scales in each draw, with stationarity, weak-dependence and moment conditions. `method="gaussian_ar"` gives conservative strong-FWER decisions under common stationary Gaussian AR(1), `0 <= phi < 1`, prespecified candidates and information configuration. It returns parameter enclosures rather than continuous p values; insufficient information or unresolved certification conservatively suppresses rejection. See the [mean API](docs/api.md) and [joint proof](docs/joint-uncertainty.md) for full restrictions.

`audit_returns`, `uncertainty_test`, `wilks_uncertainty_test`, `infer_mean` and `long_run_variance` remain available. For compatibility, `audit_returns` defaults to `studentization="fixed"`; `test_returns` defaults to `"resampled"`. The CLI retains return testing and reports:

```bash
strategy-inference test examples/demo_returns.csv \
  --method bootstrap --n-resamples 1999 --seed 17 --output result.json
```

## Documentation, research and performance

The [roadmap](docs/toolbox-roadmap.md) distinguishes implemented functionality from proposed extensions. The [research index](docs/research.md) retains historical evaluations, failed settings, frozen protocols and raw evidence. Implementations of established forecasting, bootstrap and conformal algorithms are not claims of foundational algorithmic novelty.

The v0.6 [performance notes](docs/time-series-performance.md) and [raw record](benchmarks/results/time-series-0.6.json) retain conditions, five warmed samples and source hashes. Locally, three-baseline rolling evaluation at 958 origins takes about 19 ms; comparing two candidates with 999 resamples takes about 24 ms; 100,000 streaming interval updates take about 123 ms. Input generation and imports are excluded, as is the pre-existing backtest for comparison. These are local engineering measurements, not universal speed guarantees.

Historical evidence remains in the [v0.5 page and full benchmark](https://studyer-tang.github.io/strategy-inference/library/v0.5.0/) and [v0.5 performance notes](docs/performance.md). Those timings do not measure v0.6's new interfaces. Reproduce historical formal studies at the frozen commit/tag specified in their metadata.

```bash
python -m pytest
ruff check .
python scripts/sync_protocols.py --check
python scripts/build_library_site.py --check
python scripts/build_toolbox_site.py --check
```

BSD-3-Clause license.
