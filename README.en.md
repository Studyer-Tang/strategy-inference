# strategy-inference

A Python library for time-series forecast comparison and uncertainty assessment. Run chronological backtests, select forecasters on validation data, compare out-of-sample losses under temporal dependence, and update online intervals with mature feedback. Forecast timing, candidate scores and data provenance remain explicit.

[中文](README.md) · [Online docs](https://studyer-tang.github.io/strategy-inference/library/) · [Time-series API](docs/time-series.md) · [Multi-step API](docs/multistep-api.md) · [Mean API](docs/api.md) · [Roadmap](docs/toolbox-roadmap.md) · [Research and reproduction](docs/research.md)

## Install

Requires Python 3.10+, NumPy and SciPy. The GitHub release wheel can be installed without cloning. The package has not been published to PyPI.

```bash
python -m pip install https://github.com/Studyer-Tang/strategy-inference/releases/download/v0.11.0/strategy_inference-0.11.0-py3-none-any.whl
```

Source installation from the corresponding tag:

```bash
python -m pip install 'git+https://github.com/Studyer-Tang/strategy-inference.git@v0.11.0'
```

For development, run `python -m pip install -e '.[dev]'` in a source checkout. DataFrame exports such as `to_frame()` require optional pandas: `python -m pip install 'pandas>=2'`.

## Forecast, compare and calibrate

This example generates its own data and fixes models and comparison lead in advance:

```python
import numpy as np
from strategy_inference import (
    backtest, evaluate_forecasts, compare_forecasts, adaptive_intervals,
    sequential_compare_forecasts, naive_forecast, drift_forecast, SeasonalNaive,
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
sequential = sequential_compare_forecasts(run, lead_time=1, loss="absolute")
# Consecutive one-step forecasts: issue naive's forecast before its feedback.
intervals = adaptive_intervals(
    run.actuals[:, 0], run.forecasts[:, 0, 0], alpha=0.1, scale=1.0,
)

print(run.forecasts.shape)        # (262, 3, 3): origin × lead × model
print(scores.mean_loss)           # (3, 3): mean loss for each lead and model
print(comparison.records())       # Improvement over naive, adjusted p values, decisions
print(sequential.to_dict())       # Continuous comparison snapshot for the fixed family
print(intervals.coverage)         # Realized average coverage of evaluated one-step forecasts
```

A positive `mean_improvement` means lower candidate loss than baseline loss. This example uses squared loss for bootstrap and absolute loss for sequential comparison; their targets and assumptions are described below. It demonstrates the interfaces, without establishing model superiority or coverage validity.

## Real data

```python
from strategy_inference import load_dataset, Autoregression, backtest, naive_forecast

data = load_dataset("fred_md")
run = backtest(
    data["T1"].values,
    {"naive": naive_forecast, "ar": Autoregression(lags=12, ridge=1.0)},
    initial_train_size=600, window=120,
)
```

Curated FRED-MD, Bitcoin and Oikolab archives from the [official Monash repository](https://huggingface.co/datasets/Monash-University/monash_tsf) total about 1.7 MB compressed. Each revision and SHA-256 is pinned; cache reads are verified and `offline=True` never accesses the network. No HF SDK, pandas or remote loading code is required. `read_tsf(path)` also reads local TSF/ZIP files, preserving missing positions as NaN and original time attributes.

In a source checkout, `python examples/real_data.py --dataset all` prints compact tables; `--output result.json` saves provenance and candidate scores. Window, lag structure, differencing and ridge are chosen on common chronological validation targets; MASE uses only the initial training scale. Iterations on these previously published examples are development comparisons. These CC BY 4.0 historical archives are not point-in-time market feeds; FRED-MD has supplied preprocessing and anonymous columns without release-vintage data. Details and attribution are in the [API](docs/time-series.md#真实数据与缓存).

For independent validation from a current `main` checkout, run `python benchmarks/forecast_validation.py --mode full --cache-dir .cache --output forecast-validation.json` (requires `.[dev]`). The [prespecified protocol](benchmarks/forecast_validation_protocol.json) fixes five FRED-MD series not used in the development examples, same-window [statsmodels OLS](https://www.statsmodels.org/stable/generated/statsmodels.tsa.ar_model.AutoReg.html) checks, and an AR(1) size/power experiment for the same mean-loss target. The [complete ledger](benchmarks/results/forecast-validation-0.11.json) retains individual decisions and Monte Carlo uncertainty. This is a within-archive extension, not external-dataset validation; the known-parameter oracle is a calibration reference.

`Autoregression(lags=(1, 2, 24, 168))` fits a sparse lag set. `Differenced(model, period=24)` composes seasonal differencing with any callback and restores levels. `select_forecaster(train, validation, models, ...)` returns the chosen callable and all validation scores, with fixed common targets and deterministic ties. Defaults remain dense, undifferenced AR. These independent recursive implementations take guidance from [Huang, Xu and Darlow (2026)](https://arxiv.org/abs/2606.27282) on representation and validation choices for linear forecasters.

## Multi-step intervals and mature feedback

Each forecast updates its controller only when its own target becomes observable. The batch interface retains physical origins and leads, rather than treating backtest rows as immediate feedback:

```python
from strategy_inference import multistep_intervals

walk = np.cumsum(np.random.default_rng(17).normal(size=384))
multi_run = backtest(
    walk, {"naive": naive_forecast}, initial_train_size=128, window=128, horizon=12,
)
origins = np.asarray([split.origin for split in multi_run.splits])
leads = multi_run.target_indices[0] - origins[0]  # Physical leads include gap.
multi = multistep_intervals(
    walk, multi_run.forecasts[:, :, 0], origins=origins, lead_times=leads,
    scale=np.sqrt(leads), step_size=0.1 / np.sqrt(leads), decay=0.2,
    initial_quantile=0.65, strategy="pooled", scale_decay=0.97,
    scale_source="blended", scale_share_weight=0.5,
)
print(multi.summary())
```

Here `sqrt(leads)` uses this simulation's unit-innovation random-walk scale; decreasing the learning rate by lead is an example, not an optimal-rate claim. `blended` combines own-lead RMS with shortest-lead shared RMS using prespecified scalar or per-lead weights. Issued scales remain frozen. Fixed, `horizon` and `shortest` scales are alternatives. Choose weights with training or independent validation data; the default one-half is a compromise, not an optimum.

The [independent-path study](docs/blended-scales-results.md) records benefits and costs; blending has no general efficiency advantage.

With forecasts issued at every time, all leads receive the current label at the same calendar time once the pipeline is full. Shared scales change forecast vintage and residual information; they do not provide long-lead labels earlier or remove the lead's threshold-feedback delay.

Streaming `MultiStepConformal` calls `observe(t, y[t])` before `predict(path)` on a consecutive integer observation clock; unobserved future targets stay pending at termination. `pooled` keeps one controller per lead; `interlaced` separates origin phases. Their feedback counts and learning-rate clocks differ. See the [multi-step API](docs/multistep-api.md), [methods and proofs](docs/multistep-methods.md) and [runnable example](examples/multistep.py).

## Implemented interfaces

| Interface | Inputs and outputs |
| --- | --- |
| `rolling_splits(...)`, `backtest(y, forecasters, ...)` | Finite, equally spaced one-dimensional series; retains training/test positions, forecasts, actuals and target indices. `window=None` expands history; an integer caps rolling history |
| `naive_forecast`, `SeasonalNaive(period)`, `drift_forecast` | Three transparent baselines; alternatively supply `callback(train, lead_times)` returning one forecast per requested lead |
| `forecast_loss(...)`, `evaluate_forecasts(run, ...)` | Squared, absolute and pinball losses; retains origin × lead × model losses and summarizes each lead separately. Pinball forecasts must represent the specified quantile |
| `interval_score(actual, lower, upper, alpha=...)` | Central interval score: width plus missed-distance penalties. Whole-line intervals score infinity; empty intervals are unsupported |
| `compare_forecasts(run, baseline=..., lead_time=...)` | Shared-index max bootstrap against one prespecified baseline at one lead, for a fixed candidate family; returns family/candidate decisions and diagnostics |
| `SequentialModelConfidenceSet`, `sequential_compare_forecasts(run, ...)` | Fixed model family, absolute or pinball loss and nonoverlapping feedback; returns a continuously updated, shrinking model confidence set and log e-value snapshot |
| `AdaptiveConformal`, `adaptive_intervals(...)` | One-step decaying quantile tracker with ordered complete feedback; returns intervals, empty/whole-line states and realized coverage |
| `MultiStepConformal`, `multistep_intervals(...)` | Univariate integer clock, mature multi-step feedback, pooled/interlaced states, frozen issued scales and summaries by lead |
| `test_returns(...)`, `infer_mean(...)` | Existing simultaneous or marginal inference on aligned return means |

`run.to_dict()` retains full backtest arrays and splits. `scores.losses` stores losses at every origin; `scores.to_dict()` exports summaries. Comparisons and online intervals also offer `to_dict()`. Candidate tables and long forecast tables can be exported with `to_frame()`, requiring pandas. See the [time-series API](docs/time-series.md) for options, shapes and field meanings.

## Methods and scope

Backtests follow input order and use each fold's training history. Callbacks receive separate read-only training and lead arrays. Callers must also prevent access to future information through captured variables, external state or data sources. Missing/nonfinite inputs and invalid predictions raise errors; dates are not automatically sorted, filled or resampled.

Fixed-sample comparisons use baseline loss minus candidate loss. Bootstrap approximation requires stationary, weakly dependent loss differences, suitable moments, nondegenerate variance and a fixed family. Multi-step backtests require a `lead_time`; default lags and block lengths account for overlap and sample size but remain heuristics. Backtest splits do not establish these conditions or handle hidden search and repeated monitoring.

Sequential comparison implements the strong conditional-superiority construction of [Arnold et al., JRSSB 2026](https://doi.org/10.1093/jrsssb/qkag066): a target model must have no larger expected loss than every other fixed model, conditional on information at every forecast issue. The target set may be empty. Exclusions are permanent; continue supplying all original models afterward. Each label must arrive before the next forecast; backtests require origin spacing at least the selected physical lead. Fix `bet_fraction` in `(0, .5]` beforehand; `.25` is the default, without optimality or floating-point certification claims. See the [time-series API](docs/time-series.md#连续模型置信集) for fields and a short streaming example.

One-step intervals implement the decaying update of [Angelopoulos–Barber–Bates, ICML 2024](https://proceedings.mlr.press/v235/angelopoulos24a.html), with a fixed-scale bounded residual transform. The multi-step module additionally maintains a mature-feedback queue. Its ideal recursion controls retrospective average miscoverage of matured forecasts separately at each lead, without conditional coverage at each time or simultaneous coverage of an entire path. This independent engineering implementation does not reproduce full AcMCP PID and scorecasting. Empty and whole-line sets remain explicit and thresholds are not clipped. Ordinary floating-point feedback uses the returned closed intervals, without a rounding certificate. See the [time-series API](docs/time-series.md) and [multi-step methods](docs/multistep-methods.md).

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

Runnable examples cover the [time-series workflow](examples/time_series.py) and [multi-step intervals](examples/multistep.py). The [roadmap](docs/toolbox-roadmap.md) separates implemented and planned capabilities. The [research index](docs/research.md) preserves historical evaluations, failed settings and frozen evidence; reproduce formal studies at the commit/tag in their metadata. Implementing established algorithms is not a claim of foundational novelty.

Measurements are versioned: [forecast evaluation](docs/time-series-performance.md), [multi-step intervals](docs/multistep-performance.md), and [scale blending and refactoring](docs/blended-scales-performance.md). Older measurements do not cover newly added interfaces; local timings guarantee neither universal speed nor statistical calibration.

```bash
python -m pytest
ruff check .
```

BSD-3-Clause license.
