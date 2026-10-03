# strategy-inference

A Python library for mean inference with serial dependence and candidate selection. Supply aligned return observations to obtain family tests, candidate decisions and method diagnostics.

[中文](README.md) · [Online docs](https://studyer-tang.github.io/strategy-inference/library/) · [API](docs/api.md) · [Performance](docs/performance.md) · [Methods](docs/methods.md) · [Research and reproduction](docs/research.md)

## Install

Python 3.10+ is required. The core dependencies are NumPy and SciPy. Install the GitHub v0.5.0 release wheel directly; no clone is required:

```bash
python -m pip install https://github.com/Studyer-Tang/strategy-inference/releases/download/v0.5.0/strategy_inference-0.5.0-py3-none-any.whl
```

Source installation is also available. Use a checkout for development or plotting:

```bash
python -m pip install 'git+https://github.com/Studyer-Tang/strategy-inference.git@v0.5.0'

# Development and figures:
git clone --branch v0.5.0 https://github.com/Studyer-Tang/strategy-inference.git
cd strategy-inference
python -m pip install -e '.[figures,dev]'
```

The package has not been published to PyPI. Example CSV files, research scripts and saved experiment results live in the source repository.

## Minimal working example

```python
import numpy as np
from strategy_inference import test_returns

rng = np.random.default_rng(17)
returns = rng.normal(0.0, 0.01, size=(512, 3))
returns[:, 0] += 0.0005

result = test_returns(
    returns,
    method="bootstrap",
    names=["strategy_a", "strategy_b", "strategy_c"],
    n_resamples=1999,
    seed=17,
    search_complete=True,  # This family was fixed before generating the data.
)
print(result.global_pvalue)    # Joint null: every candidate mean is nonpositive.
print(result.adjusted_pvalue)  # K adjusted candidate p values.
print(result.decisions)        # Candidate rejection decisions at the chosen alpha.
```

These are simulated single-period decimal returns. The bootstrap default recomputes the HAC scale in each draw and relies on a resampling approximation. The example demonstrates the interface; it does not provide finite-sample guarantees for arbitrary financial series.

## Core interface

```text
test_returns(returns, *, method="bootstrap", alpha=0.05, names=None, **options)
```

Inputs include NumPy arrays, numeric array-like objects, optional pandas DataFrames and `ReturnTable` objects from `read_returns_csv`. Rows are time and columns are candidates, with shape `(T, K)`. One-dimensional input is treated as one candidate. Require `T >= 8`, positive sample variance in every column and finite observations. DataFrame and CSV inputs supply column names automatically.

Observations must be equally spaced and aligned. For benchmark-adjusted returns, subtract the benchmark and transaction costs first. Missing values, nonfinite values and constant columns raise errors; rows are not silently dropped. Means and intervals retain the input's single-period units and are not automatically annualized.

The unified result is a `TestResult`:

| Attribute | Meaning |
| --- | --- |
| `names`, `mean` | Candidate names and original sample means |
| `decisions`, `global_reject` | Candidate decisions and whether any candidate is rejected |
| `adjusted_pvalue`, `global_pvalue` | Bootstrap adjusted p values; `None` for Gaussian AR |
| `parameter_intervals` | Gaussian AR temporal parameter enclosure; `None` for bootstrap |
| `diagnostics`, `details` | Method diagnostics and the underlying result |

`result.to_dict()` produces a JSON-serializable record. `result.to_frame()` provides a candidate table; install its optional dependency with `python -m pip install 'pandas>=2'`. See the [API reference](docs/api.md) for complete options and diagnostics.

## Choose a method

| Method | Purpose | Conditions and scope |
| --- | --- | --- |
| `test_returns(..., method="bootstrap")` | Test whether all means in a prespecified finite family are nonpositive | Stationarity, weak dependence, suitable moments and consistent scale estimation; asymptotic discussion keeps K fixed, and short or persistent series can impair approximation |
| `test_returns(..., method="gaussian_ar")` | Conservative simultaneous decisions with an unknown common temporal parameter | Common stationary Gaussian AR(1), `0 <= phi < 1`, unknown marginal scales and contemporaneous covariance; model-scoped strong FWER control can be very conservative |
| `infer_mean(..., method="hac")` or `"iid"` | Marginal comparisons of prespecified candidates | No selection adjustment; HAC is asymptotic, and finite-sample t validity requires independent Gaussian observations |

Bootstrap defaults are `studentization="resampled"` and `n_resamples=999`. All columns share stationary-bootstrap time indices. Options include `lags`, `block_length`, `n_resamples`, `seed` and `batch_size`; the latter controls computation batches. Default lag and block-length choices are sample-size heuristics. Re-studentization does not automatically remove finite-sample size error. This is not Hansen's SPA implementation.

`"gaussian_ar"` calls `wilks_uncertainty_test`, uses GLS inference and certifies decisions over an entire parameter enclosure. Require `0 < beta < alpha < 0.5` and prespecify the first `min(K, max_dimension)` columns and time-block scales used for parameter information. Unresolved certificates, retained unit-root endpoints or insufficient information conservatively suppress rejection. The [joint proofs](docs/joint-uncertainty.md) distinguish the ideal Gaussian distribution guarantee from algebraic certification on supplied floating-point inputs; measurement-rounding error is outside the distributional theorem.

The original `audit_returns`, `uncertainty_test`, `wilks_uncertainty_test`, `infer_mean` and `long_run_variance` interfaces remain available. For compatibility, `audit_returns` still defaults to `studentization="fixed"`; the unified bootstrap entry point defaults to `"resampled"`. `uncertainty_test` uses a prespecified reference column for its parameter set; see the [parameter-uncertainty notes](docs/parameter-uncertainty.md).

Adjustment covers the supplied family. A return matrix cannot establish search completeness or recover hidden trials, adaptive candidate generation or repeated monitoring followed by stopping. A family rejection does not establish future profitability.

## CSV and command line

```python
from strategy_inference import read_returns_csv, test_returns

table = read_returns_csv("examples/demo_returns.csv")
result = test_returns(table, method="bootstrap", n_resamples=1999, seed=17)
print(result.to_dict())
```

An optional `date` column records observation dates; other columns contain candidate returns. `read_returns_csv(..., benchmark="benchmark")` subtracts the named benchmark column. Duplicate headers, invalid date order and missing data raise errors. The example CSV contains simulated returns.

```bash
strategy-inference test examples/demo_returns.csv \
  --method bootstrap --n-resamples 1999 --seed 17 --output result.json

strategy-inference test examples/demo_returns.csv \
  --method gaussian_ar --output gaussian-ar.json
```

The original `audit` command retains its JSON and HTML reports.

## Research and development

The [research index](docs/research.md) retains evaluations, failed settings, power costs, frozen protocols, raw evidence and reproduction commands. Known-parameter tail corrections and fitted parametric replay remain research modules outside the unified testing entry point. Read the [method notes](docs/methods.md) for derivations and [references](docs/references.bib) for sources.

```bash
python -m pytest
ruff check .
python scripts/sync_protocols.py --check
```

BSD-3-Clause license.
