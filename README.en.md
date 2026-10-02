# strategy-inference

Mean-return inference under time dependence and strategy selection. Version 0.2.0.

[中文说明](README.md) · [Methods](docs/methods.md) · [Results](docs/results.md) · [References](docs/references.bib)

A positive sample mean is not sufficient evidence of a positive expected return. Serial dependence changes the uncertainty of the mean; choosing a strategy from the same history changes what a single-strategy p value can establish. This package makes both effects explicit and compares their consequences in reproducible simulations.

The target is a **fixed, finite candidate family**: does any supplied strategy have positive expected benchmark-adjusted return? Input is a finite `T × K` matrix of equally spaced, aligned, net-of-cost return differences. The joint null is `mean_j <= 0` for every candidate. This is a mean-return test, not a Sharpe-ratio test or a trading recommendation.

In the independent confirmation experiment with 50 zero-mean Gaussian AR candidates, false positives were **10.65%** with a fixed scale and **6.10%** with a resampled HAC scale (95% pointwise Monte Carlo interval **5.13%–7.24%**). Re-studentization reduced the observed rate, but only **1 of 19** simultaneous upper bounds met the frozen 7% benchmark limit. Under stronger persistence (`T=2048`, `phi=0.9`), its false-positive rate was **10.60%** (**8.84%–12.66%**). The measured improvement does not establish general calibration; see the [result discussion](docs/results.md) for the settings and limits of this comparison.

## Install

Python 3.10 or later is required. From a local checkout:

```bash
python -m pip install .
# Include Matplotlib to reproduce the figures:
python -m pip install '.[figures]'
```

For a GitHub release installation, download the 0.2.0 wheel from the [release page](https://github.com/Studyer-Tang/strategy-inference/releases), then run:

```bash
python -m pip install ./strategy_inference-0.2.0-py3-none-any.whl
```

These instructions use source or release artifacts; no PyPI publication is assumed. Use a source checkout for the example CSV, documentation and saved experiment outputs.

## Audit a candidate family

```python
from strategy_inference import audit_returns

# excess_returns: rows are time, columns are the complete candidate family.
result = audit_returns(
    excess_returns,
    studentization="resampled",
    n_resamples=1999,
    seed=17,
    search_complete=None,  # Unknown unless the researcher confirms the search record.
)
print(result.selected_name, result.global_pvalue)
print(result.to_dict())
```

The selected candidate has the largest original-sample HAC mean statistic. `global_pvalue` tests the joint null; `adjusted_pvalue` contains single-step max-adjusted candidate p values. IID and HAC p values are reported for comparison and do not account for selection on their own.

The CSV interface produces JSON and HTML reports:

```bash
strategy-inference audit examples/demo_returns.csv \
  --studentization resampled --output results/demo --seed 17
```

An optional `date` column records observation dates. If the other columns contain raw strategy and benchmark returns, `--benchmark benchmark` subtracts that benchmark once. Missing values, duplicate headers, invalid date order and constant candidates raise errors; rows are not silently dropped. Example returns are simulated.

## What the bootstrap does

All candidates share circular stationary-bootstrap row indices with geometric block lengths. Each column is centered at its observed mean to impose the least-favourable zero-mean null while retaining cross-candidate dependence.

`studentization="fixed"` remains the default for compatibility with the v0.1 baseline: every draw uses the original HAC denominator. The explicit `"resampled"` option recomputes Bartlett HAC around each draw's own mean, using the same lag count as the original statistic. Both use prespecified lag and block-length heuristics. Neither option is Hansen's SPA test, and re-studentization alone does not establish finite-sample calibration.

`stationary_mean_variance` computes the exact conditional variance of a stationary-bootstrap sample mean for the chosen index law. It diagnoses the resampling distribution; it is not the unknown population variance. Gaussian AR simulations also include known-covariance reference tests, using finite-sample mean variances and a one-dimensional Gaussian maximum integral. These references receive known DGP parameters; the audit API does not. Numerical common-factor quantiles support `p` in `[1e-12, 1-1e-12]`; analytic `K=1` and correlation 0/1 cases do not have this additional restriction. Tiny-tail relative accuracy is not guaranteed.

## Reproduce the experiments

Run from a checkout with the `figures` extra installed:

```bash
strategy-inference reproduce --study calibration --profile quick \
  --output results/calibration/quick
strategy-inference reproduce --study calibration --profile full \
  --output results/calibration/full
# Preserve and rerun the original fixed-scale baseline:
strategy-inference reproduce --study baseline --profile full --output results/full
```

The calibration study compares fixed and resampled scales on the same data and bootstrap indices. Its [frozen protocol](experiments/calibration-protocol.json) specifies the independent evaluation seed, main experiments, holdout settings, block-length sensitivity and acceptance rule. The [baseline protocol](experiments/protocol.json) is retained separately. `quick` checks the execution path and cannot qualify a method as calibrated.

The three main figures examine:

1. False positives as AR persistence changes, with one candidate.
2. False positives after selecting from nested candidate families.
3. Null rejection and signal detection under Gaussian AR, AR with standardized t₅ innovation components, and GARCH.

Each full study saves CSV counts and rates, PNG/SVG/PDF figures, and a run record with parameters, environment versions and source hashes. Read the [calibration report](results/calibration/full/report.html) and [result discussion](docs/results.md) for measured outcomes and remaining failures. The protocol specifies what to test; the report records what happened.

The completed calibration run used frozen revision `579697d`. Later guards, installed-resource lookup and numerical-domain checks account for some differences between its recorded source hashes and the final repository. The original run record is preserved; a [comparison](results/verification/frozen-source-comparison.json) found byte-identical CSVs across all seven quick-study tables. This comparison does not represent a second full run.

## Uncertainty and scope

Bootstrap p values use `(1 + exceedances)/(B + 1)`. This avoids zero estimates and sets the numerical resolution; it is not an exact randomization-test guarantee. A conditional bootstrap-tail interval describes uncertainty from a finite number of draws for the observed data.

Outer Monte Carlo Wilson intervals describe simulation uncertainty in each estimated rejection rate and are pointwise. The benchmark assessment uses simultaneous one-sided Clopper–Pearson upper bounds over 19 prespecified null cells. Those bounds concern Monte Carlo rejection rates, whereas the API's simultaneous confidence intervals concern candidate means. An upper bound above 7% fails to establish the benchmark limit; it does not alone prove that the true rate exceeds 7%. Passing the finite benchmark would not establish validity for all financial series.

The inference requires stationarity, weak dependence, suitable moments and consistent scale estimation. The theoretical scope keeps `K` fixed. Degenerate resamples and unsupported floating-point scales fail explicitly. Strong persistence, short samples, nonstationarity and heavy tails can still impair approximation quality.

Adjustment covers the supplied family. A return matrix cannot recover omitted trials, validate transaction costs, detect every form of leakage, or justify strategies generated adaptively after repeatedly examining the same data. Rejecting the family null does not establish future profitability. The project implements established methods and studies their limits; it does not claim a new theorem.

## Development

```bash
python -m pip install -e '.[figures,dev]'
python -m pytest
ruff check .
```

The [method notes](docs/methods.md) derive the statistics and connect them to the implementation. Source methods and software references are recorded in [references.bib](docs/references.bib). Code is licensed under BSD-3-Clause.
