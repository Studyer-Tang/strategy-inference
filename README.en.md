# strategy-inference

Mean-return inference under time dependence and strategy selection. Version 0.4.0.

[中文说明](README.md) · [Methods](docs/methods.md) · [Results](docs/results.md) · [Joint information study](https://studyer-tang.github.io/strategy-inference/research/joint/) · [References](docs/references.bib)

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

Install the tagged GitHub source directly:

```bash
python -m pip install 'git+https://github.com/Studyer-Tang/strategy-inference.git@v0.4.0'
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

## Tail-scale mechanism study

A separate research study implements the fixed-bandwidth AR(1) formulas from [Liu and Chan, JASA 2026](https://doi.org/10.1080/01621459.2026.2676715), computes exact centered Gaussian quadratic-form moments, and evaluates 12 frozen null cells with 5000 replicates per cell. Independent execution of the authors' original R functions agrees on 48 scalar comparisons.

For `T=512, phi=0.9, K=100, rho=0`, a reference that uses true parameters to make every marginal scale exactly unbiased still rejects **10.52%** of the time (pointwise 95% interval **9.70%–11.40%**). Its average scale ratio across all columns is **1.00015**, whereas the selected column's ratio averages **0.87302**. Correct scale expectation alone does not calibrate the tail of a selected statistic with a random denominator.

Read the [results](docs/tail-results.md), [mechanism proofs](docs/tail-mechanism.md), [oracle selection bound](docs/oracle-selection-bound.md), and [three-figure report with replicate records](results/research/tail/full/report.html). The [protocol](experiments/tail-diagnostic-protocol.json) and computation revision were frozen before evaluation.

```bash
python scripts/tail_diagnostics.py --profile full --output results/research/tail/reproduced
```

Run from a source checkout with the `figures` extra installed, using an empty output directory. Every comparison uses a known correlation matrix; only the true-variance reference has the exact Gaussian scale. This is a mechanism study, not an implementable joint test with unknown covariance. The new research modules do not alter `audit_returns`.

## Fitted replay and finite Monte Carlo resolution

A separate study implements full parametric replay within a common Gaussian AR model, separating temporal fitting, correlation fitting and correction-factor refitting. The [proofs](docs/parametric-replay.md) establish conditional validity for fixed K under mild persistence and path limits for the specific estimator and statistic in a local-unit sequence. Finite B can also impose a power ceiling on the prespecified jitter-based, independently size-matched simulation comparison.

The frozen study uses 13,000 outer datasets across independent calibration, null evaluation and power stages, 199 inner draws, and 147,000 method records. With 100 correlated candidates, fitted replay reduces rejection from **19.4% to 6.4%**; strong persistence still yields **16.6%**, and heterogeneous temporal structure yields **30.0%**. Only one correctly specified cell clears the prespecified six-cell size screen. These research helpers do not change the public audit rule.

Read the [interpretation](docs/replay-results.md), [three-figure report and records](results/research/replay/full/report.html), [protocol](experiments/parametric-replay-protocol.json), and [independent audit](results/research/replay/full/audit.json).

```bash
# Rebuild all three figures and the report without new Monte Carlo draws:
python scripts/replay_report.py --output results/research/replay/full
# Recompute in an empty output directory:
python scripts/parametric_replay.py --profile full --output results/research/replay/reproduced
python scripts/replay_report.py --output results/research/replay/reproduced
python scripts/verify_parametric_replay.py --output results/research/replay/reproduced
```

## Unknown temporal parameters and their power cost

`uncertainty_test` provides fixed-level simultaneous mean decisions under a stationary Gaussian AR model with one unknown common coefficient in `[0,1)`, unknown marginal scales and unrestricted contemporaneous covariance. A prespecified reference column supplies an exact innovation F confidence set. Integer/rational Bernstein certificates verify rejection over its entire continuous outer enclosure. A shared coverage budget plus Bonferroni gives strong FWER; the reference column may itself carry signal.

```python
from strategy_inference import uncertainty_test

result = uncertainty_test(excess_returns, alpha=0.05, beta=0.005, reference=0)
print(result.decisions, result.interval_bounds, result.phi1_retained)
```

This procedure uses a GLS mean target and returns decisions at a specified level, rather than continuous p-values. Unresolved certificates and empty confidence sets do not reject. The ideal Gaussian distribution guarantee and exact algebra on supplied binary64 inputs are distinct; measurement-rounding error is outside the proof.

The frozen experiment uses 16,000 independent stage-noise datasets, 31,000 datasets including shared mean shifts, and 114,000 method records. At `T=512, phi=.9, K=20, delta=3`, known-parameter power is **58.9%**, while this construction achieves **1.1%**. It is a reproducible conservative research procedure, not a practically competitive strategy selector. The power loss and model-misspecification results remain public.

[Report and three figures](https://studyer-tang.github.io/strategy-inference/research/uncertainty/) · [Results](docs/uncertainty-results.md) · [Proofs and prior work](docs/parameter-uncertainty.md) · [Protocol](experiments/parameter-uncertainty-protocol.json) · [Audit](results/research/uncertainty/full/audit.json)

```bash
python scripts/uncertainty_report.py --output results/research/uncertainty/full
python scripts/parameter_uncertainty.py --profile full --output results/research/uncertainty/reproduced
python scripts/verify_parameter_uncertainty.py --output results/research/uncertainty/reproduced
python scripts/uncertainty_report.py --output results/research/uncertainty/reproduced
```

Confidence-set inversion and nuisance projection follow Dufour (1990), Dufour–Neifar (2002), and Berger–Boos; Glazer–Stark (2026) provides recent work on conservative computation. The contribution is the scoped, verifiable implementation and power-cost study, not a new general inference principle.

## Information about the common temporal parameter

`wilks_uncertainty_test` uses the first prespecified `min(K,8)` columns and fixed block lengths 4, 16 and 64 in the same common stationary Gaussian AR model. Independent within-block and between-block Wishart scatters give a Wilks determinant ratio that removes unknown contemporaneous covariance. Exact integer moments, conservative outward cutoffs and continuous determinant certificates retain the shared coverage budget and GLS strong-FWER argument.

```python
from strategy_inference import wilks_uncertainty_test

result = wilks_uncertainty_test(excess_returns, alpha=0.05, beta=0.005)
print(result.decisions, result.interval_bounds)
print(result.dimension, result.phi1_retained, result.singular_fallback)
```

Singular selected covariance conservatively removes shape information; duplicate candidates do not become additional directions. The interface returns fixed-level decisions, not p-values. Wilks, moment bounds and nuisance projection are established principles. The contribution is the scoped combination, numerical certificates and frozen paired evaluation, without a universal power or optimality claim.

The fresh evaluation uses 22,000 independent noise datasets and 220,000 method records. At `T=512, phi=.9, K=20`, true-signal power improves from **1.3% to 9.4%** at delta=3 and **12.2% to 88.0%** at delta=6; G2 improves from **0.4% to 27.5%** at delta=3. Gains primarily come from joint directions. The single-column near-unit case remains ineffective, duplicate columns trigger fallback, and heterogeneous persistence produces **7.5%** false positives. The release remains a scoped conservative research baseline.

[Three figures](https://studyer-tang.github.io/strategy-inference/research/joint/) · [Proofs](docs/joint-uncertainty.md) · [Results](docs/joint-results.md) · [Protocol](experiments/joint-uncertainty-protocol.json)

```bash
python scripts/joint_report.py --output results/research/joint/full
python scripts/joint_uncertainty.py --profile full --workers 4 --output results/research/joint/reproduced
python scripts/verify_joint_uncertainty.py --output results/research/joint/reproduced
python scripts/verify_joint_bounds.py --output results/research/joint/reproduced
python scripts/joint_report.py --output results/research/joint/reproduced
```

## Uncertainty and scope

Bootstrap p values use `(1 + exceedances)/(B + 1)`. This avoids zero estimates and sets the numerical resolution; it is not an exact randomization-test guarantee. A conditional bootstrap-tail interval describes uncertainty from a finite number of draws for the observed data.

Outer Monte Carlo Wilson intervals describe simulation uncertainty in each estimated rejection rate and are pointwise. The benchmark assessment uses simultaneous one-sided Clopper–Pearson upper bounds over 19 prespecified null cells. Those bounds concern Monte Carlo rejection rates, whereas the API's simultaneous confidence intervals concern candidate means. An upper bound above 7% fails to establish the benchmark limit; it does not alone prove that the true rate exceeds 7%. Passing the finite benchmark would not establish validity for all financial series.

The bootstrap audit requires stationarity, weak dependence, suitable moments and consistent scale estimation; its theoretical scope keeps `K` fixed. Degenerate resamples and unsupported floating-point scales fail explicitly. Strong persistence, short samples, nonstationarity and heavy tails can still impair approximation quality. The finite-sample result for `uncertainty_test` uses its separate common Gaussian AR model and does not validate the bootstrap interface.

Adjustment covers the supplied family. A return matrix cannot recover omitted trials, validate transaction costs, detect every form of leakage, or justify strategies generated adaptively after repeatedly examining the same data. Rejecting the family null does not establish future profitability. The project implements established methods and studies their limits; it does not claim a new theorem.

## Development

```bash
python -m pip install -e '.[figures,dev]'
python -m pytest
ruff check .
python scripts/sync_protocols.py --check
python scripts/build_site.py --check
python scripts/build_uncertainty_site.py --check
python scripts/build_joint_site.py --check
```

The [method notes](docs/methods.md) derive the statistics and connect them to the implementation. The [research plan](docs/research-plan.md) (Chinese) connects recent papers to completed work, remaining questions and proof obligations. Source methods and software references are recorded in [references.bib](docs/references.bib). Code is licensed under BSD-3-Clause.
