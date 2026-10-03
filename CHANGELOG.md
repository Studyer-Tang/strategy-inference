# Changelog

## 0.9.0

- Add streaming `SequentialModelConfidenceSet` and a backtest adapter, based on Arnold et al. (JRSSB, 2026). Compare all prespecified models under strong conditional superiority with predictable absolute/pinball score-difference bounds and closed e-testing.
- Keep evidence in the log domain, avoid overflowing raw losses, and use the paper's O(M log M) closure adjustment. State uses O(M²) memory without observation history or resampling. Reject overlapping feedback and invalid updates atomically.
- Extend the existing example and API reference; keep historical studies in the documentation archive.

## 0.8.1

- Remove superseded scale-learning prototypes and their dedicated tests from the active tree. Keep historical source at v0.8.0 and preserve all saved results and source ZIPs. Share the current study's generation, metrics and validation, and index path records once when aggregating.
- Validate the audit input matrix once; retain strict public entry points and the same bootstrap arithmetic, random streams and workspace budget. Share interval scoring and vectorize multistep summaries while retaining scaled `math.fsum` means and invalid-interval handling. Avoid unused feedback-lane allocations.
- Load plotting only for figure-producing runs. Package the two protocols actually consumed by the installed CLI; retain canonical research protocols outside the wheel.
- Bind v0.8.0 evidence to its immutable release and add a self-contained archive. Document a five-workload, same-input performance comparison separately from historical timings; preserve public APIs and statistical scope.

## 0.8.0

- Add optional `scale_source="blended"` with fixed scalar or per-lead `scale_share_weight`: combine own-lead mature RMS with shortest-lead shared RMS, retaining the existing threshold recursion and feedback queue.
- Preserve initial scales and old-mode results at endpoint weights on the common finite-source domain. Validate both sources, including unused endpoints, and reject invalid updates atomically. Export detached source scales and fixed weights.
- Evaluate the public API on 240 new independent paths with causal fitted rolling forecasts, six stress processes and prespecified sharing weights. Report observed score/coverage tradeoffs, nominal Monte Carlo summaries and finite-score limitations; no universal efficiency or foundational novelty claim.
- Add source snapshots, three reproducible study figures, current-source runtime/memory measurements, and a self-contained frozen v0.7 archive.

## 0.7.0

- Add `MultiStepConformal` and `multistep_intervals` for univariate consecutive integer-time observations. Observe current labels before issuing future paths; update only matured forecasts and retain unobserved future targets as pending. Freeze each issued scale and interval.
- Support pooled current-state feedback and origin-phase interlaced lanes, with scalar or per-lead learning-rate constants. Document their different decay clocks and ideal-arithmetic retrospective average-miscoverage bounds; retain explicit empty/full intervals and binary64 boundary limitations.
- Add optional mature-residual EWMA RMS scales, either per horizon or shared from the shortest configured lead with fixed initial ratios. Keep per-lead miss controllers separate. Treat shared-scale efficiency as a research question, without claiming a full AcMCP implementation, conditional coverage or simultaneous path coverage.
- Add a runnable random-walk/backtest example, multi-step API documentation and recent primary references. Freeze v0.6 performance evidence against its release commit so later modules and versions do not invalidate historical source bindings.
- Report a fixed four-process study with 160 independent paths, paired interval scores, local-coverage diagnostics and three reproducible scientific figures. Retain the development-informed rate choice and its separate seeds; distinguish simulated gains over own-lead EWMA from general superiority.
- Save four source-bound batch/stream benchmarks, publish book-style documentation from validated raw records, and preserve frozen v0.5/v0.6 pages.

## 0.6.0

- Add rolling-origin evaluation with explicit gaps, bounded or expanding training windows, isolated training copies and a model-agnostic callable interface. Retain every origin, lead and forecast, including overlapping targets. Include naive, seasonal-naive and drift baselines.
- Add vectorized squared, absolute, pinball and interval scores. Keep lead-specific loss summaries and compare a prespecified family against one baseline at one lead using shared-row stationary bootstrap. Preserve nonstationarity, overlap and incomplete-search limitations.
- Add one-step adaptive conformal intervals based on Angelopoulos–Barber–Bates (ICML 2024). Use a fixed-scale bounded residual score, explicit empty/unbounded sets and ordered predict-then-feedback updates. Distinguish retrospective average coverage from conditional or delayed-feedback coverage; retain binary64 limitations.
- Document a broader toolbox roadmap with verified recent papers and separate implemented capabilities from planned research. Archive v0.5 runtime evidence against its frozen release source instead of treating future unrelated modules as part of that historical benchmark.

## 0.5.0

- Add `test_returns` and `TestResult` for consistent array/DataFrame/CSV input, column decisions, JSON records and optional pandas tables. Keep legacy interfaces and their defaults; the new bootstrap entry uses resampled HAC scales explicitly.
- Reduce resampled-bootstrap work with contiguous time-axis batches and a bounded workspace. Reuse validated mean/HAC computations and release index buffers early, while retaining float64, shared-row random streams and draw counts.
- Reuse exact prefix Gram sums and Bernstein subdivision coefficients. Seed the Student root search using a standard-library approximation, then establish the same outward cutoff by exact integer comparisons. Preserve rational intervals, determinant polynomials, decisions and certification budgets.
- Lazily load numerical interfaces; CLI help/version do not import NumPy or SciPy. Include typing markers, concise API documentation and source-bound before/after benchmarks. Retain frozen research protocols and evidence under their original revisions.

## 0.4.0

- Add `wilks_uncertainty_test`: fixed multiscale innovation contrasts and up to eight prespecified columns inform the common AR parameter. Independent Wishart scatters remove unknown contemporaneous covariance; exact integer moments and outward rational roots preserve the coverage budget. Singular selected covariance supplies no information and conservatively retains the full parameter domain.
- Certify degree-at-most-16 determinant inequalities over continuous parameter intervals, then reuse the fixed-level GLS strong-FWER argument. Preserve unresolved intervals and nonrejections; do not estimate effective rank or claim a general power ordering.
- Freeze a new paired evaluation of temporal and cross-column information, including partial nulls, severe correlation, duplicate columns, near-unit persistence and model misspecification. Save complete records, audit selected full certificates, and provide three reproducible scientific figures.


## 0.3.0

- Add `uncertainty_test`: fixed-level simultaneous mean decisions for stationary Gaussian AR columns with a common unknown coefficient in `[0,1)`, unknown marginal scales and unrestricted contemporaneous covariance. Use an exact innovation F confidence set and a shared coverage budget; document strong FWER and every model condition.
- Certify the continuous parameter domain with exact integer/rational projection, Beta/Student critical values and Bernstein bounds. Retain unresolved intervals and suppress unresolved or empty-set decisions. Distinguish input-float algebraic certificates from ideal Gaussian distribution guarantees.
- Freeze a separate independent power-cost experiment with global/partial nulls, signed correlations, unequal scales and heterogeneous-persistence diagnostics. Compare the same GLS mean target, isolate budget versus envelope costs and retain near-unit-root power losses.
- Complete the parametric-replay study, fixed-K mild-persistence proof, local-unit mechanism and finite-rank resolution diagnosis. Publish saved raw evidence, independent audits and reproducible scientific figures.

- Independently implement fixed-bandwidth AR(1) tail factors from Liu–Chan (JASA 2026), including their finite-sample target. Cross-check 48 scalar results against original author functions running in R; retain execution provenance without vendoring GPL source.
- Compute exact first and second moments of centered Gaussian Bartlett HAC through quadratic forms, with stable AR covariance and bounded matrix calculations.
- Freeze and complete a separate 12-cell, 5000-replicate mechanism study. Retain paired decisions, winners, scale errors, seeds and hashes; reproduce three scientific figures from verified evidence. Known-correlation and mean-unbiased references diagnose failures rather than providing practical unknown-parameter guarantees.
- Add scoped proofs and result notes. Preserve public bootstrap-audit defaults and previously published studies.

## 0.2.0

- Add `studentization="resampled"` to `audit_returns` and the CSV CLI. Each bootstrap draw recomputes Bartlett HAC around its own mean, with the original lag count and shared row indices. The `"fixed"` default preserves the v0.1 interpretation.
- Add `stationary_bootstrap_statistics` with bounded draw and column batches, and `stationary_mean_variance` for the exact conditional variance of stationary-bootstrap means.
- Add known-covariance Gaussian AR reference calculations for finite-sample mean variances and equicorrelated maxima. These are simulation references and do not supply unknown population parameters to the audit API.
- State the numerical reference range explicitly: common-factor maximum quantiles support probabilities in `[1e-12, 1-1e-12]`; analytic boundary cases retain their wider domain. Quadrature tolerances do not guarantee relative accuracy in arbitrarily small tails.
- Add a separate frozen calibration protocol with an independent evaluation seed, paired method comparisons, holdout scenarios, block-length diagnostics and a simultaneous Monte Carlo assessment rule. Keep the original baseline protocol and outputs. Evaluation failures are retained; the new method does not carry a general calibration claim.
- Complete the frozen confirmation run and retain its failed statistical assessment: only 1 of 19 simultaneous upper bounds meets the prespecified 7% limit. Record the original computation revision and a seven-table quick comparison against subsequent guard and resource-lookup changes.
- Record the studentization option in JSON schema version 2 and report conditional bootstrap-tail Monte Carlo uncertainty separately from statistical validity.
- Reject constant or degenerate resamples, nonpositive or nonfinite scales, and unsupported floating-point underflow or overflow before reporting inference. Do not discard failed draws or regularize them silently.
- Extend the method documentation, references and English README. Distinguish first-order assumptions from higher-order bootstrap results, and pointwise Monte Carlo intervals from simultaneous assessment bounds.

## 0.1.0

- Establish the research baseline: IID Student and Bartlett HAC mean inference, plus a fixed-scale stationary-bootstrap max test for a supplied finite candidate family.
- Use common circular row indices to retain cross-candidate dependence; report family and single-step adjusted p values, simultaneous two-sided intervals and one-sided lower bounds.
- Provide numeric CSV audits, JSON/HTML reports, and three reproducible simulations covering time dependence, candidate selection and process robustness with signal detection.
- Document input conventions, assumptions, candidate-family completeness and finite-sample limitations. Preserve the baseline as evidence rather than treating it as a calibrated practical test.
