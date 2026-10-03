# Changelog

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
