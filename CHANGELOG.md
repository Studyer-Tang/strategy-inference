# Changelog

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
