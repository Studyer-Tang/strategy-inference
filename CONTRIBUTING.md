# Contributing

Please keep changes centered on inference for a documented statistical target.
An estimator or test should state its null, sampling assumptions, units, selection
rule, and expected failure cases. Preserve the distinction between numerical
verification, simulation evidence, and a statistical validity result.

For a computational change, compare a small instance against an independent
reference. For a statistical change, add a prespecified simulation with known
truth, report rejection counts and Monte Carlo uncertainty, and retain cases
where the proposed method performs poorly. Do not tune a seed or omit a design
point because its result is inconvenient.

Run `python -m pytest` and `ruff check .`. A quick reproduction checks the plotting
and data pipeline; it does not establish calibration. The full experiment protocol
is stored in `experiments/protocol.json` and `experiments/calibration-protocol.json`.
Freeze a new protocol before evaluation and preserve completed protocols and results.
Run both `reproduce --study baseline --profile quick` and
`reproduce --study calibration --profile quick` after changing the experiment pipeline.
