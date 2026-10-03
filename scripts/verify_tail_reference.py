"""Compare Python estimates against separately generated author-R results."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from strategy_inference.tail import tail_variance

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "results/research/verification"


def verify() -> dict:
    with (DIRECTORY / "tail-reference-input.csv").open() as stream:
        inputs = list(csv.DictReader(stream))
    with (DIRECTORY / "tail-reference-r.csv").open() as stream:
        reference = list(csv.DictReader(stream))
    assert len(reference) == len({row["case"] for row in inputs}) == 8
    checks = []
    for answer in reference:
        rows = [row for row in inputs if row["case"] == answer["case"]]
        data = np.array([float(row["value"]) for row in rows])
        phi, lags = float(rows[0]["known_phi"]), int(rows[0]["bandwidth"])-1
        known_lrv = tail_variance(data, phi=phi, lags=lags)
        plugin_lrv = tail_variance(data, lags=lags)
        known_finite = tail_variance(data, phi=phi, lags=lags, target="finite_sample")
        plugin_finite = tail_variance(data, lags=lags, target="finite_sample")
        values = {
            "fitted_phi": float(plugin_lrv.phi[0]), "unadjusted": float(plugin_lrv.unadjusted[0]),
            "known_lrv": float(known_lrv.variance[0]), "plugin_lrv": float(plugin_lrv.variance[0]),
            "known_finite": float(known_finite.variance[0]), "plugin_finite": float(plugin_finite.variance[0]),
        }
        for field, value in values.items():
            other = float(answer[field])
            error = abs(value-other)
            np.testing.assert_allclose(value, other, rtol=2e-11, atol=2e-13)
            checks.append({
                "case": int(answer["case"]), "field": field, "python": value, "author_r": other,
                "absolute_error": error, "relative_error": error/max(abs(other), np.finfo(float).tiny),
            })
    return {
        "status": "passed", "cases": 8, "scalar_checks": len(checks),
        "max_absolute_error": max(row["absolute_error"] for row in checks),
        "max_relative_error": max(row["relative_error"] for row in checks),
        "scope": "author lrv.np, arma11.M and arma11.lrv functions executed in R; fixed bandwidth; paper gamma1/gamma0 supplied for fitted-parameter ratios. Not a reproduction of the author's tseries::arma fitting wrapper or adaptive Table B.7.",
        "file_hashes": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                DIRECTORY / "tail-reference-input.csv", DIRECTORY / "tail-reference-r.csv",
                ROOT / "scripts/check_tail_reference.R", Path(__file__), ROOT / "src/strategy_inference/tail.py",
            )
        },
        "checks": checks,
    }


if __name__ == "__main__":
    report = verify()
    (DIRECTORY / "tail-reference-comparison.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(f"{report['scalar_checks']} R/Python checks passed; max relative error {report['max_relative_error']:.3g}")
