"""Check deterministic identities behind the tail-mechanism note.

No calibration simulation, tuning, or changes to the frozen study are made.
Finite-grid limit discrepancies are recorded; they are not proofs of limits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from strategy_inference.inference import default_lags
from strategy_inference.quadratic_reference import gaussian_ar_hac_moments

ROOT = Path(__file__).resolve().parents[1]


def _matrices(size: int, phi: float, sigma: float, bandwidth: int) -> tuple:
    time = np.arange(size)
    distances = np.abs(time[:, None] - time[None, :])
    covariance = sigma**2 * phi**distances
    weights = np.maximum(0, 1 - distances / bandwidth)
    centering = np.eye(size) - np.ones((size, size)) / size
    quadratic = centering @ weights @ centering / size
    return covariance, weights, centering, quadratic


def _row_sum_expectation(size: int, phi: float, bandwidth: int) -> tuple[float, float]:
    """Separate scalar autocovariance calculation, with marginal variance one."""
    time = np.arange(size)
    if phi == 0:
        rows = np.ones(size)
    else:
        log_phi = math.log(phi)
        rows = 1 + phi * (-np.expm1(time * log_phi) - np.expm1((size - 1 - time) * log_phi)) / (
            1 - phi
        )
    target = float(rows.sum() / size)
    diagonal_sums = [size - target]
    for lag in range(1, bandwidth):
        centered_sum = (
            (size - lag) * phi**lag
            - (rows[:-lag].sum() + rows[lag:].sum()) / size
            + (size - lag) * target / size
        )
        diagonal_sums.append(2 * (1 - lag / bandwidth) * float(centered_sum))
    return math.fsum(diagonal_sums) / size, target


def verify() -> dict:
    source = Path(__file__).resolve()
    own_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    residuals, inequalities = {}, {}

    def close(name, actual, expected, *, relative=2e-11, absolute=2e-13):
        difference = abs(float(actual) - float(expected))
        relative_error = difference / max(abs(float(expected)), np.finfo(float).tiny)
        record = residuals.setdefault(name, {"count": 0, "max_absolute": 0.0, "max_relative": 0.0})
        record["count"] += 1
        record["max_absolute"] = max(record["max_absolute"], difference)
        record["max_relative"] = max(record["max_relative"], relative_error)
        if not math.isclose(actual, expected, rel_tol=relative, abs_tol=absolute):
            raise ArithmeticError(f"{name}: {actual!r} differs from {expected!r}.")

    def nonnegative(name, slack):
        record = inequalities.setdefault(name, {"count": 0, "minimum_slack": math.inf})
        record["count"] += 1
        record["minimum_slack"] = min(record["minimum_slack"], float(slack))
        if slack < -2e-11:
            raise ArithmeticError(f"{name}: negative slack {slack!r}.")

    dense = []
    for size in (8, 17, 31):
        for phi in (-0.85, 0.0, 0.65, 0.99):
            for bandwidth in (1, 2, size - 1):
                for sigma in (0.4, 1.7):
                    covariance, weights, centering, quadratic = _matrices(
                        size, phi, sigma, bandwidth
                    )
                    one = np.ones(size)
                    expectation = float(np.trace(quadratic @ covariance))
                    product = quadratic @ covariance
                    variance = float(2 * np.trace(product @ product))
                    mass = sigma**2 * math.fsum(
                        [1.0] + [2 * (1 - h / bandwidth) * phi**h for h in range(1, bandwidth)]
                    )
                    uncentered = float(np.trace(weights @ covariance) / size)
                    pair_loss = (
                        2
                        * sigma**2
                        / size
                        * math.fsum(h * (1 - h / bandwidth) * phi**h for h in range(1, bandwidth))
                    )
                    center_loss = (
                        2 * one @ weights @ covariance @ one / size**2
                        - (one @ weights @ one) * (one @ covariance @ one) / size**3
                    )
                    close("trace_expansion", expectation, uncentered - center_loss)
                    close(
                        "pair_and_center_decomposition", expectation, mass - pair_loss - center_loss
                    )
                    reference = gaussian_ar_hac_moments(size, phi, sigma, lags=bandwidth - 1)
                    close("production_expectation", reference.expectation, expectation)
                    close("production_variance", reference.variance, variance)
                    close("finite_mean_target", reference.target, one @ covariance @ one / size)
                    close("population_kernel_mass", reference.population_truncated_lrv, mass)
                    if phi == 0:
                        iid = sigma**2 * (1 - bandwidth / size + (bandwidth**2 - 1) / (3 * size**2))
                        close("iid_expectation_closed_form", expectation, iid)
                        if bandwidth == 1:
                            close(
                                "iid_chi_square_variance",
                                variance,
                                2 * sigma**4 * (size - 1) / size**2,
                            )
                    if phi >= 0:
                        omega = sigma**2 * (1 + phi) / (1 - phi)
                        r = bandwidth * omega / (size * mass)
                        bias = 1 - expectation / mass
                        nonnegative("nonnegative_pair_loss", pair_loss / mass)
                        nonnegative("nonnegative_center_loss", center_loss / mass)
                        nonnegative("relative_bias_upper_bound", bandwidth / size + 2 * r - bias)
                        nonnegative("relative_variance_upper_bound", 2 * r - variance / mass**2)
                        nonnegative(
                            "persistence_r_upper_bound",
                            (bandwidth + 2 * phi / (1 - phi)) / size - r,
                        )
                        short_covariance = phi ** np.abs(
                            np.arange(bandwidth)[:, None] - np.arange(bandwidth)[None, :]
                        )
                        precision_sum = np.ones(bandwidth) @ np.linalg.solve(
                            short_covariance, np.ones(bandwidth)
                        )
                        close(
                            "ar_precision_sum",
                            precision_sum,
                            (bandwidth * (1 - phi) + 2 * phi) / (1 + phi),
                        )
                        values, vectors = np.linalg.eigh(covariance)
                        square_root = (vectors * np.sqrt(values)) @ vectors.T
                        normalized = (
                            square_root
                            @ (centering @ weights @ centering)
                            @ square_root
                            / (size * mass)
                        )
                        nonnegative(
                            "quadratic_operator_bound", r - np.linalg.eigvalsh(normalized).max()
                        )
                        nonnegative(
                            "quadratic_trace_square_bound", r - np.trace(normalized @ normalized)
                        )
                    dense.append(
                        {"n_obs": size, "phi": phi, "sigma": sigma, "bandwidth": bandwidth}
                    )

    # Match the actual fixed-bandwidth T=200 example; covariance is dense here.
    size, phi, bandwidth = 200, 0.99, 5
    covariance, weights, _, quadratic = _matrices(size, phi, 1, bandwidth)
    expectation = float(np.trace(quadratic @ covariance))
    target = float(covariance.sum() / size)
    mass = 1 + 2 * math.fsum((1 - h / bandwidth) * phi**h for h in range(1, bandwidth))
    example = dict(
        n_obs=size,
        phi=phi,
        marginal_variance=1,
        bandwidth=bandwidth,
        population_mass=mass,
        uncentered_pair_expectation=float(np.trace(weights @ covariance) / size),
        centered_expectation=expectation,
        finite_target=target,
        long_run_limit=(1 + phi) / (1 - phi),
        finite_target_corrected_expectation=target * expectation / mass,
        finite_target_expectation_ratio=expectation / mass,
        long_run_corrected_expectation=((1 + phi) / (1 - phi)) * expectation / mass,
    )
    close(
        "t200_example_expectation",
        expectation,
        gaussian_ar_hac_moments(size, phi, lags=4).expectation,
    )

    # C = 11' + expm1(log(phi)*distance). A annihilates 11', so use only
    # the residual covariance in the trace rather than subtracting unit levels.
    time = np.arange(size)
    distances = np.abs(time[:, None] - time[None, :])
    walk_covariance = np.minimum(time[:, None], time[None, :])
    limit_coefficient = float(2 * np.trace(quadratic @ walk_covariance))
    close("brownian_distance_identity", limit_coefficient, -np.trace(quadratic @ distances))
    fixed_boundary = []
    for phi in (0.9, 0.99, 0.9999, 0.999999, 0.99999999, np.nextafter(1.0, 0.0)):
        delta = 1 - phi
        residual_covariance = np.expm1(math.log(phi) * distances)
        scaled = float(np.trace(quadratic @ residual_covariance) / delta)
        reference = gaussian_ar_hac_moments(size, phi, lags=bandwidth - 1)
        close("fixed_boundary_stable_trace", scaled, reference.expectation / delta)
        fixed_boundary.append(
            dict(
                phi=float(phi),
                delta=delta,
                expectation_over_delta=scaled,
                limit_coefficient=limit_coefficient,
                discrepancy=scaled - limit_coefficient,
            )
        )
    close(
        "fixed_boundary_last_grid_point",
        fixed_boundary[-1]["expectation_over_delta"],
        limit_coefficient,
    )

    # Seeded Gaussian inputs are fixed algebra fixtures, not a rejection study.
    identity_inputs = []
    rng = np.random.default_rng(7841)
    for size in (8, 31, 200):
        for phi in (-0.7, 0.0, 0.9, 0.99):
            innovations = rng.normal(size=size)
            sample = np.empty(size)
            sample[0] = innovations[0]
            for t in range(1, size):
                sample[t] = phi * sample[t - 1] + math.sqrt(1 - phi**2) * innovations[t]
            mean = sample.mean()
            centered = sample - mean
            s0 = float(centered @ centered)
            s1 = float(centered[1:] @ centered[:-1])
            rhs = (
                math.sqrt(1 - phi**2) * (sample[:-1] @ innovations[1:])
                - phi * sample[-1] ** 2
                - (size * (1 - phi) + 1) * mean**2
                + mean * (sample[0] + sample[-1])
            )
            close("plugin_finite_t_identity", s1 - phi * s0, rhs)
            identity_inputs.append(dict(n_obs=size, phi=phi, lhs=s1 - phi * s0, rhs=float(rhs)))

    local_unit = []
    for a in (0.2, 1.0, 5.0):
        limit = 1 - 2 * (a + math.expm1(-a)) / a**2
        for size in (128, 512, 2048):
            phi = 1 - a / size
            bandwidth = default_lags(size) + 1
            expectation, target = _row_sum_expectation(size, phi, bandwidth)
            mass = 1 + 2 * math.fsum((1 - h / bandwidth) * phi**h for h in range(1, bandwidth))
            reference = gaussian_ar_hac_moments(size, phi, lags=bandwidth - 1)
            close("local_unit_row_sum_expectation", expectation, reference.expectation)
            close("local_unit_row_sum_target", target, reference.target)
            ratio = expectation / mass
            local_unit.append(
                dict(
                    a=a,
                    n_obs=size,
                    phi=phi,
                    bandwidth=bandwidth,
                    relative_expectation=ratio,
                    limiting_relative_expectation=limit,
                    finite_grid_discrepancy=ratio - limit,
                )
            )

    if hashlib.sha256(source.read_bytes()).hexdigest() != own_hash:
        raise RuntimeError("Verification script changed while checking its identities.")
    return dict(
        status="passed",
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        scope="Deterministic Gaussian quadratic-form and finite-T algebra checks. Recorded limit-grid discrepancies are not proofs, calibration experiments, or tuning criteria.",
        script_sha256=own_hash,
        production_reference_sha256=hashlib.sha256(
            (ROOT / "src/strategy_inference/quadratic_reference.py").read_bytes()
        ).hexdigest(),
        numerical_max_residuals=residuals,
        inequality_checks=inequalities,
        dense_cases=dense,
        t200_example=example,
        fixed_t_boundary=fixed_boundary,
        plugin_identity_inputs=dict(
            seed=7841,
            role="Fixed seeded Gaussian algebra fixtures, not Monte Carlo calibration.",
            cases=identity_inputs,
        ),
        local_to_unity_grid=dict(
            role="Finite-grid discrepancies recorded without an asymptotic acceptance rule.",
            cases=local_unit,
        ),
        unverified_scope=[
            "No conditional-mean Monte Carlo check was run.",
            "No empirical size limit, Jensen-mixture Monte Carlo check, or plugin-rate simulation was run.",
            "No growing-K, estimated-rho, bootstrap-validity, or novelty claim follows from this script.",
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/research/verification/tail-mechanism-checks.json",
    )
    arguments = parser.parse_args()
    report = verify()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        f"Verified {len(report['dense_cases'])} dense cases; recorded fixed-T and local-to-unity grids."
    )


if __name__ == "__main__":
    main()
