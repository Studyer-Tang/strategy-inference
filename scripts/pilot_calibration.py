"""Exploratory calibration diagnostics; NEVER replace the frozen v1 results.

Examples (each cell uses independently addressed, reproducible MC replicates):
  PYTHONPATH=src .venv/bin/python scripts/pilot_calibration.py \
      --cells gaussian_high_dependence gaussian_search --blocks 16 --restudentize

These pilots help choose hypotheses for a new, separately frozen confirmation
protocol. They do not establish that a selected method has nominal size.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.stats import norm

from strategy_inference.bootstrap import (
    stationary_bootstrap_means,
    stationary_bootstrap_statistics,
    stationary_mean_variance,
)
from strategy_inference.inference import infer_mean
from strategy_inference.simulations import simulate_returns

CELLS = {
    "gaussian_high_dependence": dict(process="gaussian_ar", phi=0.8, k=1),
    "gaussian_search": dict(process="gaussian_ar", phi=0.5, k=50),
    "student_search": dict(process="student_ar", phi=0.5, k=20),
    "garch_search": dict(process="garch", phi=0.0, k=20),
}
ALPHA = 0.05
T = 512
SIGMA = 0.01
CORR = 0.35
LAGS = 5


def stream(seed: int, scenario: int, replicate: int, purpose: int) -> int:
    seq = np.random.SeedSequence(seed, spawn_key=(51, scenario, replicate, purpose))
    return int(seq.generate_state(1, dtype=np.uint64)[0])


def pvalue(observed: float, values: np.ndarray) -> float:
    return float((1 + np.count_nonzero(values >= observed)) / (len(values) + 1))


def wilson(count: int, n: int) -> tuple[float, float, float]:
    proportion = count / n
    z = float(norm.ppf(0.975))
    divisor = 1 + z**2 / n
    center = (proportion + z**2 / (2 * n)) / divisor
    radius = z * math.sqrt(proportion * (1 - proportion) / n + z**2 / (4 * n**2)) / divisor
    return proportion, max(0.0, center - radius), min(1.0, center + radius)


def csv_write(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def exact_population_mean_variance(phi: float) -> float:
    lag = np.arange(1, T)
    return float(SIGMA**2 * (1 + 2 * np.sum((1 - lag / T) * phi**lag)) / T)


def run(args: argparse.Namespace) -> dict:
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = dict(
        status="running", evidence="EXPLORATORY PILOT; not confirmation and not v1 replacement",
        n_obs=T, n_mc=args.mc, n_bootstrap=args.resamples, seed=args.seed,
        cells={name: CELLS[name] for name in args.cells}, block_lengths=args.blocks,
        hac_lags=LAGS, sigma=SIGMA, shock_correlation=CORR, burnin=512,
        restudentize=args.restudentize,
        streams="SeedSequence(seed,spawn_key=(51,scenario_id,replicate,purpose)); "
                "purpose 0=DGP, 1=common-row bootstrap, 2=Gaussian oracle. "
                "Each block length reuses the same uniforms/data and each method the same bootstrap rows.",
        methods={
            "fixed_hac": "max(mean/original HAC5 SE) vs max(centered bootstrap mean/original HAC5 SE)",
            "conditional_sb_scale": "max(mean/exact conditional SB mean SD) vs "
                                    "max(centered bootstrap mean/same fixed conditional SD)",
            "restudentized_hac": "max(mean/original HAC5 SE) vs max(centered bootstrap "
                                 "mean/recomputed HAC5 SE per bootstrap sample)",
            "oracle_gaussian": "Known finite-T Gaussian mean covariance; diagnostic only, "
                               "not implementable without known DGP parameters",
        },
        conditional_variance_formula="[gamma_circ(0)+2 sum_(h=1)^(T-1) "
                                     "(1-h/T)*(1-1/L)^h*gamma_circ(h)]/T",
        variance_ratio_reference="Known finite-T AR mean variance for Gaussian/t-innovation AR; "
                                 "sigma^2/T for stationary GARCH martingale differences. "
                                 "Student/GARCH initialization is approximate after burn-in.",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        completed_cells=[], elapsed_seconds=0.0,
    )
    metadata_path = output / "pilot-metadata.json"
    metadata_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    started = time.perf_counter()
    details, summary, contrasts = [], [], []
    methods = ["fixed_hac", "conditional_sb_scale"]
    if args.restudentize:
        methods.append("restudentized_hac")
    for name in args.cells:
        scenario = list(CELLS).index(name)
        cell = CELLS[name]
        phi, k = cell["phi"], cell["k"]
        population_variance = exact_population_mean_variance(phi)
        counts = {length: dict.fromkeys(methods, 0) for length in args.blocks}
        ratios = {length: [] for length in args.blocks}
        paired = {length: [] for length in args.blocks}
        oracle_count = 0
        cell_started = time.perf_counter()
        print(f"Pilot {name}: MC={args.mc}, B={args.resamples}, L={args.blocks}, "
              f"re-studentization={args.restudentize}", flush=True)
        for replicate in range(args.mc):
            data_seed = stream(args.seed, scenario, replicate, 0)
            bootstrap_seed = stream(args.seed, scenario, replicate, 1)
            returns = simulate_returns(T, k, process=cell["process"], phi=phi,
                                       cross_corr=CORR, sigma=SIGMA, seed=data_seed)
            hac = infer_mean(returns, method="hac", lags=LAGS)
            observed_hac = float(np.max(hac.statistic))
            winner = int(np.argmax(hac.statistic))
            oracle_p = math.nan
            if cell["process"] == "gaussian_ar":
                raw = np.random.default_rng(stream(args.seed, scenario, replicate, 2)).standard_normal((args.resamples, k + 1))
                oracle = math.sqrt(CORR) * raw[:, :1] + math.sqrt(1 - CORR) * raw[:, 1:]
                oracle_p = pvalue(float(np.max(hac.mean / math.sqrt(population_variance))), np.max(oracle, axis=1))
                oracle_count += int(oracle_p <= ALPHA)
            for length in args.blocks:
                means = stationary_bootstrap_means(returns, n_resamples=args.resamples,
                                                  block_length=length, seed=bootstrap_seed,
                                                  center=True)
                conditional_variance = stationary_mean_variance(returns, block_length=length)
                if np.any(conditional_variance <= 0):
                    raise FloatingPointError("A conditional bootstrap variance was not positive")
                conditional_sd = np.sqrt(conditional_variance)
                probabilities = dict(
                    fixed_hac=pvalue(observed_hac, np.max(means / hac.standard_error, axis=1)),
                    conditional_sb_scale=pvalue(float(np.max(hac.mean / conditional_sd)),
                                                 np.max(means / conditional_sd, axis=1)),
                )
                if args.restudentize:
                    statistics = stationary_bootstrap_statistics(
                        returns, n_resamples=args.resamples, block_length=length,
                        lags=LAGS, seed=bootstrap_seed,
                    )
                    probabilities["restudentized_hac"] = pvalue(observed_hac, np.max(statistics, axis=1))
                if k == 1 and probabilities["fixed_hac"] != probabilities["conditional_sb_scale"]:
                    raise AssertionError("Fixed positive denominators should cancel at K=1")
                ratios[length].append((
                    float(np.mean(hac.standard_error**2 / population_variance)),
                    float(np.mean(conditional_variance / population_variance)),
                    float(hac.standard_error[winner]**2 / population_variance),
                    float(conditional_variance[winner] / population_variance),
                ))
                decisions = {method: int(probability <= ALPHA) for method, probability in probabilities.items()}
                if args.restudentize:
                    paired[length].append(decisions["restudentized_hac"] - decisions["fixed_hac"])
                for method, probability in probabilities.items():
                    counts[length][method] += decisions[method]
                    details.append(dict(cell=name, process=cell["process"], phi=phi, k=k,
                                        block_length=length, replicate=replicate, method=method,
                                        pvalue=probability, reject=decisions[method],
                                        data_seed=data_seed, bootstrap_seed=bootstrap_seed,
                                        mean_hac_variance_ratio=ratios[length][-1][0],
                                        mean_conditional_variance_ratio=ratios[length][-1][1],
                                        selected_hac_variance_ratio=ratios[length][-1][2],
                                        selected_conditional_variance_ratio=ratios[length][-1][3]))
            if (replicate + 1) % max(1, args.mc // 5) == 0:
                print(f"  {replicate + 1}/{args.mc}, elapsed {time.perf_counter()-cell_started:.1f}s", flush=True)
        for length in args.blocks:
            average_ratios = np.mean(ratios[length], axis=0)
            for method, count in counts[length].items():
                rate, low, high = wilson(count, args.mc)
                summary.append(dict(cell=name, process=cell["process"], phi=phi, k=k,
                                    block_length=length, method=method, n_mc=args.mc,
                                    n_bootstrap=args.resamples, reject_count=count,
                                    rate=rate, ci_low=low, ci_high=high,
                                    mean_hac_variance_ratio=float(average_ratios[0]),
                                    mean_conditional_variance_ratio=float(average_ratios[1]),
                                    selected_hac_variance_ratio=float(average_ratios[2]),
                                    selected_conditional_variance_ratio=float(average_ratios[3])))
            if args.restudentize:
                difference = np.array(paired[length], dtype=float)
                contrasts.append(dict(cell=name, block_length=length, n_mc=args.mc,
                                      comparison="restudentized_hac minus fixed_hac",
                                      rate_difference=float(difference.mean()),
                                      paired_mc_standard_error=float(difference.std(ddof=1) / math.sqrt(args.mc)),
                                      changed_decisions=int(np.count_nonzero(difference))))
        if cell["process"] == "gaussian_ar":
            rate, low, high = wilson(oracle_count, args.mc)
            summary.append(dict(cell=name, process=cell["process"], phi=phi, k=k,
                                block_length="not_applicable", method="oracle_gaussian", n_mc=args.mc,
                                n_bootstrap=args.resamples, reject_count=oracle_count,
                                rate=rate, ci_low=low, ci_high=high,
                                mean_hac_variance_ratio=float(np.mean(ratios[args.blocks[0]], axis=0)[0]),
                                mean_conditional_variance_ratio="not_applicable",
                                selected_hac_variance_ratio="not_applicable",
                                selected_conditional_variance_ratio="not_applicable"))
        csv_write(output / "summary.csv", summary)
        csv_write(output / "replicates.csv", details)
        csv_write(output / "paired-contrasts.csv", contrasts)
        config["completed_cells"].append(name)
        config["elapsed_seconds"] = time.perf_counter() - started
        metadata_path.write_text(json.dumps(config, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    config["status"] = "complete"
    config["elapsed_seconds"] = time.perf_counter() - started
    metadata_path.write_text(json.dumps(config, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Pilot complete in {config['elapsed_seconds']:.1f}s: {output}", flush=True)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/pilot/calibration")
    parser.add_argument("--mc", type=int, default=500)
    parser.add_argument("--resamples", type=int, default=399)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--cells", nargs="+", choices=list(CELLS), default=list(CELLS))
    parser.add_argument("--blocks", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--restudentize", action="store_true")
    args = parser.parse_args()
    if args.mc < 2 or args.resamples < 1 or args.seed < 0 or any(length < 1 or length > T for length in args.blocks):
        parser.error("MC must be >=2, B>=1, seed>=0 and each block length in [1,T]")
    run(args)


if __name__ == "__main__":
    main()
