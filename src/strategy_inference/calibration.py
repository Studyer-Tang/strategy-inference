"""Independent finite-sample evaluation of two stationary-bootstrap scales."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import beta, norm, t

from ._validation import positive_integer
from .bootstrap import (
    default_block_length,
    stationary_bootstrap_means,
    stationary_bootstrap_statistics,
    stationary_mean_variance,
)
from .experiments import _bootstrap_pvalue, _progress, _stream_seed, _versions, _wilson, _write_json
from .inference import default_lags, infer_mean
from .reference import equicorrelated_max_quantile, gaussian_ar_mean_variance
from .simulations import simulate_returns


def _protocol() -> tuple[dict[str, Any], bytes, Path]:
    for path in (
        Path(__file__).resolve().parents[2] / "experiments" / "calibration-protocol.json",
        Path(__file__).resolve().parent / "protocols" / "calibration-protocol.json",
        Path(sys.prefix) / "share" / "strategy-inference" / "calibration-protocol.json",
    ):
        if path.is_file():
            raw = path.read_bytes()
            return json.loads(raw), raw, path
    raise FileNotFoundError("The frozen calibration-protocol.json was not installed.")


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty experiment table.")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _upper(count: int, total: int, *, family_alpha: float, n_cells: int) -> float:
    return (
        1.0
        if count == total
        else float(beta.ppf(1 - family_alpha / n_cells, count + 1, total - count))
    )


def _cell(
    config: dict[str, Any],
    scenario: dict[str, Any],
    *,
    study: int,
    address: int,
    stage: str,
    counts_k: list[int] | None = None,
    deltas: list[int] | None = None,
    blocks: list[float] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """One paired simulation cell, optionally reused for nested K or mean shifts."""
    n_obs = scenario.get("n_obs", config["n_obs"])
    k = scenario["n_strategies"]
    process, phi = scenario["process"], scenario["phi"]
    rho = scenario.get("cross_corr", config["cross_corr"])
    lags = default_lags(n_obs)
    block = default_block_length(n_obs)
    counts_k = [k] if counts_k is None else counts_k
    deltas = [0] if deltas is None else deltas
    blocks = [float(block)] if blocks is None else blocks
    n_mc = (
        config["n_mc"]
        if stage in ("dependence", "selection", "process")
        else config["supplement_mc"]
    )
    methods = ["iid", "hac", "fixed", "resampled"]
    if process == "gaussian_ar":
        methods.append("oracle")
    counts = {
        (length, prefix, delta, method): 0
        for length in blocks
        for prefix in counts_k
        for delta in deltas
        for method in methods
    }
    diagnostics: dict[float, list[tuple[float, float]]] = {length: [] for length in blocks}
    variance = (
        gaussian_ar_mean_variance(n_obs, phi, config["sigma"]) if process == "gaussian_ar" else None
    )
    critical = (
        {
            prefix: equicorrelated_max_quantile(1 - config["alpha"], prefix, rho)
            for prefix in counts_k
        }
        if variance is not None
        else {}
    )
    lrv_std = config["sigma"] * math.sqrt((1 + phi) / (1 - phi))
    iid_cutoff = float(t.ppf(1 - config["alpha"], n_obs - 1))
    hac_cutoff = float(norm.ppf(1 - config["alpha"]))
    started = time.perf_counter()
    print(
        f"[{stage}] {process}, T={n_obs}, K={k}, phi={phi:g}, rho={rho:g}, blocks={blocks}",
        flush=True,
    )
    for replicate in range(n_mc):
        data = simulate_returns(
            n_obs,
            k,
            process=process,
            phi=phi,
            cross_corr=rho,
            sigma=config["sigma"],
            seed=_stream_seed(config["seed"], study, address, replicate, 0),
            burnin=config["burnin"],
        )
        hac = infer_mean(data, method="hac", lags=lags)
        iid = infer_mean(data, method="iid")
        shared_seed = _stream_seed(config["seed"], study, address, replicate, 1)
        for length in blocks:
            # Identical RNG addresses and interleaved draws pair both methods.
            means = stationary_bootstrap_means(
                data,
                n_resamples=config["n_bootstrap"],
                block_length=length,
                seed=shared_seed,
                center=True,
            )
            resampled = stationary_bootstrap_statistics(
                data,
                n_resamples=config["n_bootstrap"],
                block_length=length,
                lags=lags,
                seed=shared_seed,
            )
            fixed_max = np.maximum.accumulate(means / hac.standard_error, axis=1)
            resampled_max = np.maximum.accumulate(resampled, axis=1)
            if variance is not None:
                diagnostics[length].append(
                    (
                        float(np.median(hac.standard_error**2 / variance)),
                        float(
                            np.median(
                                stationary_mean_variance(data, block_length=length) / variance
                            )
                        ),
                    )
                )
            for prefix in counts_k:
                for delta in deltas:
                    shifted = hac.statistic[:prefix].copy()
                    shifted[0] += delta * lrv_std / math.sqrt(n_obs) / hac.standard_error[0]
                    winner = int(np.argmax(shifted))
                    observed = float(shifted[winner])
                    iid_stat = iid.statistic[winner] + (
                        delta * lrv_std / math.sqrt(n_obs) / iid.standard_error[0]
                        if winner == 0
                        else 0
                    )
                    decisions = {
                        "iid": bool(iid_stat >= iid_cutoff),
                        "hac": bool(observed >= hac_cutoff),
                        "fixed": _bootstrap_pvalue(observed, fixed_max[:, prefix - 1])
                        <= config["alpha"],
                        "resampled": _bootstrap_pvalue(observed, resampled_max[:, prefix - 1])
                        <= config["alpha"],
                    }
                    if variance is not None:
                        known_stat = hac.mean[:prefix] / math.sqrt(variance)
                        known_stat[0] += delta * lrv_std / math.sqrt(n_obs * variance)
                        decisions["oracle"] = bool(known_stat.max() >= critical[prefix])
                    for method, rejected in decisions.items():
                        counts[length, prefix, delta, method] += int(rejected)
        _progress(stage, replicate + 1, n_mc, started)
    rows = []
    for (length, prefix, delta, method), count in counts.items():
        rate, low, high = _wilson(count, n_mc, config["confidence"])
        assessment = method == "resampled" and delta == 0 and stage != "sensitivity"
        rows.append(
            dict(
                stage=stage,
                process=process,
                n_obs=n_obs,
                n_strategies=prefix,
                phi=phi,
                cross_corr=rho,
                delta=delta,
                method=method,
                hac_lags=lags,
                block_length=length,
                n_mc=n_mc,
                n_bootstrap=config["n_bootstrap"],
                reject_count=count,
                rate=rate,
                ci_low=low,
                ci_high=high,
                alpha=config["alpha"],
                size_upper_simultaneous=_upper(
                    count,
                    n_mc,
                    family_alpha=1 - config["confidence"],
                    n_cells=config["n_assessment_cells"],
                )
                if assessment
                else "",
            )
        )
    diagnostic_rows = []
    for length, values in diagnostics.items():
        if values:
            matrix = np.array(values)
            diagnostic_rows.append(
                dict(
                    stage=stage,
                    process=process,
                    n_obs=n_obs,
                    n_strategies=k,
                    phi=phi,
                    cross_corr=rho,
                    block_length=length,
                    n_mc=n_mc,
                    true_mean_variance=variance,
                    hac_ratio_median=float(np.median(matrix[:, 0])),
                    hac_ratio_p10=float(np.quantile(matrix[:, 0], 0.1)),
                    hac_ratio_p90=float(np.quantile(matrix[:, 0], 0.9)),
                    bootstrap_ratio_median=float(np.median(matrix[:, 1])),
                    bootstrap_ratio_p10=float(np.quantile(matrix[:, 1], 0.1)),
                    bootstrap_ratio_p90=float(np.quantile(matrix[:, 1], 0.9)),
                )
            )
    return rows, diagnostic_rows


def run_calibration(
    output: str | Path, profile: str = "full", seed: int | None = None
) -> dict[str, Any]:
    """Execute the independently frozen protocol and retain every assessment cell."""
    protocol, raw, protocol_path = _protocol()
    if not isinstance(profile, str) or profile not in protocol["profiles"]:
        raise ValueError("profile must be 'quick' or 'full'.")
    seed = protocol["seed"] if seed is None else positive_integer(seed, "seed", 0)
    config = {
        **{key: protocol[key] for key in ("alpha", "confidence", "sigma", "cross_corr", "burnin")},
        **protocol["profiles"][profile],
        "seed": seed,
        "n_assessment_cells": protocol["assessment"]["n_assessment_cells"],
    }
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata = dict(
        schema_version=2,
        study="calibration",
        status="running",
        profile=profile,
        seed=seed,
        seed_overridden=seed != protocol["seed"],
        protocol=protocol,
        protocol_sha256=hashlib.sha256(raw).hexdigest(),
        protocol_source=protocol_path.name,
        resolved_protocol=config,
        started_at_utc=datetime.now(timezone.utc).isoformat(),
        software_versions=_versions(),
        outputs=[],
        elapsed_seconds=0.0,
    )
    try:
        git = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=False,
        )
        metadata["git_revision"] = git.stdout.strip() if git.returncode == 0 else None
    except OSError:
        metadata["git_revision"] = None
    metadata["source_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in Path(__file__).resolve().parent.glob("*.py")
        if path.stem
        in (
            "calibration",
            "bootstrap",
            "inference",
            "simulations",
            "reference",
            "_validation",
            "experiments",
        )
    }
    metadata_path = output / "run-metadata.json"
    _write_json(metadata_path, metadata)
    started = time.perf_counter()
    tables: dict[str, list[dict[str, Any]]] = {}
    diagnostics = []

    def checkpoint() -> None:
        for name, rows in tables.items():
            if rows:
                _csv(output / name, rows)
        metadata["outputs"] = [name for name, rows in tables.items() if rows]
        metadata["elapsed_seconds"] = round(time.perf_counter() - started, 6)
        _write_json(metadata_path, metadata)

    try:
        tables["figure-1-dependence.csv"] = []
        for address, phi in enumerate(protocol["figure_1"]["phi"]):
            rows, diag = _cell(
                config,
                dict(process="gaussian_ar", phi=phi, n_strategies=1),
                study=11,
                address=address,
                stage="dependence",
            )
            tables["figure-1-dependence.csv"].extend(rows)
            diagnostics.extend(diag)
            checkpoint()
        rows, diag = _cell(
            config,
            dict(
                process="gaussian_ar",
                phi=protocol["figure_2"]["phi"],
                n_strategies=max(protocol["figure_2"]["strategy_counts"]),
            ),
            study=12,
            address=0,
            stage="selection",
            counts_k=protocol["figure_2"]["strategy_counts"],
        )
        tables["figure-2-selection.csv"] = rows
        diagnostics.extend(diag)
        checkpoint()
        tables["figure-3-size.csv"], tables["figure-3-power.csv"] = [], []
        for address, process in enumerate(protocol["figure_3"]["processes"]):
            rows, diag = _cell(
                config,
                {**process, "n_strategies": protocol["figure_3"]["n_strategies"]},
                study=13,
                address=address,
                stage="process",
                deltas=protocol["figure_3"]["delta"],
            )
            tables["figure-3-size.csv"].extend(row for row in rows if row["delta"] == 0)
            tables["figure-3-power.csv"].extend(
                row for row in rows if row["method"] in ("fixed", "resampled", "oracle")
            )
            diagnostics.extend(diag)
            checkpoint()
        tables["holdout.csv"] = []
        for address, scenario in enumerate(protocol["holdout"]):
            rows, diag = _cell(config, scenario, study=14, address=address, stage="holdout")
            tables["holdout.csv"].extend(rows)
            diagnostics.extend(diag)
            checkpoint()
        tables["block-sensitivity.csv"] = []
        for address, scenario in enumerate(protocol["block_sensitivity"]["scenarios"]):
            blocks = [
                min(config["n_obs"], max(1, default_block_length(config["n_obs"]) * multiple))
                for multiple in protocol["block_sensitivity"]["block_multipliers"]
            ]
            rows, diag = _cell(
                config, scenario, study=15, address=address, stage="sensitivity", blocks=blocks
            )
            tables["block-sensitivity.csv"].extend(rows)
            diagnostics.extend(diag)
            checkpoint()
        tables["variance-diagnostics.csv"] = diagnostics
        for name, rows in tables.items():
            _csv(output / name, rows)
        assessed = [
            row
            for name, rows in tables.items()
            if name not in ("figure-3-power.csv", "variance-diagnostics.csv")
            for row in rows
            if row.get("size_upper_simultaneous") != "" and "size_upper_simultaneous" in row
        ]
        if len(assessed) != config["n_assessment_cells"]:
            raise RuntimeError("The assessment cell count differs from the frozen protocol.")
        limit = protocol["assessment"]["maximum_false_positive_rate"]
        metadata["assessment"] = dict(
            status="smoke_only"
            if profile == "quick"
            else "passed"
            if all(row["size_upper_simultaneous"] <= limit for row in assessed)
            else "failed",
            maximum_false_positive_rate=limit,
            n_cells=len(assessed),
            worst_simultaneous_upper=max(row["size_upper_simultaneous"] for row in assessed),
            failed_cells=[
                {
                    key: row[key]
                    for key in (
                        "stage",
                        "process",
                        "n_obs",
                        "n_strategies",
                        "phi",
                        "cross_corr",
                        "rate",
                        "size_upper_simultaneous",
                    )
                }
                for row in assessed
                if row["size_upper_simultaneous"] > limit
            ],
        )
        metadata["status"] = "complete"
        metadata["outputs"] = list(tables)
        metadata["file_sha256"] = {
            name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in tables
        }
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        metadata["elapsed_seconds"] = round(time.perf_counter() - started, 6)
        metadata["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(metadata_path, metadata)
    return metadata
