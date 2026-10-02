"""Run the frozen Monte Carlo protocol and export figures, data and provenance."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata as package_metadata
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import norm

from .bootstrap import default_block_length, stationary_bootstrap_means
from .inference import default_lags, infer_mean
from .plotting import plot_autocorrelation, plot_robustness, plot_selection
from .simulations import simulate_returns

CSV_COLUMNS = (
    "profile", "process", "phi", "cross_corr", "sigma", "n_strategies", "delta",
    "method", "n_obs", "n_mc", "n_bootstrap", "hac_lags", "block_length",
    "reject_count", "rate", "ci_low", "ci_high", "alpha", "confidence", "iid_limit_reference",
)


def _load_protocol() -> tuple[dict[str, Any], bytes, Path]:
    candidates = (
        Path(__file__).resolve().parents[2] / "experiments" / "protocol.json",
        Path(sys.prefix) / "share" / "strategy-inference" / "protocol.json",
    )
    for path in candidates:
        if path.is_file():
            raw = path.read_bytes()
            return json.loads(raw), raw, path
    raise FileNotFoundError("The frozen experiments/protocol.json was not installed")


def _stream_seed(seed: int, figure: int, scenario: int, replicate: int, purpose: int) -> int:
    # Do not use spawn(n_mc): every replicate has an explicit, permanent address.
    sequence = np.random.SeedSequence(seed, spawn_key=(figure, scenario, replicate, purpose))
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def _bootstrap_pvalue(observed: float, bootstrap: np.ndarray) -> float:
    return float((1 + np.count_nonzero(bootstrap >= observed)) / (len(bootstrap) + 1))


def _wilson(count: int, n: int, confidence: float) -> tuple[float, float, float]:
    rate = count / n
    z = float(norm.ppf((1 + confidence) / 2))
    denominator = 1 + z**2 / n
    center = (rate + z**2 / (2 * n)) / denominator
    radius = z * math.sqrt(rate * (1 - rate) / n + z**2 / (4 * n**2)) / denominator
    return rate, max(0.0, center - radius), min(1.0, center + radius)


def _row(
    config: dict[str, Any], count: int, method: str, *, process: str, phi: float,
    n_strategies: int, delta: int | str = "", n_mc: int | None = None,
    n_bootstrap: int | None = None, block_length: int | None = None,
    iid_limit_reference: float | str = "",
) -> dict[str, Any]:
    n_mc = config["n_mc"] if n_mc is None else n_mc
    rate, low, high = _wilson(count, n_mc, config["confidence"])
    return dict(
        profile=config["profile"], process=process, phi=phi,
        cross_corr=config["cross_corr"], sigma=config["sigma"],
        n_strategies=n_strategies, delta=delta, method=method,
        n_obs=config["n_obs"], n_mc=n_mc,
        n_bootstrap=config["n_bootstrap"] if n_bootstrap is None else n_bootstrap,
        hac_lags=config["hac_lags"],
        block_length=config["block_length"] if block_length is None else block_length,
        reject_count=int(count), rate=rate, ci_low=low, ci_high=high,
        alpha=config["alpha"], confidence=config["confidence"],
        iid_limit_reference=iid_limit_reference,
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _progress(label: str, completed: int, total: int, started: float) -> None:
    interval = max(1, total // 10)
    if completed % interval == 0 or completed == total:
        print(f"  {label}: {completed}/{total} replicates ({time.perf_counter() - started:.1f}s)", flush=True)


def _simulate(config: dict[str, Any], seed: int, process: str, phi: float, k: int) -> np.ndarray:
    return simulate_returns(
        config["n_obs"], k, process=process, phi=phi,
        cross_corr=config["cross_corr"], sigma=config["sigma"], mean=0.0,
        seed=seed, burnin=config["burnin"],
    )


def _resample(
    returns: np.ndarray, config: dict[str, Any], seed: int, *,
    block_length: int | None = None, n_bootstrap: int | None = None,
) -> np.ndarray:
    return stationary_bootstrap_means(
        returns, n_resamples=config["n_bootstrap"] if n_bootstrap is None else n_bootstrap,
        block_length=config["block_length"] if block_length is None else block_length,
        seed=seed, batch_size=config["batch_size"], center=True,
    )


def _dependence(config: dict[str, Any], protocol: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    n_mc = config["n_mc"]
    for scenario, phi in enumerate(protocol["figure_1"]["phi"]):
        started = time.perf_counter()
        counts = dict(iid_t=0, hac_z=0, stationary_bootstrap=0)
        print(f"[1/3] Autocorrelation phi={phi:g}", flush=True)
        for replicate in range(n_mc):
            returns = _simulate(config, _stream_seed(config["seed"], 1, scenario, replicate, 0),
                                "gaussian_ar", phi, 1)
            iid = infer_mean(returns, method="iid", confidence=config["confidence"])
            hac = infer_mean(returns, method="hac", lags=config["hac_lags"], confidence=config["confidence"])
            means = _resample(returns, config, _stream_seed(config["seed"], 1, scenario, replicate, 1))
            statistic = means[:, 0] / hac.standard_error[0]
            pvalue = _bootstrap_pvalue(float(hac.statistic[0]), statistic)
            counts["iid_t"] += int(iid.pvalue[0] <= config["alpha"])
            counts["hac_z"] += int(hac.pvalue[0] <= config["alpha"])
            counts["stationary_bootstrap"] += int(pvalue <= config["alpha"])
            _progress(f"phi={phi:g}", replicate + 1, n_mc, started)
        theoretical = float(norm.sf(norm.ppf(1 - config["alpha"]) / math.sqrt((1 + phi) / (1 - phi))))
        for method, count in counts.items():
            rows.append(_row(config, count, method, process="gaussian_ar", phi=phi,
                             n_strategies=1, iid_limit_reference=theoretical))
    return rows


def _selection(config: dict[str, Any], protocol: dict[str, Any]) -> list[dict[str, Any]]:
    counts_k = protocol["figure_2"]["strategy_counts"]
    phi = protocol["figure_2"]["phi"]
    k_max = max(counts_k)
    methods = ("iid_winner", "hac_winner", "bootstrap_max")
    counts = {k: dict.fromkeys(methods, 0) for k in counts_k}
    started = time.perf_counter()
    print(f"[2/3] Nested strategy search; one joint K={k_max} bootstrap per replicate", flush=True)
    for replicate in range(config["n_mc"]):
        returns = _simulate(config, _stream_seed(config["seed"], 2, 0, replicate, 0),
                            "gaussian_ar", phi, k_max)
        iid = infer_mean(returns, method="iid", confidence=config["confidence"])
        hac = infer_mean(returns, method="hac", lags=config["hac_lags"], confidence=config["confidence"])
        means = _resample(returns, config, _stream_seed(config["seed"], 2, 0, replicate, 1))
        # One common-row bootstrap, reused for every nested prefix.
        cumulative_max = np.maximum.accumulate(means / hac.standard_error, axis=1)
        for k in counts_k:
            winner = int(np.argmax(hac.statistic[:k]))
            pvalue = _bootstrap_pvalue(float(hac.statistic[winner]), cumulative_max[:, k - 1])
            counts[k]["iid_winner"] += int(iid.pvalue[winner] <= config["alpha"])
            counts[k]["hac_winner"] += int(hac.pvalue[winner] <= config["alpha"])
            counts[k]["bootstrap_max"] += int(pvalue <= config["alpha"])
        _progress("nested search", replicate + 1, config["n_mc"], started)
    return [_row(config, counts[k][method], method, process="gaussian_ar", phi=phi, n_strategies=k)
            for k in counts_k for method in methods]


def _robustness(
    config: dict[str, Any], protocol: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    size_rows, power_rows = [], []
    experiment = protocol["figure_3"]
    k = experiment["n_strategies"]
    deltas = experiment["local_signal_delta"]
    for scenario, process in enumerate(experiment["processes"]):
        phi = experiment["garch_phi"] if process == "garch" else experiment["ar_phi"]
        lrv_std = config["sigma"] * math.sqrt((1 + phi) / (1 - phi))
        counts = dict(iid_winner=0, hac_winner=0, bootstrap_max=0)
        power_counts = dict.fromkeys(deltas, 0)
        started = time.perf_counter()
        print(f"[3/3] Null size and reused local-power paths: {process}", flush=True)
        for replicate in range(config["n_mc"]):
            returns = _simulate(config, _stream_seed(config["seed"], 3, scenario, replicate, 0),
                                process, phi, k)
            iid = infer_mean(returns, method="iid", confidence=config["confidence"])
            hac = infer_mean(returns, method="hac", lags=config["hac_lags"], confidence=config["confidence"])
            means = _resample(returns, config, _stream_seed(config["seed"], 3, scenario, replicate, 1))
            bootstrap_max = np.max(means / hac.standard_error, axis=1)
            winner = int(np.argmax(hac.statistic))
            null_pvalue = _bootstrap_pvalue(float(hac.statistic[winner]), bootstrap_max)
            counts["iid_winner"] += int(iid.pvalue[winner] <= config["alpha"])
            counts["hac_winner"] += int(hac.pvalue[winner] <= config["alpha"])
            counts["bootstrap_max"] += int(null_pvalue <= config["alpha"])
            # Translation leaves centered noise, HAC scales and the bootstrap
            # distribution invariant. Only the first observed mean is shifted.
            for delta in deltas:
                shifted = hac.statistic.copy()
                shifted[0] += delta * lrv_std / math.sqrt(config["n_obs"]) / hac.standard_error[0]
                pvalue = _bootstrap_pvalue(float(np.max(shifted)), bootstrap_max)
                power_counts[delta] += int(pvalue <= config["alpha"])
            _progress(process, replicate + 1, config["n_mc"], started)
        for method, count in counts.items():
            size_rows.append(_row(config, count, method, process=process, phi=phi, n_strategies=k, delta=0))
        for delta, count in power_counts.items():
            power_rows.append(_row(config, count, "bootstrap_max", process=process, phi=phi,
                                   n_strategies=k, delta=delta))
    return size_rows, power_rows


def _block_sensitivity(config: dict[str, Any], protocol: dict[str, Any]) -> list[dict[str, Any]]:
    experiment = protocol["block_sensitivity"]
    settings = experiment["profiles"][config["profile"]]
    n_mc, n_bootstrap = settings["monte_carlo_replicates"], settings["bootstrap_resamples"]
    lengths = experiment["expected_block_lengths"]
    counts = dict.fromkeys(lengths, 0)
    started = time.perf_counter()
    print(f"[sensitivity] Fixed block lengths {lengths}; CSV only, no tuning", flush=True)
    for replicate in range(n_mc):
        returns = _simulate(config, _stream_seed(config["seed"], 4, 0, replicate, 0),
                            experiment["process"], experiment["phi"], experiment["n_strategies"])
        hac = infer_mean(returns, method="hac", lags=config["hac_lags"], confidence=config["confidence"])
        observed = float(np.max(hac.statistic))
        for scenario, length in enumerate(lengths):
            means = _resample(returns, config, _stream_seed(config["seed"], 4, 0, replicate, scenario + 1),
                              block_length=length, n_bootstrap=n_bootstrap)
            statistic = np.max(means / hac.standard_error, axis=1)
            counts[length] += int(_bootstrap_pvalue(observed, statistic) <= config["alpha"])
        _progress("block sensitivity", replicate + 1, n_mc, started)
    return [_row(config, count, "bootstrap_max", process=experiment["process"], phi=experiment["phi"],
                 n_strategies=experiment["n_strategies"], n_mc=n_mc, n_bootstrap=n_bootstrap,
                 block_length=length) for length, count in counts.items()]


def _versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for name in ("numpy", "scipy", "matplotlib", "strategy-inference"):
        try:
            versions[name] = package_metadata.version(name)
        except package_metadata.PackageNotFoundError:
            versions[name] = "source-checkout"
    return versions


def run_experiments(
    output: str | Path, profile: str = "full", seed: int = 20261002,
) -> dict[str, Any]:
    """Export exactly three main figures and complete provenance for one profile.

    All tuning constants and logical random-stream addresses are declared before
    simulation. Full is the scientific run; quick is only a development smoke
    profile. Neither numerical size errors nor inconvenient results are removed.
    """
    protocol, raw_protocol, protocol_path = _load_protocol()
    if not isinstance(profile, str) or profile not in protocol["profiles"]:
        raise ValueError("profile must be 'quick' or 'full'")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    settings = protocol["profiles"][profile]
    n_obs = settings["n_obs"]
    config = dict(profile=profile, seed=seed, n_obs=n_obs,
                  n_mc=settings["monte_carlo_replicates"], n_bootstrap=settings["bootstrap_resamples"],
                  hac_lags=default_lags(n_obs), block_length=default_block_length(n_obs),
                  batch_size=protocol["bootstrap"]["batch_size"],
                  alpha=protocol["alpha"], confidence=protocol["confidence"],
                  sigma=protocol["sigma"], cross_corr=protocol["cross_corr"], burnin=protocol["burnin"])
    metadata = dict(
        schema_version=1, status="running", started_at_utc=datetime.now(timezone.utc).isoformat(),
        seed=seed, profile=profile, resolved_protocol=config, protocol=protocol,
        protocol_sha256=hashlib.sha256(raw_protocol).hexdigest(), protocol_source=str(protocol_path),
        software_versions=_versions(), platform=platform.platform(),
        outputs=[], stage_elapsed_seconds={}, elapsed_seconds=0.0,
        interpretation="One-sided mean tests and a fixed-scale stationary-bootstrap max test; not SPA. "
                       "Wilson intervals measure pointwise Monte Carlo uncertainty. "
                       "Finite-sample size distortions are reported, not tuned away.",
    )
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=protocol_path.parent,
                                capture_output=True, text=True, check=False)
        metadata["git_revision"] = result.stdout.strip() if result.returncode == 0 else None
    except OSError:
        metadata["git_revision"] = None
    metadata_path = output / "run-metadata.json"
    _write_json(metadata_path, metadata)
    started = time.perf_counter()
    paths: list[Path] = []
    figure_settings = dict(n_obs=n_obs, n_mc=config["n_mc"], n_bootstrap=config["n_bootstrap"])

    def record_stage(name: str, stage_start: float) -> None:
        metadata["stage_elapsed_seconds"][name] = round(time.perf_counter() - stage_start, 6)
        metadata["elapsed_seconds"] = round(time.perf_counter() - started, 6)
        metadata["outputs"] = [str(path.relative_to(output)) for path in paths]
        _write_json(metadata_path, metadata)

    print(f"Frozen {profile} protocol: T={n_obs}, MC={config['n_mc']}, B={config['n_bootstrap']}, "
          f"HAC lags={config['hac_lags']}, mean block length={config['block_length']}, seed={seed}", flush=True)
    try:
        stage_started = time.perf_counter()
        dependence = _dependence(config, protocol)
        path = output / "figure-1-dependence.csv"
        _write_csv(path, dependence)
        paths.append(path)
        paths.extend(plot_autocorrelation(dependence, output, **figure_settings))
        record_stage("figure_1", stage_started)

        stage_started = time.perf_counter()
        selection = _selection(config, protocol)
        path = output / "figure-2-selection.csv"
        _write_csv(path, selection)
        paths.append(path)
        paths.extend(plot_selection(selection, output, **figure_settings))
        record_stage("figure_2", stage_started)

        stage_started = time.perf_counter()
        sizes, powers = _robustness(config, protocol)
        for name, rows in (("figure-3-size.csv", sizes), ("figure-3-power.csv", powers)):
            path = output / name
            _write_csv(path, rows)
            paths.append(path)
        paths.extend(plot_robustness(sizes, powers, output, **figure_settings))
        record_stage("figure_3", stage_started)

        stage_started = time.perf_counter()
        sensitivity = _block_sensitivity(config, protocol)
        path = output / "block-length-sensitivity.csv"
        _write_csv(path, sensitivity)
        paths.append(path)
        record_stage("block_sensitivity", stage_started)

        metadata["status"] = "complete"
        metadata["elapsed_seconds"] = round(time.perf_counter() - started, 6)
        metadata["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        metadata["file_sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        _write_json(metadata_path, metadata)
        print(f"Complete: {len(paths)} artifacts plus run-metadata.json in {output} "
              f"({metadata['elapsed_seconds']:.1f}s)", flush=True)
        return metadata
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        metadata["elapsed_seconds"] = round(time.perf_counter() - started, 6)
        metadata["outputs"] = [str(path.relative_to(output)) for path in paths]
        _write_json(metadata_path, metadata)
        raise
