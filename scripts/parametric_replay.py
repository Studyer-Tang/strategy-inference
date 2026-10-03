"""Frozen, paired Gaussian replay experiment; no production API is calibrated here."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy.signal import lfilter
from scipy.stats import beta, binomtest, t

from strategy_inference.experiments import _versions, _wilson
from strategy_inference.inference import default_lags, long_run_variance
from strategy_inference.parametric import (
    fit_equicorrelated_ar1,
    known_phi_gls_t,
    replay_pvalue,
    tail_statistics,
)
from strategy_inference.reference import (
    equicorrelated_max_quantile,
    equicorrelated_max_tail,
    gaussian_ar_mean_variance,
)

ROOT = Path(__file__).resolve().parents[1]
REPLAYS = ("known", "phi_fitted", "rho_fitted", "fitted", "frozen")
FIELDS = (
    "phase", "seed", "group", "replicate", "delta", "method", "statistic", "winner",
    "pvalue", "exceedances", "score", "reject", "matched_reject", "phi_fit", "rho_raw",
    "rho_fit", "rho_projected", "selected_phi", "selected_variance_ratio",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _rng(seed: int, phase: int, group: int, replicate: int, purpose: int):
    return np.random.default_rng(np.random.SeedSequence([seed, 31, phase, group, replicate, purpose]))


def _filter(normal: np.ndarray, phi_factor: float, phi_idio: float) -> np.ndarray:
    """Stationary initial value, with marginal variance one for each component."""
    innovations = normal.copy()
    if phi_factor == phi_idio:
        innovations[:, 1:, :] *= math.sqrt((1 - phi_factor) * (1 + phi_factor))
        return lfilter([1], [1, -phi_factor], innovations, axis=1)
    innovations[:, 1:, :1] *= math.sqrt((1 - phi_factor) * (1 + phi_factor))
    innovations[:, 1:, 1:] *= math.sqrt((1 - phi_idio) * (1 + phi_idio))
    result = np.empty_like(innovations)
    result[:, :, :1] = lfilter([1], [1, -phi_factor], innovations[:, :, :1], axis=1)
    result[:, :, 1:] = lfilter([1], [1, -phi_idio], innovations[:, :, 1:], axis=1)
    return result


def _mix(components: np.ndarray, rho: float) -> np.ndarray:
    return math.sqrt(rho) * components[:, :, :1] + math.sqrt(1 - rho) * components[:, :, 1:]


def _reference(group: dict) -> tuple[float, float]:
    """Exact variance of sqrt(T)*mean and its cross-candidate correlation."""
    size, rho = group["n_obs"], group["rho"]
    factor = size * gaussian_ar_mean_variance(size, group["phi_factor"])
    idio = size * gaussian_ar_mean_variance(size, group["phi_idio"])
    variance = rho * factor + (1 - rho) * idio
    return variance, rho * factor / variance


def _inner(group, fit, factors, rng, draws: int, batch_size: int) -> dict[str, np.ndarray]:
    size, k = group["n_obs"], group["k"]
    lags = default_lags(size)
    maxima = {name: np.empty(draws) for name in REPLAYS}
    for start in range(0, draws, batch_size):
        count = min(batch_size, draws - start)
        normal = rng.standard_normal((count, size, k + 1))
        true = _filter(normal, group["phi_factor"], group["phi_idio"])
        fitted = _filter(normal, fit.phi, fit.phi)
        known, _, _ = tail_statistics(_mix(true, group["rho"]), lags)
        phi, _, _ = tail_statistics(_mix(fitted, group["rho"]), lags)
        if fit.rho == group["rho"]:
            rho, both = known, phi
            # Frozen correction still uses this draw's centered HAC.
            _, new_factors, _ = tail_statistics(_mix(fitted, fit.rho), lags)
        else:
            rho, _, _ = tail_statistics(_mix(true, fit.rho), lags)
            both, new_factors, _ = tail_statistics(_mix(fitted, fit.rho), lags)
        frozen = both * np.sqrt(new_factors / factors)
        for name, values in zip(REPLAYS, (known, phi, rho, both, frozen), strict=True):
            maxima[name][start:start + count] = values.max(axis=1)
    return maxima


def _records(group, seed, phase, replicate, draws, batch_size, alpha, deltas):
    size, k, identifier = group["n_obs"], group["k"], group["id"]
    normal = _rng(seed, phase, identifier, replicate, 0).standard_normal((1, size, k + 1))
    data = _mix(_filter(normal, group["phi_factor"], group["phi_idio"]), group["rho"])[0]
    fit = fit_equicorrelated_ar1(data)
    lags = default_lags(size)
    base, factor, phis = tail_statistics(data[None], lags)
    variance = long_run_variance(data, lags) * factor[0]
    target, mean_rho = _reference(group)
    known_critical = equicorrelated_max_quantile(1 - alpha, k, mean_rho)
    fitted_critical = equicorrelated_max_quantile(1 - alpha, k, fit.rho)
    maxima = _inner(
        group, fit, factor[0], _rng(seed, phase, identifier, replicate, 1), draws, batch_size
    )
    jitter = float(_rng(seed, phase, identifier, replicate, 2).random())
    gls = group["phi_factor"] == group["phi_idio"]
    records = []
    for delta in deltas:
        statistics = base[0].copy()
        statistics[0] += delta * math.sqrt(target / variance[0])
        winner = int(statistics.argmax())
        observed = float(statistics[winner])
        oracle = math.sqrt(size / target) * data.mean(axis=0)
        oracle[0] += delta
        oracle_winner = int(oracle.argmax())
        common = {
            "phase": phase, "seed": seed, "group": identifier, "replicate": replicate,
            "delta": delta, "phi_fit": fit.phi, "rho_raw": fit.raw_rho, "rho_fit": fit.rho,
            "rho_projected": int(fit.projected), "winner": winner, "statistic": observed,
            "selected_phi": float(phis[0, winner]),
            "selected_variance_ratio": float(variance[winner] / target),
            "matched_reject": "", "exceedances": "",
        }
        for method, statistic, winning, critical, rho in (
            ("oracle_gaussian", float(oracle[oracle_winner]), oracle_winner, known_critical, mean_rho),
            ("gaussian_known", observed, winner, known_critical, mean_rho),
            ("gaussian_fitted", observed, winner, fitted_critical, fit.rho),
        ):
            pvalue = equicorrelated_max_tail(statistic, k, rho)
            row = common | {
                "method": method, "statistic": statistic, "winner": winning,
                "pvalue": pvalue, "score": statistic / critical, "reject": int(pvalue <= alpha),
            }
            if method == "oracle_gaussian":
                row.update(selected_phi="", selected_variance_ratio=1.0)
            records.append(row)
        for name in REPLAYS:
            exceedances = int(np.count_nonzero(maxima[name] >= observed))
            pvalue = replay_pvalue(observed, maxima[name])
            records.append(common | {
                "method": f"replay_{name}", "pvalue": pvalue, "exceedances": exceedances,
                "score": 1 - pvalue + jitter / (draws + 1), "reject": int(pvalue <= alpha),
            })
        if gls:
            alternative = data.copy()
            alternative[:, 0] += delta * math.sqrt(target / size)
            values = known_phi_gls_t(alternative, group["phi_factor"])
            winning = int(values.argmax())
            statistic = float(values[winning])
            pvalue = min(1.0, float(k * t.sf(statistic, size - 1)))
            records.append(common | {
                "method": "gls_known", "statistic": statistic, "winner": winning,
                "pvalue": pvalue, "score": statistic / t.isf(alpha / k, size - 1),
                "reject": int(pvalue <= alpha), "selected_phi": "", "selected_variance_ratio": "",
            })
    return records, maxima


def _calibration(scores: list[float], alpha: float) -> dict:
    size = len(scores)
    rank = math.ceil((1 - alpha) * (size + 1))
    if rank > size:
        raise ValueError("Too few calibration observations for the prescribed order statistic.")
    tail = size + 1 - rank
    return {
        "n": size, "rank": rank, "cutoff": float(np.sort(scores)[rank - 1]),
        "unconditional_size": tail / (size + 1),
        "conditional_size_interval": beta.ppf([0.025, 0.975], tail, rank).tolist(),
    }


def _csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summaries(records: list[dict], phase: int, group: int, alpha: float) -> tuple[list, list]:
    summaries, paired = [], []
    deltas = sorted({row["delta"] for row in records})
    methods = list(dict.fromkeys(row["method"] for row in records))
    for delta in deltas:
        selected = {method: [row for row in records if row["method"] == method and row["delta"] == delta]
                    for method in methods}
        for method, rows in selected.items():
            for mode in ("reject", "matched_reject"):
                if rows[0][mode] == "":
                    continue
                decisions = np.array([row[mode] for row in rows], dtype=bool)
                correct = decisions & np.array([row["winner"] == 0 for row in rows])
                successes, n = int(decisions.sum()), len(rows)
                _, low, high = _wilson(successes, n, 0.95)
                assessed = phase == 1 and group <= 6 and method == "replay_fitted" and mode == "reject"
                upper = (float(beta.ppf(1 - 0.05 / 6, successes + 1, n - successes))
                         if successes < n else 1.0) if assessed else ""
                summaries.append({
                    "phase": phase, "group": group, "delta": delta, "method": method, "mode": mode,
                    "n": n, "rejections": successes, "rate": successes / n, "low": low, "high": high,
                    "reject_select_signal": int(correct.sum()), "signal_rate": float(correct.mean()),
                    "family_upper": upper,
                    "passes_tolerance": int(upper <= 0.07) if assessed else "",
                })
        reference = selected["replay_known"]
        for method, rows in selected.items():
            if method == "replay_known":
                continue
            for mode in ("reject", "matched_reject"):
                if rows[0][mode] == "":
                    continue
                left = np.array([row[mode] for row in rows], dtype=int)
                right = np.array([row[mode] for row in reference], dtype=int)
                difference = left - right
                plus, minus = int((difference == 1).sum()), int((difference == -1).sum())
                risk = float(difference.mean())
                se = float(difference.std(ddof=1) / math.sqrt(len(rows))) if len(rows) > 1 else 0.0
                paired.append({
                    "phase": phase, "group": group, "delta": delta, "method": method, "mode": mode,
                    "reference": "replay_known", "n": len(rows), "method_only": plus,
                    "reference_only": minus, "risk_difference": risk,
                    "low": risk - 1.95996398454 * se, "high": risk + 1.95996398454 * se,
                    "mcnemar_p": float(binomtest(plus, plus + minus, 0.5).pvalue) if plus + minus else 1.0,
                })
    return summaries, paired


def run(profile: str, output: Path, *, batch_size: int = 32) -> dict:
    protocol_path = ROOT / "experiments/parametric-replay-protocol.json"
    protocol = json.loads(protocol_path.read_text())
    settings, alpha = protocol["profiles"][profile], protocol["alpha"]
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Existing evidence is never overwritten; choose an empty output directory.")
    output.mkdir(parents=True, exist_ok=True)
    paths = [Path(__file__).resolve(), protocol_path, *[
        ROOT / "src/strategy_inference" / name for name in (
            "parametric.py", "tail.py", "reference.py", "inference.py", "_validation.py", "experiments.py",
        )
    ]]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True
    ).stdout.strip()
    metadata = {
        "status": "running", "profile": profile, "settings": settings, "alpha": alpha,
        "git_revision": revision, "protocol_sha256": _sha(protocol_path), "environment": _versions(),
        "source_hashes": {str(path.relative_to(ROOT)): _sha(path) for path in paths},
        "batch_size": batch_size, "scope": protocol["scope"], "cells": [],
    }
    _json(output / "metadata.json", metadata)
    started = time.perf_counter()
    summaries, paired, calibration, snapshots = [], [], {}, {}
    total_records = 0
    for phase, total_key, seed_key in (
        (0, "calibration_replicates", "calibration_seed"),
        (1, "null_replicates", "null_seed"), (2, "power_replicates", "power_seed"),
    ):
        total, seed = settings[total_key], settings[seed_key]
        for group in protocol["groups"]:
            identifier = group["id"]
            if phase != 1 and identifier not in protocol["power"]["groups"]:
                continue
            deltas = protocol["power"]["standardized_mean_shifts"] if phase == 2 else [0.0]
            key = f"p{phase}-g{identifier:02d}"
            print(f"{key}: T={group['n_obs']}, K={group['k']}, R={total}, B={settings['inner_draws']}", flush=True)
            cell_records, scores = [], {}
            with (output / f"{key}.csv.gz").open("wb") as raw:
                with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
                    with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as stream:
                        writer = csv.DictWriter(stream, fieldnames=FIELDS)
                        writer.writeheader()
                        for replicate in range(total):
                            records, maxima = _records(
                                group, seed, phase, replicate, settings["inner_draws"], batch_size, alpha, deltas
                            )
                            if replicate in {0, total // 2, total - 1}:
                                snapshots[f"{key}-r{replicate:05d}"] = {
                                    name: values.tolist() for name, values in maxima.items()
                                }
                            for row in records:
                                method = row["method"]
                                if phase == 0:
                                    scores.setdefault(method, []).append(row["score"])
                                elif identifier in protocol["power"]["groups"]:
                                    row["matched_reject"] = int(row["score"] > calibration[str(identifier)][method]["cutoff"])
                            writer.writerows(records)
                            cell_records.extend(records)
                            if (replicate + 1) % 200 == 0:
                                print(f"  {replicate + 1}/{total}; elapsed {time.perf_counter()-started:.1f}s", flush=True)
            if phase == 0:
                calibration[str(identifier)] = {name: _calibration(values, alpha) for name, values in scores.items()}
                _json(output / "calibration.json", calibration)
            else:
                summary, differences = _summaries(cell_records, phase, identifier, alpha)
                summaries.extend(summary)
                paired.extend(differences)
            total_records += len(cell_records)
            target, mean_rho = _reference(group)
            metadata["cells"].append({
                "key": key, "phase": phase, "group": identifier, "n": total, "records": len(cell_records),
                "target": target, "correlation_of_means": mean_rho,
                "projected_fraction": float(np.mean([
                    row["rho_projected"] for row in cell_records if row["method"] == "oracle_gaussian"
                ])),
            })
            _json(output / "metadata.json", metadata)
    _csv(output / "summary.csv", summaries)
    _csv(output / "paired.csv", paired)
    _json(output / "inner-snapshots.json", snapshots)
    for name, digest in metadata["source_hashes"].items():
        if _sha(ROOT / name) != digest:
            raise ValueError(f"Source changed during computation: {name}; run remains incomplete.")
    metadata.update(status="complete", elapsed_seconds=time.perf_counter() - started, records=total_records)
    metadata["output_hashes"] = {
        path.name: _sha(path) for path in sorted(output.iterdir()) if path.name != "metadata.json"
    }
    _json(output / "metadata.json", metadata)
    print(f"Complete: {total_records} method records in {metadata['elapsed_seconds']:.1f}s", flush=True)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("quick", "full"), default="quick")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    run(args.profile, args.output, batch_size=args.batch_size)
