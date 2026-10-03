"""Prespecified paired experiment for a certified unknown-AR mean test."""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
from parametric_replay import _filter, _reference, _sha
from scipy.stats import binomtest, t

from strategy_inference.experiments import _versions, _wilson
from strategy_inference.parametric import fit_equicorrelated_ar1, known_phi_gls_t
from strategy_inference.uncertainty import uncertainty_test

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("gls_known", "gls_known_budget", "gls_fitted", "uncertainty")
FIELDS = (
    "phase", "seed", "group", "replicate", "delta", "method", "reject", "false_reject",
    "signal_reject", "rejection_count", "decision_mask", "max_statistic", "phi_fit",
    "ci_low", "ci_high", "ci_length", "ci_count", "ci_intervals", "ci_contains_truth", "phi1_retained",
    "empty_ci", "ci_unresolved", "certificate_unresolved", "certificate_nodes",
    "critical_squared", "low_df", "high_df",
)


def _json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _dataset(group: dict, seed: int, phase: int, replicate: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(np.random.SeedSequence([seed, 43, phase, group["id"], replicate, 0]))
    size, k, rho = group["n_obs"], group["k"], group["rho"]
    components = _filter(rng.standard_normal((1, size, k + 1)), group["phi_factor"], group["phi_idio"])[0]
    signs = np.where(np.arange(k) % 2, -1, 1) if group["signed_scaled"] else np.ones(k)
    scales = np.linspace(0.5, 1.5, k) if group["signed_scaled"] else np.ones(k)
    data = scales * (math.sqrt(rho) * components[:, :1] * signs + math.sqrt(1 - rho) * components[:, 1:])
    mean_variance, _ = _reference(group)
    return data, scales * math.sqrt(mean_variance / size)


def _records(group: dict, seed: int, phase: int, replicate: int, deltas: list[float], protocol: dict):
    base, mean_sd = _dataset(group, seed, phase, replicate)
    k, size = group["k"], group["n_obs"]
    options = {key: protocol["procedure"][key] for key in (
        "reference", "ci_depth", "certificate_depth", "max_nodes", "critical_bits",
    )}
    records, snapshots = [], []
    for delta in deltas:
        data = base.copy()
        signals = np.zeros(k, dtype=bool)
        if phase == 2:
            signals[0] = True
        elif phase == 3:
            signals[::2] = True
        data[:, signals] += delta * mean_sd[signals]
        result = uncertainty_test(data, alpha=protocol["alpha"], beta=protocol["beta"], **options)
        fitted = fit_equicorrelated_ar1(data).phi
        plugin = known_phi_gls_t(data, fitted)
        decisions = {
            "gls_fitted": k * t.sf(plugin, size - 1) <= protocol["alpha"],
            "uncertainty": result.decisions,
        }
        statistics = {"gls_fitted": float(plugin.max()), "uncertainty": ""}
        if group["in_scope"]:
            oracle = known_phi_gls_t(data, group["phi_factor"])
            decisions["gls_known"] = k * t.sf(oracle, size - 1) <= protocol["alpha"]
            decisions["gls_known_budget"] = oracle > np.nextafter(math.sqrt(float(result.critical_squared)), math.inf)
            statistics.update(gls_known=float(oracle.max()), gls_known_budget=float(oracle.max()))
        intervals = result.intervals
        common = {
            "phase": phase, "seed": seed, "group": group["id"], "replicate": replicate, "delta": delta,
            "phi_fit": fitted, "ci_low": float(intervals[0][0]) if intervals else "",
            "ci_high": float(intervals[-1][1]) if intervals else "",
            "ci_length": float(sum((high - low for low, high in intervals), start=0)),
            "ci_count": len(intervals),
            "ci_intervals": ";".join(f"{low},{high}" for low, high in intervals),
            "ci_contains_truth": int(any(low <= group["phi_factor"] <= high for low, high in intervals)),
            "phi1_retained": int(result.phi1_retained), "empty_ci": int(result.empty_ci),
            "ci_unresolved": result.ci_unresolved_cells,
            "certificate_unresolved": int(np.count_nonzero(result.certificate_unresolved)),
            "certificate_nodes": int(result.certificate_nodes.sum()),
            "critical_squared": float(result.critical_squared), "low_df": result.low_df, "high_df": result.high_df,
        }
        for method in METHODS:
            if method not in decisions:
                continue
            mask = np.asarray(decisions[method], dtype=bool)
            records.append(common | {
                "method": method, "reject": int(mask.any()), "false_reject": int(mask[~signals].any()),
                "signal_reject": int(mask[0]) if phase == 2 else int(mask[signals].any()) if phase == 3 else 0,
                "rejection_count": int(mask.sum()), "decision_mask": np.packbits(mask).tobytes().hex(),
                "max_statistic": statistics[method],
            })
        snapshots.append({
            "delta": delta, "intervals": [[str(low), str(high)] for low, high in intervals],
            "critical_squared": str(result.critical_squared), "f_cutoffs": [str(value) for value in result.f_cutoffs],
            "decisions": result.decisions.astype(int).tolist(),
            "certificate_unresolved": result.certificate_unresolved.astype(int).tolist(),
            "certificate_nodes": result.certificate_nodes.tolist(),
        })
    return records, snapshots


def _summaries(rows: list[dict], group: int, phase: int, protocol: dict):
    summaries, paired = [], []
    for delta in sorted({row["delta"] for row in rows}):
        selected = {name: [row for row in rows if row["method"] == name and row["delta"] == delta]
                    for name in METHODS}
        selected = {name: values for name, values in selected.items() if values}
        for method, values in selected.items():
            for mode in ("reject", "false_reject", "signal_reject"):
                count, total = sum(row[mode] for row in values), len(values)
                rate, low, high = _wilson(count, total, 0.95)
                summaries.append({"phase": phase, "group": group, "delta": delta, "method": method,
                                  "mode": mode, "n": total, "count": count, "rate": rate, "low": low, "high": high})
        reference = selected["uncertainty"]
        for method in protocol["comparisons"]:
            if method not in selected:
                continue
            for mode in ("reject", "signal_reject"):
                difference = np.array([left[mode] - right[mode] for left, right in
                                       zip(selected[method], reference, strict=True)])
                plus, minus = int((difference == 1).sum()), int((difference == -1).sum())
                risk = float(difference.mean())
                se = float(difference.std(ddof=1) / math.sqrt(len(difference))) if len(difference) > 1 else 0.0
                paired.append({"phase": phase, "group": group, "delta": delta, "method": method,
                               "reference": "uncertainty", "mode": mode, "n": len(difference),
                               "method_only": plus, "reference_only": minus, "risk_difference": risk,
                               "low": risk - 1.95996398454 * se, "high": risk + 1.95996398454 * se,
                               "mcnemar_p": float(binomtest(plus, plus + minus, 0.5).pvalue) if plus + minus else 1.0})
        values = selected["uncertainty"]
        for mode in ("ci_contains_truth", "phi1_retained", "empty_ci"):
            count, total = sum(row[mode] for row in values), len(values)
            rate, low, high = _wilson(count, total, 0.95)
            summaries.append({"phase": phase, "group": group, "delta": delta, "method": "confidence_set",
                              "mode": mode, "n": total, "count": count, "rate": rate, "low": low, "high": high})
    return summaries, paired


def run(profile: str, output: Path) -> dict:
    path = ROOT / "experiments/parameter-uncertainty-protocol.json"
    protocol = json.loads(path.read_text())
    settings = protocol["profiles"][profile]
    if output.exists() and any(output.iterdir()):
        raise ValueError("Existing evidence is never overwritten; choose an empty directory.")
    output.mkdir(parents=True, exist_ok=True)
    paths = [path, Path(__file__).resolve(), ROOT / "scripts/parametric_replay.py", *[
        ROOT / "src/strategy_inference" / name for name in (
            "uncertainty.py", "parametric.py", "reference.py", "_validation.py", "experiments.py", "inference.py",
        )
    ]]
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    metadata = {"status": "running", "profile": profile, "settings": settings, "protocol": protocol,
                "git_revision": revision, "environment": _versions(), "source_hashes": {
                    str(value.relative_to(ROOT)): _sha(value) for value in paths}, "cells": []}
    _json(output / "metadata.json", metadata)
    start = time.perf_counter()
    summaries, paired, snapshots, total_rows = [], [], {}, 0
    for phase, label in ((1, "null"), (2, "power"), (3, "partial")):
        total, seed = settings[f"{label}_replicates"], settings[f"{label}_seed"]
        for group in protocol["groups"]:
            identifier = group["id"]
            if phase == 2 and identifier not in protocol["power"]["groups"]:
                continue
            if phase == 3 and identifier not in protocol["partial_null"]["groups"]:
                continue
            deltas = protocol["power"]["standardized_mean_shifts"] if phase == 2 else (
                [protocol["partial_null"]["standardized_mean_shift"]] if phase == 3 else [0.0])
            key = f"p{phase}-g{identifier:02d}"
            print(f"{key}: T={group['n_obs']}, K={group['k']}, R={total}", flush=True)
            cell = []
            with (output / f"{key}.csv.gz").open("wb") as raw:
                with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
                    with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as stream:
                        writer = csv.DictWriter(stream, fieldnames=FIELDS)
                        writer.writeheader()
                        for replicate in range(total):
                            records, certificates = _records(group, seed, phase, replicate, deltas, protocol)
                            writer.writerows(records)
                            cell.extend(records)
                            if replicate in {0, total // 2, total - 1}:
                                snapshots[f"{key}-r{replicate:05d}"] = certificates
                            if (replicate + 1) % 200 == 0:
                                print(f"  {replicate + 1}/{total}; {time.perf_counter() - start:.1f}s", flush=True)
            summary, differences = _summaries(cell, identifier, phase, protocol)
            summaries.extend(summary)
            paired.extend(differences)
            total_rows += len(cell)
            metadata["cells"].append({"key": key, "group": identifier, "phase": phase,
                                      "n": total, "records": len(cell), "in_scope": group["in_scope"]})
            _json(output / "metadata.json", metadata)
    _csv(output / "summary.csv", summaries)
    _csv(output / "paired.csv", paired)
    _json(output / "certificates.json", snapshots)
    for name, digest in metadata["source_hashes"].items():
        if _sha(ROOT / name) != digest:
            raise ValueError(f"Source changed during computation: {name}; run remains incomplete.")
    metadata.update(status="complete", elapsed_seconds=time.perf_counter() - start, records=total_rows)
    metadata["output_hashes"] = {value.name: _sha(value) for value in sorted(output.iterdir()) if value.name != "metadata.json"}
    _json(output / "metadata.json", metadata)
    print(f"Complete: {total_rows} method records in {metadata['elapsed_seconds']:.1f}s", flush=True)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("quick", "full"), default="quick")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.profile, args.output)
