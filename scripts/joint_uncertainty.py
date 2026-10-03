"""Frozen paired comparison of temporal and joint parameter information."""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

import numpy as np
from parameter_uncertainty import _csv, _dataset, _json
from parametric_replay import _sha
from scipy.stats import binomtest, t

from strategy_inference.experiments import _versions, _wilson
from strategy_inference.parametric import fit_equicorrelated_ar1, known_phi_gls_t
from strategy_inference.uncertainty import uncertainty_test
from strategy_inference.wilks import wilks_uncertainty_test

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("gls_known", "gls_known_budget", "gls_fitted", "uncertainty", "wilks_scalar", "wilks_joint")
CI_METHODS = ("uncertainty", "wilks_scalar", "wilks_joint")
DECISIONS = ("reject", "false_reject", "signal_reject")
DIAGNOSTICS = ("ci_contains_truth", "phi1_retained", "empty_ci", "singular_fallback")
FIELDS = (
    "phase", "seed", "group", "replicate", "delta", "method", *DECISIONS,
    "rejection_count", "decision_mask", "max_statistic", "phi_fit", "critical_squared",
    "dimension", "ci_intervals", "ci_length", *DIAGNOSTICS, "ci_unresolved",
    "certificate_unresolved", "certificate_nodes",
)
SOURCES = (
    "experiments/joint-uncertainty-protocol.json", "scripts/joint_uncertainty.py",
    "scripts/parameter_uncertainty.py", "scripts/parametric_replay.py",
    *(f"src/strategy_inference/{name}" for name in (
        "wilks.py", "uncertainty.py", "parametric.py", "reference.py", "_validation.py", "experiments.py", "inference.py")),
)


def _serial(value):
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, dict):
        return {key: _serial(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_serial(item) for item in value]
    return value


def _certificate(result):
    value = {"intervals": [[str(low), str(high)] for low, high in result.intervals],
             "critical_squared": str(result.critical_squared),
             "decisions": result.decisions.astype(int).tolist(),
             "certificate_unresolved": result.certificate_unresolved.astype(int).tolist(),
             "certificate_nodes": result.certificate_nodes.tolist()}
    if hasattr(result, "scales"):
        value["scales"] = [_serial(asdict(scale)) for scale in result.scales]
    else:
        value["f_cutoffs"] = [str(item) for item in result.f_cutoffs]
    return value


def _records(group, seed, phase, replicate, deltas, protocol):
    base, mean_sd = _dataset(group, seed, phase, replicate)
    size, k = base.shape
    options = protocol["procedure"]
    old_options = {key: options[key] for key in ("ci_depth", "certificate_depth", "max_nodes", "critical_bits")}
    rows, snapshots = [], []
    for delta in deltas:
        data = base.copy()
        signals = np.zeros(k, dtype=bool)
        if phase == 2:
            signals[0] = True
        elif phase == 3:
            signals[::2] = True
        data[:, signals] += delta * mean_sd[signals]
        results = {
            "uncertainty": uncertainty_test(data, alpha=protocol["alpha"], beta=protocol["beta"], **old_options),
            "wilks_scalar": wilks_uncertainty_test(data, alpha=protocol["alpha"], beta=protocol["beta"],
                **(options | {"max_dimension": 1})),
            "wilks_joint": wilks_uncertainty_test(data, alpha=protocol["alpha"], beta=protocol["beta"], **options),
        }
        fitted = fit_equicorrelated_ar1(data).phi
        plugin = known_phi_gls_t(data, fitted)
        critical = results["wilks_joint"].critical_squared
        decisions = {"gls_fitted": k * t.sf(plugin, size - 1) <= protocol["alpha"],
                     **{name: result.decisions for name, result in results.items()}}
        statistics = {"gls_fitted": float(plugin.max())}
        if group["in_scope"]:
            oracle = known_phi_gls_t(data, group["phi_factor"])
            decisions |= {"gls_known": k * t.sf(oracle, size - 1) <= protocol["alpha"],
                          "gls_known_budget": oracle > np.nextafter(math.sqrt(float(critical)), math.inf)}
            statistics |= {"gls_known": float(oracle.max()), "gls_known_budget": float(oracle.max())}
        for method in METHODS:
            if method not in decisions:
                continue
            mask = decisions[method]
            result = results.get(method)
            diagnostic = {name: "" for name in FIELDS[14:]}
            if result is not None:
                diagnostic |= {
                    "dimension": getattr(result, "dimension", 1),
                    "ci_intervals": ";".join(f"{low},{high}" for low, high in result.intervals),
                    "ci_length": float(sum((high - low for low, high in result.intervals), start=0)),
                    "ci_contains_truth": int(any(low <= Fraction(group["phi_factor"]) <= high for low, high in result.intervals)),
                    "phi1_retained": int(result.phi1_retained), "empty_ci": int(result.empty_ci),
                    "singular_fallback": int(getattr(result, "singular_fallback", False)),
                    "ci_unresolved": result.ci_unresolved_cells,
                    "certificate_unresolved": int(result.certificate_unresolved.sum()),
                    "certificate_nodes": int(result.certificate_nodes.sum()),
                }
            rows.append(diagnostic | {
                "phase": phase, "seed": seed, "group": group["id"], "replicate": replicate,
                "delta": delta, "method": method, "reject": int(mask.any()),
                "false_reject": int(mask[~signals].any()), "signal_reject": int(mask[signals].any()),
                "rejection_count": int(mask.sum()), "decision_mask": np.packbits(mask).tobytes().hex(),
                "max_statistic": statistics.get(method, ""), "phi_fit": fitted,
                "critical_squared": float(critical),
            })
        snapshots.append({"delta": delta, **{name: _certificate(value) for name, value in results.items()}})
    return rows, snapshots


def _job(arguments):
    return _records(*arguments)


def _summaries(rows, group, phase, protocol):
    summaries, paired, geometry = [], [], []
    for delta in sorted({row["delta"] for row in rows}):
        selected = {name: [row for row in rows if row["delta"] == delta and row["method"] == name] for name in METHODS}
        selected = {name: values for name, values in selected.items() if values}
        for name, values in selected.items():
            for mode in (*DECISIONS, *(DIAGNOSTICS if name in CI_METHODS else ())):
                count, total = sum(row[mode] for row in values), len(values)
                rate, low, high = _wilson(count, total, .95)
                summaries.append(dict(phase=phase, group=group, delta=delta, method=name, mode=mode,
                                      n=total, count=count, rate=rate, low=low, high=high))
            if name in CI_METHODS:
                widths = np.array([row["ci_length"] for row in values])
                geometry.append(dict(phase=phase, group=group, delta=delta, method=name, n=len(values),
                    mean_width=float(widths.mean()), median_width=float(np.median(widths)),
                    q10_width=float(np.quantile(widths, .1)), q90_width=float(np.quantile(widths, .9)),
                    mean_ci_unresolved=float(np.mean([row["ci_unresolved"] for row in values])),
                    certificate_unresolved=sum(row["certificate_unresolved"] for row in values),
                    mean_certificate_nodes=float(np.mean([row["certificate_nodes"] for row in values]))))
        for name in protocol["comparisons"]:
            if name not in selected:
                continue
            for mode in DECISIONS:
                difference = np.array([left[mode] - right[mode] for left, right in
                    zip(selected[name], selected[protocol["reference"]], strict=True)])
                plus, minus = int((difference == 1).sum()), int((difference == -1).sum())
                risk = float(difference.mean())
                se = float(difference.std(ddof=1) / math.sqrt(len(difference))) if len(difference) > 1 else 0
                paired.append(dict(phase=phase, group=group, delta=delta, method=name,
                    reference=protocol["reference"], mode=mode, n=len(difference), method_only=plus,
                    reference_only=minus, risk_difference=risk, low=risk - 1.95996398454 * se,
                    high=risk + 1.95996398454 * se,
                    mcnemar_p=float(binomtest(plus, plus + minus, .5).pvalue) if plus + minus else 1.0))
    return summaries, paired, geometry


def run(profile: str, output: Path, workers: int = 1) -> dict:
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer.")
    protocol = json.loads((ROOT / SOURCES[0]).read_text())
    settings = protocol["profiles"][profile]
    if output.exists() and any(output.iterdir()):
        raise ValueError("Existing evidence is never overwritten; choose an empty directory.")
    output.mkdir(parents=True, exist_ok=True)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    metadata = dict(status="running", profile=profile, settings=settings, protocol=protocol,
        git_revision=revision, workers=workers, environment=_versions(), cells=[],
        source_hashes={name: _sha(ROOT / name) for name in SOURCES})
    _json(output / "metadata.json", metadata)
    started = time.perf_counter()
    summaries, paired, geometry, certificates = [], [], [], {}
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        for phase, label in ((1, "null"), (2, "power"), (3, "partial")):
            total, seed = settings[f"{label}_replicates"], settings[f"{label}_seed"]
            for group in protocol["groups"]:
                if phase == 2 and group["id"] not in protocol["power"]["groups"]:
                    continue
                if phase == 3 and group["id"] not in protocol["partial_null"]["groups"]:
                    continue
                deltas = protocol["power"]["standardized_mean_shifts"] if phase == 2 else (
                    [protocol["partial_null"]["standardized_mean_shift"]] if phase == 3 else [0.0])
                key, cell = f"p{phase}-g{group['id']:02d}", []
                print(f"{key}: T={group['n_obs']}, K={group['k']}, R={total}", flush=True)
                jobs = ((group, seed, phase, replicate, deltas, protocol) for replicate in range(total))
                batches = executor.map(_job, jobs, chunksize=8) if executor else map(_job, jobs)
                with (output / f"{key}.csv.gz").open("wb") as raw:
                    with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
                        with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as stream:
                            writer = csv.DictWriter(stream, fieldnames=FIELDS)
                            writer.writeheader()
                            for replicate, (records, snapshot) in enumerate(batches):
                                writer.writerows(records)
                                cell.extend(records)
                                if replicate in {0, total // 2, total - 1}:
                                    certificates[f"{key}-r{replicate:05d}"] = snapshot
                                if (replicate + 1) % 200 == 0:
                                    print(f"  {replicate + 1}/{total}; {time.perf_counter() - started:.1f}s", flush=True)
                summary, differences, shapes = _summaries(cell, group["id"], phase, protocol)
                summaries.extend(summary)
                paired.extend(differences)
                geometry.extend(shapes)
                metadata["cells"].append(dict(key=key, group=group["id"], phase=phase, n=total,
                    records=len(cell), in_scope=group["in_scope"]))
                _json(output / "metadata.json", metadata)
    finally:
        if executor:
            executor.shutdown()
    for name, rows in (("summary", summaries), ("paired", paired), ("geometry", geometry)):
        _csv(output / f"{name}.csv", rows)
    _json(output / "certificates.json", certificates)
    for name, digest in metadata["source_hashes"].items():
        if _sha(ROOT / name) != digest:
            raise ValueError(f"Source changed during computation: {name}; run remains incomplete.")
    metadata.update(status="complete", elapsed_seconds=time.perf_counter() - started,
                    records=sum(cell["records"] for cell in metadata["cells"]))
    metadata["output_hashes"] = {value.name: _sha(value) for value in sorted(output.iterdir()) if value.name != "metadata.json"}
    _json(output / "metadata.json", metadata)
    print(f"Complete: {metadata['records']} records in {metadata['elapsed_seconds']:.1f}s", flush=True)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("quick", "full"), default="quick")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    arguments = parser.parse_args()
    run(arguments.profile, arguments.output, arguments.workers)
