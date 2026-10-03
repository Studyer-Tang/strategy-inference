"""Audit every saved record and regenerate prespecified complete certificates."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
from joint_uncertainty import CI_METHODS, DECISIONS, DIAGNOSTICS, METHODS, SOURCES, _records
from parameter_uncertainty import _json
from parametric_replay import _sha
from scipy.stats import binomtest

from strategy_inference.experiments import _wilson
from strategy_inference.uncertainty import _critical_squared

ROOT = Path(__file__).resolve().parents[1]


def _read(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as stream:
        return list(csv.DictReader(stream))


def _close(left, right):
    if not math.isclose(float(left), float(right), rel_tol=3e-12, abs_tol=3e-13):
        raise ValueError(f"Numeric mismatch: {left} != {right}")


def _index(rows, keys):
    indexed = {tuple(row[key] for key in keys): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("Duplicate summary keys.")
    return indexed


def _deltas(phase, protocol):
    return protocol["power"]["standardized_mean_shifts"] if phase == 2 else (
        [protocol["partial_null"]["standardized_mean_shift"]] if phase == 3 else [0.0])


def verify(output: Path) -> dict:
    metadata = json.loads((output / "metadata.json").read_text())
    protocol = json.loads((ROOT / SOURCES[0]).read_text())
    if metadata["status"] != "complete" or metadata["protocol"] != protocol:
        raise ValueError("A complete run of the frozen protocol is required.")
    settings = protocol["profiles"][metadata["profile"]]
    if metadata["settings"] != settings or not isinstance(metadata["workers"], int) or metadata["workers"] < 1:
        raise ValueError("Run settings differ from the protocol.")
    if set(metadata["source_hashes"]) != set(SOURCES):
        raise ValueError("Computational source manifest differs from the protocol.")
    for name, digest in metadata["source_hashes"].items():
        if _sha(ROOT / name) != digest:
            raise ValueError(f"Computational source mismatch: {name}")
    groups = {group["id"]: group for group in protocol["groups"]}
    expected_cells = {(phase, identifier) for phase in (1, 2, 3) for identifier in groups
        if phase == 1 or phase == 2 and identifier in protocol["power"]["groups"]
        or phase == 3 and identifier in protocol["partial_null"]["groups"]}
    cells = metadata["cells"]
    if {(cell["phase"], cell["group"]) for cell in cells} != expected_cells or len(cells) != len(expected_cells):
        raise ValueError("Missing or duplicated prespecified cells.")
    required = {"summary.csv", "paired.csv", "geometry.csv", "certificates.json"} | {f"{cell['key']}.csv.gz" for cell in cells}
    if set(metadata["output_hashes"]) != required:
        raise ValueError("Evidence manifest differs from the protocol.")
    for name, digest in metadata["output_hashes"].items():
        if Path(name).name != name or _sha(output / name) != digest:
            raise ValueError(f"Evidence hash mismatch: {name}")
    summaries = _index(_read(output / "summary.csv"), ("phase", "group", "delta", "method", "mode"))
    paired = _index(_read(output / "paired.csv"), ("phase", "group", "delta", "method", "mode"))
    geometry = _index(_read(output / "geometry.csv"), ("phase", "group", "delta", "method"))
    certificates = json.loads((output / "certificates.json").read_text())
    seen_summaries, seen_pairs, seen_geometry, seen_certificates = set(), set(), set(), set()
    checked, regenerated = 0, 0
    for cell in cells:
        phase, group, total = cell["phase"], groups[cell["group"]], cell["n"]
        label = {1: "null", 2: "power", 3: "partial"}[phase]
        if total != settings[f"{label}_replicates"] or cell["key"] != f"p{phase}-g{group['id']:02d}" or cell["in_scope"] != group["in_scope"]:
            raise ValueError("Cell settings differ from the protocol.")
        deltas = _deltas(phase, protocol)
        methods = METHODS if group["in_scope"] else tuple(name for name in METHODS if not name.startswith("gls_known"))
        rows = _read(output / f"{cell['key']}.csv.gz")
        expected = [(rep, delta, method) for rep in range(total) for delta in deltas for method in methods]
        if len(rows) != len(expected) or len(rows) != cell["records"]:
            raise ValueError("Wrong cell record count.")
        critical = _critical_squared(group["n_obs"], group["k"],
            Fraction(protocol["alpha"]) - Fraction(protocol["beta"]), protocol["procedure"]["critical_bits"])
        by_replicate, shared = {}, {}
        for row, key in zip(rows, expected, strict=True):
            rep, delta, method = int(row["replicate"]), float(row["delta"]), row["method"]
            if (rep, delta, method) != key or int(row["phase"]) != phase or int(row["group"]) != group["id"]:
                raise ValueError("Record order or labels differ from the canonical paired design.")
            if int(row["seed"]) != settings[f"{label}_seed"]:
                raise ValueError("Wrong phase seed.")
            common = (row["phi_fit"], row["critical_squared"])
            if shared.setdefault((rep, delta), common) != common:
                raise ValueError("Paired inputs differ across methods.")
            if not math.isfinite(float(row["phi_fit"])) or not -1 < float(row["phi_fit"]) < 1:
                raise ValueError("Invalid fitted phi.")
            _close(row["critical_squared"], critical)
            raw = np.unpackbits(np.frombuffer(bytes.fromhex(row["decision_mask"]), dtype=np.uint8))
            k = group["k"]
            if len(raw) != 8 * math.ceil(k / 8) or raw[k:].any():
                raise ValueError("Malformed decision mask.")
            mask = raw[:k].astype(bool)
            signals = np.zeros(k, dtype=bool)
            if phase == 2:
                signals[0] = True
            elif phase == 3:
                signals[::2] = True
            for mode, actual in zip((*DECISIONS, "rejection_count"),
                                   (mask.any(), mask[~signals].any(), mask[signals].any(), mask.sum()), strict=True):
                if int(row[mode]) != int(actual):
                    raise ValueError(f"Decision summary mismatch: {mode}")
            if method in CI_METHODS:
                intervals = [tuple(Fraction(item) for item in pair.split(",")) for pair in row["ci_intervals"].split(";") if pair]
                if any(not 0 <= low <= high <= 1 for low, high in intervals) or any(
                    left[1] >= right[0] for left, right in zip(intervals, intervals[1:], strict=False)):
                    raise ValueError("Invalid or unmerged confidence set.")
                _close(row["ci_length"], sum((high - low for low, high in intervals), start=0))
                truth = Fraction(group["phi_factor"])
                diagnostic = {"ci_contains_truth": any(low <= truth <= high for low, high in intervals),
                              "phi1_retained": any(high == 1 for _, high in intervals), "empty_ci": not intervals}
                for mode, actual in diagnostic.items():
                    if int(row[mode]) != int(actual):
                        raise ValueError(f"Confidence diagnostic mismatch: {mode}")
                dim = min(k, protocol["procedure"]["max_dimension"]) if method == "wilks_joint" else 1
                if int(row["dimension"]) != dim or int(row["singular_fallback"]) not in (0, 1):
                    raise ValueError("Invalid dimension or singular flag.")
                if (diagnostic["phi1_retained"] or not intervals) and mask.any():
                    raise ValueError("Conservative boundary or empty-set guard failed.")
                if not 0 <= int(row["certificate_unresolved"]) <= k or not 0 <= int(row["certificate_nodes"]) <= k * protocol["procedure"]["max_nodes"]:
                    raise ValueError("Invalid certificate counts.")
                if int(row["ci_unresolved"]) < 0:
                    raise ValueError("Invalid unresolved-cell count.")
            elif any(row[name] for name in ("dimension", "ci_intervals", "ci_length", *DIAGNOSTICS, "ci_unresolved", "certificate_unresolved", "certificate_nodes")):
                raise ValueError("A GLS comparison was incorrectly assigned a confidence set.")
            if method not in CI_METHODS and not math.isfinite(float(row["max_statistic"])):
                raise ValueError("Invalid GLS comparison statistic.")
            by_replicate.setdefault(rep, []).append(row)
            checked += 1
        for rep in {0, total // 2, total - 1}:
            key = f"{cell['key']}-r{rep:05d}"
            actual, snapshot = _records(group, settings[f"{label}_seed"], phase, rep, deltas, protocol)
            if certificates[key] != snapshot:
                raise ValueError("Full determinant/interval/decision certificate regeneration differed.")
            seen_certificates.add(key)
            for saved, generated in zip(by_replicate[rep], actual, strict=True):
                for name, value in generated.items():
                    if isinstance(value, (float, np.floating)):
                        _close(saved[name], value)
                    elif str(value) != saved[name]:
                        raise ValueError(f"Regenerated field mismatch: {name}")
            regenerated += len(deltas)
        for delta in deltas:
            selected = {method: [row for row in rows if row["method"] == method and float(row["delta"]) == delta] for method in methods}
            for method, values in selected.items():
                for mode in (*DECISIONS, *(DIAGNOSTICS if method in CI_METHODS else ())):
                    key = (str(phase), str(group["id"]), str(delta), method, mode)
                    saved = summaries[key]
                    count = sum(int(row[mode]) for row in values)
                    if int(saved["count"]) != count or int(saved["n"]) != total:
                        raise ValueError("Summary count mismatch.")
                    for name, value in zip(("rate", "low", "high"), _wilson(count, total, .95), strict=True):
                        _close(saved[name], value)
                    seen_summaries.add(key)
                if method in CI_METHODS:
                    key = (str(phase), str(group["id"]), str(delta), method)
                    saved, widths = geometry[key], np.array([float(row["ci_length"]) for row in values])
                    if int(saved["n"]) != total:
                        raise ValueError("Geometry count mismatch.")
                    for name, value in {
                        "mean_width": widths.mean(), "median_width": np.median(widths),
                        "q10_width": np.quantile(widths, .1), "q90_width": np.quantile(widths, .9),
                        "mean_ci_unresolved": np.mean([int(row["ci_unresolved"]) for row in values]),
                        "certificate_unresolved": sum(int(row["certificate_unresolved"]) for row in values),
                        "mean_certificate_nodes": np.mean([int(row["certificate_nodes"]) for row in values]),
                    }.items():
                        _close(saved[name], value)
                    seen_geometry.add(key)
            for method in protocol["comparisons"]:
                if method not in selected:
                    continue
                for mode in DECISIONS:
                    key = (str(phase), str(group["id"]), str(delta), method, mode)
                    saved = paired[key]
                    difference = np.array([int(a[mode]) - int(b[mode]) for a, b in zip(selected[method], selected[protocol["reference"]], strict=True)])
                    plus, minus = int((difference == 1).sum()), int((difference == -1).sum())
                    se = difference.std(ddof=1) / math.sqrt(total) if total > 1 else 0
                    if saved["reference"] != protocol["reference"] or int(saved["n"]) != total or int(saved["method_only"]) != plus or int(saved["reference_only"]) != minus:
                        raise ValueError("Paired count mismatch.")
                    for name, value in {"risk_difference": difference.mean(),
                        "low": difference.mean() - 1.95996398454 * se,
                        "high": difference.mean() + 1.95996398454 * se,
                        "mcnemar_p": binomtest(plus, plus + minus, .5).pvalue if plus + minus else 1}.items():
                        _close(saved[name], value)
                    seen_pairs.add(key)
    if seen_summaries != set(summaries) or seen_pairs != set(paired) or seen_geometry != set(geometry) or seen_certificates != set(certificates):
        raise ValueError("Missing or extra evidence summaries/certificates.")
    if checked != metadata["records"]:
        raise ValueError("Metadata record total mismatch.")
    result = dict(status="passed", records_checked=checked, shifted_datasets_regenerated=regenerated,
        complete_ci_calls_regenerated=3 * regenerated, summaries_checked=len(summaries),
        paired_checked=len(paired), geometry_checked=len(geometry),
        scope="Every record: hash, canonical design, masks, intervals and arithmetic. Prespecified snapshots: complete data and rational determinant/GLS certificate regeneration. Other numerical certificates are not rerun.",
        metadata_sha256=_sha(output / "metadata.json"), auditor_sha256=_sha(Path(__file__)))
    _json(output / "audit.json", result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    verify(parser.parse_args().output)
