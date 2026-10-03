"""Verify saved evidence, arithmetic and complete selected certificates."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
from parameter_uncertainty import METHODS, _records
from parametric_replay import _sha
from scipy.stats import binomtest

from strategy_inference.experiments import _wilson
from strategy_inference.uncertainty import _critical_squared, _projection

ROOT = Path(__file__).resolve().parents[1]


def _read(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as stream:
        return list(csv.DictReader(stream))


def _close(left, right):
    if not math.isclose(float(left), float(right), rel_tol=3e-12, abs_tol=3e-13):
        raise ValueError(f"Numeric mismatch: {left} != {right}")


def verify(output: Path) -> dict:
    metadata = json.loads((output / "metadata.json").read_text())
    if metadata["status"] != "complete":
        raise ValueError("Only completed runs can be audited.")
    protocol = metadata["protocol"]
    groups = {group["id"]: group for group in protocol["groups"]}
    expected_protocol = json.loads((ROOT / "experiments/parameter-uncertainty-protocol.json").read_text())
    if protocol != expected_protocol:
        raise ValueError("The recorded protocol differs from its frozen source.")
    settings = protocol["profiles"][metadata["profile"]]
    if metadata["settings"] != settings:
        raise ValueError("Run settings differ from the prespecified profile.")
    expected_cells = {(phase, group["id"]) for phase in (1, 2, 3) for group in groups.values()
                      if phase == 1 or phase == 2 and group["id"] in protocol["power"]["groups"]
                      or phase == 3 and group["id"] in protocol["partial_null"]["groups"]}
    actual_cells = {(cell["phase"], cell["group"]) for cell in metadata["cells"]}
    if actual_cells != expected_cells or len(metadata["cells"]) != len(expected_cells):
        raise ValueError("Missing or duplicated prespecified cells.")
    required = {"summary.csv", "paired.csv", "certificates.json"} | {f"{cell['key']}.csv.gz" for cell in metadata["cells"]}
    if set(metadata["output_hashes"]) != required:
        raise ValueError("Evidence manifest is incomplete or contains extra files.")
    for name, digest in metadata["output_hashes"].items():
        if Path(name).name != name or _sha(output / name) != digest:
            raise ValueError(f"Evidence hash mismatch: {name}")
    for name, digest in metadata["source_hashes"].items():
        if _sha(ROOT / name) != digest:
            raise ValueError(f"Computational source mismatch: {name}")
    summary_rows, paired_rows = _read(output / "summary.csv"), _read(output / "paired.csv")
    summaries = {(int(row["phase"]), int(row["group"]), float(row["delta"]), row["method"], row["mode"]): row for row in summary_rows}
    paired = {(int(row["phase"]), int(row["group"]), float(row["delta"]), row["method"], row["mode"]): row for row in paired_rows}
    if len(summaries) != len(summary_rows) or len(paired) != len(paired_rows):
        raise ValueError("Duplicate summaries or paired results.")
    certificates = json.loads((output / "certificates.json").read_text())
    checked, regenerated, seen_summaries, seen_pairs = 0, 0, set(), set()
    for cell in metadata["cells"]:
        phase, group, total = cell["phase"], groups[cell["group"]], cell["n"]
        label = {1: "null", 2: "power", 3: "partial"}[phase]
        deltas_expected = protocol["power"]["standardized_mean_shifts"] if phase == 2 else (
            [protocol["partial_null"]["standardized_mean_shift"]] if phase == 3 else [0.0])
        methods_expected = METHODS if group["in_scope"] else ("gls_fitted", "uncertainty")
        if total != settings[f"{label}_replicates"] or cell["key"] != f"p{phase}-g{group['id']:02d}" or cell["in_scope"] != group["in_scope"]:
            raise ValueError("Cell settings differ from the protocol.")
        rows = _read(output / f"{cell['key']}.csv.gz")
        if len(rows) != cell["records"]:
            raise ValueError("Wrong cell record count.")
        unique = set()
        by_replicate = {}
        shared = {}
        expected_order = [(rep, delta, method) for rep in range(total) for delta in deltas_expected for method in methods_expected]
        projection = _projection(group["n_obs"])
        critical = _critical_squared(group["n_obs"], group["k"], Fraction(protocol["alpha"]) - Fraction(protocol["beta"]), protocol["procedure"]["critical_bits"])
        for row, expected_key in zip(rows, expected_order, strict=True):
            rep, delta, method = int(row["replicate"]), float(row["delta"]), row["method"]
            key = (rep, delta, method)
            if key in unique or not 0 <= rep < total or method not in METHODS:
                raise ValueError("Duplicate or invalid record.")
            if key != expected_key:
                raise ValueError("Record order differs from the canonical paired design.")
            if delta not in deltas_expected or method not in methods_expected or int(row["seed"]) != settings[f"{label}_seed"]:
                raise ValueError("Wrong prespecified shift, method or seed.")
            unique.add(key)
            diagnostic = {name: value for name, value in row.items() if name not in (
                "method", "reject", "false_reject", "signal_reject", "rejection_count", "decision_mask", "max_statistic",
            )}
            if shared.setdefault((rep, delta), diagnostic) != diagnostic:
                raise ValueError("Paired methods use different diagnostic inputs.")
            _close(row["critical_squared"], critical)
            if (int(row["low_df"]), int(row["high_df"])) != (projection.q, projection.s):
                raise ValueError("Wrong projection degrees of freedom.")
            if not math.isfinite(float(row["phi_fit"])) or not -1 < float(row["phi_fit"]) < 1:
                raise ValueError("Invalid fitted phi.")
            if method != "uncertainty" and not math.isfinite(float(row["max_statistic"])):
                raise ValueError("Nonfinite GLS statistic.")
            if not 0 <= int(row["certificate_unresolved"]) <= group["k"] or not 0 <= int(row["certificate_nodes"]) <= group["k"] * protocol["procedure"]["max_nodes"]:
                raise ValueError("Invalid certificate diagnostic count.")
            by_replicate.setdefault(rep, []).append(row)
            if int(row["phase"]) != phase or int(row["group"]) != group["id"]:
                raise ValueError("Wrong group or phase label.")
            mask_raw = np.unpackbits(np.frombuffer(bytes.fromhex(row["decision_mask"]), dtype=np.uint8))
            if len(mask_raw) != 8 * math.ceil(group["k"] / 8) or mask_raw[group["k"]:].any():
                raise ValueError("Malformed decision mask.")
            mask = mask_raw[:group["k"]].astype(bool)
            signals = np.zeros(group["k"], dtype=bool)
            if phase == 2:
                signals[0] = True
            elif phase == 3:
                signals[::2] = True
            for label, actual in (("reject", mask.any()), ("false_reject", mask[~signals].any()),
                                  ("signal_reject", mask[signals].any()), ("rejection_count", mask.sum())):
                if int(row[label]) != int(actual):
                    raise ValueError(f"Wrong decision summary: {label}")
            intervals = [tuple(Fraction(value) for value in item.split(",")) for item in row["ci_intervals"].split(";") if item]
            if any(not 0 <= low <= high <= 1 for low, high in intervals):
                raise ValueError("Invalid confidence interval.")
            if any(left[1] >= right[0] for left, right in zip(intervals, intervals[1:], strict=False)):
                raise ValueError("Confidence intervals are unsorted or unmerged.")
            if int(row["ci_count"]) != len(intervals) or int(row["empty_ci"]) != int(not intervals):
                raise ValueError("Wrong confidence-set count.")
            _close(row["ci_length"], sum((high - low for low, high in intervals), start=0))
            if intervals:
                _close(row["ci_low"], intervals[0][0])
                _close(row["ci_high"], intervals[-1][1])
            truth = Fraction.from_float(group["phi_factor"])
            covered, boundary = any(low <= truth <= high for low, high in intervals), any(high == 1 for _, high in intervals)
            if int(row["ci_contains_truth"]) != int(covered) or int(row["phi1_retained"]) != int(boundary):
                raise ValueError("Wrong coverage or boundary diagnostic.")
            if method == "uncertainty" and (boundary or not intervals) and mask.any():
                raise ValueError("A conservative boundary/empty-set guard was violated.")
            checked += 1
        deltas = sorted({float(row["delta"]) for row in rows})
        methods = [method for method in METHODS if any(row["method"] == method for row in rows)]
        if len(unique) != total * len(deltas) * len(methods):
            raise ValueError("Missing records.")
        for rep in {0, total // 2, total - 1}:
            actual, certificate = _records(group, int(by_replicate[rep][0]["seed"]), phase, rep, deltas, protocol)
            if certificate != certificates[f"{cell['key']}-r{rep:05d}"]:
                raise ValueError("A full certificate regeneration differed.")
            for recorded, regenerated_row in zip(by_replicate[rep], actual, strict=True):
                for name, value in regenerated_row.items():
                    if isinstance(value, (float, np.floating)):
                        _close(recorded[name], value)
                    elif str(value) != recorded[name]:
                        raise ValueError(f"Regenerated field mismatch: {name}")
            regenerated += len(deltas)
        for delta in deltas:
            selected = {method: [row for row in rows if row["method"] == method and float(row["delta"]) == delta]
                        for method in methods}
            for method, records in selected.items():
                modes = ("reject", "false_reject", "signal_reject")
                if method == "uncertainty":
                    modes += ("ci_contains_truth", "phi1_retained", "empty_ci")
                for mode in modes:
                    summary_method = "confidence_set" if mode.startswith("ci_") or mode in ("phi1_retained", "empty_ci") else method
                    key = (phase, group["id"], delta, summary_method, mode)
                    saved = summaries[key]
                    count = sum(int(row[mode]) for row in records)
                    rate, low, high = _wilson(count, total, 0.95)
                    if int(saved["count"]) != count or int(saved["n"]) != total:
                        raise ValueError("Wrong summary count.")
                    for name, value in (("rate", rate), ("low", low), ("high", high)):
                        _close(saved[name], value)
                    seen_summaries.add(key)
            for method in protocol["comparisons"]:
                if method not in selected:
                    continue
                for mode in ("reject", "signal_reject"):
                    key = (phase, group["id"], delta, method, mode)
                    saved = paired[key]
                    if saved["reference"] != "uncertainty":
                        raise ValueError("Wrong paired reference method.")
                    diff = np.array([int(left[mode]) - int(right[mode]) for left, right in
                                     zip(selected[method], selected["uncertainty"], strict=True)])
                    plus, minus = int((diff == 1).sum()), int((diff == -1).sum())
                    se = diff.std(ddof=1) / math.sqrt(total) if total > 1 else 0.0
                    if (int(saved["n"]), int(saved["method_only"]), int(saved["reference_only"])) != (total, plus, minus):
                        raise ValueError("Wrong paired counts.")
                    pvalue = float(binomtest(plus, plus + minus, 0.5).pvalue) if plus + minus else 1.0
                    for name, value in (("risk_difference", diff.mean()), ("low", diff.mean() - 1.95996398454 * se),
                                        ("high", diff.mean() + 1.95996398454 * se), ("mcnemar_p", pvalue)):
                        _close(saved[name], value)
                    seen_pairs.add(key)
    if seen_summaries != set(summaries) or seen_pairs != set(paired) or checked != metadata["records"]:
        raise ValueError("Summary coverage or overall record count differs.")
    result = {"status": "passed", "records_checked": checked, "datasets_and_certificates_regenerated": regenerated,
              "summary_rows_checked": len(summaries), "paired_rows_checked": len(paired),
              "metadata_sha256": _sha(output / "metadata.json"), "auditor_sha256": _sha(Path(__file__)),
              "scope": "All hashes, masks, confidence intervals, rate intervals and paired arithmetic; three independent datasets per cell with every mean shift and exact certificates regenerated. Other certificates are not all rerun."}
    (output / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(verify(parser.parse_args().output), indent=2))
