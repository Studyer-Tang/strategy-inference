"""Read-only audit of frozen Gaussian replay evidence, with a separate audit file.

Every raw record is checked against its recorded decisions and analytic law.
All saved inner snapshots are checked; their selected replicates are regenerated
at a different batch partition. This does not rerun every inner simulation or
establish statistical validity beyond the experiment's stated model.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import beta, binomtest, norm, t

from strategy_inference.reference import equicorrelated_max_quantile, equicorrelated_max_tail

ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _member(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Evidence member escapes its directory: {name!r}.")
    return path


def _read_csv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


class Checks:
    def __init__(self) -> None:
        self.counts: dict[str, int] = {}
        self.residuals: dict[str, dict] = {}

    def require(self, name: str, condition: bool, detail: str) -> None:
        if not condition:
            raise ArithmeticError(f"{name}: {detail}")
        self.counts[name] = self.counts.get(name, 0) + 1

    def close(self, name: str, actual: object, expected: object) -> None:
        left, right = float(actual), float(expected)
        if not math.isfinite(left) or not math.isfinite(right):
            raise ArithmeticError(f"{name}: non-finite comparison {left!r}, {right!r}.")
        error = abs(left - right)
        record = self.residuals.setdefault(name, {"count": 0, "max_absolute": 0.0, "max_scaled": 0.0})
        record["count"] += 1
        record["max_absolute"] = max(record["max_absolute"], error)
        record["max_scaled"] = max(record["max_scaled"], error / max(1.0, abs(right)))
        if not math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12):
            raise ArithmeticError(f"{name}: {left!r} differs from {right!r}.")


def _reference(group: dict) -> tuple[float, float]:
    """Independent lag-sum calculation, not the runner's mean-variance helper."""
    size = group["n_obs"]

    def mass(phi):
        return math.fsum([1.0] + [2 * (1 - h / size) * phi**h for h in range(1, size)])

    common, idio = mass(group["phi_factor"]), mass(group["phi_idio"])
    variance = group["rho"] * common + (1 - group["rho"]) * idio
    return variance, group["rho"] * common / variance


def _row_key(row: dict) -> tuple[int, int, float, str, str]:
    return int(row["phase"]), int(row["group"]), float(row["delta"]), row["method"], row["mode"]


def _summary(rows: list[dict], phase: int, group: int, delta: float, method: str, mode: str) -> dict:
    decisions = np.array([int(row[mode]) for row in rows], dtype=int)
    count, size = int(decisions.sum()), len(rows)
    rate, z = count / size, float(norm.ppf(0.975))
    denominator = 1 + z**2 / size
    middle = (rate + z**2 / (2 * size)) / denominator
    radius = z / denominator * math.sqrt(rate * (1 - rate) / size + z**2 / (4 * size**2))
    correct = sum(int(row[mode]) and int(row["winner"]) == 0 for row in rows)
    assessed = phase == 1 and group <= 6 and method == "replay_fitted" and mode == "reject"
    upper = (
        float(beta.ppf(1 - 0.05 / 6, count + 1, size - count)) if count < size else 1.0
    ) if assessed else ""
    return {
        "phase": phase, "group": group, "delta": delta, "method": method, "mode": mode,
        "n": size, "rejections": count, "rate": rate, "low": middle - radius, "high": middle + radius,
        "reject_select_signal": correct, "signal_rate": correct / size, "family_upper": upper,
        "passes_tolerance": int(upper <= 0.07) if assessed else "",
    }


def _paired(left: list[dict], right: list[dict], phase: int, group: int,
            delta: float, method: str, mode: str) -> dict:
    difference = np.array([int(row[mode]) for row in left]) - [int(row[mode]) for row in right]
    plus, minus = int((difference == 1).sum()), int((difference == -1).sum())
    risk = float(difference.mean())
    se = float(difference.std(ddof=1) / math.sqrt(len(left))) if len(left) > 1 else 0.0
    return {
        "phase": phase, "group": group, "delta": delta, "method": method, "mode": mode,
        "reference": "replay_known", "n": len(left), "method_only": plus, "reference_only": minus,
        "risk_difference": risk, "low": risk - norm.ppf(0.975) * se,
        "high": risk + norm.ppf(0.975) * se,
        "mcnemar_p": float(binomtest(plus, plus + minus, 0.5).pvalue) if plus + minus else 1.0,
    }


def _compare_record(checks: Checks, name: str, actual: dict, expected: dict) -> None:
    checks.require(name + "_schema", set(actual) == set(expected), "record fields differ")
    for field, value in expected.items():
        if isinstance(value, str):
            checks.require(name + "_text", actual[field] == value, f"{field} differs")
        elif isinstance(value, (int, np.integer)):
            checks.require(name + "_integer", actual[field] == str(value), f"{field} differs")
        else:
            checks.close(name + "_numeric", actual[field], value)


def _load_runner():
    source = ROOT / "scripts/parametric_replay.py"
    specification = importlib.util.spec_from_file_location("frozen_replay_for_audit", source)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _calibration(checks: Checks, scores: list[float], result: dict, alpha: float) -> None:
    size = len(scores)
    rank = math.ceil((1 - alpha) * (size + 1))
    tail = size + 1 - rank
    checks.require("calibration_size", result["n"] == size, "n differs")
    checks.require("calibration_rank", result["rank"] == rank <= size, "order statistic differs")
    checks.close("calibration_cutoff", result["cutoff"], sorted(scores)[rank - 1])
    checks.close("calibration_unconditional_size", result["unconditional_size"], tail / (size + 1))
    expected = beta.ppf([0.025, 0.975], tail, rank)
    checks.require("calibration_interval_shape", len(result["conditional_size_interval"]) == 2, "interval shape")
    for actual, value in zip(result["conditional_size_interval"], expected, strict=True):
        checks.close("calibration_beta_interval", actual, value)


def verify(output: Path, *, batch_size: int = 7) -> dict:
    output = output.resolve()
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    checks = Checks()
    started = time.perf_counter()
    metadata_path = output / "metadata.json"
    metadata_digest = _sha(metadata_path)
    metadata = json.loads(metadata_path.read_text())
    checks.require("completed", metadata["status"] == "complete", "calculation is incomplete")
    source_hashes = metadata["source_hashes"]
    for name, digest in source_hashes.items():
        checks.require("calculation_source_hash", _sha(_member(ROOT, name)) == digest, name)
    protocol_path = ROOT / "experiments/parametric-replay-protocol.json"
    checks.require("protocol_hash", _sha(protocol_path) == metadata["protocol_sha256"], "protocol changed")
    protocol = json.loads(protocol_path.read_text())
    settings = protocol["profiles"][metadata["profile"]]
    checks.require("profile_settings", settings == metadata["settings"], "profile differs")
    checks.close("alpha", metadata["alpha"], protocol["alpha"])
    alpha, draws = protocol["alpha"], settings["inner_draws"]
    runner = _load_runner()
    regenerate_batch = batch_size if batch_size != metadata["batch_size"] else batch_size + 1
    groups = {group["id"]: group for group in protocol["groups"]}
    checks.require("unique_groups", len(groups) == len(protocol["groups"]), "duplicate group ID")
    seeds = [
        profile[name] for profile in protocol["profiles"].values()
        for name in ("calibration_seed", "null_seed", "power_seed")
    ]
    checks.require("separate_stage_seeds", len(seeds) == len(set(seeds)), "stage/profile seed reuse")
    old = json.loads((ROOT / "experiments/tail-diagnostic-protocol.json").read_text())
    checks.require("previous_study_seeds", not set(seeds) & {old["root_seed"], old["quick_seed"]}, "old seed reuse")
    expected_cells = []
    for phase, n_key, seed_key in (
        (0, "calibration_replicates", "calibration_seed"),
        (1, "null_replicates", "null_seed"), (2, "power_replicates", "power_seed"),
    ):
        for group in protocol["groups"]:
            if phase != 1 and group["id"] not in protocol["power"]["groups"]:
                continue
            expected_cells.append((phase, group, settings[n_key], settings[seed_key]))
    raw_names = {f"p{phase}-g{group['id']:02d}.csv.gz" for phase, group, _, _ in expected_cells}
    raw_names.update({"calibration.json", "inner-snapshots.json", "summary.csv", "paired.csv"})
    checks.require("raw_output_members", set(metadata["output_hashes"]) == raw_names, "raw output set differs")
    for name, digest in metadata["output_hashes"].items():
        checks.require("raw_output_hash", _sha(_member(output, name)) == digest, name)
    cells = {cell["key"]: cell for cell in metadata["cells"]}
    checks.require("metadata_cells", len(cells) == len(expected_cells) == len(metadata["cells"]), "cell counts differ")
    calibration = json.loads((output / "calibration.json").read_text())
    checks.require("calibration_groups", set(calibration) == {str(group) for group in protocol["power"]["groups"]}, "groups differ")
    snapshots = json.loads((output / "inner-snapshots.json").read_text())
    expected_snapshot_keys = {
        f"p{phase}-g{group['id']:02d}-r{replicate:05d}"
        for phase, group, total, _ in expected_cells
        for replicate in {0, total // 2, total - 1}
    }
    checks.require("snapshot_members", set(snapshots) == expected_snapshot_keys, "snapshot selection differs")
    expected_summaries, expected_pairs = {}, {}
    total_records, mc_records, regenerated = 0, 0, 0
    for phase, group, total, seed in expected_cells:
        identifier, size, k = group["id"], group["n_obs"], group["k"]
        key = f"p{phase}-g{identifier:02d}"
        records = _read_csv(output / f"{key}.csv.gz")
        methods = [
            method for method in protocol["methods"]
            if method != "gls_known" or group["phi_factor"] == group["phi_idio"]
        ]
        deltas = protocol["power"]["standardized_mean_shifts"] if phase == 2 else [0.0]
        width = len(methods) * len(deltas)
        checks.require("cell_record_count", len(records) == total * width, key)
        cell = cells[key]
        checks.require("cell_metadata_count", cell["n"] == total and cell["records"] == len(records), key)
        checks.require("cell_metadata_address", cell["phase"] == phase and cell["group"] == identifier, key)
        target, mean_rho = _reference(group)
        checks.close("finite_mean_target", cell["target"], target)
        checks.close("correlation_of_means", cell["correlation_of_means"], mean_rho)
        known_critical = equicorrelated_max_quantile(1 - alpha, k, mean_rho)
        gls_critical = float(t.isf(alpha / k, size - 1))
        matched = phase != 0 and identifier in protocol["power"]["groups"]
        selected: dict[tuple[float, str], list[dict]] = {}
        print(f"Auditing {key}: {len(records)} method records; {total} outer replicates", flush=True)
        for replicate in range(total):
            block = records[replicate * width:(replicate + 1) * width]
            first = block[0]
            rho, raw_rho = float(first["rho_fit"]), float(first["rho_raw"])
            phi, projected = float(first["phi_fit"]), int(first["rho_projected"])
            checks.require("fitted_phi_domain", math.isfinite(phi) and abs(phi) < 1, key)
            checks.require("rho_projection", rho == float(np.clip(raw_rho, 0, 1)), key)
            checks.require("rho_projection_flag", projected == int(rho != raw_rho), key)
            fitted_critical = equicorrelated_max_quantile(1 - alpha, k, rho)
            jitter = float(np.random.default_rng(
                np.random.SeedSequence([seed, 31, phase, identifier, replicate, 2])
            ).random())
            order = [(delta, method) for delta in deltas for method in methods]
            for row, (delta, method) in zip(block, order, strict=True):
                checks.require("raw_schema", set(row) == set(runner.FIELDS), key)
                checks.require(
                    "raw_address", (int(row["phase"]), int(row["seed"]), int(row["group"]),
                                    int(row["replicate"])) == (phase, seed, identifier, replicate), key,
                )
                checks.require("raw_method_order", row["method"] == method, key)
                checks.close("raw_shift", row["delta"], delta)
                for field in ("phi_fit", "rho_raw", "rho_fit", "rho_projected"):
                    checks.require("constant_fit_across_methods_and_shifts", row[field] == first[field], key)
                winner = int(row["winner"])
                checks.require("winner_domain", 0 <= winner < k, key)
                pvalue, statistic = float(row["pvalue"]), float(row["statistic"])
                checks.require("pvalue_domain", math.isfinite(pvalue) and 0 <= pvalue <= 1, key)
                checks.require("finite_statistic", math.isfinite(statistic), key)
                checks.require("raw_reject", int(row["reject"]) == int(pvalue <= alpha), key)
                if method.startswith("replay_"):
                    count = int(row["exceedances"])
                    checks.require("monte_carlo_count_domain", 0 <= count <= draws, key)
                    checks.close("monte_carlo_rank_p", pvalue, (count + 1) / (draws + 1))
                    checks.close("monte_carlo_jitter_score", row["score"], 1 - pvalue + jitter / (draws + 1))
                    mc_records += 1
                else:
                    checks.require("analytic_count_blank", row["exceedances"] == "", key)
                    if method == "gls_known":
                        expected_p = min(1.0, k * float(t.sf(statistic, size - 1)))
                        critical = gls_critical
                    else:
                        reference_rho = rho if method == "gaussian_fitted" else mean_rho
                        expected_p = equicorrelated_max_tail(statistic, k, reference_rho)
                        critical = fitted_critical if method == "gaussian_fitted" else known_critical
                    checks.close("analytic_pvalue", pvalue, expected_p)
                    checks.close("analytic_score", row["score"], statistic / critical)
                if method == "oracle_gaussian":
                    checks.require("oracle_phi_blank", row["selected_phi"] == "", key)
                    checks.close("oracle_scale", row["selected_variance_ratio"], 1)
                elif method == "gls_known":
                    checks.require("gls_tail_fields_blank", row["selected_phi"] == row["selected_variance_ratio"] == "", key)
                else:
                    fitted_column = float(row["selected_phi"])
                    ratio = float(row["selected_variance_ratio"])
                    checks.require("selected_phi_domain", math.isfinite(fitted_column) and abs(fitted_column) < 1, key)
                    checks.require("selected_variance_domain", math.isfinite(ratio) and ratio > 0, key)
                if matched:
                    cutoff = calibration[str(identifier)][method]["cutoff"]
                    checks.require("matched_reject", int(row["matched_reject"]) == int(float(row["score"]) > cutoff), key)
                else:
                    checks.require("matched_reject_blank", row["matched_reject"] == "", key)
                selected.setdefault((float(delta), method), []).append(row)

            snapshot_key = f"{key}-r{replicate:05d}"
            if snapshot_key in snapshots:
                saved = snapshots[snapshot_key]
                checks.require("snapshot_methods", set(saved) == set(runner.REPLAYS), snapshot_key)
                for name, values in saved.items():
                    checks.require("snapshot_inner_count", len(values) == draws, snapshot_key)
                    array = np.asarray(values, dtype=float)
                    checks.require("snapshot_finite", np.isfinite(array).all(), snapshot_key)
                    for row in block:
                        if row["method"] == f"replay_{name}":
                            count = int(np.count_nonzero(array >= float(row["statistic"])))
                            checks.require("saved_snapshot_exceedances", count == int(row["exceedances"]), snapshot_key)
                regenerated_rows, maxima = runner._records(
                    group, seed, phase, replicate, draws, regenerate_batch, alpha, deltas
                )
                checks.require("regenerated_row_count", len(regenerated_rows) == len(block), snapshot_key)
                for name, actual in maxima.items():
                    checks.require("regenerated_inner_count", len(actual) == len(saved[name]), snapshot_key)
                    for value, expected in zip(actual, saved[name], strict=True):
                        checks.close("regenerated_inner_maximum", value, expected)
                for row, expected in zip(block, regenerated_rows, strict=True):
                    if matched:
                        expected["matched_reject"] = int(
                            expected["score"] > calibration[str(identifier)][expected["method"]]["cutoff"]
                        )
                    _compare_record(checks, "regenerated_raw_record", row, expected)
                regenerated += 1

        oracle_rows = [row for row in records if row["method"] == "oracle_gaussian"]
        checks.close("projected_fraction", cell["projected_fraction"],
                     sum(int(row["rho_projected"]) for row in oracle_rows) / len(oracle_rows))
        if phase == 0:
            for method in methods:
                values = [float(row["score"]) for row in selected[(0.0, method)]]
                _calibration(checks, values, calibration[str(identifier)][method], alpha)
            checks.require("calibration_methods", set(calibration[str(identifier)]) == set(methods), key)
        else:
            for delta in deltas:
                reference = selected[(float(delta), "replay_known")]
                for method in methods:
                    rows = selected[(float(delta), method)]
                    for mode in ("reject", "matched_reject"):
                        if rows[0][mode] == "":
                            continue
                        identity = (phase, identifier, float(delta), method, mode)
                        expected_summaries[identity] = _summary(rows, phase, identifier, delta, method, mode)
                        if method != "replay_known":
                            expected_pairs[identity] = _paired(rows, reference, phase, identifier, delta, method, mode)
        total_records += len(records)

    checks.require("total_method_records", total_records == metadata["records"], "total differs")
    for filename, expected in (("summary.csv", expected_summaries), ("paired.csv", expected_pairs)):
        actual = _read_csv(output / filename)
        identities = [_row_key(row) for row in actual]
        checks.require(filename + "_identities", len(identities) == len(set(identities)) == len(expected), "duplicate/missing rows")
        checks.require(filename + "_members", set(identities) == set(expected), "row selection differs")
        for row, identity in zip(actual, identities, strict=True):
            _compare_record(checks, filename.removesuffix(".csv"), row, expected[identity])

    # A presentation builder may add derived files, which are deliberately not
    # part of the frozen raw-member set. Only attest the calculation evidence.
    checks.require("metadata_unchanged", _sha(metadata_path) == metadata_digest, "metadata changed during audit")
    for name, digest in metadata["output_hashes"].items():
        checks.require("raw_unchanged_during_audit", _sha(_member(output, name)) == digest, name)
    for name, digest in source_hashes.items():
        checks.require("source_unchanged_during_audit", _sha(_member(ROOT, name)) == digest, name)
    return {
        "status": "passed", "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started, "output": str(output),
        "profile": metadata["profile"], "calculation_git_revision": metadata["git_revision"],
        "calculation_metadata_sha256": metadata_digest,
        "calculation_source_hashes": source_hashes, "raw_output_hashes": metadata["output_hashes"],
        "audit_source_sha256": _sha(Path(__file__).resolve()),
        "audit_environment": {"numpy": np.__version__, "scipy": scipy.__version__},
        "method_records": total_records, "monte_carlo_records": mc_records,
        "cells": len(expected_cells), "regenerated_snapshot_replicates": regenerated,
        "recorded_inner_batch_size": metadata["batch_size"], "regeneration_batch_size": regenerate_batch,
        "checks": checks.counts, "numeric_residuals": checks.residuals,
        "scope": (
            "All hashed raw records, decisions, analytic probabilities, calibration, summaries and paired "
            "statistics checked. Every saved inner snapshot checked and its selected replicate regenerated. "
            "Other inner draws were not rerun. This is an evidence-integrity audit, not a proof of statistical "
            "validity or originality. Derived presentation files are outside the frozen raw hash set."
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Completed calculation evidence directory.")
    parser.add_argument("--audit", type=Path, help="Separate audit JSON; default is OUTPUT/audit.json.")
    parser.add_argument("--batch-size", type=int, default=7, help="Regeneration partition, changed if equal to the recorded partition.")
    arguments = parser.parse_args()
    destination = arguments.audit or arguments.output / "audit.json"
    if destination.exists():
        raise ValueError("Existing audit evidence is not overwritten; choose a new --audit path.")
    destination = destination.resolve()
    protected = arguments.output.resolve() / "metadata.json"
    if destination == protected:
        raise ValueError("The audit destination must not replace calculation metadata.")
    result = verify(arguments.output, batch_size=arguments.batch_size)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(
        f"Audit passed: {result['method_records']} records, "
        f"{result['regenerated_snapshot_replicates']} regenerated replicates; {destination}", flush=True,
    )
