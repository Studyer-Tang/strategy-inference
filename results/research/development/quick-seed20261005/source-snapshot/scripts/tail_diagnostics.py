"""Reproduce known-model tail-scale diagnostics; not a production joint test."""

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
from scipy.stats import binomtest

from strategy_inference.experiments import _versions, _wilson
from strategy_inference.inference import default_lags
from strategy_inference.quadratic_reference import gaussian_ar_hac_moments
from strategy_inference.reference import equicorrelated_max_quantile
from strategy_inference.tail import ar1_tail_factor

ROOT = Path(__file__).resolve().parents[1]
METHODS = (
    "oracle", "bartlett", "tail_known_lrv", "tail_plugin_lrv",
    "tail_known_finite", "tail_plugin_finite", "mean_unbiased_oracle",
)
LABELS = (
    "Known variance", "Bartlett", "Tail: known AR, LRV", "Tail: fitted AR, LRV",
    "Tail: known AR, finite T", "Tail: fitted AR, finite T", "Mean-unbiased scale",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, content: dict) -> None:
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _components(seed: int, group: dict, start: int, count: int) -> np.ndarray:
    size, columns = group["n_obs"], max(group["k"]) + 1
    innovations = np.empty((count, size, columns))
    scale = math.sqrt((1 - group["phi"]) * (1 + group["phi"]))
    for row, replicate in enumerate(range(start, start + count)):
        rng = np.random.default_rng(np.random.SeedSequence([seed, 21, group["id"], replicate, 0]))
        innovations[row] = rng.standard_normal((size, columns))
    innovations[:, 1:, :] *= scale
    return lfilter([1], [1, -group["phi"]], innovations, axis=1)


def _batch_scales(data: np.ndarray, lags: int, phi: float, moments) -> tuple:
    count, size, columns = data.shape
    means = data.mean(axis=1)
    centered = data - means[:, None, :]
    gamma0 = np.einsum("btk,btk->bk", centered, centered) / size
    gamma1 = np.einsum("btk,btk->bk", centered[:, 1:], centered[:, :-1]) / size
    fitted = gamma1 / gamma0
    raw = gamma0.copy()
    for lag in range(1, lags + 1):
        raw += 2 * (1 - lag / (lags + 1)) * np.einsum(
            "btk,btk->bk", centered[:, lag:], centered[:, :-lag]
        ) / size
    known_lrv = float(ar1_tail_factor(phi, lags + 1))
    known_finite = float(ar1_tail_factor(phi, lags + 1, target="finite_sample", n_obs=size))
    plugin_lrv = ar1_tail_factor(fitted.ravel(), lags + 1).reshape(count, columns)
    plugin_finite = ar1_tail_factor(
        fitted.ravel(), lags + 1, target="finite_sample", n_obs=size
    ).reshape(count, columns)
    scales = np.stack((
        np.full_like(raw, moments.target), raw, known_lrv * raw, plugin_lrv * raw,
        known_finite * raw, plugin_finite * raw, raw * moments.target / moments.expectation,
    ), axis=1)
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("Invalid scale: no simulation replicate may be discarded.")
    statistics = np.sqrt(size) * means[:, None, :] / np.sqrt(scales)
    return statistics, scales, fitted


def run(profile: str, output: Path, *, batch_size: int = 32) -> dict:
    protocol_path = ROOT / "experiments/tail-diagnostic-protocol.json"
    protocol = json.loads(protocol_path.read_text())
    total, alpha = protocol["profiles"][profile], protocol["alpha"]
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty; existing evidence is not overwritten.")
    output.mkdir(parents=True, exist_ok=True)
    source_paths = [
        Path(__file__).resolve(), protocol_path,
        *[ROOT / "src/strategy_inference" / name for name in (
            "tail.py", "quadratic_reference.py", "reference.py", "inference.py",
            "_validation.py", "experiments.py",
        )],
    ]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True
    ).stdout.strip()
    metadata = {
        "status": "running", "profile": profile, "n_mc": total,
        "interpretation": protocol["scope"], "protocol_sha256": _sha(protocol_path),
        "git_revision": revision, "environment": _versions(),
        "source_hashes": {str(path.relative_to(ROOT)): _sha(path) for path in source_paths},
        "batch_size": batch_size,
    }
    _write_json(output / "metadata.json", metadata)
    started = time.perf_counter()
    summaries, paired_rows, moments_rows = [], [], []
    for group in protocol["groups"]:
        size, phi, identifier = group["n_obs"], group["phi"], group["id"]
        lags = default_lags(size)
        moments = gaussian_ar_hac_moments(size, phi, lags=lags)
        moments_rows.append({
            "group": identifier, "n_obs": size, "phi": phi, "lags": lags,
            "target": moments.target, "expectation": moments.expectation,
            "variance": moments.variance, "population_mass": moments.population_truncated_lrv,
            "long_run_limit": moments.long_run_limit,
        })
        cells = []
        for rho in group["rho"]:
            for columns in group["k"]:
                if group.get("skip_duplicate_single") and columns == 1 and rho != group["rho"][0]:
                    continue
                key = f"g{identifier:02d}-k{columns:03d}-r{round(100*rho):02d}"
                raw_file = (output / f"{key}.csv.gz").open("wb")
                gz_file = gzip.GzipFile(filename="", fileobj=raw_file, mode="wb", mtime=0)
                stream = io.TextIOWrapper(gz_file, encoding="utf-8", newline="")
                fields = ["replicate", "root_seed", "group", "k", "rho", "critical"]
                fields += [f"{method}_{field}" for method in METHODS for field in (
                    "statistic", "reject", "winner", "variance_ratio", "fitted_phi",
                )]
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                cells.append({
                    "key": key, "k": columns, "rho": rho, "stream": stream, "raw_file": raw_file,
                    "writer": writer, "critical": equicorrelated_max_quantile(1-alpha, columns, rho),
                    "decisions": np.empty((total, len(METHODS)), dtype=bool),
                    "ratio_sum": np.zeros(len(METHODS)), "ratio_squares": np.zeros(len(METHODS)),
                })
        print(f"Group {identifier}: T={size}, phi={phi:.6f}, {len(cells)} cells", flush=True)
        try:
            for start in range(0, total, batch_size):
                count = min(batch_size, total - start)
                components = _components(protocol["root_seed"], group, start, count)
                for rho in group["rho"]:
                    data = math.sqrt(rho) * components[:, :, :1] + math.sqrt(1-rho) * components[:, :, 1:]
                    statistics, scales, fitted = _batch_scales(data, lags, phi, moments)
                    for cell in (item for item in cells if item["rho"] == rho):
                        columns = cell["k"]
                        winners = statistics[:, :, :columns].argmax(axis=2)
                        selected = np.take_along_axis(statistics, winners[:, :, None], axis=2)[:, :, 0]
                        selected_scales = np.take_along_axis(scales, winners[:, :, None], axis=2)[:, :, 0]
                        selected_phi = np.take_along_axis(
                            np.broadcast_to(fitted[:, None, :], statistics.shape),
                            winners[:, :, None], axis=2,
                        )[:, :, 0]
                        decisions = selected >= cell["critical"]
                        cell["decisions"][start:start+count] = decisions
                        ratios = selected_scales / moments.target
                        cell["ratio_sum"] += ratios.sum(axis=0)
                        cell["ratio_squares"] += (ratios**2).sum(axis=0)
                        for row in range(count):
                            record = {
                                "replicate": start+row, "root_seed": protocol["root_seed"],
                                "group": identifier, "k": columns, "rho": rho, "critical": cell["critical"],
                            }
                            for method_index, method in enumerate(METHODS):
                                record.update({
                                    f"{method}_statistic": float(selected[row, method_index]),
                                    f"{method}_reject": int(decisions[row, method_index]),
                                    f"{method}_winner": int(winners[row, method_index]),
                                    f"{method}_variance_ratio": float(ratios[row, method_index]),
                                    f"{method}_fitted_phi": float(selected_phi[row, method_index]),
                                })
                            cell["writer"].writerow(record)
                if start == 0 or (start + count) % 1000 < batch_size:
                    print(f"  {start+count}/{total} replicates", flush=True)
        finally:
            for cell in cells:
                cell["stream"].close()
                cell["raw_file"].close()
        for cell in cells:
            for index, method in enumerate(METHODS):
                rejected = int(cell["decisions"][:, index].sum())
                rate, lower, upper = _wilson(rejected, total, 0.95)
                mean_ratio = cell["ratio_sum"][index] / total
                summaries.append({
                    "cell": cell["key"], "group": identifier, "n_obs": size, "phi": phi,
                    "k": cell["k"], "rho": cell["rho"], "method": method, "n_mc": total,
                    "reject_count": rejected, "rate": rate, "ci_low": lower, "ci_high": upper,
                    "selected_variance_ratio_mean": mean_ratio,
                })
                if method == "bartlett":
                    continue
                raw_decision = cell["decisions"][:, 1]
                other = cell["decisions"][:, index]
                plus = int(np.count_nonzero(other & ~raw_decision))
                minus = int(np.count_nonzero(raw_decision & ~other))
                difference = (plus-minus) / total
                se = math.sqrt(max(0, (plus+minus-total*difference**2)/(total-1)/total))
                paired_rows.append({
                    "cell": cell["key"], "method": method, "reference": "bartlett", "n_mc": total,
                    "only_method_rejects": plus, "only_reference_rejects": minus,
                    "risk_difference": difference, "ci_low_normal": difference-1.96*se,
                    "ci_high_normal": difference+1.96*se,
                    "mcnemar_p_unadjusted": float(binomtest(plus, plus+minus, 0.5).pvalue) if plus+minus else 1.0,
                })
    for filename, rows in (
        ("summary.csv", summaries), ("paired.csv", paired_rows), ("moments.csv", moments_rows)
    ):
        with (output / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if any(_sha(path) != metadata["source_hashes"][str(path.relative_to(ROOT))] for path in source_paths):
        raise RuntimeError("Calculation source changed during the run; evidence remains incomplete.")
    metadata.update({"status": "complete", "elapsed_seconds": time.perf_counter()-started})
    metadata["output_hashes"] = {path.name: _sha(path) for path in sorted(output.iterdir()) if path.name != "metadata.json"}
    _write_json(output / "metadata.json", metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("quick", "full"), default="quick")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    arguments = parser.parse_args()
    output = arguments.output or ROOT / "results/research/tail" / arguments.profile
    run(arguments.profile, output, batch_size=arguments.batch_size)


if __name__ == "__main__":
    main()
