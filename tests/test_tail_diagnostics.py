"""Small, paired replay checks for the research-only diagnostic runner."""

import copy
import csv
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import binomtest, norm

from strategy_inference.inference import default_lags, long_run_variance
from strategy_inference.quadratic_reference import gaussian_ar_hac_moments
from strategy_inference.reference import gaussian_ar_mean_variance
from strategy_inference.tail import tail_variance

PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture
def runner():
    source = PROJECT / "scripts/tail_diagnostics.py"
    specification = importlib.util.spec_from_file_location("tail_diagnostics_review", source)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


@pytest.fixture
def tiny_runner(runner, tmp_path, monkeypatch):
    # Hash checks operate on real temporary copies; production sources and the
    # formal protocol are never rewritten by these tests.
    temporary_root = tmp_path / "snapshot"
    source = temporary_root / "scripts/tail_diagnostics.py"
    source.parent.mkdir(parents=True)
    source.write_bytes((PROJECT / "scripts/tail_diagnostics.py").read_bytes())
    package = temporary_root / "src/strategy_inference"
    package.mkdir(parents=True)
    for name in (
        "tail.py",
        "quadratic_reference.py",
        "reference.py",
        "inference.py",
        "_validation.py",
        "experiments.py",
    ):
        (package / name).write_bytes((PROJECT / "src/strategy_inference" / name).read_bytes())
    protocol = copy.deepcopy(
        json.loads((PROJECT / "experiments/tail-diagnostic-protocol.json").read_text())
    )
    protocol.update(
        profiles={"quick": 19, "full": 19},
        root_seed=8721,
        quick_seed=8722,
        groups=[
            {"id": 91, "n_obs": 24, "phi": 0.7, "k": [1, 3], "rho": [0, 0.35, 1]},
            {"id": 92, "n_obs": 16, "phi": -0.4, "k": [2], "rho": [0]},
        ],
    )
    protocol_path = temporary_root / "experiments/tail-diagnostic-protocol.json"
    protocol_path.parent.mkdir()
    protocol_path.write_text(json.dumps(protocol))
    monkeypatch.setattr(runner, "ROOT", temporary_root)
    monkeypatch.setattr(runner, "__file__", str(source))

    def known_revision(arguments, **kwargs):
        assert arguments == ["git", "rev-parse", "HEAD"]
        assert kwargs["cwd"] == temporary_root
        return SimpleNamespace(stdout="a" * 40 + "\n")

    monkeypatch.setattr(runner.subprocess, "run", known_revision)
    return runner, protocol, tmp_path / "output"


def _read_csv(path):
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _production_scales(data, phi, lags, moments):
    raw = long_run_variance(data, lags)
    fitted = tail_variance(data, lags=lags)
    scales = np.stack(
        [
            np.full(data.shape[1], moments.target),
            raw,
            tail_variance(data, phi=phi, lags=lags).variance,
            fitted.variance,
            tail_variance(data, phi=phi, lags=lags, target="finite_sample").variance,
            tail_variance(data, lags=lags, target="finite_sample").variance,
            raw * moments.target / moments.expectation,
        ]
    )
    return scales, fitted.phi


@pytest.mark.parametrize("phi", [-0.8, 0, 0.97])
def test_components_use_stationary_initial_draw_and_addressed_innovations(runner, phi):
    seed, start, count = 4891, 7, 4
    group = {"id": 41, "n_obs": 16, "phi": phi, "k": [1, 3]}
    observed = runner._components(seed, group, start, count)
    for row, replicate in enumerate(range(start, start + count)):
        independent = np.random.default_rng(
            np.random.SeedSequence([seed, 21, group["id"], replicate, 0])
        ).standard_normal((16, 4))
        # X_0 has marginal variance one, while later innovations have 1-phi^2.
        np.testing.assert_array_equal(observed[row, 0], independent[0])
        residual = observed[row, 1:] - phi * observed[row, :-1]
        np.testing.assert_allclose(
            residual, math.sqrt(1 - phi**2) * independent[1:], rtol=2e-14, atol=2e-15
        )


def test_replicate_addresses_are_independent_of_batch_partition(runner):
    group = {"id": 42, "n_obs": 24, "phi": 0.7, "k": [1, 3]}
    observed = runner._components(992, group, 0, 11)
    for batch_size in [1, 4, 6]:
        chunks = [
            runner._components(992, group, start, min(batch_size, 11 - start))
            for start in range(0, 11, batch_size)
        ]
        np.testing.assert_array_equal(np.concatenate(chunks), observed)
    np.testing.assert_array_equal(runner._components(992, group, 8, 1)[0], observed[8])
    assert not np.array_equal(observed, runner._components(993, group, 0, 11))
    assert not np.array_equal(observed, runner._components(992, {**group, "id": 43}, 0, 11))


@pytest.mark.parametrize("phi", [-0.6, 0, 0.95])
@pytest.mark.parametrize("lags", [0, 3])
def test_batch_scales_match_production_estimates_for_every_draw(runner, phi, lags):
    group = {"id": 43, "n_obs": 32, "phi": phi, "k": [3]}
    components = runner._components(4908, group, 0, 9)
    data = math.sqrt(0.35) * components[:, :, :1] + math.sqrt(0.65) * components[:, :, 1:]
    moments = gaussian_ar_hac_moments(32, phi, lags=lags)
    statistics, scales, fitted = runner._batch_scales(data, lags, phi, moments)
    for row in range(len(data)):
        independent_scales, independent_fitted = _production_scales(data[row], phi, lags, moments)
        np.testing.assert_allclose(scales[row], independent_scales, rtol=1e-13, atol=1e-14)
        np.testing.assert_allclose(fitted[row], independent_fitted, rtol=1e-13, atol=1e-14)
        independent_statistic = data[row].mean(axis=0) / np.sqrt(independent_scales / 32)
        np.testing.assert_allclose(statistics[row], independent_statistic, rtol=1e-13, atol=1e-14)
        # target is T*Var(mean): using Var(mean) here would inflate Z by sqrt(T).
        np.testing.assert_allclose(scales[row, 0], 32 * gaussian_ar_mean_variance(32, phi))
    pieces = [
        runner._batch_scales(data[first : first + 4], lags, phi, moments)
        for first in range(0, 9, 4)
    ]
    for index, original in enumerate((statistics, scales, fitted)):
        np.testing.assert_array_equal(np.concatenate([part[index] for part in pieces]), original)


def test_small_run_replays_nested_k_rho_and_all_method_logs(tiny_runner):
    runner, protocol, output = tiny_runner
    metadata = runner.run("quick", output, batch_size=5)
    assert metadata["status"] == "complete"
    assert metadata["root_seed"] == protocol["quick_seed"]
    groups = {group["id"]: group for group in protocol["groups"]}
    cached = {
        identifier: runner._components(protocol["quick_seed"], group, 0, 19)
        for identifier, group in groups.items()
    }
    for path in output.glob("*.csv.gz"):
        records = _read_csv(path)
        assert [int(record["replicate"]) for record in records] == list(range(19))
        first = records[0]
        group = groups[int(first["group"])]
        columns, rho = int(first["k"]), float(first["rho"])
        if columns == 1 or rho == 1:
            assert float(first["critical"]) == pytest.approx(norm.ppf(1 - protocol["alpha"]))
        lags = default_lags(group["n_obs"])
        moments = gaussian_ar_hac_moments(group["n_obs"], group["phi"], lags=lags)
        components = cached[group["id"]]
        data = math.sqrt(rho) * components[:, :, :1] + math.sqrt(1 - rho) * components[:, :, 1:]
        for replicate, record in enumerate(records):
            # Every rho and nested K is reconstructed from the same component
            # replicate, independently of the runner's recorded winner.
            scales, fitted = _production_scales(data[replicate], group["phi"], lags, moments)
            scores = data[replicate].mean(axis=0) / np.sqrt(scales / group["n_obs"])
            assert int(record["root_seed"]) == protocol["quick_seed"]
            for index, method in enumerate(runner.METHODS):
                winner = int(np.argmax(scores[index, :columns]))
                expected_score = scores[index, winner]
                assert int(record[f"{method}_winner"]) == winner
                assert float(record[f"{method}_statistic"]) == pytest.approx(
                    expected_score, rel=1e-13, abs=1e-14
                )
                assert int(record[f"{method}_reject"]) == int(
                    expected_score >= float(record["critical"])
                )
                assert float(record[f"{method}_variance_ratio"]) == pytest.approx(
                    scales[index, winner] / moments.target, rel=1e-13
                )
                assert float(record[f"{method}_fitted_phi"]) == pytest.approx(
                    fitted[winner], rel=1e-13, abs=1e-14
                )
                assert float(record[f"{method}_column_mean_ratio"]) == pytest.approx(
                    scales[index, :columns].mean() / moments.target, rel=1e-13
                )
    for name, expected_hash in metadata["output_hashes"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected_hash
    assert set(metadata["output_hashes"]) == {path.name for path in output.iterdir()} - {
        "metadata.json"
    }
    for name, expected_hash in metadata["source_hashes"].items():
        assert hashlib.sha256((runner.ROOT / name).read_bytes()).hexdigest() == expected_hash

    for record in _read_csv(output / "moments.csv"):
        size, phi, lags = int(record["n_obs"]), float(record["phi"]), int(record["lags"])
        distances = np.abs(np.arange(size)[:, None] - np.arange(size)[None, :])
        covariance = phi**distances
        centering = np.eye(size) - np.ones((size, size)) / size
        weights = np.maximum(0, 1 - distances / (lags + 1))
        product = (centering @ weights @ centering / size) @ covariance
        assert float(record["target"]) == pytest.approx(covariance.sum() / size, rel=1e-13)
        assert float(record["expectation"]) == pytest.approx(np.trace(product), rel=1e-13)
        assert float(record["variance"]) == pytest.approx(
            2 * np.trace(product @ product), rel=1e-13
        )


def test_summaries_and_paired_uncertainty_recompute_from_saved_records(tiny_runner):
    runner, _, output = tiny_runner
    runner.run("quick", output, batch_size=5)
    logs = {path.name.removesuffix(".csv.gz"): _read_csv(path) for path in output.glob("*.csv.gz")}
    for row in _read_csv(output / "summary.csv"):
        records, method, total = logs[row["cell"]], row["method"], int(row["n_mc"])
        count = sum(int(record[f"{method}_reject"]) for record in records)
        rate, z = count / total, norm.ppf(0.975)
        denominator = 1 + z**2 / total
        midpoint = (rate + z**2 / (2 * total)) / denominator
        radius = z / denominator * math.sqrt(rate * (1 - rate) / total + z**2 / (4 * total**2))
        assert int(row["reject_count"]) == count
        assert float(row["rate"]) == pytest.approx(rate)
        assert float(row["ci_low"]) == pytest.approx(midpoint - radius, abs=2e-16)
        assert float(row["ci_high"]) == pytest.approx(midpoint + radius, abs=2e-16)
        for source, summary in [
            ("variance_ratio", "selected_variance_ratio_mean"),
            ("column_mean_ratio", "column_mean_variance_ratio"),
        ]:
            expected = math.fsum(float(record[f"{method}_{source}"]) for record in records) / total
            assert float(row[summary]) == pytest.approx(expected, rel=2e-15)
    for row in _read_csv(output / "paired.csv"):
        records, method = logs[row["cell"]], row["method"]
        differences = np.array(
            [int(record[f"{method}_reject"]) - int(record["bartlett_reject"]) for record in records]
        )
        plus, minus = int((differences == 1).sum()), int((differences == -1).sum())
        assert row["reference"] == "bartlett"
        assert int(row["only_method_rejects"]) == plus
        assert int(row["only_reference_rejects"]) == minus
        difference = differences.mean()
        standard_error = differences.std(ddof=1) / math.sqrt(len(differences))
        assert float(row["risk_difference"]) == pytest.approx(difference)
        assert float(row["ci_low_normal"]) == pytest.approx(difference - 1.96 * standard_error)
        assert float(row["ci_high_normal"]) == pytest.approx(difference + 1.96 * standard_error)
        pvalue = binomtest(plus, plus + minus).pvalue if plus + minus else 1
        assert float(row["mcnemar_p_unadjusted"]) == pytest.approx(pvalue)


def test_gzip_and_recorded_results_are_deterministic_across_batch_sizes(tiny_runner):
    runner, _, first = tiny_runner
    second = first.with_name("other-output")
    runner.run("quick", first, batch_size=1)
    runner.run("quick", second, batch_size=6)
    for path in first.iterdir():
        if path.name == "metadata.json":
            continue
        assert path.read_bytes() == (second / path.name).read_bytes(), path.name
        if path.suffix == ".gz":
            assert path.read_bytes()[4:8] == b"\0\0\0\0"  # mtime is fixed, not wall time.


def test_quick_and_full_use_disjoint_root_seeds(tiny_runner):
    runner, protocol, quick = tiny_runner
    full = quick.with_name("full-output")
    quick_metadata = runner.run("quick", quick, batch_size=6)
    full_metadata = runner.run("full", full, batch_size=6)
    assert quick_metadata["root_seed"] == protocol["quick_seed"]
    assert full_metadata["root_seed"] == protocol["root_seed"]
    first_quick = next(quick.glob("*.csv.gz"))
    assert first_quick.read_bytes() != (full / first_quick.name).read_bytes()
    assert all(
        int(row["root_seed"]) == protocol["root_seed"] for row in _read_csv(full / first_quick.name)
    )


def test_completed_evidence_is_not_overwritten(tiny_runner, monkeypatch):
    runner, _, output = tiny_runner
    runner.run("quick", output, batch_size=5)
    previous = {path.name: path.read_bytes() for path in output.iterdir()}

    def forbidden_draw(*args, **kwargs):
        raise AssertionError("The overwrite guard must run before simulation.")

    monkeypatch.setattr(runner, "_components", forbidden_draw)
    with pytest.raises(ValueError, match="not overwritten"):
        runner.run("quick", output, batch_size=5)
    assert previous == {path.name: path.read_bytes() for path in output.iterdir()}


def test_source_mutation_preserves_incomplete_evidence(tiny_runner, monkeypatch):
    runner, _, output = tiny_runner
    original = runner._batch_scales
    changed = False

    def mutate_snapshot(*args, **kwargs):
        nonlocal changed
        answer = original(*args, **kwargs)
        if not changed:
            source = runner.ROOT / "src/strategy_inference/tail.py"
            source.write_bytes(source.read_bytes() + b"\n# Deliberate test mutation.\n")
            changed = True
        return answer

    monkeypatch.setattr(runner, "_batch_scales", mutate_snapshot)
    with pytest.raises(RuntimeError, match="source changed"):
        runner.run("quick", output, batch_size=5)
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["status"] != "complete"
    assert "output_hashes" not in metadata


def test_invalid_scale_aborts_without_discarding_replicates(tiny_runner, monkeypatch):
    runner, _, output = tiny_runner
    original = runner._components

    def constant_component(*args, **kwargs):
        return np.zeros_like(original(*args, **kwargs))

    monkeypatch.setattr(runner, "_components", constant_component)
    with np.errstate(divide="ignore", invalid="ignore"), pytest.raises(ValueError):
        runner.run("quick", output, batch_size=5)
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["status"] != "complete"
    assert "output_hashes" not in metadata
    assert not (output / "summary.csv").exists()
