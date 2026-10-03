"""Independent checks for the research runner; formal evidence is never changed."""

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
from scipy.stats import beta, binomtest, norm, t

from strategy_inference.inference import default_lags
from strategy_inference.parametric import fit_equicorrelated_ar1, known_phi_gls_t, tail_statistics
from strategy_inference.reference import equicorrelated_max_quantile, equicorrelated_max_tail
from strategy_inference.tail import tail_variance

PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture
def runner():
    source = PROJECT / "scripts/parametric_replay.py"
    specification = importlib.util.spec_from_file_location("parametric_replay_review", source)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


@pytest.fixture
def group():
    return {"id": 1, "n_obs": 16, "k": 3, "phi_factor": 0.7, "phi_idio": 0.7, "rho": 0.35}


@pytest.fixture
def tiny_runner(runner, tmp_path, monkeypatch):
    # Source mutation tests use copies and never touch the live repository.
    snapshot = tmp_path / "snapshot"
    source = snapshot / "scripts/parametric_replay.py"
    source.parent.mkdir(parents=True)
    source.write_bytes((PROJECT / "scripts/parametric_replay.py").read_bytes())
    package = snapshot / "src/strategy_inference"
    package.mkdir(parents=True)
    for name in ("parametric.py", "tail.py", "reference.py", "inference.py", "_validation.py", "experiments.py"):
        (package / name).write_bytes((PROJECT / "src/strategy_inference" / name).read_bytes())
    protocol = copy.deepcopy(json.loads((PROJECT / "experiments/parametric-replay-protocol.json").read_text()))
    settings = {
        "null_replicates": 5, "calibration_replicates": 24, "power_replicates": 5,
        "inner_draws": 19, "calibration_seed": 8351, "null_seed": 8352, "power_seed": 8353,
    }
    protocol["profiles"] = {
        "quick": settings,
        "full": {**settings, "calibration_seed": 8451, "null_seed": 8452, "power_seed": 8453},
    }
    protocol["groups"] = [
        {"id": 1, "n_obs": 16, "k": 2, "phi_factor": 0.6, "phi_idio": 0.6, "rho": 0.35},
        {"id": 7, "n_obs": 16, "k": 2, "phi_factor": 0.8, "phi_idio": 0.1, "rho": 0.35},
    ]
    protocol["power"]["groups"] = [1, 7]
    protocol["power"]["standardized_mean_shifts"] = [1.0, 2.0]
    protocol_path = snapshot / "experiments/parametric-replay-protocol.json"
    protocol_path.parent.mkdir()
    protocol_path.write_text(json.dumps(protocol))
    monkeypatch.setattr(runner, "ROOT", snapshot)
    monkeypatch.setattr(runner, "__file__", str(source))

    def known_revision(arguments, **kwargs):
        assert arguments == ["git", "rev-parse", "HEAD"]
        assert kwargs["cwd"] == snapshot
        return SimpleNamespace(stdout="a" * 40 + "\n")

    monkeypatch.setattr(runner.subprocess, "run", known_revision)
    return runner, protocol, tmp_path / "output"


def _read_csv(path):
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _independent_filter(normal, phi_factor, phi_idio):
    result = np.empty_like(normal)
    result[:, 0] = normal[:, 0]
    coefficients = np.full(normal.shape[2], phi_idio)
    coefficients[0] = phi_factor
    for time in range(1, normal.shape[1]):
        result[:, time] = coefficients * result[:, time - 1] + np.sqrt(
            (1 - coefficients) * (1 + coefficients)
        ) * normal[:, time]
    return result


def _independent_data(runner, group, seed, phase, replicate):
    normal = np.random.default_rng(
        np.random.SeedSequence([seed, 31, phase, group["id"], replicate, 0])
    ).standard_normal((1, group["n_obs"], group["k"] + 1))
    components = _independent_filter(normal, group["phi_factor"], group["phi_idio"])
    return (
        math.sqrt(group["rho"]) * components[:, :, :1]
        + math.sqrt(1 - group["rho"]) * components[:, :, 1:]
    )[0]


@pytest.mark.parametrize("factor,idio", [(0.0, 0.0), (0.99, 0.99), (-0.8, 0.6)])
def test_component_filter_preserves_stationary_initial_and_heterogeneous_innovations(runner, factor, idio):
    normal = np.random.default_rng(813).normal(size=(7, 16, 4))
    actual = runner._filter(normal, factor, idio)
    np.testing.assert_array_equal(actual[:, 0], normal[:, 0])
    np.testing.assert_allclose(actual, _independent_filter(normal, factor, idio), rtol=0, atol=0)
    np.testing.assert_array_equal(normal, np.random.default_rng(813).normal(size=normal.shape))


@pytest.mark.parametrize("factor,idio,rho", [(0.9, 0.9, 0.35), (0.98, 0.3, 0.35), (-0.7, 0.8, 0.8)])
def test_reference_uses_exact_variance_and_correlation_of_means(runner, factor, idio, rho):
    size = 24
    distances = np.abs(np.arange(size)[:, None] - np.arange(size)[None, :])
    common, independent = factor**distances, idio**distances
    expected = (rho * common + (1 - rho) * independent).sum() / size
    expected_rho = rho * common.sum() / (size * expected)
    actual = runner._reference({"n_obs": size, "rho": rho, "phi_factor": factor, "phi_idio": idio})
    np.testing.assert_allclose(actual, [expected, expected_rho], rtol=2e-14)
    if factor != idio:
        assert not math.isclose(actual[1], rho, rel_tol=0.01)


@pytest.mark.parametrize("fitted_rho", [0.35, 0.6])
def test_inner_batch_partition_is_bitwise_invariant(runner, group, fitted_rho):
    group = {**group, "phi_idio": -0.2}
    fit = SimpleNamespace(phi=0.4, rho=fitted_rho)
    factors = np.array([0.7, 1.5, 3.0])
    original = runner._inner(group, fit, factors, np.random.default_rng(8631), 11, 11)
    for batch in [1, 4, 7]:
        actual = runner._inner(group, fit, factors, np.random.default_rng(8631), 11, batch)
        for name in runner.REPLAYS:
            np.testing.assert_array_equal(actual[name], original[name])


@pytest.mark.parametrize("fitted_rho", [0.35, 0.6])
def test_every_replay_and_frozen_reuse_matches_direct_statistic_recomputation(runner, group, fitted_rho):
    group = {**group, "phi_idio": -0.2}
    fit = SimpleNamespace(phi=-0.4, rho=fitted_rho)
    factors = np.array([0.7, 1.5, 3.0])
    draws = 9
    normal = np.random.default_rng(8632).standard_normal((draws, group["n_obs"], group["k"] + 1))
    true = _independent_filter(normal, group["phi_factor"], group["phi_idio"])
    fitted = _independent_filter(normal, fit.phi, fit.phi)
    data = {
        "known": runner._mix(true, group["rho"]),
        "phi_fitted": runner._mix(fitted, group["rho"]),
        "rho_fitted": runner._mix(true, fit.rho),
        "fitted": runner._mix(fitted, fit.rho),
        "frozen": runner._mix(fitted, fit.rho),
    }
    actual = runner._inner(group, fit, factors, np.random.default_rng(8632), draws, 4)
    for name in runner.REPLAYS:
        statistics, _, _ = tail_statistics(
            data[name], default_lags(group["n_obs"]), factors if name == "frozen" else None
        )
        np.testing.assert_allclose(actual[name], statistics.max(axis=1), rtol=2e-14, atol=2e-15)


def test_power_reuses_centered_fits_and_null_draws_but_recomputes_the_winner(runner, group, monkeypatch):
    original = runner._inner
    calls = []

    def count_inner(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, "_inner", count_inner)
    seed, phase, replicate = 9751, 2, 3
    deltas = [0.0, 1.0, 5.0]
    records, maxima = runner._records(group, seed, phase, replicate, 19, 4, 0.05, deltas)
    assert len(calls) == 1
    data = _independent_data(runner, group, seed, phase, replicate)
    target, mean_rho = runner._reference(group)
    lags = default_lags(group["n_obs"])
    fit = fit_equicorrelated_ar1(data)
    for delta in deltas:
        shifted = data.copy()
        shifted[:, 0] += delta * math.sqrt(target / len(data))
        reference = tail_variance(shifted, lags=lags, target="finite_sample")
        scores = shifted.mean(axis=0) / np.sqrt(reference.variance / len(data))
        winner, maximum = int(scores.argmax()), float(scores.max())
        shifted_fit = fit_equicorrelated_ar1(shifted)
        assert shifted_fit.phi == pytest.approx(fit.phi, rel=2e-14, abs=2e-15)
        assert shifted_fit.rho == pytest.approx(fit.rho, rel=2e-14, abs=2e-15)
        for row in [entry for entry in records if entry["delta"] == delta]:
            method = row["method"]
            if method == "oracle_gaussian":
                oracle = shifted.mean(axis=0) * math.sqrt(len(data) / target)
                assert row["statistic"] == pytest.approx(oracle.max(), rel=2e-14, abs=2e-15)
                assert row["winner"] == int(oracle.argmax())
                assert row["pvalue"] == pytest.approx(equicorrelated_max_tail(oracle.max(), group["k"], mean_rho))
            elif method == "gls_known":
                values = known_phi_gls_t(shifted, group["phi_factor"])
                assert row["statistic"] == pytest.approx(values.max(), rel=2e-14, abs=2e-15)
                assert row["winner"] == int(values.argmax())
                assert row["pvalue"] == pytest.approx(min(1.0, group["k"] * t.sf(values.max(), len(data) - 1)))
            else:
                assert row["statistic"] == pytest.approx(maximum, rel=2e-14, abs=2e-15)
                assert row["winner"] == winner
                assert row["selected_phi"] == pytest.approx(reference.phi[winner], rel=2e-14, abs=2e-15)
                assert row["selected_variance_ratio"] == pytest.approx(reference.variance[winner] / target, rel=2e-14)
                if method.startswith("replay_"):
                    name = method.removeprefix("replay_")
                    count = np.count_nonzero(maxima[name] >= maximum)
                    assert row["exceedances"] == count
                    assert row["pvalue"] == (count + 1) / 20
                else:
                    rho = mean_rho if method == "gaussian_known" else fit.rho
                    assert row["pvalue"] == pytest.approx(equicorrelated_max_tail(maximum, group["k"], rho))
                    critical = equicorrelated_max_quantile(0.95, group["k"], rho)
                    assert row["score"] == pytest.approx(maximum / critical, rel=2e-14)
        assert all(row["phi_fit"] == fit.phi for row in records)


@pytest.mark.parametrize("size,alpha", [(19, 0.05), (24, 0.05), (2000, 0.05), (24, 0.2)])
def test_calibration_order_statistic_and_beta_law_parameters(runner, size, alpha):
    scores = np.random.default_rng(815).uniform(size=size)
    actual = runner._calibration(scores.tolist(), alpha)
    rank = math.ceil((1 - alpha) * (size + 1))
    tail = size + 1 - rank
    assert actual["rank"] == rank
    assert actual["cutoff"] == np.sort(scores)[rank - 1]
    assert actual["unconditional_size"] == tail / (size + 1)
    np.testing.assert_array_equal(actual["conditional_size_interval"], beta.ppf([0.025, 0.975], tail, rank))


def test_calibration_refuses_inadequate_null_sample(runner):
    with pytest.raises(ValueError, match="Too few calibration"):
        runner._calibration(list(range(18)), 0.05)


def test_protocol_stages_and_development_have_distinct_seed_addresses(runner):
    protocol = json.loads((PROJECT / "experiments/parametric-replay-protocol.json").read_text())
    seeds = [
        settings[name]
        for settings in protocol["profiles"].values()
        for name in ("calibration_seed", "null_seed", "power_seed")
    ]
    assert len(seeds) == len(set(seeds))
    old = json.loads((PROJECT / "experiments/tail-diagnostic-protocol.json").read_text())
    assert not set(seeds).intersection({old["root_seed"], old["quick_seed"]})
    reference = runner._rng(9021, 1, 4, 3, 0).normal(size=20)
    np.testing.assert_array_equal(reference, runner._rng(9021, 1, 4, 3, 0).normal(size=20))
    for arguments in [(9022, 1, 4, 3, 0), (9021, 2, 4, 3, 0), (9021, 1, 5, 3, 0),
                      (9021, 1, 4, 4, 0), (9021, 1, 4, 3, 1)]:
        assert not np.array_equal(reference, runner._rng(*arguments).normal(size=20))


def test_tiny_run_records_hashes_mc_counts_and_omits_inapplicable_gls(tiny_runner):
    runner, protocol, output = tiny_runner
    metadata = runner.run("quick", output, batch_size=4)
    assert metadata["status"] == "complete"
    assert metadata["git_revision"] == "a" * 40
    assert metadata["settings"] == protocol["profiles"]["quick"]
    assert len(metadata["cells"]) == 6
    groups = {group["id"]: group for group in protocol["groups"]}
    total_records = 0
    for cell in metadata["cells"]:
        phase, identifier = cell["phase"], cell["group"]
        group = groups[identifier]
        records = _read_csv(output / f"{cell['key']}.csv.gz")
        total_records += len(records)
        assert len(records) == cell["records"]
        assert set(records[0]) == set(runner.FIELDS)
        settings = protocol["profiles"]["quick"]
        seed = settings[["calibration_seed", "null_seed", "power_seed"][phase]]
        assert all(int(row["seed"]) == seed and int(row["phase"]) == phase for row in records)
        assert all(int(row["group"]) == identifier for row in records)
        gls = group["phi_factor"] == group["phi_idio"]
        assert ("gls_known" in {row["method"] for row in records}) == gls
        assert {int(row["replicate"]) for row in records} == set(range(cell["n"]))
        for row in records:
            assert int(row["reject"]) == int(float(row["pvalue"]) <= protocol["alpha"])
            if row["method"].startswith("replay_"):
                exceedances = int(row["exceedances"])
                assert 0 <= exceedances <= settings["inner_draws"]
                assert float(row["pvalue"]) == (exceedances + 1) / (settings["inner_draws"] + 1)
                width = 1 / (settings["inner_draws"] + 1)
                jittered = float(row["score"]) - (1 - float(row["pvalue"]))
                assert 0 <= jittered < width
            if phase == 0:
                assert row["matched_reject"] == ""
        expected_target, expected_rho = runner._reference(group)
        assert cell["target"] == expected_target
        assert cell["correlation_of_means"] == expected_rho
    assert metadata["records"] == total_records
    snapshots = json.loads((output / "inner-snapshots.json").read_text())
    for cell in metadata["cells"]:
        path = output / f"{cell['key']}.csv.gz"
        records = _read_csv(path)
        for replicate in {0, cell["n"] // 2, cell["n"] - 1}:
            simulated = snapshots[f"{cell['key']}-r{replicate:05d}"]
            assert set(simulated) == set(runner.REPLAYS)
            for name, values in simulated.items():
                assert len(values) == protocol["profiles"]["quick"]["inner_draws"]
                rows = [
                    row for row in records
                    if int(row["replicate"]) == replicate and row["method"] == f"replay_{name}"
                ]
                for row in rows:
                    count = sum(value >= float(row["statistic"]) for value in values)
                    assert count == int(row["exceedances"])
    for name, digest in metadata["source_hashes"].items():
        assert hashlib.sha256((runner.ROOT / name).read_bytes()).hexdigest() == digest
    for name, digest in metadata["output_hashes"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    assert set(metadata["output_hashes"]) == {path.name for path in output.iterdir()} - {"metadata.json"}


def test_saved_summaries_calibration_and_paired_results_recompute_from_logs(tiny_runner):
    runner, protocol, output = tiny_runner
    runner.run("quick", output, batch_size=4)
    logs = {path.name.removesuffix(".csv.gz"): _read_csv(path) for path in output.glob("*.csv.gz")}
    calibration = json.loads((output / "calibration.json").read_text())
    for identifier, methods in calibration.items():
        records = logs[f"p0-g{int(identifier):02d}"]
        for method, result in methods.items():
            scores = [float(row["score"]) for row in records if row["method"] == method]
            expected = runner._calibration(scores, protocol["alpha"])
            assert result == expected
    for key, records in logs.items():
        if key.startswith("p0"):
            continue
        for row in records:
            cutoff = calibration[row["group"]][row["method"]]["cutoff"]
            assert int(row["matched_reject"]) == int(float(row["score"]) > cutoff)
    for row in _read_csv(output / "summary.csv"):
        key = f"p{row['phase']}-g{int(row['group']):02d}"
        selected = [
            value for value in logs[key]
            if value["method"] == row["method"] and float(value["delta"]) == float(row["delta"])
        ]
        decisions = np.array([int(value[row["mode"]]) for value in selected])
        count, size = int(decisions.sum()), len(decisions)
        rate, z = count / size, norm.ppf(0.975)
        denominator = 1 + z**2 / size
        middle = (rate + z**2 / (2 * size)) / denominator
        radius = z / denominator * math.sqrt(rate * (1 - rate) / size + z**2 / (4 * size**2))
        assert int(row["rejections"]) == count
        assert int(row["n"]) == size
        assert float(row["rate"]) == rate
        assert float(row["low"]) == pytest.approx(middle - radius, abs=2e-16)
        assert float(row["high"]) == pytest.approx(middle + radius, abs=2e-16)
        correct = sum(int(value[row["mode"]]) and int(value["winner"]) == 0 for value in selected)
        assert int(row["reject_select_signal"]) == correct
        assert float(row["signal_rate"]) == correct / size
        assessed = (
            row["phase"] == "1" and int(row["group"]) <= 6
            and row["method"] == "replay_fitted" and row["mode"] == "reject"
        )
        if assessed:
            upper = beta.ppf(1 - 0.05 / 6, count + 1, size - count) if count < size else 1
            assert float(row["family_upper"]) == upper
            assert int(row["passes_tolerance"]) == int(upper <= 0.07)
        else:
            assert row["family_upper"] == row["passes_tolerance"] == ""
    paired = _read_csv(output / "paired.csv")
    assert {row["mode"] for row in paired} == {"reject", "matched_reject"}
    for row in paired:
        records = logs[f"p{row['phase']}-g{int(row['group']):02d}"]
        left = [
            int(value[row["mode"]]) for value in records
            if value["method"] == row["method"] and float(value["delta"]) == float(row["delta"])
        ]
        right = [
            int(value[row["mode"]]) for value in records
            if value["method"] == "replay_known" and float(value["delta"]) == float(row["delta"])
        ]
        differences = np.array(left) - right
        plus, minus = int((differences == 1).sum()), int((differences == -1).sum())
        assert row["reference"] == "replay_known"
        assert int(row["method_only"]) == plus
        assert int(row["reference_only"]) == minus
        risk = differences.mean()
        se = differences.std(ddof=1) / math.sqrt(len(differences))
        assert float(row["risk_difference"]) == risk
        assert float(row["low"]) == pytest.approx(risk - norm.ppf(0.975) * se, abs=2e-12)
        assert float(row["high"]) == pytest.approx(risk + norm.ppf(0.975) * se, abs=2e-12)
        expected = binomtest(plus, plus + minus, 0.5).pvalue if plus + minus else 1
        assert float(row["mcnemar_p"]) == expected


def test_tiny_run_evidence_is_byte_identical_across_inner_batch_partitions(tiny_runner):
    runner, _, first = tiny_runner
    second = first.with_name("second-output")
    runner.run("quick", first, batch_size=1)
    runner.run("quick", second, batch_size=7)
    for path in first.iterdir():
        if path.name != "metadata.json":
            assert path.read_bytes() == (second / path.name).read_bytes(), path.name
        if path.suffix == ".gz":
            assert path.read_bytes()[4:8] == b"\0\0\0\0"


def test_existing_evidence_is_rejected_before_generation(tiny_runner, monkeypatch):
    runner, _, output = tiny_runner
    output.mkdir()
    sentinel = output / "metadata.json"
    sentinel.write_text('{"status":"running"}\n')

    def forbidden(*args, **kwargs):
        raise AssertionError("No draw may happen before the overwrite guard.")

    monkeypatch.setattr(runner, "_records", forbidden)
    previous = sentinel.read_bytes()
    with pytest.raises(ValueError, match="never overwritten"):
        runner.run("quick", output)
    assert sentinel.read_bytes() == previous


def test_draw_failure_preserves_running_status_and_refuses_resume(tiny_runner, monkeypatch):
    runner, _, output = tiny_runner

    def invalid(*args, **kwargs):
        raise ValueError("Deliberate invalid draw; no selection of successful draws.")

    monkeypatch.setattr(runner, "_records", invalid)
    with pytest.raises(ValueError, match="Deliberate invalid draw"):
        runner.run("quick", output)
    assert json.loads((output / "metadata.json").read_text())["status"] == "running"
    assert list(output.glob("*.csv.gz"))
    assert not (output / "summary.csv").exists()
    with pytest.raises(ValueError, match="never overwritten"):
        runner.run("quick", output)


def test_source_change_is_detected_before_completion(tiny_runner, monkeypatch):
    runner, _, output = tiny_runner
    original = runner._records
    changed = False

    def mutate_copy(*args, **kwargs):
        nonlocal changed
        result = original(*args, **kwargs)
        if not changed:
            source = runner.ROOT / "src/strategy_inference/parametric.py"
            source.write_bytes(source.read_bytes() + b"\n# Deliberate mutation of a test copy.\n")
            changed = True
        return result

    monkeypatch.setattr(runner, "_records", mutate_copy)
    with pytest.raises(ValueError, match="Source changed during computation"):
        runner.run("quick", output, batch_size=4)
    assert json.loads((output / "metadata.json").read_text())["status"] == "running"
    assert (output / "inner-snapshots.json").exists()


@pytest.mark.parametrize("batch_size", [0, -1])
def test_invalid_batch_size_does_not_create_evidence(tiny_runner, batch_size):
    runner, _, output = tiny_runner
    with pytest.raises(ValueError, match="batch_size must be positive"):
        runner.run("quick", output, batch_size=batch_size)
    assert not output.exists()
