"""Benchmark-specific protocol, causal selection and independent-reference checks."""

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import norm


@pytest.fixture
def benchmark():
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "forecast_validation.py"
    spec = importlib.util.spec_from_file_location("forecast_validation_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("n,phi", [(8, 0.0), (17, 0.6), (256, 0.9)])
def test_oracle_variance_matches_full_stationary_covariance(benchmark, n, phi):
    distances = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    covariance = phi**distances
    expected = np.ones(n) @ covariance @ np.ones(n) / n**2
    assert benchmark.mean_variance(n, phi) == pytest.approx(expected, rel=2e-15)


@pytest.mark.parametrize("successes", [0, 2, 64, 128])
def test_wilson_intervals_match_statsmodels_including_boundaries(benchmark, successes):
    proportions = pytest.importorskip("statsmodels.stats.proportion")
    expected = proportions.proportion_confint(successes, 128, alpha=0.05, method="wilson")
    actual = benchmark.wilson([True] * successes + [False] * (128 - successes))
    np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=2e-16)
    with pytest.raises(ValueError):
        benchmark.wilson([])


def test_simulation_preserves_loss_target_pairing_seeds_and_oracle(benchmark, monkeypatch):
    protocol = copy.deepcopy(benchmark.PROTOCOL)
    protocol["simulation"].update(n_observations=32, replications={"quick": 3, "full": 3})
    monkeypatch.setattr(benchmark, "PROTOCOL", protocol)
    observed = []

    def compare(run, **options):
        losses = (run.actuals[:, :, None] - run.forecasts) ** 2
        difference = losses[:, 0, 0] - losses[:, 0, 1]
        np.testing.assert_allclose(difference, -run.actuals[:, 0], rtol=2e-15, atol=2e-15)
        assert run.n_folds == 32 and options["loss"] == "squared"
        assert options["baseline"] == "plus" and options["lead_time"] == 1
        observed.append((run.actuals[:, 0].copy(), options["seed"]))
        return SimpleNamespace(inference=SimpleNamespace(global_reject=difference.mean() > 0.1))

    monkeypatch.setattr(benchmark, "compare_forecasts", compare)
    cells = benchmark.simulation()
    assert len(cells) == 4
    for before, after in zip(observed[::2], observed[1::2], strict=True):
        np.testing.assert_allclose(after[0] - before[0], -0.15, rtol=0, atol=5e-16)
        assert before[1] == after[1]
    for cell in cells:
        bits = cell["decisions"]
        differences = np.asarray(bits["bootstrap"], float) - np.asarray(bits["oracle"], float)
        assert cell["paired_rate_difference"] == differences.mean()
        assert cell["paired_mcse"] == pytest.approx(np.std(differences, ddof=1) / np.sqrt(3))
        offset = protocol["simulation"]["phi"].index(cell["phi"]) * 6
        selected = observed[offset + (0 if cell["delta"] == 0 else 1) : offset + 6 : 2]
        expected = [bool(-values.mean() / np.sqrt(cell["oracle_mean_variance"]) > norm.isf(0.05))
                    for values, _ in selected]
        assert bits["oracle"] == expected
    json.dumps(cells, allow_nan=False)


def test_short_simulation_runs_actual_forecast_comparison(benchmark, monkeypatch):
    protocol = copy.deepcopy(benchmark.PROTOCOL)
    protocol["simulation"].update(n_observations=32, replications={"quick": 2, "full": 2})
    protocol["inference"]["n_resamples"] = 9
    monkeypatch.setattr(benchmark, "PROTOCOL", protocol)
    first, repeated = benchmark.simulation(), benchmark.simulation()
    assert first == repeated
    assert all(len(cell["decisions"]["bootstrap"]) == 2 for cell in first)


def test_test_labels_never_select_model_and_reference_histories_end_before_target(benchmark, monkeypatch):
    values = np.random.default_rng(290).normal(size=728).cumsum()
    reference_histories, comparisons = [], []

    def reference(histories, lags, options):
        reference_histories.append([row.copy() for row in histories])
        return {"passed": True}

    def compare(run, **options):
        comparisons.append(run)
        assert options["baseline"] == "naive" and options["loss"] == "absolute"
        return SimpleNamespace(to_dict=lambda: {"sample_size": run.n_folds})

    monkeypatch.setattr(benchmark, "_reference_ar", reference)
    monkeypatch.setattr(benchmark, "compare_forecasts", compare)
    first = benchmark._panel_case({"T2": SimpleNamespace(values=values)}, "T2")
    changed = values.copy()
    changed[first["test"][0] :] += 1000
    second = benchmark._panel_case({"T2": SimpleNamespace(values=changed)}, "T2")
    assert first["selection"] == second["selection"]
    assert first["scores"] != second["scores"]
    assert first["train"] == [0, 509] and first["validation"] == [509, 618]
    assert first["test"] == [618, 728] and first["n_test_targets"] == 110
    assert len(first["configurations"]) == 4
    assert [row["model"] for row in first["scores"]] == ["naive", "seasonal", "drift", "selected_ar"]
    for histories in reference_histories[:2]:
        for history, target in zip(histories, range(618, 728), strict=True):
            np.testing.assert_array_equal(history, values[target - 120 : target])
    np.testing.assert_array_equal(comparisons[0].target_indices[:, 0], np.arange(618, 728))
    assert first["mase_training_denominator"] == second["mase_training_denominator"]


@pytest.mark.parametrize("problem", ["length", "missing"])
def test_planned_series_are_not_dropped_or_imputed(benchmark, problem):
    values = np.ones(728)
    if problem == "length":
        values = values[:-1]
    else:
        values[650] = np.nan
    with pytest.raises(ValueError, match="no series replaced"):
        benchmark._panel_case({"T2": SimpleNamespace(values=values)}, "T2")


@pytest.mark.parametrize("lags", [12, (1, 2, 12)])
def test_ols_forecasts_match_external_dense_and_sparse_reference(benchmark, lags):
    pytest.importorskip("statsmodels")
    values = np.random.default_rng(770).normal(size=100)
    histories = [values[:80], values[10:90]]
    result = benchmark._reference_ar(histories, lags, dict(repetitions=2, rtol=1e-10, atol=1e-10))
    assert result["passed"] and result["max_absolute_difference"] < 1e-10
    assert all(len(times) == 2 for times in result["timing_seconds"].values())


def test_dirty_protocol_blocks_execution(benchmark, monkeypatch):
    monkeypatch.setattr(benchmark.subprocess, "check_output", lambda *args, **kwargs: " M protocol")
    with pytest.raises(RuntimeError, match="Commit the runner and protocol"):
        benchmark._source()


def test_existing_output_is_rejected_before_benchmark_runs(benchmark, tmp_path, monkeypatch):
    destination = tmp_path / "saved.json"
    destination.write_text("original", encoding="utf-8")
    monkeypatch.setattr(benchmark, "_source", lambda: pytest.fail("Must not start benchmark"))
    with pytest.raises(SystemExit) as error:
        benchmark.main(["--output", str(destination)])
    assert error.value.code == 2 and destination.read_text() == "original"
