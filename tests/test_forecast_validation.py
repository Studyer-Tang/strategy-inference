"""Benchmark-specific protocol, causal selection and independent-reference checks."""

import copy
import hashlib
import importlib.util
import json
import subprocess
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


@pytest.fixture
def source_checkout(benchmark, tmp_path, monkeypatch):
    """A real Git index exercises staged, unstaged and untracked provenance."""
    runner = tmp_path / "benchmarks" / "forecast_validation.py"
    runner.parent.mkdir()
    runner.write_bytes(Path(benchmark.__file__).read_bytes())
    protocol = runner.with_name("forecast_validation_protocol.json")
    protocol.write_text(json.dumps(benchmark.PROTOCOL), encoding="utf-8")
    core = tmp_path / "src" / "strategy_inference" / "__init__.py"
    core.parent.mkdir(parents=True)
    core.write_text("value = 1\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'fixture'\n", encoding="utf-8")

    def git(*arguments):
        return subprocess.check_output(["git", *arguments], cwd=tmp_path, text=True).strip()

    git("init", "--quiet")
    git("add", ".")
    git("-c", "user.name=Benchmark test", "-c", "user.email=benchmark@example.invalid",
        "commit", "--quiet", "-m", "Committed benchmark inputs")
    monkeypatch.setattr(benchmark, "PROTOCOL_PATH", protocol)
    monkeypatch.setattr(benchmark, "__file__", str(runner))
    monkeypatch.setattr(benchmark.strategy_inference, "__file__", str(core))
    return SimpleNamespace(root=tmp_path, runner=runner, protocol=protocol, core=core, git=git)


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
        expected = [
            bool(-values.mean() / np.sqrt(cell["oracle_mean_variance"]) > norm.isf(0.05))
            for values, _ in selected
        ]
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


def test_test_labels_never_select_model_and_reference_histories_end_before_target(
    benchmark, monkeypatch
):
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
    assert [row["model"] for row in first["scores"]] == [
        "naive",
        "seasonal",
        "drift",
        "selected_ar",
    ]
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


@pytest.mark.parametrize("mode", ["quick", "full"])
@pytest.mark.parametrize("name", ["runner", "protocol"])
def test_dirty_runner_or_protocol_blocks_execution(benchmark, source_checkout, mode, name):
    path = getattr(source_checkout, name)
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(RuntimeError, match="Commit the runner and protocol"):
        benchmark._source(mode)


def test_external_install_is_rejected_before_recording_checkout_hashes(
    benchmark, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        benchmark.strategy_inference, "__file__", str(tmp_path / "other" / "__init__.py")
    )
    monkeypatch.setattr(
        benchmark.subprocess,
        "check_output",
        lambda *args, **kwargs: pytest.fail(
            "External package must be rejected before provenance is recorded"
        ),
    )
    with pytest.raises(RuntimeError, match="this checkout's editable package"):
        benchmark._source()


def test_core_edits_are_visible_even_with_unchanged_commit(benchmark, source_checkout, monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    before = benchmark._source()
    source_checkout.core.write_text("value = 2\n", encoding="utf-8")
    after = benchmark._source()
    assert before["commit"] == after["commit"]
    assert before["core_sha256"] != after["core_sha256"]
    assert list(after["core_sha256"]) == ["src/strategy_inference/__init__.py"]
    assert not before["dirty"] and before["changes"] == []
    assert after["dirty"] and after["run_kind"] == "development"
    assert after["changes"] == [dict(status=" M", path="src/strategy_inference/__init__.py")]
    assert after["input_sha256"]["src/strategy_inference/__init__.py"] == hashlib.sha256(
        source_checkout.core.read_bytes()
    ).hexdigest()
    assert after["thread_environment"]["OMP_NUM_THREADS"] == "1"
    assert after["package_path"] == str(source_checkout.core.parent)
    json.dumps(after, allow_nan=False)


@pytest.mark.parametrize("relative_path,staged", [
    ("src/strategy_inference/__init__.py", False),
    ("src/strategy_inference/__init__.py", True),
    ("src/strategy_inference/new model.py", False),
    ("src/strategy_inference/protocols/new.json", False),
    ("pyproject.toml", False),
    ("requirements-dev.txt", False),
    ("uv.lock", False),
])
def test_full_rejects_changed_runtime_inputs_and_quick_records_them(
    benchmark, source_checkout, relative_path, staged
):
    path = source_checkout.root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("changed input\n", encoding="utf-8")
    if staged:
        source_checkout.git("add", relative_path)
    quick = benchmark._source("quick")
    assert quick["dirty"] and quick["run_kind"] == "development"
    assert [change["path"] for change in quick["changes"]] == [relative_path]
    assert quick["input_sha256"][relative_path] == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(RuntimeError, match="Full validation requires committed") as error:
        benchmark._source("full")
    assert relative_path in str(error.value)


def test_full_records_committed_inputs_and_ignores_unrelated_outputs(benchmark, source_checkout):
    (source_checkout.root / "result.json").write_text("{}", encoding="utf-8")
    source = benchmark._source("full")
    assert source["run_kind"] == "formal" and not source["dirty"]
    assert source["formal_eligible"] and source["index_flags"] == [] and source["symlinks"] == []
    assert source["changes"] == []
    assert source["commit"] == source_checkout.git("rev-parse", "HEAD")
    assert source["protocol_commit"] == source["commit"]
    for path, fingerprint in source["input_sha256"].items():
        committed = subprocess.check_output(
            ["git", "show", f"HEAD:{path}"], cwd=source_checkout.root,
        )
        actual = (source_checkout.root / path).read_bytes()
        assert committed == actual
        assert fingerprint == hashlib.sha256(actual).hexdigest()
    assert set(source["sha256"]) <= set(source["input_sha256"])
    assert set(source["core_sha256"]) <= set(source["input_sha256"])
    assert "pyproject.toml" in source["input_sha256"]
    assert "result.json" not in source["input_sha256"]


@pytest.mark.parametrize("relative_path", ["src/strategy_inference/ignored.py", "uv.lock"])
def test_ignored_source_or_dependencies_cannot_be_reported_as_committed(
    benchmark, source_checkout, relative_path
):
    (source_checkout.root / ".gitignore").write_text(relative_path + "\n", encoding="utf-8")
    source_checkout.git("add", ".gitignore")
    source_checkout.git(
        "-c", "user.name=Benchmark test", "-c", "user.email=benchmark@example.invalid",
        "commit", "--quiet", "-m", "Ignore a local benchmark input",
    )
    path = source_checkout.root / relative_path
    path.write_text("ignored local input\n", encoding="utf-8")
    assert source_checkout.git("status", "--porcelain") == ""
    quick = benchmark._source("quick")
    assert quick["dirty"] and quick["changes"] == [dict(status="!!", path=relative_path)]
    assert relative_path in quick["input_sha256"]
    with pytest.raises(RuntimeError, match="Full validation requires committed"):
        benchmark._source("full")


@pytest.mark.parametrize("flags", [
    ("assume-unchanged",), ("skip-worktree",), ("assume-unchanged", "skip-worktree"),
])
@pytest.mark.parametrize("modified", [False, True])
def test_index_flags_cannot_hide_changes_in_formal_validation(
    benchmark, source_checkout, flags, modified
):
    relative = "src/strategy_inference/__init__.py"
    for flag in flags:
        source_checkout.git("update-index", f"--{flag}", relative)
    if modified:
        source_checkout.core.write_text("value = 2\n", encoding="utf-8")
    assert source_checkout.git("status", "--porcelain") == ""
    quick = benchmark._source("quick")
    assert not quick["dirty"] and quick["changes"] == []
    assert not quick["formal_eligible"] and quick["run_kind"] == "development"
    assert quick["index_flags"] == [dict(
        path=relative, assume_unchanged="assume-unchanged" in flags,
        skip_worktree="skip-worktree" in flags,
    )]
    assert quick["input_sha256"][relative] == hashlib.sha256(
        source_checkout.core.read_bytes()
    ).hexdigest()
    with pytest.raises(RuntimeError, match="Git index flags") as error:
        benchmark._source("full")
    assert relative in str(error.value)


@pytest.mark.parametrize("external", [False, True])
def test_committed_symlinks_are_explicitly_ineligible_for_formal_validation(
    benchmark, source_checkout, external
):
    target = source_checkout.core
    if external:
        target = source_checkout.root.parent / f"{source_checkout.root.name}-external.py"
        target.write_text("external_value = 1\n", encoding="utf-8")
    link = source_checkout.core.with_name("linked.py")
    link.symlink_to(target)
    source_checkout.git("add", "src")
    source_checkout.git(
        "-c", "user.name=Benchmark test", "-c", "user.email=benchmark@example.invalid",
        "commit", "--quiet", "-m", "Committed source symlink",
    )
    assert source_checkout.git("status", "--porcelain") == ""
    relative = "src/strategy_inference/linked.py"
    quick = benchmark._source("quick")
    assert not quick["dirty"] and quick["changes"] == [] and not quick["formal_eligible"]
    assert quick["symlinks"] == [dict(path=relative, target=str(target))]
    assert quick["input_sha256"][relative] == hashlib.sha256(target.read_bytes()).hexdigest()
    with pytest.raises(RuntimeError, match="regular input files") as error:
        benchmark._source("full")
    assert relative in str(error.value)


def test_renamed_source_paths_are_recorded_without_losing_spaces(benchmark, source_checkout):
    source_checkout.git("mv", "src/strategy_inference/__init__.py",
                        "src/strategy_inference/renamed model.py")
    source = benchmark._source("quick")
    assert source["changes"] == [dict(
        status="R ", path="src/strategy_inference/renamed model.py",
        previous_path="src/strategy_inference/__init__.py",
    )]
    with pytest.raises(RuntimeError, match="Full validation requires committed"):
        benchmark._source("full")


def test_cli_full_rejects_dirty_source_before_running_benchmarks(
    benchmark, source_checkout, monkeypatch
):
    source_checkout.core.write_text("value = 2\n", encoding="utf-8")
    monkeypatch.setattr(benchmark, "simulation", lambda *_: pytest.fail("Must not simulate"))
    monkeypatch.setattr(benchmark, "panel", lambda **_: pytest.fail("Must not load data"))
    output = source_checkout.root / "result.json"
    with pytest.raises(RuntimeError, match="Full validation requires committed"):
        benchmark.main(["--mode", "full", "--output", str(output)])
    assert not output.exists()


def test_cli_quick_serializes_development_provenance(benchmark, source_checkout, monkeypatch):
    source_checkout.core.write_text("value = 2\n", encoding="utf-8")
    monkeypatch.setattr(benchmark, "simulation", lambda mode: [dict(mode=mode)])
    output = source_checkout.root / "result.json"
    assert benchmark.main(["--suite", "simulation", "--output", str(output)]) == 0
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["mode"] == "quick" and record["simulation"] == [dict(mode="quick")]
    assert record["source"]["dirty"] and record["source"]["run_kind"] == "development"
    assert record["source"]["changes"][0]["path"] == "src/strategy_inference/__init__.py"


def test_cli_full_serializes_formal_provenance(benchmark, source_checkout, monkeypatch):
    monkeypatch.setattr(benchmark, "simulation", lambda mode: [dict(mode=mode)])
    output = source_checkout.root / "result.json"
    assert benchmark.main([
        "--mode", "full", "--suite", "simulation", "--output", str(output),
    ]) == 0
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["mode"] == "full" and record["simulation"] == [dict(mode="full")]
    assert record["source"]["run_kind"] == "formal" and not record["source"]["dirty"]


@pytest.mark.parametrize("committed", [False, True])
def test_cli_full_never_writes_report_if_inputs_change_during_run(
    benchmark, source_checkout, monkeypatch, committed
):
    def simulation(mode):
        source_checkout.core.write_text("value = 2\n", encoding="utf-8")
        if committed:
            source_checkout.git("add", "src")
            source_checkout.git(
                "-c", "user.name=Benchmark test", "-c", "user.email=benchmark@example.invalid",
                "commit", "--quiet", "-m", "Changed benchmark inputs during execution",
            )
        return [dict(mode=mode)]

    monkeypatch.setattr(benchmark, "simulation", simulation)
    output = source_checkout.root / "result.json"
    with pytest.raises(RuntimeError):
        benchmark.main(["--mode", "full", "--suite", "simulation", "--output", str(output)])
    assert not output.exists()


def test_existing_output_is_rejected_before_benchmark_runs(benchmark, tmp_path, monkeypatch):
    destination = tmp_path / "saved.json"
    destination.write_text("original", encoding="utf-8")
    monkeypatch.setattr(benchmark, "_source", lambda: pytest.fail("Must not start benchmark"))
    with pytest.raises(SystemExit) as error:
        benchmark.main(["--output", str(destination)])
    assert error.value.code == 2 and destination.read_text() == "original"
