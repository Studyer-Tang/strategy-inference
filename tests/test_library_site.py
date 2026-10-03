"""Bind the library page to saved evidence without executing any benchmark."""

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "hac_512x20": ("hac", 512, 20, 0),
    "bootstrap_fixed_512x20": ("fixed", 512, 20, 1999),
    "bootstrap_resampled_512x20": ("resampled", 512, 20, 1999),
    "bootstrap_resampled_1024x100": ("resampled", 1024, 100, 499),
    "gaussian_ar_512x20": ("gaussian_ar", 512, 20, 0),
    "gaussian_ar_2048x100": ("gaussian_ar", 2048, 100, 0),
}


@pytest.fixture
def builder(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    return importlib.import_module("build_library_site")


@pytest.fixture
def evidence(builder, monkeypatch, tmp_path):
    root = tmp_path / "repository"
    package = root / "src/strategy_inference"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('__version__ = "0.5.0"\n')
    (package / "inference.py").write_text("# Saved source fixture.\n")
    runner = root / "benchmarks/run.py"
    runner.parent.mkdir()
    runner.write_text(f"CASES = {CASES!r}\n")
    monkeypatch.setattr(builder, "ROOT", root)
    baseline, candidate, comparisons = [], [], []
    for name, (mode, t, k, b) in CASES.items():
        if mode == "gaussian_ar":
            fingerprint = {"exact_evidence_sha256": "a" * 64}
            errors = {}
        else:
            keys = ("long_run_variance",) if mode == "hac" else (
                "mean", "standard_error", "statistic", "simultaneous_ci_low",
                "simultaneous_ci_high",
            )
            fingerprint = {"numeric": {key: [1.0] * k for key in keys}}
            if mode != "hac":
                fingerprint.update(decisions=[False] * k, adjusted_pvalue=[0.5] * k,
                                   global_pvalue=0.5)
            errors = {key: 0.0 for key in keys}
        for cells, version, warm in ((baseline, "0.4.0", 100.0), (candidate, "0.5.0", 50.0)):
            cells.append(dict(case=name, mode=mode, n_obs=t, k=k, n_resamples=b,
                code_version=version, import_ms=2.0, api_load_ms=3.0, cold_call_ms=10.0,
                first_use_ms=15.0, warm_call_ms=[warm] * 7, warm_median_ms=warm,
                unified_api_warm_median_ms=(
                    51.0 if version == "0.5.0" and mode != "hac" else None
                ), process_peak_rss_mib=20.0, traced_call_peak_mib=1.0,
                fingerprint=json.loads(json.dumps(fingerprint))))
        comparisons.append(dict(status="passed", warm_speedup=2.0,
            maximum_numeric_errors=errors,
            exact_decisions_and_pvalues=True if mode in ("fixed", "resampled") else None,
            exact_certificates=True if mode == "gaussian_ar" else None))
    report = dict(schema_version=1, candidate_source_sha256=builder._source_hashes(),
        benchmark_source_sha256=builder._sha(runner.read_bytes()), repeats=7,
        baseline=baseline, candidate=candidate, comparisons=comparisons,
        platform='<script>alert("unsafe")</script>', python="3.10 fixture",
        numpy="fixture", scipy="fixture", blas_threads=1)
    source = root / "benchmarks/results/library-0.5.json"
    source.parent.mkdir()
    source.write_text(json.dumps(report))
    return source, report


def _save(source, report):
    source.write_text(json.dumps(report))


def _files(directory):
    return {str(path.relative_to(directory)): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in directory.rglob("*") if path.is_file()}


def _git(root, *arguments):
    return subprocess.run(
        ["git", "-C", str(root), *arguments], check=True, capture_output=True,
    ).stdout.decode().strip()


@pytest.fixture
def frozen_evidence(builder, evidence, monkeypatch):
    """Use actual local Git objects, without touching the production checkout."""
    _git(builder.ROOT, "init")
    _git(builder.ROOT, "add", "src", "benchmarks/run.py")
    _git(builder.ROOT, "-c", "user.name=Library fixture", "-c",
         "user.email=fixture@example.invalid", "commit", "-m", "Frozen v0.5 fixture")
    monkeypatch.setattr(builder, "RELEASE_COMMIT", _git(builder.ROOT, "rev-parse", "HEAD"))
    return evidence


def test_verified_build_and_checks_are_read_only(builder, evidence, tmp_path):
    source, _ = evidence
    destination = tmp_path / "site"
    source_before = _files(builder.ROOT)
    builder.build_site(source, destination)
    assert (destination / "benchmark.json").read_bytes() == source.read_bytes()
    page = (destination / "index.html").read_text()
    assert page.count("<tr>") == 7  # Header plus six evidence rows.
    assert "&lt;script&gt;" in page and '<script>alert("unsafe")</script>' not in page
    assert "/blob/v0.5.0/docs/api.md" in page
    assert "不含 Python/NumPy 启动" in page
    assert "max-width:100%; overflow-x:auto" in page
    destination_before = _files(destination)
    builder.build_site(source, destination, check=True)
    assert _files(destination) == destination_before
    assert _files(builder.ROOT) == source_before
    missing = tmp_path / "missing-site"
    with pytest.raises(ValueError, match="differs"):
        builder.build_site(source, missing, check=True)
    assert not missing.exists()


@pytest.mark.parametrize("change", ["modify", "add", "remove", "runner"])
def test_changed_current_sources_are_rejected(builder, evidence, tmp_path, change):
    source, _ = evidence
    package = builder.ROOT / "src/strategy_inference"
    if change == "modify":
        (package / "inference.py").write_text("# Changed.\n")
    elif change == "add":
        (package / "added.py").write_text("# New source.\n")
    elif change == "remove":
        (package / "inference.py").unlink()
    else:
        (builder.ROOT / "benchmarks/run.py").write_text("# Different runner.\n")
    with pytest.raises(ValueError, match="SHA-256"):
        builder.build_site(source, tmp_path / "site")
    assert not (tmp_path / "site").exists()


@pytest.mark.parametrize("change", ["source_hash", "runner_hash", "code_version", "current_version"])
def test_source_and_release_bindings_are_required(builder, evidence, tmp_path, change):
    source, report = evidence
    if change == "source_hash":
        report["candidate_source_sha256"]["inference.py"] = "changed"
    elif change == "runner_hash":
        report["benchmark_source_sha256"] = "changed"
    elif change == "code_version":
        report["candidate"][0]["code_version"] = "0.4.0"
    else:
        (builder.ROOT / "src/strategy_inference/__init__.py").write_text('__version__ = "0.6.0"\n')
        report["candidate_source_sha256"] = builder._source_hashes()
    _save(source, report)
    with pytest.raises(ValueError):
        builder.build_site(source, tmp_path / "site")


def test_frozen_git_release_ignores_new_current_modules_version_and_runner(
    builder, frozen_evidence, tmp_path,
):
    source, _ = frozen_evidence
    destination = tmp_path / "archive"
    builder.build_site(source, destination)
    archive_before = _files(destination)
    package = builder.ROOT / "src/strategy_inference"
    (package / "forecasting.py").write_text("# New module, outside the frozen release.\n")
    (package / "inference.py").write_text("# Changed current implementation.\n")
    (package / "__init__.py").write_text('__version__ = "0.6.0"\n')
    (builder.ROOT / "benchmarks/run.py").write_text("# New benchmark protocol.\n")
    current_before = _files(builder.ROOT)
    builder.build_site(source, destination, check=True)
    assert _files(destination) == archive_before
    assert _files(builder.ROOT) == current_before
    builder.build_site(source, destination)
    assert {key: value[0] for key, value in _files(destination).items()} == {
        key: value[0] for key, value in archive_before.items()
    }


@pytest.mark.parametrize("changed", ["source", "runner"])
def test_report_rejects_different_frozen_git_objects(
    builder, frozen_evidence, monkeypatch, tmp_path, changed,
):
    source, _ = frozen_evidence
    path = builder.ROOT / (
        "src/strategy_inference/inference.py" if changed == "source" else "benchmarks/run.py"
    )
    path.write_text(path.read_text() + "# Altered release fixture.\n")
    _git(builder.ROOT, "add", str(path.relative_to(builder.ROOT)))
    _git(builder.ROOT, "-c", "user.name=Library fixture", "-c",
         "user.email=fixture@example.invalid", "commit", "-m", "Different release objects")
    monkeypatch.setattr(builder, "RELEASE_COMMIT", _git(builder.ROOT, "rev-parse", "HEAD"))
    before = _files(builder.ROOT)
    with pytest.raises(ValueError, match="SHA-256"):
        builder.build_site(source, tmp_path / "archive")
    assert not (tmp_path / "archive").exists()
    assert _files(builder.ROOT) == before


def test_frozen_git_release_version_is_checked_independently_of_current_version(
    builder, frozen_evidence, monkeypatch, tmp_path,
):
    source, report = frozen_evidence
    init = builder.ROOT / "src/strategy_inference/__init__.py"
    init.write_text('__version__ = "0.6.0"\n')
    _git(builder.ROOT, "add", "src/strategy_inference/__init__.py")
    _git(builder.ROOT, "-c", "user.name=Library fixture", "-c",
         "user.email=fixture@example.invalid", "commit", "-m", "Wrong release version")
    monkeypatch.setattr(builder, "RELEASE_COMMIT", _git(builder.ROOT, "rev-parse", "HEAD"))
    report["candidate_source_sha256"] = builder._source_hashes()
    _save(source, report)
    init.write_text('__version__ = "0.5.0"\n')
    with pytest.raises(ValueError, match="Frozen release library version"):
        builder.build_site(source, tmp_path / "archive")


def test_missing_frozen_commit_fails_clearly_and_without_writes(
    builder, frozen_evidence, monkeypatch, tmp_path,
):
    source, _ = frozen_evidence
    monkeypatch.setattr(builder, "RELEASE_COMMIT", "f" * 40)
    before = _files(builder.ROOT)
    with pytest.raises(ValueError, match="fetch this commit and its Git objects"):
        builder.build_site(source, tmp_path / "archive", check=True)
    assert not (tmp_path / "archive").exists()
    assert _files(builder.ROOT) == before


def test_check_cli_defaults_to_versioned_archive_and_preserves_landing(
    builder, frozen_evidence, monkeypatch,
):
    source, _ = frozen_evidence
    landing = builder.ROOT / "docs/library"
    landing.mkdir(parents=True)
    (landing / "index.html").write_text("Current toolbox landing, retained.\n")
    (landing / "benchmark.json").write_text("Old snapshot, retained.\n")
    builder.build_site(source, landing / "v0.5.0")
    before = _files(builder.ROOT)
    monkeypatch.setattr(sys, "argv", ["build_library_site.py", "--check"])
    builder.main()
    assert _files(builder.ROOT) == before


@pytest.mark.parametrize("change", ["truncate", "append", "reorder", "duplicate", "shape"])
def test_six_cells_and_their_ordered_parameters_are_bound(builder, evidence, tmp_path, change):
    source, report = evidence
    if change == "truncate":
        report["comparisons"].pop()
    elif change == "append":
        report["candidate"].append(report["candidate"][0])
    elif change == "reorder":
        report["candidate"][0], report["candidate"][1] = report["candidate"][1], report["candidate"][0]
    elif change == "duplicate":
        report["candidate"][1] = report["candidate"][0]
    else:
        report["candidate"][0]["k"] = 1
    _save(source, report)
    with pytest.raises(ValueError):
        builder.build_site(source, tmp_path / "site")


@pytest.mark.parametrize("change", ["nan", "inf", "bool", "zero", "median", "first_use", "speedup", "repeats"])
def test_numeric_and_derived_timing_tampering_is_rejected(builder, evidence, tmp_path, change):
    source, report = evidence
    cell = report["candidate"][0]
    if change in ("nan", "inf", "bool", "zero"):
        cell["warm_call_ms"][0] = {"nan": float("nan"), "inf": float("inf"),
                                   "bool": True, "zero": 0}[change]
    elif change == "median":
        cell["warm_median_ms"] = 10.0
    elif change == "first_use":
        cell["first_use_ms"] = 5.0
    elif change == "speedup":
        report["comparisons"][0]["warm_speedup"] = 10.0
    else:
        cell["warm_call_ms"].pop()
    _save(source, report)
    with pytest.raises(ValueError):
        builder.build_site(source, tmp_path / "site")


@pytest.mark.parametrize("change", ["status", "decision", "pvalue", "certificate", "numeric",
                                    "error", "pvalue_flag", "certificate_flag"])
def test_result_fingerprints_and_comparisons_are_checked(builder, evidence, tmp_path, change):
    source, report = evidence
    if change == "status":
        report["comparisons"][0]["status"] = "failed"
    elif change == "decision":
        report["candidate"][1]["fingerprint"]["decisions"][0] = True
    elif change == "pvalue":
        report["candidate"][1]["fingerprint"]["adjusted_pvalue"][0] = 0.01
    elif change == "certificate":
        report["candidate"][4]["fingerprint"]["exact_evidence_sha256"] = "b" * 64
    elif change == "numeric":
        report["candidate"][0]["fingerprint"]["numeric"]["long_run_variance"][0] = 2.0
    elif change == "error":
        report["comparisons"][0]["maximum_numeric_errors"]["long_run_variance"] = 1.0
    elif change == "pvalue_flag":
        report["comparisons"][1]["exact_decisions_and_pvalues"] = None
    else:
        report["comparisons"][4]["exact_certificates"] = None
    _save(source, report)
    with pytest.raises(ValueError):
        builder.build_site(source, tmp_path / "site")


def test_output_assets_paths_and_links_do_not_damage_existing_files(builder, evidence, tmp_path):
    source, _ = evidence
    with pytest.raises(ValueError, match="overlap"):
        builder.build_site(source, source.parent / "site")
    alias = tmp_path / "source-link.json"
    alias.symlink_to(source)
    with pytest.raises(ValueError, match="symlink"):
        builder.build_site(alias, tmp_path / "site")
    destination = tmp_path / "site"
    builder.build_site(source, destination)
    original = _files(destination)
    extra = destination / "keep.txt"
    extra.write_text("retained")
    with pytest.raises(ValueError, match="unexpected"):
        builder.build_site(source, destination)
    assert extra.read_text() == "retained"
    extra.unlink()
    index = destination / "index.html"
    index.write_text("modified")
    before = _files(destination)
    with pytest.raises(ValueError, match="differs"):
        builder.build_site(source, destination, check=True)
    assert _files(destination) == before
    index.unlink()
    index.symlink_to(source)
    with pytest.raises(ValueError, match="symlinks"):
        builder.build_site(source, destination)
    index.unlink()
    index.hardlink_to(source)
    source_before = source.read_bytes()
    with pytest.raises(ValueError, match="hardlinks"):
        builder.build_site(source, destination)
    assert source.read_bytes() == source_before
    index.unlink()
    with pytest.raises(ValueError, match="differs"):
        builder.build_site(source, destination, check=True)
    assert original["benchmark.json"] == _files(destination)["benchmark.json"]


def test_saved_release_schema_renders_without_running_benchmarks(builder):
    source = ROOT / "benchmarks/results/library-0.5.json"
    before = source.read_bytes(), source.stat().st_mtime_ns
    raw, report, _ = builder._snapshot(source)
    page = builder._render(report, builder._sha(raw))
    assert page.count(b"<tr>") == 7
    assert b"strategy_inference-0.5.0-py3-none-any.whl" in page
    assert builder.RELEASE_COMMIT.encode() in page
    assert (source.read_bytes(), source.stat().st_mtime_ns) == before
