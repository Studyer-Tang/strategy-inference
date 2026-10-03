"""Bind the toolbox page to saved evidence and protect every managed target."""

import hashlib
import html
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _git(root, *arguments):
    return subprocess.check_output(["git", "-C", str(root), *arguments], stderr=subprocess.PIPE)


@pytest.fixture
def builder():
    spec = importlib.util.spec_from_file_location(
        "toolbox_builder_fixture",
        ROOT / "scripts/build_toolbox_site.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def evidence(builder, monkeypatch, tmp_path):
    root = tmp_path / "repository"
    package = root / "src/strategy_inference"
    package.mkdir(parents=True)
    # Fixtures retain release blobs, independently of the installed version or
    # new modules in the current checkout. No historical Python is executed.
    tree = _git(ROOT, "ls-tree", "-r", "--name-only", builder.RELEASE_COMMIT, "--",
                "src/strategy_inference").decode().splitlines()
    for path in tree:
        if path.endswith(".py") and Path(path).parent == Path("src/strategy_inference"):
            (package / Path(path).name).write_bytes(
                _git(ROOT, "show", f"{builder.RELEASE_COMMIT}:{path}")
            )
    runner = root / "benchmarks/time_series.py"
    runner.parent.mkdir()
    runner.write_bytes(_git(ROOT, "show", f"{builder.RELEASE_COMMIT}:benchmarks/time_series.py"))
    report = json.loads((ROOT / "benchmarks/results/time-series-0.6.json").read_bytes())
    source = root / "benchmarks/results/time-series-0.6.json"
    source.parent.mkdir()
    source.write_text(json.dumps(report))
    archive = root / "docs/library/v0.5.0"
    archive.mkdir(parents=True)
    (archive / "index.html").write_bytes(b"Frozen v0.5 page.\n")
    (archive / "benchmark.json").write_bytes(b"Frozen v0.5 raw evidence.\n")
    # The active landing and its legacy snapshots are not this builder's outputs.
    landing = root / "docs/library"
    (landing / "index.html").write_bytes(b"Current landing must remain unchanged.\n")
    (landing / "benchmark-0.6.json").write_bytes(b"Legacy root benchmark must remain unchanged.\n")
    (landing / "benchmark.json").write_bytes(b"Legacy v0.5 root benchmark.\n")
    monkeypatch.setattr(builder, "ROOT", root)
    return source, report


@pytest.fixture
def frozen_repository(builder, evidence, monkeypatch):
    _git(builder.ROOT, "init", "-q")
    _git(builder.ROOT, "config", "user.name", "Frozen fixture")
    _git(builder.ROOT, "config", "user.email", "fixture@example.invalid")
    _git(builder.ROOT, "add", "src", "benchmarks/time_series.py")
    _git(builder.ROOT, "commit", "-qm", "v0.6 source fixture")
    monkeypatch.setattr(builder, "RELEASE_COMMIT", _git(builder.ROOT, "rev-parse", "HEAD").decode().strip())
    return evidence


def _snapshot(root):
    snapshot = {}
    for path in root.rglob("*"):
        key = str(path.relative_to(root))
        if path.is_symlink():
            snapshot[key] = ("symlink", str(path.readlink()), path.lstat().st_mtime_ns)
        elif path.is_file():
            snapshot[key] = ("file", path.read_bytes(), path.stat().st_mtime_ns)
    return snapshot


def _save(source, report):
    source.write_text(json.dumps(report))


def test_build_copies_exact_evidence_and_check_is_readonly_with_archive_retained(builder, evidence):
    source, report = evidence
    source_before = _snapshot(builder.ROOT / "src")
    benchmarks_before = _snapshot(builder.ROOT / "benchmarks")
    archive = builder.ROOT / "docs/library/v0.5.0"
    archive_before = _snapshot(archive)
    landing = builder.ROOT / "docs/library"
    root_before = {
        name: ((landing / name).read_bytes(), (landing / name).stat().st_mtime_ns)
        for name in ("index.html", "benchmark-0.6.json", "benchmark.json")
    }
    builder.build()
    site = builder.ROOT / "docs/library/v0.6.0"
    assert (site / "benchmark-0.6.json").read_bytes() == source.read_bytes()
    page = (site / "index.html").read_text()
    for case in report["cases"]:
        assert f"{case['warm_median_seconds'] * 1000:.3f}" in page
    assert 'href="../v0.5.0/"' in page
    assert {
        name: ((landing / name).read_bytes(), (landing / name).stat().st_mtime_ns)
        for name in root_before
    } == root_before
    assert _snapshot(builder.ROOT / "src") == source_before
    assert _snapshot(builder.ROOT / "benchmarks") == benchmarks_before
    assert _snapshot(archive) == archive_before
    before = _snapshot(builder.ROOT)
    builder.build(check=True)
    assert _snapshot(builder.ROOT) == before


@pytest.mark.parametrize("change", ["missing", "changed"])
def test_check_never_creates_or_repairs_assets(builder, evidence, change):
    if change == "changed":
        builder.build()
        (builder.ROOT / "docs/library/v0.6.0/benchmark-0.6.json").write_bytes(b"changed")
    before = _snapshot(builder.ROOT)
    with pytest.raises(ValueError, match="differs"):
        builder.build(check=True)
    assert _snapshot(builder.ROOT) == before


@pytest.mark.parametrize(
    "change", ["modify_source", "add_source", "remove_source", "runner", "current_version"]
)
def test_changed_source_runner_and_runtime_version_are_rejected_before_writing(
    builder, evidence, change
):
    source, report = evidence
    package = builder.ROOT / "src/strategy_inference"
    if change == "modify_source":
        (package / "model_selection.py").write_text("# Changed candidate source.\n")
    elif change == "add_source":
        (package / "new_module.py").write_text("# New candidate source.\n")
    elif change == "remove_source":
        (package / "model_selection.py").unlink()
    elif change == "runner":
        (builder.ROOT / "benchmarks/time_series.py").write_text("# Different benchmark runner.\n")
    else:
        (package / "__init__.py").write_text('__version__ = "0.7.0"\n')
        # Keep source hashes consistent to exercise the version binding itself.
        report["candidate_source_sha256"] = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in package.glob("*.py")
        }
        _save(source, report)
    before = _snapshot(builder.ROOT)
    with pytest.raises(ValueError):
        builder.build()
    assert _snapshot(builder.ROOT) == before


@pytest.mark.parametrize(
    "change",
    [
        "package_version",
        "source_hash",
        "runner_hash",
        "repeats",
        "case_order",
        "median",
        "zero_timing",
        "bool_timing",
        "negative_peak",
        "bool_peak",
        "peak_below_current",
        "cell_version",
        "cell_numpy",
        "warmup",
        "global_threads",
        "cell_threads",
        "resample_budget",
        "loss_dimensions",
        "backtest_horizon",
        "conformal_options",
        "comparison_exclusion",
    ],
)
def test_inconsistent_report_fields_are_rejected_without_touching_outputs(
    builder, evidence, change
):
    source, report = evidence
    if change == "package_version":
        report["package_version"] = "0.5.0"
    elif change == "source_hash":
        report["candidate_source_sha256"]["model_selection.py"] = "0" * 64
    elif change == "runner_hash":
        report["benchmark_source_sha256"] = "0" * 64
    elif change == "repeats":
        report["repeats"] = 4
    elif change == "case_order":
        report["cases"][0], report["cases"][1] = report["cases"][1], report["cases"][0]
    elif change == "median":
        report["cases"][0]["warm_median_seconds"] *= 2
    elif change == "zero_timing":
        report["cases"][0]["warm_seconds"][0] = 0
    elif change == "bool_timing":
        report["cases"][0]["warm_seconds"][0] = True
    elif change == "negative_peak":
        report["cases"][0]["traced_peak_bytes"] = -1
    elif change == "bool_peak":
        report["cases"][0]["traced_peak_bytes"] = True
    elif change == "peak_below_current":
        report["cases"][0]["traced_peak_bytes"] = report["cases"][0]["traced_current_bytes"] - 1
    elif change == "cell_version":
        report["cases"][0]["package_version"] = "0.5.0"
    elif change == "cell_numpy":
        report["cases"][0]["numpy_version"] = "different"
    elif change == "warmup":
        report["warmup_calls"] = 0
    elif change == "global_threads":
        report["blas_thread_environment"]["OMP_NUM_THREADS"] = "2"
    elif change == "cell_threads":
        report["cases"][0]["thread_environment"]["OMP_NUM_THREADS"] = "2"
    elif change == "resample_budget":
        report["cases"][2]["settings"]["comparison_options"]["n_resamples"] = 998
    elif change == "loss_dimensions":
        report["cases"][0]["settings"]["n_models"] = 21
    elif change == "backtest_horizon":
        report["cases"][1]["settings"]["backtest_options"]["horizon"] = 11
    elif change == "conformal_options":
        report["cases"][3]["settings"]["options"]["decay"] = 0.5
    else:
        report["cases"][2]["settings"]["existing_backtest_excluded"] = False
    _save(source, report)
    before = _snapshot(builder.ROOT)
    with pytest.raises(ValueError):
        builder.build()
    assert _snapshot(builder.ROOT) == before


def test_environment_metadata_is_escaped_in_rendered_html(builder, evidence):
    source, report = evidence
    source_before = source.read_bytes(), source.stat().st_mtime_ns
    report["platform"] = '<img src=x onerror="alert(1)">'
    report["python"] = "<script>alert(1)</script> fixture"
    report["numpy"] = "<svg/onload=alert(1)>"
    # Environment strings are free-form display metadata, so test the renderer
    # directly without claiming that changed runtime versions validate evidence.
    page = builder.render(report).decode()
    for value in (report["platform"], report["python"].split()[0], report["numpy"]):
        assert html.escape(value) in page and value not in page
    assert (source.read_bytes(), source.stat().st_mtime_ns) == source_before


@pytest.mark.parametrize("name", ["index.html", "benchmark-0.6.json"])
@pytest.mark.parametrize("kind", ["broken_symlink", "symlink", "hardlink", "directory"])
def test_every_unsafe_target_is_rejected_before_any_managed_file_is_written(
    builder,
    evidence,
    tmp_path,
    name,
    kind,
):
    source, _ = evidence
    site = builder.ROOT / "docs/library/v0.6.0"
    builder.build()
    target = site / name
    target.unlink()
    protected = tmp_path / "protected.txt"
    protected.write_bytes(b"External contents must not be overwritten.\n")
    if kind == "broken_symlink":
        target.symlink_to(tmp_path / "missing.txt")
    elif kind == "symlink":
        target.symlink_to(protected)
    elif kind == "hardlink":
        target.hardlink_to(protected)
    else:
        target.mkdir()
    before = _snapshot(builder.ROOT)
    outside_before = protected.read_bytes(), protected.stat().st_mtime_ns
    for check in (True, False):
        with pytest.raises(ValueError):
            builder.build(check=check)
        assert _snapshot(builder.ROOT) == before
        assert (protected.read_bytes(), protected.stat().st_mtime_ns) == outside_before
        assert source.is_file()


@pytest.mark.parametrize("directory", ["docs", "docs/library", "docs/library/v0.6.0"])
def test_symlinked_parent_directory_cannot_redirect_managed_writes(
    builder, evidence, tmp_path, directory
):
    builder.build()
    target = builder.ROOT / directory
    outside = tmp_path / "outside"
    target.rename(outside)
    target.symlink_to(outside, target_is_directory=True)
    before = _snapshot(builder.ROOT)
    outside_before = _snapshot(outside)
    with pytest.raises(ValueError):
        builder.build()
    assert _snapshot(builder.ROOT) == before
    assert _snapshot(outside) == outside_before


def test_real_saved_report_is_read_without_running_or_rewriting_measurements(builder):
    report_path = ROOT / "benchmarks/results/time-series-0.6.json"
    before = report_path.read_bytes(), report_path.stat().st_mtime_ns
    raw, report = builder._read_performance()
    assert raw == before[0]
    assert report["package_version"] == "0.6.0"
    assert builder.render(report).startswith(b"<!doctype html>")
    assert (report_path.read_bytes(), report_path.stat().st_mtime_ns) == before


@pytest.mark.parametrize("change", ["source", "new_module", "remove_module", "runner", "version"])
def test_frozen_git_release_ignores_changed_current_checkout(builder, frozen_repository, change):
    source, _ = frozen_repository
    package = builder.ROOT / "src/strategy_inference"
    builder.build()
    expected = _snapshot(builder.ROOT / "docs/library")
    if change == "source":
        (package / "model_selection.py").write_text("# New implementation.\n")
    elif change == "new_module":
        (package / "multistep.py").write_text("# New v0.7 module.\n")
    elif change == "remove_module":
        (package / "model_selection.py").unlink()
    elif change == "runner":
        (builder.ROOT / "benchmarks/time_series.py").write_text("# New runner.\n")
    else:
        (package / "__init__.py").write_text('__version__ = "0.7.0"\n')
    before = _snapshot(builder.ROOT)
    raw, _ = builder._read_performance()
    assert raw == source.read_bytes()
    builder.build(check=True)
    assert _snapshot(builder.ROOT) == before
    builder.build()
    assert {
        name: content[:2] for name, content in _snapshot(builder.ROOT / "docs/library").items()
    } == {name: content[:2] for name, content in expected.items()}


@pytest.mark.parametrize("change", ["source", "new_module", "remove_module", "runner", "version"])
def test_modified_release_git_blobs_do_not_validate_old_evidence(
    builder, frozen_repository, monkeypatch, change
):
    source, report = frozen_repository
    package = builder.ROOT / "src/strategy_inference"
    if change == "source":
        (package / "model_selection.py").write_text("# Altered release.\n")
    elif change == "new_module":
        (package / "unexpected.py").write_text("# Altered source dictionary.\n")
    elif change == "remove_module":
        (package / "model_selection.py").unlink()
    elif change == "runner":
        (builder.ROOT / "benchmarks/time_series.py").write_text("# Altered release runner.\n")
    else:
        (package / "__init__.py").write_text('__version__ = "0.7.0"\n')
        report["candidate_source_sha256"] = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in package.glob("*.py")
        }
        _save(source, report)
    _git(builder.ROOT, "add", "-A", "src", "benchmarks/time_series.py")
    _git(builder.ROOT, "commit", "-qm", "Altered purported release")
    monkeypatch.setattr(builder, "RELEASE_COMMIT", _git(builder.ROOT, "rev-parse", "HEAD").decode().strip())
    before = _snapshot(builder.ROOT)
    with pytest.raises(ValueError, match="frozen"):
        builder.build()
    assert _snapshot(builder.ROOT) == before


@pytest.mark.parametrize("object_kind", ["missing", "tree"])
def test_invalid_release_object_is_rejected_readonly(
    builder, frozen_repository, monkeypatch, object_kind
):
    release = "0" * 40 if object_kind == "missing" else _git(
        builder.ROOT, "rev-parse", "HEAD^{tree}"
    ).decode().strip()
    monkeypatch.setattr(builder, "RELEASE_COMMIT", release)
    before = _snapshot(builder.ROOT)
    with pytest.raises(ValueError, match="fetch.*Git objects" if object_kind == "missing" else "Git commit"):
        builder.build(check=True)
    assert _snapshot(builder.ROOT) == before


def test_symlink_release_source_is_rejected_before_any_output(
    builder, frozen_repository, monkeypatch
):
    source = builder.ROOT / "src/strategy_inference/model_selection.py"
    source.unlink()
    source.symlink_to("inference.py")
    _git(builder.ROOT, "add", "src")
    _git(builder.ROOT, "commit", "-qm", "Invalid release source type")
    monkeypatch.setattr(builder, "RELEASE_COMMIT", _git(builder.ROOT, "rev-parse", "HEAD").decode().strip())
    before = _snapshot(builder.ROOT)
    with pytest.raises(ValueError, match="ordinary Git files"):
        builder.build()
    assert _snapshot(builder.ROOT) == before
