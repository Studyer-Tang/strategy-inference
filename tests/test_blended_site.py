"""Saved-evidence packaging rejects altered numbers, provenance and missing paths."""

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def builder(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "build_blended_site", REPO / "scripts/build_blended_site.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "repo"
    source = json.loads((REPO / module.STUDY / "results.json").read_text())
    paths = set(source["source_sha256"]) | {
        module.BENCHMARK,
        "benchmarks/blended_scales.py",
        "scripts/build_blended_site.py",
        "scripts/plot_blended_study.py",
        "scripts/plot_scale_study.py",
    }
    paths |= {
        p.relative_to(REPO).as_posix() for p in (REPO / module.STUDY).rglob("*") if p.is_file()
    }
    for relative in paths:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / relative, target)
    archive = root / "docs/library/v0.7.0/index.html"
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(b"Preserved v0.7 book and source-bound timing archive.\n")
    legacy = root / "docs/library/multistep-benchmark.json"
    legacy.write_bytes(b"Preserved legacy timing.\n")
    monkeypatch.setattr(module, "ROOT", root)
    return module


def _update(path, mutation):
    value = json.loads(path.read_text())
    mutation(value)
    path.write_text(json.dumps(value, allow_nan=False) + "\n")


def _snapshot(root):
    return {
        p.relative_to(root).as_posix(): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in root.rglob("*")
        if p.is_file()
    }


def test_build_copies_exact_downloads_and_check_leaves_archive_and_inputs_untouched(builder):
    old = _snapshot(builder.ROOT / "docs/library")
    outputs = builder.build()
    assert len(outputs) == 12
    page = (builder.ROOT / "docs/library/index.html").read_text()
    assert "5 个评分均值最低" in page
    assert "名义 MC 汇总" in page and "总体期望可能无穷" in page
    assert "0.754 秒" in page and "20.4 KiB" in page and "12.6%" in page
    assert "v0.7.0/" in page and "pilot-01" not in page and "pilot-02" not in page
    assert page.count('<div class="table-scroll"') == 3
    assert 'loading="lazy"' in page and '<html lang="zh-CN">' in page
    downloads = builder.ROOT / "docs/library/research/blended"
    assert (downloads / "source.zip").read_bytes() == (
        builder.ROOT / builder.STUDY / "source.zip"
    ).read_bytes()
    assert (downloads / "protocol.json").read_bytes() == (
        builder.ROOT / builder.PROTOCOL
    ).read_bytes()
    assert (downloads / "benchmark.json").read_bytes() == (
        builder.ROOT / builder.BENCHMARK
    ).read_bytes()
    assert json.loads((downloads / "site-manifest.json").read_text())["package_version"] == "0.8.0"
    for relative, before in old.items():
        assert _snapshot(builder.ROOT / "docs/library")[relative] == before
    snapshot = _snapshot(builder.ROOT)
    builder.build(check=True)
    assert _snapshot(builder.ROOT) == snapshot


@pytest.mark.parametrize(
    "change",
    ["aggregate", "paired", "omitted", "duplicate", "negative", "seed", "fingerprint", "version"],
)
def test_ledger_changes_are_rejected_before_page_write(builder, change):
    def mutation(data):
        if change == "aggregate":
            data["aggregate"][0]["mean_interval_score"]["mean"] += 0.01
        elif change == "paired":
            data["paired_contrasts"][0]["score_difference"]["lower"] += 0.01
        elif change == "omitted":
            data["records"].pop()
        elif change == "duplicate":
            data["records"][1] = data["records"][0]
        elif change == "negative":
            data["records"][0]["mean_interval_score"] = -1
        elif change == "seed":
            data["records"][0]["seed"] += 1
        elif change == "fingerprint":
            data["input_fingerprints"].pop()
        else:
            data["package_version"] = "0.7.0"

    _update(builder.ROOT / builder.STUDY / "results.json", mutation)
    snapshot = _snapshot(builder.ROOT)
    with pytest.raises(ValueError):
        builder.build()
    assert _snapshot(builder.ROOT) == snapshot


@pytest.mark.parametrize(
    "change",
    ["version", "median", "runs", "threads", "options", "counts", "pending", "input", "source"],
)
def test_benchmark_provenance_and_task_changes_are_rejected(builder, change):
    def mutation(data):
        case = data["cases"][3]
        if change == "version":
            data["package_version"] = "0.7.0"
        elif change == "median":
            case["warm_median_seconds"] += 0.1
        elif change == "runs":
            case["warm_seconds"].pop()
        elif change == "threads":
            case["thread_environment"]["OMP_NUM_THREADS"] = "2"
        elif change == "options":
            case["settings"]["options"]["scale_share_weight"] = 0.25
        elif change == "counts":
            case["terminal_state"]["states"][0]["n_updates"] -= 1
        elif change == "pending":
            case["terminal_state"]["pending"].pop()
        elif change == "input":
            case["settings"]["inputs"]["actual"]["sha256"] = "f" * 64
        else:
            data["candidate_source_sha256"]["multistep.py"] = "f" * 64

    _update(builder.ROOT / builder.BENCHMARK, mutation)
    with pytest.raises(ValueError):
        builder.build()
    assert not (builder.ROOT / "docs/library/index.html").exists()


@pytest.mark.parametrize(
    "name",
    [
        "src/strategy_inference/multistep.py",
        "benchmarks/blended_scales.py",
        "scripts/plot_blended_study.py",
    ],
)
def test_current_source_hash_must_match_every_evidence_binding(builder, name):
    path = builder.ROOT / name
    path.write_text(path.read_text() + "\n# Altered source.\n")
    with pytest.raises(ValueError):
        builder.build()


@pytest.mark.parametrize(
    "name",
    ["source.zip", "source-manifest.json", "environment.json", "figures/score-difference.svg"],
)
def test_saved_archive_environment_and_figure_bytes_are_bound(builder, name):
    path = builder.ROOT / builder.STUDY / name
    if name == "source.zip":
        path.write_bytes(path.read_bytes() + b"altered")
    elif name == "source-manifest.json":
        _update(path, lambda d: d.update(archive_sha256="f" * 64))
    elif name == "environment.json":
        _update(path, lambda d: d.update(source_manifest_sha256="f" * 64))
    else:
        path.write_bytes(path.read_bytes() + b"\n<!-- altered -->\n")
    with pytest.raises(ValueError):
        builder.build()


def test_missing_managed_asset_is_detected_without_touching_unmanaged_archive(builder):
    builder.build()
    extra = builder.ROOT / "docs/library/old-asset.txt"
    extra.write_text("Unmanaged old asset.\n")
    builder.build(check=True)
    managed = builder.ROOT / "docs/library/research/blended/local-coverage-error.svg"
    managed.unlink()
    snapshot = _snapshot(builder.ROOT)
    with pytest.raises(ValueError, match="stale or missing"):
        builder.build(check=True)
    assert _snapshot(builder.ROOT) == snapshot


def test_symlinked_managed_output_cannot_overwrite_the_old_archive(builder):
    page = builder.ROOT / "docs/library/index.html"
    old = builder.ROOT / "docs/library/v0.7.0/index.html"
    before = old.read_bytes()
    page.symlink_to(old)
    with pytest.raises(ValueError, match="ordinary unlinked"):
        builder.build()
    assert old.read_bytes() == before


def test_symlinked_evidence_is_rejected_even_if_bytes_match(builder):
    path = builder.ROOT / builder.STUDY / "source.zip"
    target = path.with_name("same-bytes.zip")
    path.rename(target)
    path.symlink_to(target)
    with pytest.raises(ValueError, match="symlinks"):
        builder.build()


def test_duplicate_json_fields_are_rejected(builder):
    path = builder.ROOT / builder.BENCHMARK
    path.write_text('{"schema_version": 1, "schema_version": 1}')
    with pytest.raises(ValueError, match="Duplicate JSON"):
        builder.build()
