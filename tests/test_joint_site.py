"""Preserve evidence bindings and make site checking read-only."""

import importlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def builder(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    return importlib.import_module("build_joint_site")


@pytest.fixture
def source(builder, tmp_path):
    path = tmp_path / "evidence"
    path.mkdir()
    names = {"summary.csv", "paired.csv", "geometry.csv", "certificates.json", "p1-g01.csv.gz"}
    for name in names:
        (path / name).write_bytes(b"saved evidence")
    metadata = dict(status="complete", protocol={"study": "joint-uncertainty"},
        records=1, cells=[{"key": "p1-g01"}],
        output_hashes={name: builder._sha((path / name).read_bytes()) for name in names})
    (path / "metadata.json").write_text(json.dumps(metadata))
    audit = dict(status="passed", records_checked=1,
        metadata_sha256=builder._sha((path / "metadata.json").read_bytes()),
        auditor_sha256=builder._sha((ROOT / "scripts/verify_joint_uncertainty.py").read_bytes()))
    (path / "audit.json").write_text(json.dumps(audit))
    bounds = dict(status="passed", metadata_sha256=audit["metadata_sha256"],
        certificates_sha256=metadata["output_hashes"]["certificates.json"],
        verifier_sha256=builder._sha((ROOT / "scripts/verify_joint_bounds.py").read_bytes()))
    (path / "bounds-audit.json").write_text(json.dumps(bounds))
    (path / "report.html").write_text('<a href="geometry.csv">data</a><a href="../../../../docs/joint-uncertainty.md">proof</a><img src="figure-1-joint-size.png">')
    figures = {f"{stem}.{extension}" for stem in builder.FIGURES for extension in ("png", "svg", "pdf")}
    for name in figures:
        (path / name).write_bytes(b"saved figure")
    presentation = dict(evidence_metadata_sha256=audit["metadata_sha256"],
        audit_sha256=builder._sha((path / "audit.json").read_bytes()),
        bounds_audit_sha256=builder._sha((path / "bounds-audit.json").read_bytes()),
        renderer_sha256=builder._sha((ROOT / "scripts/joint_report.py").read_bytes()),
        output_hashes={name: builder._sha((path / name).read_bytes()) for name in figures | {"report.html"}})
    (path / "presentation.json").write_text(json.dumps(presentation))
    return path


def test_verified_copy_and_read_only_check(builder, source, tmp_path):
    destination = tmp_path / "site"
    builder.build_site(source, destination)
    for path in source.iterdir():
        assert (destination / path.name).read_bytes() == path.read_bytes()
    index = (destination / "index.html").read_text()
    assert "https://github.com/Studyer-Tang/strategy-inference/blob/main/docs/joint-uncertainty.md" in index
    assert 'href="geometry.csv"' in index
    before = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in destination.iterdir()}
    builder.build_site(source, destination, check=True)
    assert before == {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in destination.iterdir()}


@pytest.mark.parametrize("name", ["summary.csv", "paired.csv", "geometry.csv", "certificates.json",
    "p1-g01.csv.gz", "figure-1-joint-size.png", "figure-2-joint-information.svg", "report.html"])
def test_evidence_tampering_is_rejected(builder, source, tmp_path, name):
    (source / name).write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        builder.build_site(source, tmp_path / "site")


@pytest.mark.parametrize("file,key", [("metadata.json", "status"), ("audit.json", "metadata_sha256"),
    ("audit.json", "auditor_sha256"), ("presentation.json", "renderer_sha256"),
    ("presentation.json", "audit_sha256"), ("presentation.json", "bounds_audit_sha256"),
    ("bounds-audit.json", "metadata_sha256"), ("bounds-audit.json", "certificates_sha256"),
    ("bounds-audit.json", "verifier_sha256")])
def test_provenance_tampering_is_rejected(builder, source, tmp_path, file, key):
    value = json.loads((source / file).read_text())
    value[key] = "changed"
    (source / file).write_text(json.dumps(value))
    with pytest.raises(ValueError):
        builder.build_site(source, tmp_path / "site")


def test_missing_asset_and_destination_extra_are_rejected(builder, source, tmp_path):
    destination = tmp_path / "site"
    builder.build_site(source, destination)
    (destination / "unexpected.txt").write_text("retain")
    with pytest.raises(ValueError, match="unexpected"):
        builder.build_site(source, destination)
    assert (destination / "unexpected.txt").read_text() == "retain"
    (destination / "unexpected.txt").unlink()
    (destination / "geometry.csv").unlink()
    with pytest.raises(ValueError, match="differs"):
        builder.build_site(source, destination, check=True)


def test_symlink_and_overlapping_paths_are_rejected(builder, source, tmp_path):
    with pytest.raises(ValueError, match="overlap"):
        builder.build_site(source, source / "site")
    (source / "summary.csv").unlink()
    external = tmp_path / "external.csv"
    external.write_text("saved evidence")
    (source / "summary.csv").symlink_to(external)
    with pytest.raises(ValueError, match="symlink"):
        builder.build_site(source, tmp_path / "site")
