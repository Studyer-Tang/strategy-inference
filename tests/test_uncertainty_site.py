"""Small saved-evidence fixtures; publishing never runs the experiment."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT = Path(__file__).resolve().parents[1]


def _sha(contents):
    return hashlib.sha256(contents).hexdigest()


def _json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


@pytest.fixture
def site(monkeypatch):
    monkeypatch.syspath_prepend(str(PROJECT / "scripts"))
    specification = importlib.util.spec_from_file_location(
        "uncertainty_site_review", PROJECT / "scripts/build_uncertainty_site.py"
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


@pytest.fixture
def saved(site, tmp_path, monkeypatch):
    root = tmp_path / "repo"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "uncertainty_report.py").write_bytes(b"fixture renderer\n")
    (scripts / "verify_parameter_uncertainty.py").write_bytes(b"fixture auditor\n")
    monkeypatch.setattr(site, "ROOT", root)
    source = root / "results/research/uncertainty/full"
    source.mkdir(parents=True)
    raw = {
        "summary.csv": b"method,rate\nuncertainty,0.0\n",
        "paired.csv": b"method,difference\ngls_known,0.1\n",
        "certificates.json": b'{"fixed": "fixture"}\n',
        "p1-g01.csv.gz": b"tiny fixture, not a generated Gaussian experiment",
    }
    report = """<!doctype html><html lang="zh-CN"><style>p{color:#302c26}</style>
<h1>未知参数的研究</h1><p>literal ../../../../docs/proof.md remains unchanged.</p>
<img src='figure-1-uncertainty-size.png'>
<a href="figure-1-uncertainty-size.svg">SVG</a><a href="figure-1-uncertainty-size.pdf">PDF</a>
<a HREF="../../../../docs/parameter-uncertainty.md?raw=1&amp;x=2#proof">证明</a>
<a href=../../../../experiments/parameter-uncertainty-protocol.json>协议</a>
<a href="summary.csv?download=1&amp;x=2#table">摘要</a><a href='paired.csv'>配对</a>
<a href="metadata.json">记录</a><a href="certificates.json">证书</a><a href="audit.json">审计</a>
<a href="p1-g01.csv.gz">原始</a><a href="#notes">注</a><a href="https://example.org/paper">文献</a>
</html>""".encode()
    presentation_files = {"report.html": report}
    presentation_files.update(
        {
            f"{stem}.{extension}": f"fixture {stem} {extension}\n".encode()
            for stem in site.FIGURES
            for extension in ("png", "svg", "pdf")
        }
    )
    for name, contents in (raw | presentation_files).items():
        (source / name).write_bytes(contents)
    metadata = {
        "status": "complete",
        "protocol": {"study": "parameter-uncertainty"},
        "records": 4,
        "cells": [{"key": "p1-g01"}],
        "output_hashes": {name: _sha(contents) for name, contents in raw.items()},
    }
    _json(source / "metadata.json", metadata)
    audit = {
        "status": "passed",
        "records_checked": 4,
        "metadata_sha256": _sha((source / "metadata.json").read_bytes()),
        "auditor_sha256": _sha((scripts / "verify_parameter_uncertainty.py").read_bytes()),
    }
    _json(source / "audit.json", audit)
    presentation = {
        "evidence_metadata_sha256": audit["metadata_sha256"],
        "audit_sha256": _sha((source / "audit.json").read_bytes()),
        "renderer_sha256": _sha((scripts / "uncertainty_report.py").read_bytes()),
        "output_hashes": {name: _sha(contents) for name, contents in presentation_files.items()},
    }
    _json(source / "presentation.json", presentation)
    return SimpleNamespace(root=root, source=source, destination=root / "docs/research/uncertainty")


def _rebind(source):
    audit = json.loads((source / "audit.json").read_text())
    audit["metadata_sha256"] = _sha((source / "metadata.json").read_bytes())
    _json(source / "audit.json", audit)
    presentation = json.loads((source / "presentation.json").read_text())
    presentation["evidence_metadata_sha256"] = audit["metadata_sha256"]
    presentation["audit_sha256"] = _sha((source / "audit.json").read_bytes())
    _json(source / "presentation.json", presentation)


def _change_report(saved, contents):
    (saved.source / "report.html").write_bytes(contents)
    presentation = json.loads((saved.source / "presentation.json").read_text())
    presentation["output_hashes"]["report.html"] = _sha(contents)
    _json(saved.source / "presentation.json", presentation)


def test_site_preserves_verified_bytes_and_rewrites_only_repository_links(site, saved):
    before = {path.name: path.read_bytes() for path in saved.source.iterdir()}
    site.build_site(saved.source, saved.destination)
    index = (saved.destination / "index.html").read_text()
    assert (
        "https://github.com/Studyer-Tang/strategy-inference/blob/main/docs/parameter-uncertainty.md?raw=1&amp;x=2#proof"
        in index
    )
    assert (
        'href="https://github.com/Studyer-Tang/strategy-inference/blob/main/experiments/parameter-uncertainty-protocol.json"'
        in index
    )
    assert "literal ../../../../docs/proof.md remains unchanged." in index
    assert "<img src='figure-1-uncertainty-size.png'>" in index
    assert 'href="summary.csv?download=1&amp;x=2#table"' in index
    assert 'href="https://example.org/paper"' in index
    assert {path.name: path.read_bytes() for path in saved.source.iterdir()} == before
    for name, contents in before.items():
        assert (saved.destination / name).read_bytes() == contents
    assert (saved.destination / ".nojekyll").read_bytes() == b""
    site.build_site(saved.source, saved.destination, check=True)


def test_check_is_read_only_and_does_not_create_a_missing_site(site, saved, monkeypatch):
    with pytest.raises(ValueError, match="differs"):
        site.build_site(saved.source, saved.destination, check=True)
    assert not saved.destination.exists()
    site.build_site(saved.source, saved.destination)

    def unexpected_write(*args, **kwargs):
        raise AssertionError("check mode must never write")

    monkeypatch.setattr(Path, "write_bytes", unexpected_write)
    site.build_site(saved.source, saved.destination, check=True)


@pytest.mark.parametrize("name", ["summary.csv", "report.html", "figure-3-uncertainty-power.pdf"])
def test_changed_source_bytes_are_rejected_before_writing(site, saved, name):
    (saved.source / name).write_bytes(b"changed after hashing")
    with pytest.raises(ValueError, match="SHA-256"):
        site.build_site(saved.source, saved.destination)
    assert not saved.destination.exists()


@pytest.mark.parametrize(
    "filename,field",
    [
        ("audit.json", "metadata_sha256"),
        ("audit.json", "auditor_sha256"),
        ("presentation.json", "evidence_metadata_sha256"),
        ("presentation.json", "audit_sha256"),
        ("presentation.json", "renderer_sha256"),
    ],
)
def test_provenance_must_bind_the_exact_evidence_and_sources(site, saved, filename, field):
    path = saved.source / filename
    value = json.loads(path.read_text())
    value[field] = "0" * 64
    _json(path, value)
    with pytest.raises(ValueError):
        site.build_site(saved.source, saved.destination)
    assert not saved.destination.exists()


@pytest.mark.parametrize(
    "filename,updates",
    [
        ("metadata.json", {"status": "running"}),
        ("metadata.json", {"protocol": {"study": "another-study"}}),
        ("metadata.json", {"cells": [{"key": "../outside"}]}),
        ("metadata.json", {"cells": [{"key": "p1-g01"}, {"key": "p1-g01"}]}),
        ("audit.json", {"status": "failed"}),
        ("audit.json", {"records_checked": 3}),
    ],
)
def test_incomplete_wrong_or_inconsistent_records_are_rejected(site, saved, filename, updates):
    path = saved.source / filename
    value = json.loads(path.read_text())
    value.update(updates)
    _json(path, value)
    _rebind(saved.source)
    with pytest.raises(ValueError):
        site.build_site(saved.source, saved.destination)


@pytest.mark.parametrize("filename", ["metadata.json", "presentation.json"])
def test_manifest_cannot_add_an_outside_or_unexpected_file(site, saved, filename):
    path = saved.source / filename
    value = json.loads(path.read_text())
    value["output_hashes"]["../outside"] = "0" * 64
    _json(path, value)
    _rebind(saved.source)
    with pytest.raises(ValueError, match="manifest"):
        site.build_site(saved.source, saved.destination)


@pytest.mark.parametrize("filename", ["metadata.json", "audit.json", "presentation.json"])
def test_metadata_must_be_a_JSON_object(site, saved, filename):
    _json(saved.source / filename, [])
    with pytest.raises(ValueError, match="JSON objects"):
        site.build_site(saved.source, saved.destination)


@pytest.mark.parametrize(
    "reference",
    [
        "../summary.csv",
        "/tmp/outside",
        "unverified.csv",
        "%2e%2e%2foutside",
        "../../../../docs/../outside",
        "../../../../outside.txt",
        "folder%5cfile.csv",
        "file:///tmp/outside",
        "javascript:alert(1)",
    ],
)
def test_html_cannot_refer_to_unverified_or_outside_local_resources(site, saved, reference):
    _change_report(saved, f'<a href="{reference}">invalid</a>'.encode())
    with pytest.raises(ValueError):
        site.build_site(saved.source, saved.destination)


def test_base_element_is_rejected(site, saved):
    _change_report(saved, b'<base href="https://example.org/"><a href="summary.csv">table</a>')
    with pytest.raises(ValueError, match="base"):
        site.build_site(saved.source, saved.destination)


def test_source_file_symlink_is_rejected_even_when_its_bytes_match(site, saved, tmp_path):
    path = saved.source / "summary.csv"
    outside = tmp_path / "outside.csv"
    outside.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        site.build_site(saved.source, saved.destination)


def test_source_and_destination_directory_symlinks_are_rejected(site, saved, tmp_path):
    alias = tmp_path / "source-alias"
    alias.symlink_to(saved.source, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic"):
        site.build_site(alias, saved.destination)
    target = tmp_path / "target"
    target.mkdir()
    saved.destination.parent.mkdir(parents=True)
    saved.destination.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic"):
        site.build_site(saved.source, saved.destination)
    assert not list(target.iterdir())


def test_destination_symlink_extra_file_and_changed_bytes_fail_check(site, saved, tmp_path):
    site.build_site(saved.source, saved.destination)
    path = saved.destination / "summary.csv"
    original = path.read_bytes()
    path.write_bytes(b"site changed")
    with pytest.raises(ValueError, match="differs"):
        site.build_site(saved.source, saved.destination, check=True)
    path.write_bytes(original)
    extra = saved.destination / "unexpected.txt"
    extra.write_text("preserve this file")
    with pytest.raises(ValueError, match="Unexpected"):
        site.build_site(saved.source, saved.destination)
    assert extra.read_text() == "preserve this file"
    extra.unlink()
    outside = tmp_path / "outside.csv"
    outside.write_bytes(original)
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        site.build_site(saved.source, saved.destination)


@pytest.mark.parametrize("relative", [".", "nested/site"])
def test_source_and_destination_must_not_overlap(site, saved, relative):
    with pytest.raises(ValueError, match="overlap"):
        site.build_site(saved.source, saved.source / relative)


def test_cli_build_and_check_use_the_saved_evidence_only(site, saved, monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_uncertainty_site.py",
            "--source",
            str(saved.source),
            "--destination",
            str(saved.destination),
        ],
    )
    site.main()
    assert "built:" in capsys.readouterr().out
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_uncertainty_site.py",
            "--source",
            str(saved.source),
            "--destination",
            str(saved.destination),
            "--check",
        ],
    )
    site.main()
    assert "matches its verified source" in capsys.readouterr().out
