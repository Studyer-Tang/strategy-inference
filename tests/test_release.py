"""Release packaging cannot silently upload stale or damaged artifacts."""

import importlib.util
import io
import tarfile
import zipfile
from pathlib import Path

import pytest


def _write_sdist(
    path,
    *,
    member_name="strategy_inference-0.5.0/PKG-INFO",
    metadata=b"Name: strategy-inference\nVersion: 0.5.0\n",
    member_type=tarfile.REGTYPE,
    copies=1,
):
    with tarfile.open(path, "w:gz") as archive:
        for _ in range(copies):
            member = tarfile.TarInfo(member_name)
            member.type = member_type
            if member.isfile():
                member.size = len(metadata)
                archive.addfile(member, io.BytesIO(metadata))
            else:
                member.linkname = "elsewhere/PKG-INFO"
                archive.addfile(member)


@pytest.fixture
def release(tmp_path):
    source = Path(__file__).resolve().parents[1] / "scripts/prepare_release.py"
    spec = importlib.util.spec_from_file_location("prepare_release_fixture", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "source"
    (root / "src/strategy_inference").mkdir(parents=True)
    (root / "pyproject.toml").write_text('version = "0.5.0"\n')
    (root / "src/strategy_inference/__init__.py").write_text('__version__ = "0.5.0"\n')
    (root / "CHANGELOG.md").write_text("# Changelog\n\n## 0.5.0\n\n- New API.\n\n## 0.4.0\n\n- Older work.\n")
    dist = tmp_path / "dist"
    dist.mkdir()
    wheel = dist / "strategy_inference-0.5.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("strategy_inference-0.5.0.dist-info/METADATA", "Name: strategy-inference\nVersion: 0.5.0\n")
    _write_sdist(dist / "strategy_inference-0.5.0.tar.gz")
    return module, root, dist, tmp_path / "notes.md"


def test_verified_artifacts_have_reproducible_checksums_and_current_notes(release):
    module, root, dist, notes = release
    module.prepare("v0.5.0", dist, notes, root=root)
    before = (dist / "SHA256SUMS").read_bytes(), notes.read_bytes()
    module.prepare("v0.5.0", dist, notes, root=root)
    assert before == ((dist / "SHA256SUMS").read_bytes(), notes.read_bytes())
    assert len(before[0].splitlines()) == 2
    assert "New API." in notes.read_text() and "Older work." not in notes.read_text()
    assert "releases/download/v0.5.0/" in notes.read_text()


@pytest.mark.parametrize("corruption", [
    "tag", "runtime_version", "missing_archive", "extra_artifact", "wheel_metadata",
    "damaged_sdist", "truncated_sdist", "renamed_old_sdist", "sdist_version",
    "sdist_name", "sdist_symlink", "sdist_hardlink", "duplicate_sdist_metadata",
])
def test_inconsistent_release_is_rejected_before_writing(release, corruption):
    module, root, dist, notes = release
    sdist = dist / "strategy_inference-0.5.0.tar.gz"
    tag = "v0.4.0" if corruption == "tag" else "v0.5.0"
    if corruption == "runtime_version":
        (root / "src/strategy_inference/__init__.py").write_text('__version__ = "0.4.0"\n')
    elif corruption == "missing_archive":
        (dist / "strategy_inference-0.5.0.tar.gz").unlink()
    elif corruption == "extra_artifact":
        (dist / "unexpected.whl").write_text("stale")
    elif corruption == "wheel_metadata":
        with zipfile.ZipFile(dist / "strategy_inference-0.5.0-py3-none-any.whl", "w") as archive:
            archive.writestr("strategy_inference-0.5.0.dist-info/METADATA", "Name: strategy-inference\nVersion: 0.4.0\n")
    elif corruption == "damaged_sdist":
        sdist.write_bytes(b"not a gzip archive")
    elif corruption == "truncated_sdist":
        sdist.write_bytes(sdist.read_bytes()[:20])
    elif corruption == "renamed_old_sdist":
        _write_sdist(sdist, member_name="strategy_inference-0.4.0/PKG-INFO",
            metadata=b"Name: strategy-inference\nVersion: 0.4.0\n")
    elif corruption == "sdist_version":
        _write_sdist(sdist, metadata=b"Name: strategy-inference\nVersion: 0.4.0\n")
    elif corruption == "sdist_name":
        _write_sdist(sdist, metadata=b"Name: different-package\nVersion: 0.5.0\n")
    elif corruption == "sdist_symlink":
        _write_sdist(sdist, member_type=tarfile.SYMTYPE)
    elif corruption == "sdist_hardlink":
        _write_sdist(sdist, member_type=tarfile.LNKTYPE)
    elif corruption == "duplicate_sdist_metadata":
        _write_sdist(sdist, copies=2)
    with pytest.raises(ValueError):
        module.prepare(tag, dist, notes, root=root)
    assert not notes.exists() and not (dist / "SHA256SUMS").exists()
