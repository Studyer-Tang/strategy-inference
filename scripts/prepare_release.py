"""Validate release artifacts and write checksums/notes without network access."""

from __future__ import annotations

import argparse
import hashlib
import re
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def prepare(tag: str, dist: Path, notes: Path, *, root: Path = ROOT):
    text = (root / "pyproject.toml").read_text()
    match = re.search(r'^version = "([0-9]+\.[0-9]+\.[0-9]+)"$', text, re.MULTILINE)
    if match is None or tag != f"v{match[1]}":
        raise ValueError("Release tag must match the stable package version.")
    version = match[1]
    code_version = re.search(
        r'^__version__ = "([^"]+)"$',
        (root / "src/strategy_inference/__init__.py").read_text(),
        re.MULTILINE,
    )
    if code_version is None or code_version[1] != version:
        raise ValueError("Package metadata and runtime versions differ.")
    wheel = f"strategy_inference-{version}-py3-none-any.whl"
    names = (wheel, f"strategy_inference-{version}.tar.gz")
    files = [dist / name for name in names]
    if not all(path.is_file() and not path.is_symlink() for path in files):
        raise ValueError("Both a regular wheel and a source archive are required.")
    if {p.name for p in dist.iterdir() if p.is_file()} - {*names, "SHA256SUMS"}:
        raise ValueError("The release directory contains unexpected artifacts.")
    with zipfile.ZipFile(files[0]) as archive:
        record = BytesParser().parsebytes(
            archive.read(f"strategy_inference-{version}.dist-info/METADATA")
        )
    if record["Name"] != "strategy-inference" or record["Version"] != version:
        raise ValueError("Wheel metadata differs from the release version.")
    try:
        with tarfile.open(files[1], "r:gz") as archive:
            metadata_name = f"strategy_inference-{version}/PKG-INFO"
            members = [member for member in archive.getmembers() if member.name == metadata_name]
            if len(members) != 1 or not members[0].isfile():
                raise ValueError("Source archive must contain one regular root PKG-INFO file.")
            with archive.extractfile(members[0]) as metadata:
                record = BytesParser().parsebytes(metadata.read())
    except (tarfile.TarError, OSError, EOFError) as exc:
        raise ValueError("Source archive is not a readable gzip tar archive.") from exc
    if record["Name"] != "strategy-inference" or record["Version"] != version:
        raise ValueError("Source archive metadata differs from the release version.")
    changelog = (root / "CHANGELOG.md").read_text()
    section = re.search(
        rf"^## {re.escape(version)}\n(.*?)(?=^## |\Z)", changelog, re.MULTILINE | re.DOTALL
    )
    if section is None:
        raise ValueError("A changelog section is required for this release.")
    hashes = "".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n" for path in files
    )
    api = "time-series.md" if (root / "docs/time-series.md").is_file() else "api.md"
    performance = (
        "time-series-performance.md"
        if (root / "docs/time-series-performance.md").is_file()
        else "performance.md"
    )
    body = f"{section[1].strip()}\n\nInstall the wheel without cloning the research repository:\n\n```bash\npython -m pip install 'https://github.com/Studyer-Tang/strategy-inference/releases/download/{tag}/{wheel}'\n```\n\n[Library documentation](https://studyer-tang.github.io/strategy-inference/library/) · [API](https://github.com/Studyer-Tang/strategy-inference/blob/{tag}/docs/{api}) · [Performance](https://github.com/Studyer-Tang/strategy-inference/blob/{tag}/docs/{performance})\n"
    (dist / "SHA256SUMS").write_text(hashes)
    notes.write_text(body)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--dist", type=Path, required=True)
    parser.add_argument("--notes", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.tag, args.dist, args.notes)


if __name__ == "__main__":
    main()
