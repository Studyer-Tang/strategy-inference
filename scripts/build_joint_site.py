"""Package the audited joint-information report without recomputing research."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_uncertainty_site import (
    _directory,
    _hashes,
    _read,
    _rewrite_report,
    _sha,
    _verify_resources,
)
from joint_report import FIGURES

ROOT = Path(__file__).resolve().parents[1]


def _snapshot(source):
    provenance = {name: _read(source, name) for name in ("metadata.json", "audit.json", "bounds-audit.json", "presentation.json")}
    metadata, audit, bounds, presentation = (json.loads(provenance[name]) for name in provenance)
    if not all(isinstance(value, dict) for value in (metadata, audit, bounds, presentation)):
        raise ValueError("Metadata must contain JSON objects.")
    if metadata.get("status") != "complete" or metadata.get("protocol", {}).get("study") != "joint-uncertainty":
        raise ValueError("A completed joint-uncertainty study is required.")
    cells = metadata.get("cells")
    if not isinstance(cells, list) or not cells or any(not isinstance(cell, dict) or "key" not in cell for cell in cells):
        raise ValueError("An explicit cell manifest is required.")
    keys = [cell["key"] for cell in cells]
    if len(keys) != len(set(keys)):
        raise ValueError("The cell manifest contains duplicate keys.")
    raw_names = {"summary.csv", "paired.csv", "geometry.csv", "certificates.json"} | {f"{key}.csv.gz" for key in keys}
    digest = _sha(provenance["metadata.json"])
    if audit.get("status") != "passed" or audit.get("metadata_sha256") != digest or audit.get("records_checked") != metadata.get("records"):
        raise ValueError("A passed audit must be bound to these exact records and metadata.")
    scripts = _directory(ROOT / "scripts", required=True)
    if audit.get("auditor_sha256") != _sha(_read(scripts, "verify_joint_uncertainty.py")):
        raise ValueError("Auditor source differs from its evidence binding.")
    if bounds.get("status") != "passed" or bounds.get("metadata_sha256") != digest or bounds.get("certificates_sha256") != metadata["output_hashes"].get("certificates.json"):
        raise ValueError("Bounds audit differs from the evidence binding.")
    if bounds.get("verifier_sha256") != _sha(_read(scripts, "verify_joint_bounds.py")):
        raise ValueError("Bounds verifier source differs from its evidence binding.")
    if presentation.get("evidence_metadata_sha256") != digest or presentation.get("audit_sha256") != _sha(provenance["audit.json"]):
        raise ValueError("Presentation provenance differs from evidence or audit.")
    if presentation.get("bounds_audit_sha256") != _sha(provenance["bounds-audit.json"]):
        raise ValueError("Presentation differs from its bounds audit.")
    if presentation.get("renderer_sha256") != _sha(_read(scripts, "joint_report.py")):
        raise ValueError("Renderer source differs from its presentation binding.")
    names = {"report.html"} | {f"{stem}.{extension}" for stem in FIGURES for extension in ("png", "svg", "pdf")}
    files = _hashes(source, metadata.get("output_hashes"), raw_names)
    files.update(_hashes(source, presentation.get("output_hashes"), names))
    files.update(provenance)
    if any(_read(source, name) != contents for name, contents in files.items()):
        raise ValueError("Evidence changed during the snapshot.")
    return files


def build_site(source: Path, destination: Path, *, check: bool = False):
    source = _directory(source, required=True)
    destination = _directory(destination, required=False)
    if source == destination or source.is_relative_to(destination) or destination.is_relative_to(source):
        raise ValueError("Source and site directories must not overlap.")
    files = _snapshot(source)
    expected = files | {"index.html": _rewrite_report(files["report.html"], set(files)), ".nojekyll": b""}
    if destination.exists():
        if any(path.is_symlink() or not path.is_file() for path in destination.iterdir()):
            raise ValueError("The managed site contains a symlink or non-file entry.")
        if {path.name for path in destination.iterdir()} - set(expected):
            raise ValueError("The managed site contains unexpected files.")
    if check:
        if any(not (destination / name).is_file() or (destination / name).read_bytes() != contents for name, contents in expected.items()):
            raise ValueError("Generated site differs from its verified source.")
    else:
        destination.mkdir(parents=True, exist_ok=True)
        for name, contents in expected.items():
            (destination / name).write_bytes(contents)
    _verify_resources(expected["index.html"], destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "results/research/joint/full")
    parser.add_argument("--destination", type=Path, default=ROOT / "docs/research/joint")
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    try:
        build_site(arguments.source, arguments.destination, check=arguments.check)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, f"Joint site {'check' if arguments.check else 'build'} failed: {exc}\n")
    print(f"Joint site {'matches its verified source' if arguments.check else 'built'}: {arguments.destination.resolve()}")


if __name__ == "__main__":
    main()
