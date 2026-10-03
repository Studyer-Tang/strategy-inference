"""Package the saved uncertainty report for Pages, without recomputing research.

Only repository documentation links change in index.html. The original report,
figures, raw evidence and provenance remain byte-identical downloadable files.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from pathlib import Path, PurePosixPath
from urllib.parse import quote, unquote, urlsplit

from build_site import ATTRIBUTE, Resources

ROOT = Path(__file__).resolve().parents[1]
GITHUB = "https://github.com/Studyer-Tang/strategy-inference/blob/main/"
FIGURES = (
    "figure-1-uncertainty-size",
    "figure-2-uncertainty-boundary",
    "figure-3-uncertainty-power",
)


def _sha(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def _name(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value) is None:
        raise ValueError(f"Invalid managed file name: {value!r}")
    return value


def _directory(path: Path, *, required: bool) -> Path:
    absolute = path.absolute()
    if any(value.is_symlink() for value in (absolute, *absolute.parents)):
        raise ValueError("Source and destination directories must not contain symbolic links.")
    resolved = absolute.resolve()
    if (required and not resolved.is_dir()) or (resolved.exists() and not resolved.is_dir()):
        raise ValueError(f"Expected a directory: {path}")
    return resolved


def _read(source: Path, name: str) -> bytes:
    path = source / _name(name)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular, non-symlink source file: {name}")
    return path.read_bytes()


def _hashes(source: Path, manifest: object, expected: set[str]) -> dict[str, bytes]:
    if not isinstance(manifest, dict) or set(manifest) != expected:
        raise ValueError("An evidence or presentation manifest has unexpected members.")
    files = {}
    for name, digest in manifest.items():
        _name(name)
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"Invalid SHA-256 value: {name}")
        contents = _read(source, name)
        if _sha(contents) != digest:
            raise ValueError(f"Source SHA-256 does not match its manifest: {name}")
        files[name] = contents
    return files


def _snapshot(source: Path) -> dict[str, bytes]:
    provenance = {
        name: _read(source, name) for name in ("metadata.json", "audit.json", "presentation.json")
    }
    metadata = json.loads(provenance["metadata.json"])
    audit = json.loads(provenance["audit.json"])
    presentation = json.loads(provenance["presentation.json"])
    if not all(isinstance(value, dict) for value in (metadata, audit, presentation)):
        raise ValueError("Evidence, audit and presentation metadata must be JSON objects.")
    if metadata.get("status") != "complete":
        raise ValueError("A completed uncertainty computation is required.")
    protocol = metadata.get("protocol")
    if not isinstance(protocol, dict) or protocol.get("study") != "parameter-uncertainty":
        raise ValueError("The saved computation must be the parameter-uncertainty study.")
    cells = metadata.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("The computation must contain an explicit nonempty cell manifest.")
    if any(not isinstance(cell, dict) or "key" not in cell for cell in cells):
        raise ValueError("Every cell manifest entry must contain a file key.")
    keys = [_name(cell["key"]) for cell in cells]
    if len(keys) != len(set(keys)):
        raise ValueError("The cell manifest contains duplicate names.")
    raw_names = {"summary.csv", "paired.csv", "certificates.json"} | {
        f"{key}.csv.gz" for key in keys
    }
    metadata_digest = _sha(provenance["metadata.json"])
    if audit.get("status") != "passed" or audit.get("metadata_sha256") != metadata_digest:
        raise ValueError("The passed audit must be bound to this exact metadata.")
    if audit.get("records_checked") != metadata.get("records"):
        raise ValueError("The audit record count differs from the saved computation.")
    scripts = _directory(ROOT / "scripts", required=True)
    if audit.get("auditor_sha256") != _sha(_read(scripts, "verify_parameter_uncertainty.py")):
        raise ValueError("The audit differs from the saved auditor's current source.")
    if presentation.get("evidence_metadata_sha256") != metadata_digest or presentation.get(
        "audit_sha256"
    ) != _sha(provenance["audit.json"]):
        raise ValueError("Presentation provenance differs from its evidence or audit.")
    if presentation.get("renderer_sha256") != _sha(_read(scripts, "uncertainty_report.py")):
        raise ValueError("Presentation renderer SHA-256 differs from its current source.")
    presentation_names = {"report.html"} | {
        f"{stem}.{extension}" for stem in FIGURES for extension in ("png", "svg", "pdf")
    }
    files = _hashes(source, metadata.get("output_hashes"), raw_names)
    files.update(_hashes(source, presentation.get("output_hashes"), presentation_names))
    files.update(provenance)
    # The snapshot is immutable even when another process is editing the source.
    if any(_read(source, name) != contents for name, contents in files.items()):
        raise ValueError("The saved source changed while taking the snapshot.")
    return files


def _reference(reference: str, verified: set[str]) -> str:
    parts = urlsplit(reference)
    if parts.scheme and parts.scheme not in ("http", "https", "mailto"):
        raise ValueError(f"Unsupported report URL scheme: {reference}")
    if parts.scheme or parts.netloc or not parts.path:
        return reference
    local = unquote(parts.path)
    if local.startswith("../../../../"):
        relative = local[len("../../../../") :]
        path = PurePosixPath(relative)
        if (
            not path.parts
            or path.parts[0] not in ("docs", "experiments")
            or path.is_absolute()
            or ".." in path.parts
            or "\\" in relative
        ):
            raise ValueError(f"Unexpected repository link: {reference}")
        return (
            GITHUB
            + quote(path.as_posix(), safe="/")
            + ("?" + parts.query if parts.query else "")
            + ("#" + parts.fragment if parts.fragment else "")
        )
    name = _name(local)
    if name not in verified:
        raise ValueError(f"The report refers to an unverified local resource: {reference}")
    return reference


def _rewrite_report(contents: bytes, verified: set[str]) -> bytes:
    text = contents.decode("utf-8")
    resources = Resources(text)

    def replace(match):
        if match.group("name").lower() not in ("href", "src") or not match.group("assignment"):
            return match.group(0)
        value = match.group("quoted") if match.group("quote") else match.group("bare")
        if value is None:
            return match.group(0)
        original = html.unescape(value)
        changed = _reference(original, verified)
        if changed == original:
            return match.group(0)
        delimiter = match.group("quote") or '"'
        return (
            match.group("prefix")
            + match.group("name")
            + match.group("assignment")
            + delimiter
            + html.escape(changed, quote=True)
            + delimiter
        )

    for offset, raw in reversed(resources.tags):
        text = text[:offset] + ATTRIBUTE.sub(replace, raw) + text[offset + len(raw) :]
    for reference in Resources(text).references:
        _reference(reference, verified)
    return text.encode("utf-8")


def _verify_resources(index: bytes, destination: Path) -> None:
    for reference in Resources(index.decode("utf-8")).references:
        parts = urlsplit(reference)
        if parts.scheme or parts.netloc or not parts.path:
            continue
        _read(destination, unquote(parts.path))


def build_site(source: Path, destination: Path, *, check: bool = False) -> None:
    source = _directory(source, required=True)
    destination = _directory(destination, required=False)
    if (
        source == destination
        or source.is_relative_to(destination)
        or destination.is_relative_to(source)
    ):
        raise ValueError("The source and generated site directories must not overlap.")
    files = _snapshot(source)
    expected = dict(files)
    expected["index.html"] = _rewrite_report(files["report.html"], set(files))
    expected[".nojekyll"] = b""
    if destination.exists():
        present = {path.name for path in destination.iterdir()}
        if any(path.is_symlink() or not path.is_file() for path in destination.iterdir()):
            raise ValueError("The managed site contains a symlink or non-file entry.")
        if present - set(expected):
            raise ValueError(
                f"Unexpected files in the managed site: {sorted(present - set(expected))}"
            )
    if check:
        for name, contents in expected.items():
            if not (destination / name).is_file() or (destination / name).read_bytes() != contents:
                raise ValueError(f"Generated site differs from its verified source: {name}")
    else:
        destination.mkdir(parents=True, exist_ok=True)
        for name, contents in expected.items():
            (destination / name).write_bytes(contents)
    _verify_resources(expected["index.html"], destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "results/research/uncertainty/full")
    parser.add_argument(
        "--destination", "--output", type=Path, default=ROOT / "docs/research/uncertainty"
    )
    parser.add_argument(
        "--check", action="store_true", help="Compare saved generated bytes without writing"
    )
    args = parser.parse_args()
    try:
        build_site(args.source, args.destination, check=args.check)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"Uncertainty site {'check' if args.check else 'build'} failed: {exc}\n")
    print(
        f"Uncertainty site {'matches its verified source' if args.check else 'built'}: {args.destination.resolve()}"
    )


if __name__ == "__main__":
    main()
