"""Package an existing calibration report for GitHub Pages, without recomputing it.

Run ``python scripts/build_site.py`` after the full report is complete. CI can
use ``python scripts/build_site.py --check`` to verify the committed site.
``--source`` and ``--destination`` support local previews of a saved quick run.
Only the existing report's resource paths change; its text and styling remain.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit, urlunsplit

FIGURES = (
    "figure-1-calibration-dependence",
    "figure-2-calibration-selection",
    "figure-3-calibration-processes",
)
TABLES = (
    "figure-1-dependence.csv",
    "figure-2-selection.csv",
    "figure-3-size.csv",
    "figure-3-power.csv",
    "holdout.csv",
    "block-sensitivity.csv",
    "variance-diagnostics.csv",
)
ASSETS = tuple(f"{stem}.{suffix}" for stem in FIGURES for suffix in ("pdf", "svg", "png")) + (
    *TABLES,
    "run-metadata.json",
)
ATTRIBUTE = re.compile(
    r"(?P<prefix>\s+)(?P<name>[^\s=/>]+)(?P<assignment>\s*=\s*)?"
    r"(?:(?P<quote>[\"'])(?P<quoted>.*?)(?P=quote)|(?P<bare>[^\s>]+))?",
    re.IGNORECASE | re.DOTALL,
)


class Resources(HTMLParser):
    """Locate resource attributes while preserving all other HTML bytes."""

    def __init__(self, text: str):
        super().__init__(convert_charrefs=False)
        self.tags: list[tuple[int, str]] = []
        self.references: list[str] = []
        self.offsets = [0]
        for line in text.splitlines(keepends=True):
            self.offsets.append(self.offsets[-1] + len(line))
        self.feed(text)
        self.close()

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "base":
            raise ValueError("The saved report must use local resource paths, without a base tag.")
        raw = self.get_starttag_text()
        if raw is None:
            raise ValueError("Cannot locate a report HTML tag.")
        line, column = self.getpos()
        self.tags.append((self.offsets[line - 1] + column, raw))
        self.references.extend(value for name, value in attrs if name in ("href", "src") and value)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)


def _local_path(reference: str) -> str | None:
    parts = urlsplit(reference)
    if parts.scheme or parts.netloc or not parts.path:
        return None
    return unquote(parts.path)


def _rewrite_report(text: str) -> bytes:
    resources = Resources(text)

    def replace(match):
        if match.group("name").lower() not in ("href", "src") or not match.group("assignment"):
            return match.group(0)
        raw = match.group("quoted") if match.group("quote") else match.group("bare")
        if raw is None:
            return match.group(0)
        reference = html.unescape(raw)
        local = _local_path(reference)
        if local is None:
            return match.group(0)
        path = PurePosixPath(local)
        if path.is_absolute() or len(path.parts) != 1 or path.name not in ASSETS:
            raise ValueError(f"Unexpected local report resource: {reference}")
        parts = urlsplit(reference)
        rewritten = urlunsplit(("", "", f"assets/{path.name}", parts.query, parts.fragment))
        quote = match.group("quote") or '"'
        return (
            match.group("prefix")
            + match.group("name")
            + match.group("assignment")
            + quote
            + html.escape(rewritten, quote=True)
            + quote
        )

    for first, raw in reversed(resources.tags):
        changed = ATTRIBUTE.sub(replace, raw)
        text = text[:first] + changed + text[first + len(raw) :]
    rewritten = Resources(text)
    before_external = [ref for ref in resources.references if _local_path(ref) is None]
    after_external = [ref for ref in rewritten.references if _local_path(ref) is None]
    if before_external != after_external:
        raise ValueError("External report links changed during site generation.")
    for reference in rewritten.references:
        local = _local_path(reference)
        if local is not None and local not in {f"assets/{name}" for name in ASSETS}:
            raise ValueError(f"A report resource was not relocated: {reference}")
    return text.encode("utf-8")


def _snapshot(source: Path) -> tuple[dict[str, bytes], bytes]:
    metadata_bytes = (source / "run-metadata.json").read_bytes()
    metadata = json.loads(metadata_bytes)
    if not isinstance(metadata, dict) or metadata.get("status") != "complete":
        raise ValueError("The calibration metadata status must be complete before building a site.")
    if metadata.get("study") != "calibration":
        raise ValueError("The site requires a calibration study.")
    hashes = metadata.get("file_sha256")
    outputs = metadata.get("outputs")
    if (
        not isinstance(hashes, dict)
        or not isinstance(outputs, list)
        or not all(isinstance(name, str) for name in outputs)
    ):
        raise ValueError("The run metadata must contain output names and their SHA-256 hashes.")
    required = set(ASSETS) - {"run-metadata.json"} | {"report.html"}
    if not required <= hashes.keys() or not required <= set(outputs):
        raise ValueError("The run record must include all 9 figures, 7 tables and report.html.")
    files = {}
    for name, recorded in hashes.items():
        if (
            not isinstance(name, str)
            or PurePosixPath(name).name != name
            or "\\" in name
            or not isinstance(recorded, str)
            or re.fullmatch(r"[0-9a-fA-F]{64}", recorded) is None
        ):
            raise ValueError("An output hash or file name in the run record is invalid.")
        contents = (source / name).read_bytes()
        actual = hashlib.sha256(contents).hexdigest()
        if actual != recorded.lower():
            raise ValueError(f"Source SHA-256 does not match the run record: {name}")
        files[name] = contents
    files["run-metadata.json"] = metadata_bytes
    index = _rewrite_report(files["report.html"].decode("utf-8"))
    if (source / "run-metadata.json").read_bytes() != metadata_bytes:
        raise ValueError("The run metadata changed while taking the source snapshot.")
    return {name: files[name] for name in ASSETS}, index


def _verify_resources(index: bytes, destination: Path) -> None:
    for reference in Resources(index.decode("utf-8")).references:
        local = _local_path(reference)
        if local is None:
            continue
        path = destination / local
        if not path.is_file() or not path.resolve().is_relative_to(destination.resolve()):
            raise ValueError(f"A local site resource is missing or outside the site: {reference}")


def build_site(source: Path, destination: Path, *, check: bool = False) -> None:
    source, destination = source.resolve(), destination.resolve()
    if source == destination:
        raise ValueError("The site destination must differ from the experiment source.")
    files, index = _snapshot(source)
    assets = destination / "assets"
    if assets.is_symlink():
        raise ValueError("The managed assets directory must not be a symbolic link.")
    if assets.exists():
        extra = {path.name for path in assets.iterdir()} - set(ASSETS)
        if extra:
            raise ValueError(f"Unexpected files in the managed assets directory: {sorted(extra)}")
    expected = {Path("index.html"): index, Path(".nojekyll"): b""}
    expected.update({Path("assets") / name: contents for name, contents in files.items()})
    if any((destination / relative).is_symlink() for relative in expected):
        raise ValueError("Generated site files must not be symbolic links.")
    if check:
        for relative, contents in expected.items():
            path = destination / relative
            if not path.is_file() or path.read_bytes() != contents:
                raise ValueError(f"Generated site differs from its verified source: {relative}")
    else:
        assets.mkdir(parents=True, exist_ok=True)
        for relative, contents in expected.items():
            (destination / relative).write_bytes(contents)
    _verify_resources(index, destination)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=root / "results" / "calibration" / "full")
    parser.add_argument("--destination", "--output", type=Path, default=root / "docs")
    parser.add_argument("--check", action="store_true", help="Verify the site without writing files")
    args = parser.parse_args()
    try:
        build_site(args.source, args.destination, check=args.check)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Site {'check' if args.check else 'build'} failed: {exc}\n")
    action = "matches its verified source" if args.check else "built from verified saved results"
    print(f"Site {action}: {args.destination.resolve()} ({len(ASSETS)} assets)")


if __name__ == "__main__":
    main()
