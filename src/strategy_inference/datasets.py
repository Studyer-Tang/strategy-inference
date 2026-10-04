"""Small, pinned real-world archives and a local TSF reader; no remote code."""

from __future__ import annotations

import hashlib
import io
import os
import re
import tempfile
import warnings
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import NamedTuple
from urllib.request import Request, urlopen

import numpy as np
from numpy.typing import NDArray

_REVISION = "58aafbe2712ff481c014f562e42723f2820fd5d4"
_REPOSITORY = "https://huggingface.co/datasets/Monash-University/monash_tsf"
_MAX_BYTES = 32 * 1024 * 1024
_EMPTY_VALUE = re.compile(r"(^|,)\s*(?=,|$)")
_MISSING_VALUE = re.compile(r"(?:^|,)\s*\?\s*(?=,|$)")


class _Archive(NamedTuple):
    filename: str
    sha256: str
    size: int
    source: str


_ARCHIVES = {
    "fred_md": _Archive(
        "fred_md_dataset.zip",
        "305c0edd2b5e97159c6339be4990ea97fdf86772c1edef2a0dfc836bf29f45c3",
        169107,
        "https://zenodo.org/records/4654833",
    ),
    "bitcoin": _Archive(
        "bitcoin_dataset_with_missing_values.zip",
        "aca43bae943dbf24617e885b5165f2493578fb195a888702ec85947e727f015f",
        220403,
        "https://zenodo.org/records/5121965",
    ),
    "oikolab_weather": _Archive(
        "oikolab_weather_dataset.zip",
        "6f0d2dce3a5aa17c26627ea6107cbf4e5b10050f8a1648d6d03306cd0777f735",
        1326101,
        "https://zenodo.org/records/5184708",
    ),
}


@dataclass(frozen=True)
class TimeSeries:
    """An original series: float64 values, NaN for ``?``, unmodified attributes.

    Returned arrays are read-only. A missing timestamp stays ``None``; this
    reader does not invent a calendar, timezone, or observation-release dates.
    """

    name: str
    values: NDArray[np.float64]
    attributes: Mapping[str, str | float | datetime]

    @property
    def start_timestamp(self) -> datetime | None:
        value = self.attributes.get("start_timestamp")
        return value if isinstance(value, datetime) else None


@dataclass(frozen=True)
class TimeSeriesDataset:
    """Original series and TSF headers, with byte-level provenance when loaded."""

    name: str
    series: tuple[TimeSeries, ...]
    metadata: Mapping[str, str]
    sha256: str
    source: str | None = None
    license: str | None = None
    revision: str | None = None

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.series)

    @property
    def frequency(self) -> str | None:
        return self.metadata.get("frequency")

    def __getitem__(self, name: str) -> TimeSeries:
        for item in self.series:
            if item.name == name:
                return item
        raise KeyError(name)


def available_datasets() -> tuple[str, ...]:
    """Names of curated Monash archives available through ``load_dataset``."""
    return tuple(_ARCHIVES)


def _parse(content: bytes, *, name: str, digest: str, encoding: str) -> TimeSeriesDataset:
    attributes, records, metadata, names = [], [], {}, set()
    in_data = False
    for number, raw in enumerate(io.StringIO(content.decode(encoding)), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            if line.startswith("@"):
                if in_data:
                    raise ValueError("header after @data")
                parts = line.split()
                key = parts[0][1:].lower()
                if key == "attribute":
                    if len(parts) != 3 or parts[2] not in {"string", "numeric", "date"}:
                        raise ValueError("expected @attribute name string|numeric|date")
                    if parts[1] in {item[0] for item in attributes}:
                        raise ValueError("duplicate attribute")
                    attributes.append((parts[1], parts[2]))
                elif key == "data":
                    if len(parts) != 1 or not attributes:
                        raise ValueError("@data requires attributes and no arguments")
                    in_data = True
                else:
                    if len(parts) < 2 or key in metadata:
                        raise ValueError("empty or duplicate header")
                    value = " ".join(parts[1:])
                    if key in {"missing", "equallength"} and value not in {"true", "false"}:
                        raise ValueError("expected true or false")
                    if key == "horizon" and (not value.isdigit() or int(value) < 1):
                        raise ValueError("horizon must be a positive integer")
                    metadata[key] = value
                continue
            if not in_data:
                raise ValueError("record before @data")
            fields = line.split(":", len(attributes))
            if len(fields) != len(attributes) + 1:
                raise ValueError("record does not match attributes")
            parsed = {}
            for (key, kind), field in zip(attributes, fields[:-1], strict=True):
                field = field.strip()
                if not field:
                    raise ValueError("empty attribute")
                if kind == "date":
                    parsed[key] = datetime.strptime(field, "%Y-%m-%d %H-%M-%S")
                elif kind == "numeric":
                    parsed[key] = float(field)
                    if not np.isfinite(parsed[key]):
                        raise ValueError("nonfinite numeric attribute")
                else:
                    parsed[key] = field
            values_text = fields[-1]
            if _EMPTY_VALUE.search(values_text):
                raise ValueError("empty series value")
            missing_tokens = values_text.count("?")
            if missing_tokens and missing_tokens != sum(
                1 for _ in _MISSING_VALUE.finditer(values_text)
            ):
                raise ValueError("a missing token must be exactly '?'")
            # fromstring parses in C without a Python float/list per observation.
            # Older NumPy warns instead of raising on trailing malformed text.
            with warnings.catch_warnings():
                warnings.simplefilter("error", DeprecationWarning)
                values = np.fromstring(values_text.replace("?", "nan"), sep=",", dtype=float)
            if values.size != values_text.count(",") + 1 or np.isinf(values).any():
                raise ValueError("empty, malformed or infinite series value")
            missing = int(np.isnan(values).sum())
            if missing != missing_tokens:
                raise ValueError("only '?' may represent a missing value")
            if missing and metadata.get("missing") == "false":
                raise ValueError("missing values contradict @missing false")
            item_name = str(parsed.get("series_name", len(records) + 1))
            if item_name in names:
                raise ValueError("duplicate series name")
            if metadata.get("equallength") == "true" and records:
                if values.size != records[0].values.size:
                    raise ValueError("unequal lengths contradict @equallength true")
            names.add(item_name)
            values.flags.writeable = False
            records.append(TimeSeries(item_name, values, MappingProxyType(parsed)))
        except (ValueError, OverflowError, DeprecationWarning) as exc:
            raise ValueError(f"Invalid TSF at line {number}: {exc}.") from exc
    if not records:
        raise ValueError("TSF must contain @data and at least one nonempty series.")
    return TimeSeriesDataset(
        metadata.get("relation", name), tuple(records), MappingProxyType(metadata), digest
    )


def _read(content: bytes, *, name: str, encoding: str) -> TimeSeriesDataset:
    digest = hashlib.sha256(content).hexdigest()
    if zipfile.is_zipfile(io.BytesIO(content)):
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            entries = [entry for entry in archive.infolist() if not entry.is_dir()]
            if len(entries) != 1 or not entries[0].filename.lower().endswith(".tsf"):
                raise ValueError("Archive must contain exactly one TSF file.")
            if entries[0].file_size > _MAX_BYTES:
                raise ValueError("Uncompressed TSF exceeds the 32 MiB limit.")
            content = archive.read(entries[0])  # No extraction or path execution.
    return _parse(content, name=name, digest=digest, encoding=encoding)


def read_tsf(path: str | os.PathLike[str], *, encoding: str = "cp1252") -> TimeSeriesDataset:
    """Read a local TSF or single-TSF ZIP (at most 32 MiB, also uncompressed).

    The default encoding follows the archive authors' parser. Local UTF-8 files
    can use ``encoding="utf-8"``. Missing observations remain in place as NaN;
    backtesting requires an explicitly chosen, finite, equally spaced series.
    Attributes absent from the input remain absent. Without ``series_name``,
    names are one-based record numbers. No filtering or transformation is done.
    """
    path = Path(path)
    with path.open("rb") as stream:
        content = stream.read(_MAX_BYTES + 1)
    if len(content) > _MAX_BYTES:
        raise ValueError("TSF input exceeds the 32 MiB limit.")
    return _read(content, name=path.stem, encoding=encoding)


def load_dataset(
    name: str,
    *,
    cache_dir: str | os.PathLike[str] | None = None,
    offline: bool = False,
    timeout: float = 30.0,
) -> TimeSeriesDataset:
    """Download a pinned, SHA-256 verified Monash ZIP, or read its cached copy.

    Only data bytes are read: no ``datasets``, pandas, HF SDK, or remote Python
    is needed. Every cached read verifies the hash. A corrupt cache raises;
    remove that file to download again. ``offline=True`` never accesses the
    network. Downloads are size-bounded, verified, then atomically cached.

    The archives are historical research snapshots (CC BY 4.0), not current or
    point-in-time market feeds. FRED-MD carries the archive's supplied numeric
    preprocessing and anonymous T1…T107 names. No release-vintage is provided.
    """
    if not isinstance(name, str) or name not in _ARCHIVES:
        raise ValueError(f"Unknown dataset; choose one of {available_datasets()}.")
    if not isinstance(offline, bool):
        raise ValueError("offline must be a bool.")
    if (
        not np.isscalar(timeout)
        or np.ma.isMaskedArray(timeout)
        or np.asarray(timeout).dtype.kind not in "iuf"
    ):
        raise ValueError("timeout must be a finite positive number.")
    try:
        timeout = float(timeout)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("timeout must be a finite positive number.") from exc
    if not np.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be a finite positive number.")
    spec = _ARCHIVES[name]
    if cache_dir is None:
        cache_dir = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / (
            "strategy-inference/datasets"
        )
    directory = Path(cache_dir)
    path = directory / f"{name}-{_REVISION[:12]}.zip"
    cached = path.exists()
    if cached:
        with path.open("rb") as stream:
            content = stream.read(spec.size + 1)
    elif offline:
        raise FileNotFoundError(f"Dataset is not cached: {path}")
    else:
        url = f"{_REPOSITORY}/resolve/{_REVISION}/data/{spec.filename}"
        request = Request(url, headers={"User-Agent": "strategy-inference-datasets"})
        with urlopen(request, timeout=timeout) as response:
            content = response.read(spec.size + 1)
    if len(content) != spec.size or hashlib.sha256(content).hexdigest() != spec.sha256:
        raise ValueError(f"Dataset size or SHA-256 mismatch: {path if cached else name}")
    result = _read(content, name=name, encoding="cp1252")
    if not cached:
        directory.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=directory, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(content)
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return replace(
        result, name=name, source=spec.source, license="CC-BY-4.0", revision=_REVISION
    )
