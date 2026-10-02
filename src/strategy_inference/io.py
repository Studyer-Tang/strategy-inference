"""Strict CSV input without implicit deletion or calendar resampling."""

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ._validation import as_returns


@dataclass(frozen=True)
class ReturnTable:
    values: NDArray[np.float64]
    names: tuple[str, ...]
    dates: tuple[str, ...] | None


def read_returns_csv(path: str | Path, *, benchmark: str | None = None) -> ReturnTable:
    """Read numeric candidate columns and an optional increasing ISO date column.

    Calendar spacing is not used as a proxy for equally spaced trading sessions.
    The caller is responsible for the observation frequency and point-in-time data.
    """
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError("The CSV is empty.") from exc
        header = [name.strip() for name in header]
        if not header or any(not name for name in header) or len(set(header)) != len(header):
            raise ValueError("CSV column names must be distinct and nonempty.")
        date_index = header.index("date") if "date" in header else None
        numeric_indices = [index for index in range(len(header)) if index != date_index]
        names = tuple(header[index] for index in numeric_indices)
        if not names:
            raise ValueError("The CSV contains no candidate columns.")
        if benchmark is not None and (benchmark not in names or len(names) == 1):
            raise ValueError(
                "benchmark must name a numeric column alongside at least one candidate."
            )
        rows, dates = [], []
        previous_date = None
        for line, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise ValueError(
                    f"CSV line {line}: expected {len(header)} fields, found {len(row)}."
                )
            try:
                rows.append([float(row[index]) for index in numeric_indices])
            except ValueError as exc:
                raise ValueError(
                    f"CSV line {line}: every return must be numeric and nonempty."
                ) from exc
            if date_index is not None:
                label = row[date_index].strip()
                try:
                    observed_date = datetime.fromisoformat(label)
                    ordered = previous_date is None or observed_date > previous_date
                except (ValueError, TypeError) as exc:
                    raise ValueError(
                        f"CSV line {line}: date must be a consistent ISO date or timestamp."
                    ) from exc
                if not ordered:
                    raise ValueError(f"CSV line {line}: dates must be strictly increasing.")
                previous_date = observed_date
                dates.append(label)
    data = np.asarray(rows, dtype=np.float64)
    if benchmark is not None and data.ndim == 2:
        index = names.index(benchmark)
        data = np.delete(data - data[:, index, None], index, axis=1)
        names = tuple(name for name in names if name != benchmark)
    return ReturnTable(as_returns(data), names, tuple(dates) if date_index is not None else None)
