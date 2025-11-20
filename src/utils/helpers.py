"""Generic helper utilities for CalexaScraping."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator, Sequence, TypeVar

T = TypeVar("T")


def ensure_dir(path: Path) -> Path:
    """Create a directory (and parents) if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp(fmt: str = "%Y%m%d_%H%M%S") -> str:
    """Return a timestamp string used for filenames."""
    return datetime.now().strftime(fmt)


def chunked(iterable: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    """Yield chunks of *size* from the provided iterable."""
    if size <= 0:
        raise ValueError("size must be greater than zero")
    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]


def read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    """Load a CSV file into a list of dictionaries."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV no encontrado: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        return [dict(row) for row in reader]
