"""Dataclasses used throughout the scraping pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List

from PIL import Image

from src.utils.helpers import read_csv_rows


@dataclass(slots=True)
class Expediente:
    numero_expediente: str
    materia: str | None = None
    fecha_desde: str | None = None
    fecha_hasta: str | None = None
    extra: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_row(cls, row: dict[str, str]) -> "Expediente":
        numero = row.get("numero_expediente") or row.get("expediente")
        if not numero:
            raise ValueError("CSV debe contener la columna numero_expediente")

        materia = row.get("materia") or row.get("materia_proceso")
        fecha_desde = row.get("fecha_desde")
        fecha_hasta = row.get("fecha_hasta")

        extra = {
            key: value
            for key, value in row.items()
            if key
            not in {
                "numero_expediente",
                "expediente",
                "materia",
                "materia_proceso",
                "fecha_desde",
                "fecha_hasta",
            }
            and value
        }

        return cls(
            numero_expediente=numero.strip(),
            materia=materia.strip() if materia else None,
            fecha_desde=fecha_desde.strip() if fecha_desde else None,
            fecha_hasta=fecha_hasta.strip() if fecha_hasta else None,
            extra=extra,
        )


@dataclass(slots=True)
class CapturedCaptcha:
    image: Image.Image
    image_path: Path | None
    variance: float | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CaptchaPrediction:
    text: str
    solver: str
    confidence: float | None = None
    image_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def load_expedientes(csv_path: Path) -> List[Expediente]:
    """Load expedientes input file."""
    rows = read_csv_rows(csv_path)
    return [Expediente.from_row(row) for row in rows]


__all__ = ["Expediente", "CaptchaPrediction", "CapturedCaptcha", "load_expedientes"]
