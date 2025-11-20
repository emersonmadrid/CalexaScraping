"""Application-wide configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from src.utils.helpers import ensure_dir

load_dotenv()


@dataclass(frozen=True, slots=True)
class PathSettings:
    root: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = root / "data"
    inputs_dir: Path = data_dir / "inputs"
    temp_dir: Path = data_dir / "temp"
    logs_dir: Path = root / "logs"
    training_dir: Path = data_dir / "training"
    drivers_dir: Path = root / "drivers"

    def __post_init__(self) -> None:
        ensure_dir(self.data_dir)
        ensure_dir(self.inputs_dir)
        ensure_dir(self.temp_dir)
        ensure_dir(self.logs_dir)
        ensure_dir(self.training_dir)


@dataclass(slots=True)
class AppSettings:
    base_url: str
    headless: bool
    max_captcha_attempts: int
    expedientes_csv: Path
    chromedriver_path: Path
    temp_dir: Path
    logs_dir: Path
    training_dir: Path
    prefer_cnn: bool
    prefer_ml: bool


def load_settings() -> AppSettings:
    """Read settings from environment providing sensible fallbacks."""
    paths = PathSettings()

    def _bool(name: str, default: bool) -> bool:
        value = os.getenv(name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}

    base_url = os.getenv(
        "CEJ_BASE_URL", "https://cej.pj.gob.pe/cej/forms/busquedaform.html"
    )

    chromedriver_override = os.getenv("CHROMEDRIVER")
    chromedriver_path = (
        Path(chromedriver_override)
        if chromedriver_override
        else paths.drivers_dir / "chromedriver.exe"
    )

    return AppSettings(
        base_url=base_url,
        headless=_bool("BROWSER_HEADLESS", False),
        max_captcha_attempts=int(os.getenv("CAPTCHA_ATTEMPTS", "3")),
        expedientes_csv=Path(
            os.getenv(
                "EXPEDIENTES_CSV",
                str(paths.inputs_dir / "expedientes.csv"),
            )
        ),
        chromedriver_path=chromedriver_path,
        temp_dir=paths.temp_dir,
        logs_dir=paths.logs_dir,
        training_dir=paths.training_dir,
        prefer_cnn=_bool("CAPTCHA_USE_CNN", True),
        prefer_ml=_bool("CAPTCHA_USE_ML", True),
    )


__all__ = ["AppSettings", "PathSettings", "load_settings"]
