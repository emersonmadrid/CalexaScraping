"""Shared CAPTCHA utilities."""

from __future__ import annotations

import base64
import math
import re
import time
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Protocol

import numpy as np
import requests
from PIL import Image

from src.config import selectors
from src.models.data_models import CaptchaPrediction, CapturedCaptcha
from src.utils.helpers import ensure_dir, timestamp
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


@dataclass(slots=True)
class CaptureDiagnostics:
    attempts: int = 0
    last_error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseCaptchaSolver(Protocol):
    """Common interface implemented by every solver."""

    name: str
    priority: int

    def predict(self, capture: CapturedCaptcha) -> CaptchaPrediction | None: ...


class CaptchaCaptureService:
    """Centralizes all the Selenium logic to capture CAPTCHA images."""

    def __init__(
        self,
        temp_dir: Path,
        *,
        variance_threshold: float = 8.0,
        timeout: int = 12,
    ) -> None:
        self.temp_dir = ensure_dir(temp_dir / "captchas")
        self.diagnostic_dir = ensure_dir(temp_dir / "diagnostico")
        self.variance_threshold = variance_threshold
        self.timeout = timeout

    def capture_image(
        self,
        driver,
        *,
        max_retries: int = 3,
    ) -> CapturedCaptcha | None:
        """Capture the CAPTCHA element and returns a PIL image."""
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.support.ui import WebDriverWait

        diagnostics = CaptureDiagnostics()

        for attempt in range(1, max_retries + 1):
            diagnostics.attempts = attempt
            try:
                LOGGER.debug("Capturando CAPTCHA (intento %s/%s)", attempt, max_retries)
                captcha_img = WebDriverWait(driver, self.timeout).until(
                    EC.presence_of_element_located((By.ID, selectors.CAPTCHA_IMAGE))
                )

                WebDriverWait(driver, self.timeout).until(
                    lambda d: captcha_img.get_attribute("src")
                    and len(captcha_img.get_attribute("src")) > 50
                )

                driver.execute_script(
                    """
                    return arguments[0].complete &&
                           arguments[0].naturalHeight > 0 &&
                           arguments[0].naturalWidth > 0;
                    """,
                    captcha_img,
                )

                time.sleep(0.25)

                image = self._capture_via_screenshot(captcha_img)

                if not image:
                    image = self._capture_from_src(driver, captcha_img.get_attribute("src") or "")

                if not image:
                    diagnostics.last_error = "No se pudo capturar imagen"
                    continue

                grayscale = image.convert("L")
                variance = float(np.array(grayscale).var())

                if variance < self.variance_threshold:
                    diagnostics.last_error = f"Varianza insuficiente: {variance:.2f}"
                    LOGGER.debug("Varianza baja (%.2f), reintentando...", variance)
                    self.reload(driver)
                    time.sleep(0.5)
                    continue

                image_path = self._save_capture(image, prefix=f"captcha_{timestamp()}")
                metadata = {
                    "attempt": attempt,
                    "size": image.size,
                    "mode": image.mode,
                    "variance": variance,
                }

                diagnostics.metadata.update(metadata)
                LOGGER.debug("CAPTCHA capturado correctamente en %s", image_path)

                return CapturedCaptcha(
                    image=image,
                    image_path=image_path,
                    variance=variance,
                    metadata=diagnostics.metadata,
                )

            except Exception as exc:  # pragma: no cover - depende del sitio
                diagnostics.last_error = str(exc)
                LOGGER.warning("Error capturando CAPTCHA: %s", exc)
                self.reload(driver)
                time.sleep(1)

        LOGGER.error(
            "No se pudo capturar el CAPTCHA tras %s intentos (%s)",
            diagnostics.attempts,
            diagnostics.last_error,
        )
        return None

    def reload(self, driver) -> None:
        """Click the reload button to request a new CAPTCHA."""
        from selenium.webdriver.common.by import By

        try:
            boton = driver.find_element(By.ID, selectors.RELOAD_CAPTCHA_BUTTON)
            boton.click()
        except Exception as exc:  # pragma: no cover - depende del sitio
            LOGGER.debug("No se pudo recargar el CAPTCHA: %s", exc)

    def diagnose(self, driver) -> dict[str, Any]:
        """Return detailed information about the CAPTCHA element."""
        from selenium.webdriver.common.by import By

        info: dict[str, Any] = {}
        try:
            captcha_img = driver.find_element(By.ID, selectors.CAPTCHA_IMAGE)
            info["size"] = captcha_img.size
            info["location"] = captcha_img.location
            info["src_length"] = len(captcha_img.get_attribute("src") or "")
            info["is_displayed"] = captcha_img.is_displayed()
        except Exception as exc:  # pragma: no cover
            info["error"] = str(exc)
        return info

    def _capture_via_screenshot(self, element) -> Image.Image | None:
        try:
            png_bytes = element.screenshot_as_png
            return Image.open(BytesIO(png_bytes))
        except Exception as exc:  # pragma: no cover - depende de Selenium
            LOGGER.debug("Screenshot directo falló: %s", exc)
            return None

    def _capture_from_src(self, driver, src: str) -> Image.Image | None:
        if not src:
            return None

        try:
            if src.startswith("data:image"):
                match = re.search(r"base64,(.+)", src)
                if not match:
                    return None
                data = base64.b64decode(match.group(1))
                return Image.open(BytesIO(data))

            if src.startswith("http"):
                cookies = {cookie["name"]: cookie["value"] for cookie in driver.get_cookies()}
                response = requests.get(src, cookies=cookies, timeout=10)
                if response.status_code == 200 and len(response.content) > 100:
                    return Image.open(BytesIO(response.content))
        except Exception as exc:  # pragma: no cover
            LOGGER.debug("Captura desde src falló: %s", exc)
        return None

    def _save_capture(self, image: Image.Image, prefix: str) -> Path:
        path = self.temp_dir / f"{prefix}.png"
        image.save(path)
        return path


__all__ = [
    "CaptchaCaptureService",
    "BaseCaptchaSolver",
    "CaptureDiagnostics",
]
