"""High level coordinator that tries multiple CAPTCHA solvers."""

from __future__ import annotations

import time
from typing import Iterable, List, Sequence

from src.models.data_models import CaptchaPrediction, CapturedCaptcha
from src.services.captcha_common import BaseCaptchaSolver, CaptchaCaptureService
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


class CaptchaManager:
    def __init__(
        self,
        capture_service: CaptchaCaptureService,
        solvers: Sequence[BaseCaptchaSolver],
        *,
        max_attempts: int = 3,
    ) -> None:
        self.capture_service = capture_service
        self.solvers: List[BaseCaptchaSolver] = sorted(
            solvers,
            key=lambda solver: solver.priority,
        )
        self.max_attempts = max_attempts

    def solve(self, driver) -> CaptchaPrediction | None:
        """Try every solver until one produces a prediction."""
        for attempt in range(1, self.max_attempts + 1):
            capture = self.capture_service.capture_image(driver)
            if not capture:
                LOGGER.debug("Intento %s: no se capturó imagen", attempt)
                continue

            for solver in self.solvers:
                try:
                    prediction = solver.predict(capture)
                except Exception as exc:  # pragma: no cover - depende de libs externas
                    LOGGER.exception(
                        "Solver %s falló durante la predicción: %s", solver.name, exc
                    )
                    continue

                if prediction:
                    prediction.metadata.setdefault("attempt", attempt)
                    LOGGER.info(
                        "Solver %s resolvió el CAPTCHA en intento %s con '%s'",
                        solver.name,
                        attempt,
                        prediction.text,
                    )
                    return prediction

            self.capture_service.reload(driver)
            time.sleep(1)

        LOGGER.error("No se pudo resolver el CAPTCHA tras %s intentos", self.max_attempts)
        return None

    @property
    def solver_names(self) -> list[str]:
        return [solver.name for solver in self.solvers]


__all__ = ["CaptchaManager"]
