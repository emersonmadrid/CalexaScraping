"""CLI entry point for the CalexaScraping automation."""

from __future__ import annotations

import sys
import time
from pathlib import Path

from src.config.settings import AppSettings, load_settings
from src.models.data_models import Expediente, load_expedientes
from src.services.browser_manager import BrowserManager
from src.services.captcha_common import CaptchaCaptureService
from src.services.captcha_manager import CaptchaManager
from src.services.captcha_solver import CaptchaSolverOptimizado
from src.services.captcha_solver_cnn import CaptchaSolverCNN
from src.services.captcha_solver_ml import CaptchaSolverML
from src.services.form_filler import FormFiller
from src.utils.logger import configure_logging, get_logger

LOGGER = get_logger(__name__)


def build_captcha_manager(settings: AppSettings) -> CaptchaManager:
    capture_service = CaptchaCaptureService(settings.temp_dir)

    solvers = []

    cnn_solver = CaptchaSolverCNN(settings.training_dir / "cnn")
    if cnn_solver.modelo:
        solvers.append(cnn_solver)

    ml_solver = CaptchaSolverML(settings.training_dir)
    if ml_solver.patrones:
        solvers.append(ml_solver)

    solvers.append(CaptchaSolverOptimizado())

    if not solvers:
        raise RuntimeError("No hay solvers disponibles para resolver el CAPTCHA")

    return CaptchaManager(
        capture_service,
        solvers,
        max_attempts=settings.max_captcha_attempts,
    )


def procesar_expedientes(settings: AppSettings, expedientes: list[Expediente]) -> None:
    if not expedientes:
        LOGGER.warning("No se encontraron expedientes para procesar")
        return

    captcha_manager = build_captcha_manager(settings)
    form_filler = FormFiller(captcha_manager)

    with BrowserManager(settings) as driver:
        driver.get(settings.base_url)
        time.sleep(2)

        for expediente in expedientes:
            LOGGER.info("Procesando expediente %s", expediente.numero_expediente)
            exito = form_filler.llenar_formulario(driver, expediente)

            if not exito:
                LOGGER.warning("No se pudo completar la búsqueda para %s", expediente.numero_expediente)

            driver.get(settings.base_url)
            time.sleep(2)


def main() -> int:
    settings = load_settings()
    configure_logging(log_dir=settings.logs_dir)

    try:
        expedientes = load_expedientes(settings.expedientes_csv)
    except Exception as exc:
        LOGGER.error("No se pudo cargar el archivo de expedientes: %s", exc)
        return 1

    try:
        procesar_expedientes(settings, expedientes)
    except KeyboardInterrupt:
        LOGGER.warning("Ejecución interrumpida por el usuario")
        return 130
    except Exception as exc:
        LOGGER.exception("Error inesperado: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
