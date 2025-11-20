from __future__ import annotations

import time
from pathlib import Path

from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from src.config import selectors
from src.models.data_models import Expediente
from src.services.captcha_manager import CaptchaManager
from src.utils.helpers import ensure_dir, timestamp
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


class FormFiller:
    def __init__(self, captcha_manager: CaptchaManager):
        self.captcha_manager = captcha_manager

    def llenar_formulario(
        self,
        driver,
        expediente: Expediente,
        *,
        max_reintentos: int = 3,
    ) -> bool:
        """Lena el formulario completo con validación."""
        for intento in range(1, max_reintentos + 1):
            try:
                LOGGER.info("Llenando formulario (intento %s/%s)", intento, max_reintentos)
                self._ingresar_expediente(driver, expediente)

                prediction = self.captcha_manager.solve(driver)
                if not prediction:
                    LOGGER.warning("CAPTCHA no resuelto en intento %s", intento)
                    driver.refresh()
                    time.sleep(2)
                    continue

                captcha_input = WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.ID, selectors.CAPTCHA_INPUT))
                )
                captcha_input.clear()
                captcha_input.send_keys(prediction.text)
                LOGGER.info("CAPTCHA ingresado con solver %s", prediction.solver)

                buscar_btn = driver.find_element(By.ID, selectors.BUSCAR_BUTTON)
                buscar_btn.click()
                time.sleep(3)

                if self.verificar_resultado(driver):
                    return True

                LOGGER.warning("Resultado no concluyente, reintentando")
                driver.refresh()
                time.sleep(2)
            except Exception as exc:
                LOGGER.error("Error llenando formulario: %s", exc)
                if intento < max_reintentos:
                    driver.refresh()
                    time.sleep(2)

        LOGGER.error("Formulario no pudo completarse después de %s intentos", max_reintentos)
        return False

    def _ingresar_expediente(self, driver, expediente: Expediente) -> None:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.NAME, selectors.EXPEDIENTE_INPUT))
        )
        campo = driver.find_element(By.NAME, selectors.EXPEDIENTE_INPUT)
        campo.clear()
        campo.send_keys(expediente.numero_expediente)
        LOGGER.debug("Ingresado expediente %s", expediente.numero_expediente)

    def verificar_resultado(self, driver) -> bool:
        """Verifica el resultado de la búsqueda."""
        try:
            time.sleep(2)
            resultados_dir = ensure_dir(Path("data") / "temp" / "resultados")
            screenshot_path = resultados_dir / f"resultado_{timestamp()}.png"
            driver.save_screenshot(str(screenshot_path))
            LOGGER.info("Resultado guardado en %s", screenshot_path)

            page_source = driver.page_source.lower()

            if "código de captcha incorrecto" in page_source or "captcha incorrecto" in page_source:
                LOGGER.warning("CAPTCHA incorrecto reportado por el sitio")
                return False

            if (
                "no se encontraron registros" in page_source
                or "no se encontraron resultados" in page_source
            ):
                LOGGER.info("Búsqueda ejecutada sin resultados")
                return True

            if "expediente" in page_source or "resultado" in page_source:
                LOGGER.info("¡Búsqueda exitosa!")
                return True

            LOGGER.warning("Resultado incierto, revisar screenshot")
            return False
        except Exception as exc:
            LOGGER.error("Error verificando resultado: %s", exc)
            return False

    def extraer_datos(self, driver):
        """Placeholder para extracción de datos de la página de resultados."""
        try:
            from selenium.webdriver.common.by import By

            page_source = driver.page_source
            try:
                driver.find_element(By.TAG_NAME, "table")
                LOGGER.info("Tabla de resultados detectada")
            except Exception:
                LOGGER.warning("No se encontró tabla de resultados")
            return {"html": page_source}
        except Exception as exc:
            LOGGER.error("Error extrayendo datos: %s", exc)
            return None
