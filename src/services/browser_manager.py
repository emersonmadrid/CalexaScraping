"""Browser management utilities."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Optional

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

from src.config.settings import AppSettings
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


class BrowserManager(AbstractContextManager):
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.driver: Optional[webdriver.Chrome] = None

    def __enter__(self):
        self.driver = self.iniciar_navegador(headless=self.settings.headless)
        return self.driver

    def __exit__(self, exc_type, exc, tb):
        if self.driver:
            self.driver.quit()
            LOGGER.debug("Navegador cerrado")
            self.driver = None

    def iniciar_navegador(self, headless: bool | None = None) -> webdriver.Chrome:
        """Inicia y configura el navegador Chrome."""
        headless = self.settings.headless if headless is None else headless

        if not self.settings.chromedriver_path.exists():
            raise FileNotFoundError(
                f"ChromeDriver no encontrado en: {self.settings.chromedriver_path}"
            )

        options = Options()
        if headless:
            options.add_argument("--headless=new")

        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1920,1080")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)

        service = Service(str(self.settings.chromedriver_path))
        LOGGER.debug("Iniciando ChromeDriver en %s", self.settings.chromedriver_path)
        driver = webdriver.Chrome(service=service, options=options)

        driver.execute_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )
        LOGGER.info("Navegador iniciado correctamente (headless=%s)", headless)
        return driver
