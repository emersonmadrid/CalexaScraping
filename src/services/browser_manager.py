# src/services/browser_manager.py
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import os

class BrowserManager:
    def __init__(self):
        # Ruta ABSOLUTA para evitar problemas
        current_dir = os.getcwd()
        self.chromedriver_path = os.path.join(current_dir, 'drivers', 'chromedriver.exe')
        print(f"🔧 Directorio actual: {current_dir}")
        print(f"🔧 ChromeDriver path: {self.chromedriver_path}")
        print(f"🔧 ChromeDriver existe: {os.path.exists(self.chromedriver_path)}")

    def iniciar_navegador(self, headless=False):
        """Inicia y configura el navegador Chrome"""
        try:
            # Verificar que el archivo existe
            if not os.path.exists(self.chromedriver_path):
                raise FileNotFoundError(f"ChromeDriver no encontrado en: {self.chromedriver_path}")
            
            chrome_options = Options()
            
            if headless:
                chrome_options.add_argument("--headless")
            
            # Configuraciones básicas
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--window-size=1920,1080")
            
            print("🔧 Iniciando ChromeDriver...")
            service = Service(self.chromedriver_path)
            driver = webdriver.Chrome(service=service, options=chrome_options)
            
            # Evitar detección
            driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            print("✅ Navegador iniciado correctamente")
            return driver
            
        except Exception as e:
            print(f"❌ Error iniciando navegador: {e}")
            raise