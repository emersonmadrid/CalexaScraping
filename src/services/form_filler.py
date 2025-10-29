# src/services/form_filler.py
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from src.config.selectors import *

class FormFiller:
    def __init__(self, captcha_solver):
        self.captcha_solver = captcha_solver

    def llenar_formulario(self, driver, datos_expediente, max_reintentos=3):
        """
        Llena el formulario completo con validación y reintentos
        """
        for intento in range(max_reintentos):
            try:
                print(f"\n📝 Llenando formulario (intento {intento + 1}/{max_reintentos})...")
                
                # 1. Esperar que la página esté lista
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.NAME, EXPEDIENTE_INPUT))
                )
                
                # 2. Llenar número de expediente
                expediente_input = driver.find_element(By.NAME, EXPEDIENTE_INPUT)
                expediente_input.clear()
                time.sleep(0.5)
                expediente_input.send_keys(datos_expediente['numero_expediente'])
                print(f"✅ Expediente: {datos_expediente['numero_expediente']}")
                
                # 3. Esperar que el CAPTCHA esté visible
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.ID, "captcha_image"))
                )
                time.sleep(1)  # Pequeña espera para que cargue completamente
                
                # 4. Resolver CAPTCHA
                texto_captcha = self.captcha_solver.resolver_captcha(driver)
                
                if texto_captcha:
                    # Ingresar CAPTCHA
                    captcha_input = WebDriverWait(driver, 5).until(
                        EC.presence_of_element_located((By.ID, CAPTCHA_INPUT))
                    )
                    captcha_input.clear()
                    time.sleep(0.5)
                    captcha_input.send_keys(texto_captcha)
                    print(f"✅ CAPTCHA ingresado: {texto_captcha}")
                    
                    # 4. Hacer clic en buscar
                    buscar_btn = driver.find_element(By.ID, BUSCAR_BUTTON)
                    buscar_btn.click()
                    print("🔍 Búsqueda enviada...")
                    time.sleep(3)
                    
                    # 5. Verificar resultado
                    return self.verificar_resultado(driver)
                else:
                    print(f"❌ No se pudo resolver CAPTCHA en intento {intento + 1}")
                    if intento < max_reintentos - 1:
                        print("🔄 Recargando página...")
                        driver.refresh()
                        time.sleep(3)
                    
            except Exception as e:
                print(f"❌ Error en intento {intento + 1}: {e}")
                if intento < max_reintentos - 1:
                    driver.refresh()
                    time.sleep(3)
        
        print("❌ Formulario no pudo ser completado después de todos los intentos")
        return False

    def verificar_resultado(self, driver):
        """
        Verifica el resultado de la búsqueda
        """
        try:
            time.sleep(2)
            
            # Guardar screenshot del resultado
            import os
            os.makedirs("data/temp/resultados", exist_ok=True)
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = f"data/temp/resultados/resultado_{timestamp}.png"
            driver.save_screenshot(screenshot_path)
            print(f"📸 Resultado guardado: {screenshot_path}")
            
            # Analizar la página
            page_source = driver.page_source.lower()
            
            # Verificar diferentes escenarios
            if "código de captcha incorrecto" in page_source or "captcha incorrecto" in page_source:
                print("❌ CAPTCHA incorrecto - La búsqueda no se realizó")
                return False
            
            elif "no se encontraron registros" in page_source or "no se encontraron resultados" in page_source:
                print("✅ Búsqueda exitosa - No hay resultados para este expediente")
                return True
            
            elif "expediente" in page_source or "resultado" in page_source:
                print("🎉 ¡Búsqueda exitosa! - Se encontraron resultados")
                return True
            
            else:
                print("⚠️ Resultado incierto - Revisa el screenshot")
                return False
                
        except Exception as e:
            print(f"❌ Error verificando resultado: {e}")
            return False

    def extraer_datos(self, driver):
        """
        Extrae los datos del expediente de la página de resultados
        """
        try:
            # TODO: Implementar extracción según la estructura de la página de resultados
            # Por ahora, solo capturamos el HTML
            page_source = driver.page_source
            
            # Buscar tabla de resultados
            from selenium.webdriver.common.by import By
            try:
                tabla = driver.find_element(By.TAG_NAME, "table")
                # Extraer datos de la tabla
                # ... implementar según estructura
                print("✅ Datos extraídos")
                return {"html": page_source}
            except:
                print("⚠️ No se encontró tabla de resultados")
                return {"html": page_source}
                
        except Exception as e:
            print(f"❌ Error extrayendo datos: {e}")
            return None