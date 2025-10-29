# src/services/form_filler.py
from selenium.webdriver.common.by import By
import time
import os
from src.config.selectors import *

# CAMBIA ESTA LÍNEA:
# from services.captcha_solver import CaptchaSolver
# POR ESTA:
from services.captcha_solver import CaptchaSolverAvanzado

class FormFiller:
    def __init__(self, captcha_solver):
        self.captcha_solver = captcha_solver

    def llenar_formulario(self, driver, datos_expediente):
        """Llena el formulario del CEJ"""
        try:
            print("📝 Llenando formulario...")
            
            # 1. Llenar expediente
            expediente_input = driver.find_element(By.NAME, EXPEDIENTE_INPUT)
            expediente_input.clear()
            expediente_input.send_keys(datos_expediente['numero_expediente'])
            print(f"✅ Expediente: {datos_expediente['numero_expediente']}")
            
            # 2. Resolver CAPTCHA
            if self.resolver_captcha(driver):
                print("✅ Formulario listo")
                return True
            return False
            
        except Exception as e:
            print(f"❌ Error llenando formulario: {e}")
            return False

    def resolver_captcha(self, driver, max_intentos=3):
        """Resuelve el CAPTCHA con reintentos"""
        for intento in range(max_intentos):
            try:
                print(f"🔍 CAPTCHA intento {intento + 1}/{max_intentos}")
                
                captcha_image = driver.find_element(By.ID, CAPTCHA_IMAGE)
                texto = self.captcha_solver.resolver_captcha(captcha_image)
                
                if texto:
                    captcha_input = driver.find_element(By.ID, CAPTCHA_INPUT)
                    captcha_input.clear()
                    captcha_input.send_keys(texto)
                    print(f"✅ CAPTCHA ingresado: {texto}")
                    return True
                else:
                    print("🔄 Refrescando CAPTCHA...")
                    reload_btn = driver.find_element(By.ID, RELOAD_CAPTCHA_BUTTON)
                    reload_btn.click()
                    time.sleep(2)
                    
            except Exception as e:
                print(f"❌ Error en CAPTCHA: {e}")
        
        print("❌ No se pudo resolver el CAPTCHA")
        return False

    def buscar(self, driver):
        """Hace clic en el botón buscar"""
        try:
            buscar_btn = driver.find_element(By.ID, BUSCAR_BUTTON)
            buscar_btn.click()
            print("✅ Búsqueda iniciada")
            time.sleep(5)
            return True
        except Exception as e:
            print(f"❌ Error en búsqueda: {e}")
            return False