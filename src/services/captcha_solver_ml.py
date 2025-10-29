#!/usr/bin/env python3
# src/services/captcha_solver_ml.py
"""
Solver de CAPTCHA con enfoque de Machine Learning
Entrena un modelo con CAPTCHAs etiquetados manualmente
"""
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import os
import pickle
import json
from collections import Counter
from datetime import datetime

class CaptchaSolverML:
    def __init__(self):
        """
        Solver que aprende de ejemplos etiquetados
        """
        print("🔧 Inicializando solver ML...")
        
        # Intentar cargar EasyOCR como fallback
        self.usar_easyocr = False
        try:
            import easyocr
            self.reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            self.usar_easyocr = True
            print("✅ EasyOCR disponible como fallback")
        except:
            print("⚠️ EasyOCR no disponible")
        
        # Directorio de entrenamiento
        self.training_dir = "data/training"
        self.model_path = "data/training/captcha_patterns.pkl"
        
        # Cargar patrones si existen
        self.patrones = self.cargar_patrones()
        
        print(f"✅ Solver ML listo ({len(self.patrones)} patrones)")
    
    def cargar_patrones(self):
        """Carga patrones de CAPTCHAs conocidos"""
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                patrones = pickle.load(f)
                print(f"📚 {len(patrones)} patrones cargados")
                return patrones
        return {}
    
    def guardar_patrones(self):
        """Guarda patrones aprendidos"""
        os.makedirs(self.training_dir, exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.patrones, f)
        print(f"💾 {len(self.patrones)} patrones guardados")
    
    def obtener_imagen_captcha(self, driver):
        """Obtiene imagen del CAPTCHA"""
        from selenium.webdriver.common.by import By
        
        try:
            captcha_img = driver.find_element(By.ID, "captcha_image")
            png_bytes = captcha_img.screenshot_as_png
            imagen = Image.open(BytesIO(png_bytes))
            return imagen
        except Exception as e:
            print(f"❌ Error capturando: {e}")
            return None
    
    def calcular_hash_visual(self, imagen_pil):
        """
        Calcula un hash perceptual de la imagen
        Para detectar CAPTCHAs similares/idénticos
        """
        # Redimensionar a tamaño fijo
        img_small = imagen_pil.resize((32, 32), Image.Resampling.LANCZOS)
        
        # Convertir a escala de grises
        img_gray = img_small.convert('L')
        
        # Convertir a array y normalizar
        pixels = np.array(img_gray).flatten()
        
        # Hash simple: promedio de pixeles
        avg = pixels.mean()
        hash_bits = ''.join(['1' if p > avg else '0' for p in pixels])
        
        return hash_bits
    
    def distancia_hamming(self, hash1, hash2):
        """Calcula similitud entre dos hashes"""
        if len(hash1) != len(hash2):
            return 100
        return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
    
    def buscar_en_patrones(self, imagen_pil, umbral=100):
        """
        Busca si este CAPTCHA ya está en los patrones
        
        Returns:
            str or None: Texto si se encontró coincidencia
        """
        if not self.patrones:
            return None
        
        img_hash = self.calcular_hash_visual(imagen_pil)
        
        # Buscar coincidencias cercanas
        coincidencias = []
        for hash_guardado, texto in self.patrones.items():
            distancia = self.distancia_hamming(img_hash, hash_guardado)
            if distancia < umbral:
                coincidencias.append((distancia, texto))
        
        if coincidencias:
            # Ordenar por distancia
            coincidencias.sort(key=lambda x: x[0])
            mejor = coincidencias[0]
            print(f"   🎯 Coincidencia en BD (distancia={mejor[0]}): '{mejor[1]}'")
            return mejor[1]
        
        return None
    
    def agregar_a_entrenamiento(self, imagen_pil, texto_correcto):
        """
        Agrega un CAPTCHA etiquetado al conjunto de entrenamiento
        """
        # Guardar imagen original
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(f"{self.training_dir}/images", exist_ok=True)
        
        ruta_img = f"{self.training_dir}/images/{texto_correcto}_{timestamp}.png"
        imagen_pil.save(ruta_img)
        
        # Agregar a patrones
        img_hash = self.calcular_hash_visual(imagen_pil)
        self.patrones[img_hash] = texto_correcto
        
        # Guardar
        self.guardar_patrones()
        
        print(f"   📚 Agregado a entrenamiento: '{texto_correcto}'")
        print(f"   💾 Imagen: {ruta_img}")
    
    def preprocesar_para_ocr(self, imagen_pil):
        """Preprocesamiento específico"""
        img = np.array(imagen_pil)
        
        # Convertir a escala de grises
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        # Redimensionar MUCHO más grande
        scale = 5
        width = int(gray.shape[1] * scale)
        height = int(gray.shape[0] * scale)
        large = cv2.resize(gray, (width, height), interpolation=cv2.INTER_CUBIC)
        
        # Denoise agresivo
        denoised = cv2.fastNlMeansDenoising(large, None, 15, 7, 21)
        
        # Contraste mejorado
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Umbral adaptativo
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 2
        )
        
        return binary
    
    def ocr_fallback(self, imagen_pil):
        """OCR con EasyOCR como fallback"""
        if not self.usar_easyocr:
            return None
        
        try:
            # Preprocesar
            img_proc = self.preprocesar_para_ocr(imagen_pil)
            
            # Aplicar OCR
            resultados = self.reader.readtext(
                img_proc,
                detail=0,
                paragraph=False,
                text_threshold=0.4,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            )
            
            if resultados:
                texto = ''.join(resultados).strip().upper()
                texto = ''.join(c for c in texto if c.isalnum())
                return texto if 3 <= len(texto) <= 8 else None
                
        except Exception as e:
            print(f"   ⚠️ OCR fallback falló: {e}")
        
        return None
    
    def resolver_captcha(self, driver, modo_entrenamiento=False):
        """
        Resuelve CAPTCHA con aprendizaje
        
        Args:
            driver: WebDriver de Selenium
            modo_entrenamiento: Si es True, pide confirmación manual
        
        Returns:
            str or None: Texto del CAPTCHA
        """
        print("\n🔍 Resolviendo CAPTCHA...")
        
        # 1. Capturar imagen
        imagen = self.obtener_imagen_captcha(driver)
        if not imagen:
            return None
        
        # Guardar para referencia
        os.makedirs("data/temp/captchas", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ruta_temp = f"data/temp/captchas/captcha_{timestamp}.png"
        imagen.save(ruta_temp)
        print(f"   💾 Guardado: {ruta_temp}")
        
        # 2. Buscar en patrones conocidos
        texto_patron = self.buscar_en_patrones(imagen)
        
        if texto_patron:
            if modo_entrenamiento:
                print(f"   ❓ ¿Es correcto '{texto_patron}'? (s/n): ", end='')
                if input().lower() == 's':
                    return texto_patron
            else:
                return texto_patron
        
        # 3. Intentar OCR
        print("   🔍 Intentando OCR...")
        texto_ocr = self.ocr_fallback(imagen)
        
        if texto_ocr:
            print(f"   📖 OCR detectó: '{texto_ocr}'")
            
            if modo_entrenamiento:
                print(f"   ❓ ¿Es correcto '{texto_ocr}'? (s/n/c=corregir): ", end='')
                respuesta = input().lower()
                
                if respuesta == 's':
                    self.agregar_a_entrenamiento(imagen, texto_ocr)
                    return texto_ocr
                elif respuesta == 'c':
                    print("   💡 Ingresa el texto correcto: ", end='')
                    texto_correcto = input().upper().strip()
                    self.agregar_a_entrenamiento(imagen, texto_correcto)
                    return texto_correcto
            else:
                return texto_ocr
        
        # 4. Modo manual si nada funcionó
        if modo_entrenamiento:
            print("   ❌ No se pudo resolver automáticamente")
            print("   💡 Ingresa el texto manualmente: ", end='')
            texto_manual = input().upper().strip()
            
            if texto_manual:
                self.agregar_a_entrenamiento(imagen, texto_manual)
                return texto_manual
        
        return None
    
    def entrenar_interactivo(self, driver, num_ejemplos=20):
        """
        Modo de entrenamiento interactivo
        Recolecta ejemplos etiquetados para mejorar el modelo
        """
        print(f"\n{'='*60}")
        print(f"📚 MODO ENTRENAMIENTO INTERACTIVO")
        print(f"{'='*60}")
        print(f"Objetivo: {num_ejemplos} ejemplos")
        print(f"Actual: {len(self.patrones)} ejemplos en BD")
        print()
        
        for i in range(num_ejemplos):
            print(f"\n{'='*50}")
            print(f"📋 Ejemplo {i+1}/{num_ejemplos}")
            print(f"{'='*50}")
            
            texto = self.resolver_captcha(driver, modo_entrenamiento=True)
            
            if texto:
                print(f"✅ Registrado: '{texto}'")
            else:
                print(f"⏭️ Saltado")
            
            # Recargar CAPTCHA
            if i < num_ejemplos - 1:
                self.recargar_captcha(driver)
                import time
                time.sleep(2)
        
        print(f"\n{'='*60}")
        print(f"✅ ENTRENAMIENTO COMPLETADO")
        print(f"Total de ejemplos: {len(self.patrones)}")
        print(f"{'='*60}")
    
    def recargar_captcha(self, driver):
        """Recarga el CAPTCHA"""
        try:
            from selenium.webdriver.common.by import By
            import time
            
            reload_btn = driver.find_element(By.ID, "btnReload")
            reload_btn.click()
            time.sleep(1.5)
        except Exception as e:
            print(f"   ⚠️ No se pudo recargar: {e}")