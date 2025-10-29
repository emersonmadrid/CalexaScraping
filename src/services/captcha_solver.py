# src/services/captcha_solver_optimizado.py
import cv2
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import pytesseract
import os
import base64
import re

class CaptchaSolverOptimizado:
    def __init__(self):
        """
        Solver optimizado con múltiples métodos de captura
        """
        print("🔧 Inicializando solver optimizado...")
        
        # Configurar Tesseract si está disponible
        self.usar_tesseract = True
        try:
            pytesseract.get_tesseract_version()
            print("✅ Tesseract OCR detectado")
        except:
            self.usar_tesseract = False
            print("⚠️ Tesseract no disponible, usando solo EasyOCR")
        
        # Configurar EasyOCR
        try:
            import easyocr
            self.reader = easyocr.Reader(['en'], gpu=False)
            self.usar_easyocr = True
            print("✅ EasyOCR listo")
        except:
            self.usar_easyocr = False
            print("⚠️ EasyOCR no disponible")
        
        print("✅ Solver listo")
    
    def obtener_imagen_captcha(self, driver):
        """
        Obtiene la imagen del CAPTCHA probando múltiples métodos
        """
        from selenium.webdriver.common.by import By
        
        try:
            # Encontrar el elemento imagen
            captcha_img = driver.find_element(By.ID, "captcha_image")
            
            # MÉTODO 1: Screenshot directo (más confiable)
            print("   📸 Método 1: Screenshot directo...")
            try:
                png_bytes = captcha_img.screenshot_as_png
                imagen = Image.open(BytesIO(png_bytes))
                print(f"   ✅ Imagen capturada: {imagen.size}, modo: {imagen.mode}")
                return imagen
            except Exception as e:
                print(f"   ❌ Screenshot falló: {e}")
            
            # MÉTODO 2: Desde src (data URI o URL)
            print("   🔗 Método 2: Desde atributo src...")
            try:
                img_src = captcha_img.get_attribute("src")
                
                if not img_src:
                    print("   ❌ No hay src")
                else:
                    print(f"   📝 Src encontrado: {img_src[:100]}...")
                    
                    # Si es data URI
                    if img_src.startswith("data:image"):
                        print("   🔍 Procesando data URI...")
                        # Extraer la parte base64
                        match = re.search(r'base64,(.+)', img_src)
                        if match:
                            img_data = base64.b64decode(match.group(1))
                            imagen = Image.open(BytesIO(img_data))
                            print(f"   ✅ Imagen desde data URI: {imagen.size}")
                            return imagen
                    
                    # Si es URL HTTP
                    elif img_src.startswith("http"):
                        print("   🌐 Descargando desde URL...")
                        # Obtener cookies para mantener sesión
                        cookies = {cookie['name']: cookie['value'] 
                                 for cookie in driver.get_cookies()}
                        
                        response = requests.get(img_src, cookies=cookies, timeout=10)
                        if response.status_code == 200:
                            imagen = Image.open(BytesIO(response.content))
                            print(f"   ✅ Imagen descargada: {imagen.size}")
                            return imagen
            except Exception as e:
                print(f"   ❌ Src falló: {e}")
            
            # MÉTODO 3: Screenshot de toda la página y recortar
            print("   ✂️ Método 3: Screenshot y recorte...")
            try:
                # Obtener ubicación y tamaño del elemento
                location = captcha_img.location
                size = captcha_img.size
                
                # Screenshot de toda la página
                png_full = driver.get_screenshot_as_png()
                imagen_full = Image.open(BytesIO(png_full))
                
                # Recortar CAPTCHA
                left = location['x']
                top = location['y']
                right = left + size['width']
                bottom = top + size['height']
                
                imagen = imagen_full.crop((left, top, right, bottom))
                print(f"   ✅ Imagen recortada: {imagen.size}")
                return imagen
            except Exception as e:
                print(f"   ❌ Recorte falló: {e}")
            
            print("   ❌ Todos los métodos de captura fallaron")
            return None
            
        except Exception as e:
            print(f"   💥 Error general: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def preprocesar_imagen(self, imagen_pil):
        """
        Preprocesamiento mejorado para CAPTCHA del Poder Judicial
        """
        # Convertir a numpy array
        img = np.array(imagen_pil)
        
        # Convertir a RGB si es necesario
        if len(img.shape) == 2:  # Escala de grises
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:  # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        
        # Convertir a escala de grises
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Redimensionar MÁS GRANDE (mejor para distinguir P de R)
        scale = 4  # Aumentado de 3 a 4
        width = int(gray.shape[1] * scale)
        height = int(gray.shape[0] * scale)
        gray_large = cv2.resize(gray, (width, height), interpolation=cv2.INTER_CUBIC)
        
        # Denoise más agresivo
        denoised = cv2.fastNlMeansDenoising(gray_large, None, 10, 7, 21)
        
        # Preparar lista de resultados
        resultados = []
        
        # 1. Umbral simple con varios niveles
        for umbral in [100, 127, 150]:
            _, thresh = cv2.threshold(denoised, umbral, 255, cv2.THRESH_BINARY)
            resultados.append((f"thresh_{umbral}", thresh))
        
        # 2. Umbral Otsu
        _, thresh_otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        resultados.append(("thresh_otsu", thresh_otsu))
        
        # 3. Umbral adaptativo (varios tamaños de bloque)
        for block_size in [11, 15, 19]:
            thresh_adapt = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, block_size, 2
            )
            resultados.append((f"adapt_{block_size}", thresh_adapt))
        
        # 4. Morfología para limpiar
        kernel = np.ones((2, 2), np.uint8)
        morph = cv2.morphologyEx(thresh_otsu, cv2.MORPH_CLOSE, kernel)
        resultados.append(("morph", morph))
        
        return resultados
    
    def limpiar_texto(self, texto):
        """Limpia y valida texto extraído"""
        if not texto:
            return None
        
        # Limpiar
        texto = ''.join(c for c in texto if c.isalnum()).upper()
        
        # Validar longitud
        if not (3 <= len(texto) <= 8):
            return None
        
        return texto
    
    def ocr_tesseract(self, imagen_cv):
        """OCR con Tesseract - configuración optimizada"""
        if not self.usar_tesseract:
            return None
        try:
            # PSM 8 = Tratar como una sola palabra
            # OEM 3 = Default engine mode
            # tessedit_char_whitelist = Solo estos caracteres
            config = '--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            texto = pytesseract.image_to_string(imagen_cv, config=config)
            return texto.strip()
        except Exception as e:
            print(f"      ⚠️ Tesseract error: {e}")
            return None
    
    def ocr_easyocr(self, imagen_cv):
        """OCR con EasyOCR - configuración optimizada"""
        if not self.usar_easyocr:
            return None
        try:
            # Configuraciones más estrictas para mejor precisión
            resultados = self.reader.readtext(
                imagen_cv,
                detail=0,
                paragraph=False,
                text_threshold=0.5,  # Más estricto (antes 0.3)
                low_text=0.4,         # Más estricto
                link_threshold=0.4,
                canvas_size=2560,     # Mayor resolución
                mag_ratio=1.5
            )
            return ''.join(resultados).strip() if resultados else None
        except Exception as e:
            print(f"      ⚠️ EasyOCR error: {e}")
            return None
    
    def resolver_captcha(self, driver, max_intentos=3):
        """
        Resuelve el CAPTCHA con múltiples estrategias
        """
        for intento in range(max_intentos):
            try:
                print(f"\n🔍 Intento {intento + 1}/{max_intentos}")
                
                # 1. Obtener imagen
                imagen = self.obtener_imagen_captcha(driver)
                
                if not imagen:
                    print("❌ No se pudo capturar la imagen")
                    if intento < max_intentos - 1:
                        self.recargar_captcha(driver)
                    continue
                
                # Guardar original
                os.makedirs("data/temp/captchas", exist_ok=True)
                ruta_original = f"data/temp/captchas/original_{intento}.png"
                imagen.save(ruta_original)
                print(f"   💾 Original guardado: {ruta_original}")
                
                # 2. Preprocesar
                imagenes_procesadas = self.preprocesar_imagen(imagen)
                print(f"   🔧 {len(imagenes_procesadas)} versiones procesadas")
                
                # 3. Aplicar OCR
                todos_textos = []
                
                for nombre, img_proc in imagenes_procesadas:
                    # Guardar versión procesada
                    cv2.imwrite(f"data/temp/captchas/{nombre}_{intento}.png", img_proc)
                    
                    # Tesseract
                    texto_tess = self.ocr_tesseract(img_proc)
                    if texto_tess:
                        limpio = self.limpiar_texto(texto_tess)
                        if limpio:
                            print(f"      📖 Tesseract ({nombre}): {limpio}")
                            todos_textos.append(limpio)
                    
                    # EasyOCR
                    texto_easy = self.ocr_easyocr(img_proc)
                    if texto_easy:
                        limpio = self.limpiar_texto(texto_easy)
                        if limpio:
                            print(f"      📖 EasyOCR ({nombre}): {limpio}")
                            todos_textos.append(limpio)
                
                # 4. Elegir mejor resultado con análisis inteligente
                if todos_textos:
                    from collections import Counter
                    contador = Counter(todos_textos)
                    
                    # Mostrar todos los resultados
                    print(f"   📊 Resultados encontrados:")
                    for texto, frecuencia in contador.most_common():
                        print(f"      • '{texto}': {frecuencia} veces")
                    
                    # Si hay empate o resultados similares, aplicar lógica
                    mejor = self.elegir_mejor_captcha(contador, todos_textos)
                    
                    if mejor:
                        print(f"   🎯 CAPTCHA elegido: '{mejor}'")
                        return mejor
                else:
                    print("   ❌ No se pudo extraer texto")
                
                # 5. Recargar si hay más intentos
                if intento < max_intentos - 1:
                    self.recargar_captcha(driver)
                    
            except Exception as e:
                print(f"   💥 Error en intento {intento + 1}: {e}")
                import traceback
                traceback.print_exc()
        
        print("❌ CAPTCHA no resuelto después de todos los intentos")
        return None
    
    def elegir_mejor_captcha(self, contador, todos_textos):
        """
        Elige el mejor CAPTCHA aplicando lógica inteligente
        
        Args:
            contador: Counter con frecuencias
            todos_textos: Lista de todos los textos detectados
        
        Returns:
            str: Mejor CAPTCHA
        """
        # Si solo hay uno, usarlo
        if len(contador) == 1:
            return contador.most_common(1)[0][0]
        
        # Obtener los 2 más comunes
        top2 = contador.most_common(2)
        primero, freq1 = top2[0]
        
        # Si el primero tiene mayoría clara (>50%), usarlo
        total = sum(contador.values())
        if freq1 / total > 0.5:
            return primero
        
        # Si hay empate o frecuencias similares, comparar letra por letra
        if len(top2) > 1:
            segundo, freq2 = top2[1]
            
            # Si las frecuencias son iguales o muy cercanas
            if abs(freq1 - freq2) <= 1:
                print(f"   ⚖️ Empate técnico: '{primero}' vs '{segundo}'")
                
                # Comparar letra por letra con todos los resultados
                mejor = self.consenso_por_posicion(todos_textos)
                if mejor:
                    print(f"   🔍 Consenso letra por letra: '{mejor}'")
                    return mejor
        
        # Por defecto, el más común
        return primero
    
    def consenso_por_posicion(self, textos):
        """
        Crea consenso letra por letra
        
        Args:
            textos: Lista de textos detectados
        
        Returns:
            str: Texto consensuado
        """
        from collections import Counter
        
        # Filtrar solo textos de la misma longitud
        longitudes = Counter([len(t) for t in textos])
        longitud_comun = longitudes.most_common(1)[0][0]
        textos_filtrados = [t for t in textos if len(t) == longitud_comun]
        
        if not textos_filtrados:
            return None
        
        # Consenso letra por letra
        resultado = []
        for i in range(longitud_comun):
            letras_posicion = [t[i] for t in textos_filtrados]
            letra_mas_comun = Counter(letras_posicion).most_common(1)[0][0]
            resultado.append(letra_mas_comun)
        
        return ''.join(resultado)
    
    def recargar_captcha(self, driver):
        """Recarga el CAPTCHA"""
        try:
            from selenium.webdriver.common.by import By
            import time
            
            print("   🔄 Recargando CAPTCHA...")
            reload_btn = driver.find_element(By.ID, "btnReload")
            reload_btn.click()
            time.sleep(2)
            print("   ✅ CAPTCHA recargado")
        except Exception as e:
            print(f"   ⚠️ No se pudo recargar: {e}")