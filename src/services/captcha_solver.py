# src/services/captcha_solver_avanzado.py
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import os
import easyocr
from collections import Counter

class CaptchaSolverAvanzado:
    def __init__(self):
        print("🔧 Inicializando OCR avanzado...")
        self.reader = easyocr.Reader(['en'])
        print("✅ OCR avanzado listo")
    
    def mejorar_imagen(self, imagen_pil):
        """Mejora la imagen para mejor reconocimiento"""
        # Convertir a OpenCV
        img_cv = cv2.cvtColor(np.array(imagen_pil), cv2.COLOR_RGB2BGR)
        gris = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Diferentes técnicas de mejora
        tecnicas = []
        
        # 1. Contraste normal
        normal = cv2.convertScaleAbs(gris, alpha=1.5, beta=0)
        tecnicas.append(normal)
        
        # 2. Alto contraste
        alto_contraste = cv2.convertScaleAbs(gris, alpha=2.5, beta=0)
        tecnicas.append(alto_contraste)
        
        # 3. Binario Otsu
        _, binario = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        tecnicas.append(binario)
        
        # 4. Binario inverso (a veces funciona mejor)
        _, binario_inv = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        tecnicas.append(binario_inv)
        
        return tecnicas
    
    def corregir_inteligentemente(self, texto):
        """Correcciones inteligentes basadas en patrones"""
        if not texto:
            return None
            
        texto = texto.replace(' ', '').upper()
        print(f"🔍 OCR vio: '{texto}'")
        
        # Si es muy corto, descartar
        if len(texto) < 3:
            return None
        
        # Correcciones específicas para confusiones comunes
        correcciones = {
            '0': 'O', '1': 'I', '2': 'Z', '5': 'S', '6': 'G', '8': 'B',
            'C': 'G', 'G': 'C', 'I': '1', 'O': '0', 'S': '5', 'Z': '2',
            'B': '8', 'Q': 'O', 'D': 'O', 'U': 'V', 'V': 'U',
            'X': 'Y', 'Y': 'X',  # Intercambiar X e Y
        }
        
        # Aplicar correcciones
        texto_corregido = ''
        for letra in texto:
            texto_corregido += correcciones.get(letra, letra)
        
        # Solo mantener letras y números
        texto_final = ''.join(c for c in texto_corregido if c.isalnum())
        
        # Aceptar 3-5 caracteres
        if 3 <= len(texto_final) <= 5:
            print(f"🔧 Corregido a: '{texto_final}'")
            return texto_final
            
        return None
    
    def resolver_captcha(self, imagen_element):
        """Resuelve CAPTCHA con técnicas avanzadas"""
        try:
            # Capturar imagen
            captcha_png = imagen_element.screenshot_as_png
            imagen_pil = Image.open(BytesIO(captcha_png))
            
            # Guardar para análisis
            import random
            archivo = f"data/temp/captcha_avanzado_{random.randint(1000,9999)}.png"
            os.makedirs("data/temp", exist_ok=True)
            imagen_pil.save(archivo)
            print(f"📷 CAPTCHA guardado: {archivo}")
            
            # Probar múltiples técnicas de mejora
            imagenes_mejoradas = self.mejorar_imagen(imagen_pil)
            todos_textos = []
            
            print("🔍 Probando técnicas avanzadas...")
            
            for i, img_mejorada in enumerate(imagenes_mejoradas):
                try:
                    # Probar diferentes configuraciones de EasyOCR
                    configuraciones = [
                        {},  # Normal
                        {'text_threshold': 0.3, 'low_text': 0.2},  # Más sensible
                        {'text_threshold': 0.6, 'low_text': 0.4},  # Menos sensible
                    ]
                    
                    for config in configuraciones:
                        textos = self.reader.readtext(img_mejorada, detail=0, **config)
                        for texto in textos:
                            corregido = self.corregir_inteligentemente(texto)
                            if corregido:
                                todos_textos.append(corregido)
                                print(f"  ✅ Técnica {i+1}: '{corregido}'")
                except Exception as e:
                    print(f"  ❌ Técnica {i+1} falló: {e}")
            
            # Elegir el mejor resultado
            if todos_textos:
                contador = Counter(todos_textos)
                texto_ganador, veces = contador.most_common(1)[0]
                print(f"🎯 CAPTCHA resuelto: '{texto_ganador}' (apareció {veces} veces)")
                return texto_ganador
            else:
                print("❌ No se pudo resolver el CAPTCHA")
                return None
                
        except Exception as e:
            print(f"💥 Error: {e}")
            return None