#!/usr/bin/env python3
# entrenar_manual.py
"""
Entrenamiento 100% manual - más rápido para empezar
El CAPTCHA se muestra en el navegador y tú lo escribes
"""
import sys
sys.path.append('src')

from selenium.webdriver.common.by import By
from services.browser_manager import BrowserManager
from PIL import Image
from io import BytesIO
import os
import pickle
import time
import cv2
import numpy as np

class EntrenadorManual:
    def __init__(self):
        self.training_dir = "data/training"
        self.patrones_file = f"{self.training_dir}/captcha_patterns.pkl"
        self.patrones = self.cargar_patrones()
        
    def cargar_patrones(self):
        """Carga patrones existentes"""
        if os.path.exists(self.patrones_file):
            with open(self.patrones_file, 'rb') as f:
                patrones = pickle.load(f)
                print(f"📚 {len(patrones)} patrones cargados de sesiones anteriores")
                return patrones
        return {}
    
    def guardar_patrones(self):
        """Guarda patrones"""
        os.makedirs(self.training_dir, exist_ok=True)
        with open(self.patrones_file, 'wb') as f:
            pickle.dump(self.patrones, f)
        print(f"💾 {len(self.patrones)} patrones guardados")
    
    def calcular_hash(self, imagen_pil):
        """Calcula hash perceptual simple"""
        # Redimensionar
        img_small = imagen_pil.resize((32, 32), Image.Resampling.LANCZOS)
        
        # Escala de grises
        img_gray = img_small.convert('L')
        
        # Hash basado en promedio
        pixels = np.array(img_gray).flatten()
        avg = pixels.mean()
        hash_bits = ''.join(['1' if p > avg else '0' for p in pixels])
        
        return hash_bits
    
    def capturar_captcha(self, driver):
        """Captura el CAPTCHA actual"""
        try:
            captcha_img = driver.find_element(By.ID, "captcha_image")
            png_bytes = captcha_img.screenshot_as_png
            return Image.open(BytesIO(png_bytes))
        except Exception as e:
            print(f"❌ Error capturando: {e}")
            return None
    
    def guardar_ejemplo(self, imagen, texto):
        """Guarda un ejemplo etiquetado"""
        # Calcular hash
        img_hash = self.calcular_hash(imagen)
        
        # Guardar en patrones
        self.patrones[img_hash] = texto
        
        # Guardar imagen física
        os.makedirs(f"{self.training_dir}/images", exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        ruta = f"{self.training_dir}/images/{texto}_{timestamp}.png"
        imagen.save(ruta)
        
        return img_hash
    
    def recargar_captcha(self, driver):
        """Recarga el CAPTCHA"""
        try:
            reload_btn = driver.find_element(By.ID, "btnReload")
            reload_btn.click()
            time.sleep(1.5)
        except:
            pass
    
    def entrenar(self, driver, num_ejemplos=50):
        """
        Entrenamiento manual interactivo
        """
        print(f"\n{'='*60}")
        print(f"📚 ENTRENAMIENTO MANUAL")
        print(f"{'='*60}")
        print(f"Meta: {num_ejemplos} ejemplos")
        print(f"Actual: {len(self.patrones)} ejemplos")
        print()
        print("Instrucciones:")
        print("  • Mira el CAPTCHA en el navegador")
        print("  • Escribe exactamente lo que ves (mayúsculas)")
        print("  • Enter para continuar")
        print("  • Escribe 'saltar' para omitir uno difícil")
        print("  • Ctrl+C para terminar antes")
        print()
        
        guardados = 0
        saltados = 0
        
        try:
            for i in range(num_ejemplos):
                print(f"\n{'─'*50}")
                print(f"📋 Ejemplo {i+1}/{num_ejemplos}")
                print(f"{'─'*50}")
                
                # Capturar
                imagen = self.capturar_captcha(driver)
                
                if not imagen:
                    print("❌ No se pudo capturar, recargando...")
                    self.recargar_captcha(driver)
                    time.sleep(2)
                    continue
                
                # Mostrar info
                print(f"📏 Tamaño: {imagen.size}")
                print()
                print("👁️  MIRA EL CAPTCHA EN EL NAVEGADOR")
                print()
                
                # Pedir texto
                print("💬 Escribe el texto del CAPTCHA: ", end='')
                texto = input().strip().upper()
                
                # Validar
                if not texto:
                    print("⏭️ Vacío, saltando...")
                    self.recargar_captcha(driver)
                    saltados += 1
                    continue
                
                if texto.lower() == 'saltar':
                    print("⏭️ Saltado")
                    self.recargar_captcha(driver)
                    saltados += 1
                    continue
                
                # Limpiar texto
                texto = ''.join(c for c in texto if c.isalnum())
                
                if not (3 <= len(texto) <= 8):
                    print(f"⚠️ Longitud inusual: {len(texto)} caracteres")
                    print("¿Continuar de todas formas? (s/n): ", end='')
                    if input().lower() != 's':
                        self.recargar_captcha(driver)
                        continue
                
                # Guardar
                img_hash = self.guardar_ejemplo(imagen, texto)
                guardados += 1
                
                print(f"✅ Guardado: '{texto}'")
                print(f"   Hash: {img_hash[:16]}...")
                print(f"   Total: {len(self.patrones)} patrones")
                
                # Guardar cada 5 ejemplos
                if guardados % 5 == 0:
                    self.guardar_patrones()
                    print(f"   💾 Checkpoint guardado")
                
                # Recargar para siguiente
                if i < num_ejemplos - 1:
                    self.recargar_captcha(driver)
                    time.sleep(1)
        
        except KeyboardInterrupt:
            print("\n\n⚠️ Interrumpido por el usuario")
        
        # Guardar final
        self.guardar_patrones()
        
        # Resumen
        print(f"\n{'='*60}")
        print(f"📊 RESUMEN")
        print(f"{'='*60}")
        print(f"Guardados: {guardados}")
        print(f"Saltados: {saltados}")
        print(f"Total en BD: {len(self.patrones)}")
        print(f"Ubicación: {self.training_dir}/")
        print(f"{'='*60}")
        
        return guardados

def main():
    print("=" * 60)
    print("🎓 ENTRENAMIENTO MANUAL DE CAPTCHA")
    print("=" * 60)
    print()
    print("Este método es más rápido que depender del OCR.")
    print("Tú ves el CAPTCHA y lo escribes directamente.")
    print()
    
    print("¿Cuántos ejemplos quieres entrenar? (recomendado: 50-100): ", end='')
    try:
        num = int(input())
    except:
        num = 50
        print(f"Usando valor por defecto: {num}")
    
    if num <= 0:
        print("❌ Número inválido")
        return
    
    print()
    print("💡 Consejos:")
    print("  • Usa buena iluminación en tu pantalla")
    print("  • Tómate tu tiempo, la precisión es importante")
    print("  • Salta CAPTCHAs muy difíciles de leer")
    print("  • Guarda cada 5 ejemplos automáticamente")
    print()
    
    print("¿Continuar? (s/n): ", end='')
    if input().lower() != 's':
        print("❌ Cancelado")
        return
    
    driver = None
    
    try:
        # Iniciar navegador
        print("\n🌐 Iniciando navegador...")
        from services.browser_manager import BrowserManager
        browser = BrowserManager()
        driver = browser.iniciar_navegador(headless=False)
        
        print("🔗 Navegando a CEJ...")
        driver.get("https://cej.pj.gob.pe/cej/forms/busquedaform.html")
        time.sleep(3)
        
        # Maximizar ventana para ver bien el CAPTCHA
        driver.maximize_window()
        
        print("✅ Navegador listo")
        print()
        input("👁️  VERIFICA QUE PUEDES VER EL CAPTCHA CLARAMENTE. Presiona Enter para comenzar...")
        
        # Entrenar
        entrenador = EntrenadorManual()
        guardados = entrenador.entrenar(driver, num)
        
        if guardados > 0:
            print()
            print("✅ ¡Entrenamiento completado!")
            print()
            print("🚀 Próximos pasos:")
            print("   1. Ejecuta: python test_modelo_entrenado.py")
            print("   2. Verifica la precisión del modelo")
            print("   3. Si es >80%, úsalo en producción")
            print("   4. Si es <80%, entrena más ejemplos")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if driver:
            input("\n⏸️  Presiona Enter para cerrar...")
            driver.quit()

if __name__ == "__main__":
    main()