#!/usr/bin/env python3
# test_modelo_entrenado.py
"""
Prueba la precisión del modelo entrenado
"""
import sys
sys.path.append('src')

from services.captcha_solver_ml import CaptchaSolverML
from services.browser_manager import BrowserManager
from selenium.webdriver.common.by import By
from PIL import Image
from io import BytesIO
import time

def main():
    print("=" * 60)
    print("🧪 TEST DEL MODELO ENTRENADO")
    print("=" * 60)
    
    # Cargar solver
    solver = CaptchaSolverML()
    
    if len(solver.patrones) == 0:
        print("\n❌ No hay patrones entrenados")
        print("💡 Ejecuta primero: python entrenar_manual.py")
        return
    
    print(f"\n📚 Patrones en base de datos: {len(solver.patrones)}")
    print()
    print("¿Cuántas pruebas realizar? (recomendado: 10): ", end='')
    try:
        num_pruebas = int(input())
    except:
        num_pruebas = 10
    
    driver = None
    
    try:
        # Iniciar
        print("\n🌐 Iniciando navegador...")
        browser = BrowserManager()
        driver = browser.iniciar_navegador(headless=False)
        
        print("🔗 Navegando a CEJ...")
        driver.get("https://cej.pj.gob.pe/cej/forms/busquedaform.html")
        time.sleep(3)
        driver.maximize_window()
        
        # Pruebas
        resultados = {
            'aciertos': 0,
            'fallos': 0,
            'no_encontrados': 0
        }
        
        print(f"\n🎯 Iniciando {num_pruebas} pruebas...")
        print()
        
        for i in range(num_pruebas):
            print(f"\n{'='*50}")
            print(f"Prueba {i+1}/{num_pruebas}")
            print(f"{'='*50}")
            
            # Capturar
            try:
                captcha_img = driver.find_element(By.ID, "captcha_image")
                png_bytes = captcha_img.screenshot_as_png
                imagen = Image.open(BytesIO(png_bytes))
            except:
                print("❌ Error capturando")
                continue
            
            # Buscar en patrones
            img_hash = solver.calcular_hash_visual(imagen)
            encontrado = None
            mejor_distancia = 999
            
            for hash_guardado, texto in solver.patrones.items():
                distancia = solver.distancia_hamming(img_hash, hash_guardado)
                if distancia < mejor_distancia:
                    mejor_distancia = distancia
                    encontrado = texto
            
            print(f"🔍 Búsqueda:")
            if mejor_distancia < 100:
                print(f"   ✅ Coincidencia (distancia={mejor_distancia}): '{encontrado}'")
                print()
                print("👁️  MIRA EL CAPTCHA EN EL NAVEGADOR")
                print()
                print(f"❓ ¿Es correcto '{encontrado}'? (s/n): ", end='')
                respuesta = input().lower()
                
                if respuesta == 's':
                    print("✅ ACIERTO")
                    resultados['aciertos'] += 1
                else:
                    print("❌ FALLO")
                    resultados['fallos'] += 1
                    print("💡 ¿Cuál era el correcto?: ", end='')
                    correcto = input().upper().strip()
                    if correcto:
                        print(f"📝 Debería ser '{correcto}' no '{encontrado}'")
            else:
                print(f"   ❌ No encontrado (distancia mínima={mejor_distancia})")
                resultados['no_encontrados'] += 1
                print()
                print("💡 ¿Quieres agregarlo? Escribe el texto (o Enter para saltar): ", end='')
                texto = input().upper().strip()
                if texto:
                    solver.patrones[img_hash] = texto
                    solver.guardar_patrones()
                    print(f"✅ Agregado: '{texto}'")
            
            # Recargar
            if i < num_pruebas - 1:
                try:
                    reload_btn = driver.find_element(By.ID, "btnReload")
                    reload_btn.click()
                    time.sleep(1.5)
                except:
                    pass
        
        # Resultados
        total = resultados['aciertos'] + resultados['fallos']
        precision = (resultados['aciertos'] / total * 100) if total > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"📊 RESULTADOS")
        print(f"{'='*60}")
        print(f"Aciertos: {resultados['aciertos']}")
        print(f"Fallos: {resultados['fallos']}")
        print(f"No encontrados: {resultados['no_encontrados']}")
        print(f"Precisión: {precision:.1f}%")
        print()
        
        if precision >= 80:
            print("🎉 ¡Excelente! El modelo está listo para producción")
        elif precision >= 60:
            print("👍 Bueno, pero entrena más ejemplos para mejorar")
        else:
            print("⚠️ Necesitas entrenar más ejemplos")
        
        print(f"\nPatrones actuales: {len(solver.patrones)}")
        print(f"{'='*60}")
        
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