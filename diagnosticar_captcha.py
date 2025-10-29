#!/usr/bin/env python3
# test_captcha_visual.py
"""
Prueba el CAPTCHA solver y muestra resultados visuales
"""
import sys
sys.path.append('src')

from services.browser_manager import BrowserManager
from services.captcha_solver import CaptchaSolverOptimizado
from selenium.webdriver.common.by import By
import time

def main():
    print("=" * 60)
    print("🔍 TEST VISUAL DE CAPTCHA SOLVER")
    print("=" * 60)
    
    driver = None
    
    try:
        # Iniciar
        print("\n🌐 Iniciando navegador...")
        browser = BrowserManager()
        driver = browser.iniciar_navegador(headless=False)
        
        print("🔗 Navegando a CEJ...")
        driver.get("https://cej.pj.gob.pe/cej/forms/busquedaform.html")
        time.sleep(3)
        
        # Solver
        solver = CaptchaSolverOptimizado()
        
        # Probar varias veces
        num_pruebas = 5
        resultados = []
        
        print(f"\n🎯 Realizando {num_pruebas} pruebas...\n")
        
        for i in range(num_pruebas):
            print(f"{'='*60}")
            print(f"📋 PRUEBA {i+1}/{num_pruebas}")
            print(f"{'='*60}")
            
            # Resolver
            texto = solver.resolver_captcha(driver, max_intentos=1)
            
            if texto:
                print(f"✅ Resultado: '{texto}'")
                resultados.append(texto)
                
                # Mostrar las imágenes guardadas
                print(f"\n💡 Revisa las imágenes en:")
                print(f"   data/temp/captchas/original_{i}.png")
                print(f"   Y las versiones procesadas")
                
                # Preguntar si es correcto
                print(f"\n❓ ¿Es correcto '{texto}'? (s/n/r=recargar): ", end='')
                respuesta = input().lower()
                
                if respuesta == 's':
                    print("✅ Correcto!")
                elif respuesta == 'n':
                    print("❌ Incorrecto")
                    print("💡 ¿Cuál era el texto correcto?: ", end='')
                    correcto = input().upper()
                    print(f"📝 Anotado: '{correcto}' vs '{texto}'")
                elif respuesta == 'r':
                    print("🔄 Recargando...")
                    solver.recargar_captcha(driver)
                    time.sleep(2)
                    continue
            else:
                print("❌ No se pudo resolver")
            
            # Recargar para siguiente prueba
            if i < num_pruebas - 1:
                print("\n🔄 Recargando para siguiente prueba...")
                solver.recargar_captcha(driver)
                time.sleep(2)
        
        # Resumen
        print(f"\n{'='*60}")
        print("📊 RESUMEN")
        print(f"{'='*60}")
        print(f"Pruebas realizadas: {num_pruebas}")
        print(f"CAPTCHAs resueltos: {len(resultados)}")
        print(f"Tasa de éxito: {len(resultados)/num_pruebas*100:.1f}%")
        
        if resultados:
            print(f"\nResultados obtenidos:")
            for i, r in enumerate(resultados, 1):
                print(f"  {i}. {r}")
        
        print(f"\n💡 Todas las imágenes están en: data/temp/captchas/")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        input("\n⏸️  Presiona Enter para cerrar...")
        if driver:
            driver.quit()

if __name__ == "__main__":
    main()