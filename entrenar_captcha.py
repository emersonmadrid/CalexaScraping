#!/usr/bin/env python3
# entrenar_captcha.py
"""
Script para entrenar el solver con ejemplos reales
"""
import sys
sys.path.append('src')

from services.browser_manager import BrowserManager
from services.captcha_solver_ml import CaptchaSolverML
import time

def main():
    print("=" * 60)
    print("📚 ENTRENAMIENTO DE CAPTCHA SOLVER")
    print("=" * 60)
    print()
    print("Este script te ayudará a crear una base de datos de")
    print("CAPTCHAs etiquetados para mejorar la precisión.")
    print()
    print("Proceso:")
    print("1. El sistema captura el CAPTCHA")
    print("2. Intenta resolverlo con OCR")
    print("3. Tú confirmas o corriges el resultado")
    print("4. Se guarda en la base de datos de entrenamiento")
    print()
    
    print("¿Cuántos ejemplos quieres recolectar? (recomendado: 50+): ", end='')
    try:
        num_ejemplos = int(input())
    except:
        num_ejemplos = 20
        print(f"Usando valor por defecto: {num_ejemplos}")
    
    print("\n¿Continuar? (s/n): ", end='')
    if input().lower() != 's':
        print("❌ Cancelado")
        return
    
    driver = None
    
    try:
        # Iniciar
        print("\n🌐 Iniciando navegador...")
        browser = BrowserManager()
        driver = browser.iniciar_navegador(headless=False)
        
        print("🔗 Navegando a CEJ...")
        driver.get("https://cej.pj.gob.pe/cej/forms/busquedaform.html")
        time.sleep(3)
        
        # Solver ML
        solver = CaptchaSolverML()
        
        # Entrenar
        solver.entrenar_interactivo(driver, num_ejemplos)
        
        # Estadísticas
        print(f"\n📊 ESTADÍSTICAS:")
        print(f"   Total de patrones: {len(solver.patrones)}")
        print(f"   Ubicación: {solver.training_dir}")
        print()
        print("💡 Ahora puedes usar el solver entrenado en tus búsquedas")
        print("   El sistema recordará estos CAPTCHAs automáticamente")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Cancelado por el usuario")
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