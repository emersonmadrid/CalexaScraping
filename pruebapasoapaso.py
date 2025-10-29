# prueba_avanzada.py
import os
import sys
import time

sys.path.append('src')

print("🚀 PRUEBA SISTEMA AVANZADO")
print("=" * 40)

try:
    from selenium.webdriver.common.by import By
    from services.browser_manager import BrowserManager
    from services.captcha_solver import CaptchaSolverAvanzado
    from services.form_filler import FormFiller
    
    print("✅ Cargando sistema avanzado...")
    
    # 1. Inicializar
    navegador = BrowserManager()
    solver_avanzado = CaptchaSolverAvanzado()
    llenador = FormFiller(solver_avanzado)
    
    # 2. Abrir página
    driver = navegador.iniciar_navegador()
    driver.get("https://cej.pj.gob.pe/cej/forms/busquedaform.html")
    time.sleep(3)
    print("✅ Página cargada")
    
    # 3. Datos de prueba
    datos = {'numero_expediente': '01234-2024-0-0401-JR-PE-01'}
    
    # 4. Llenar formulario completo
    print("\n📝 Iniciando proceso completo...")
    if llenador.llenar_formulario(driver, datos):
        print("✅ Formulario llenado")
        
        # 5. Buscar
        print("\n🔍 Ejecutando búsqueda...")
        if llenador.buscar(driver):
            print("✅ Búsqueda completada")
            
            # 6. Verificar resultado
            driver.save_screenshot('data/temp/resultado_avanzado.png')
            print("📸 Resultado guardado: data/temp/resultado_avanzado.png")
            
            # Análisis automático
            pagina = driver.page_source.lower()
            if "incorrecto" in pagina:
                print("❌ CAPTCHA incorrecto")
            elif "no se encontraron" in pagina:
                print("✅ Búsqueda exitosa (sin resultados)")
            else:
                print("🎉 ¡Posible éxito! Revisa el screenshot")
        else:
            print("❌ Error en búsqueda")
    else:
        print("❌ Error llenando formulario")
    
    print("\n💡 Proceso terminado")
    input("Presiona Enter para cerrar: ")
    driver.quit()
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()