#!/usr/bin/env python3
# pruebapasoapaso.py - ACTUALIZADO
"""
Script de prueba paso a paso con el sistema actualizado
"""
import sys
import os
sys.path.append('src')

import time
from selenium.webdriver.common.by import By

print("🚀 PRUEBA PASO A PASO - SISTEMA ACTUALIZADO")
print("=" * 60)

driver = None

try:
    # 1. Importar módulos
    print("\n📦 Importando módulos...")
    from services.browser_manager import BrowserManager
    from services.captcha_solver import CaptchaSolverOptimizado
    from services.form_filler import FormFiller
    print("✅ Módulos importados")
    
    # 2. Inicializar componentes
    print("\n🔧 Inicializando componentes...")
    navegador = BrowserManager()
    solver = CaptchaSolverOptimizado()
    llenador = FormFiller(solver)
    print("✅ Componentes listos")
    
    # 3. Abrir navegador
    print("\n🌐 Abriendo navegador...")
    driver = navegador.iniciar_navegador(headless=False)
    print("✅ Navegador abierto")
    
    # 4. Navegar a CEJ
    print("\n🔗 Navegando a CEJ...")
    driver.get("https://cej.pj.gob.pe/cej/forms/busquedaform.html")
    time.sleep(3)
    print(f"✅ Página cargada: {driver.title}")
    
    # 5. Datos de prueba
    datos = {
        'numero_expediente': '00001-2024-0-0401-JR-PE-01'
    }
    print(f"\n📋 Datos de prueba:")
    print(f"   Expediente: {datos['numero_expediente']}")
    
    # 6. Llenar formulario
    print("\n📝 Llenando formulario...")
    exito = llenador.llenar_formulario(driver, datos)
    
    # 7. Resultado
    print("\n" + "=" * 60)
    if exito:
        print("🎉 ¡ÉXITO! Búsqueda completada")
        print("\n💡 Revisa:")
        print("   - Screenshots en: data/temp/resultados/")
        print("   - CAPTCHAs procesados en: data/temp/captchas/")
    else:
        print("❌ La búsqueda falló")
        print("\n💡 Para diagnosticar, ejecuta:")
        print("   python diagnosticar_captcha.py")
    print("=" * 60)
    
    input("\n⏸️  Presiona Enter para cerrar el navegador...")
    
except KeyboardInterrupt:
    print("\n\n⚠️ Cancelado por el usuario")
except ImportError as e:
    print(f"\n❌ Error de importación: {e}")
    print("\n💡 Solución:")
    print("   1. Verifica que estés en el directorio correcto")
    print("   2. Ejecuta: python verificar_instalacion.py")
    print("   3. Instala dependencias faltantes: pip install -r requirements.txt")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    if driver:
        print("\n🔒 Cerrando navegador...")
        driver.quit()
        print("✅ Navegador cerrado")

print("\n👋 Script terminado")