# diagnostico.py
import os
import sys

print("🔍 DIAGNÓSTICO DE ESTRUCTURA")
print("=" * 50)

print(f"📂 Directorio actual: {os.getcwd()}")
print(f"🐍 Python version: {sys.version}")

# Verificar estructura actual
estructura = {
    'drivers/chromedriver.exe': os.path.exists('drivers/chromedriver.exe'),
    'src/': os.path.exists('src/'),
    'src/services/': os.path.exists('src/services/'),
    'tests/': os.path.exists('tests/'),
    'data/': os.path.exists('data/')
}

print("\n📁 ESTRUCTURA ACTUAL:")
for archivo, existe in estructura.items():
    status = '✅' if existe else '❌'
    print(f"   {status} {archivo}")

# Verificar archivos específicos
if os.path.exists('drivers/chromedriver.exe'):
    print(f"\n🔧 ChromeDriver: {os.path.abspath('drivers/chromedriver.exe')}")
    print(f"   Tamaño: {os.path.getsize('drivers/chromedriver.exe')} bytes")
else:
    print("\n❌ ChromeDriver no encontrado en drivers/")

# Verificar imports
print("\n🔧 PROBANDO IMPORTS...")
try:
    from selenium import webdriver
    print("✅ selenium instalado")
except ImportError:
    print("❌ selenium NO instalado")

try:
    import easyocr
    print("✅ easyocr instalado")
except ImportError:
    print("❌ easyocr NO instalado")

print("\n🎯 PRÓXIMOS PASOS:")
if not all(estructura.values()):
    print("1. Completar estructura faltante")
else:
    print("1. Ejecutar pruebas de integración")