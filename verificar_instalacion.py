#!/usr/bin/env python3
# verificar_instalacion.py
"""
Verifica que todas las dependencias estén instaladas correctamente
"""
import sys
import os

def verificar_modulo(nombre_modulo, nombre_paquete=None):
    """Verifica si un módulo está instalado"""
    if nombre_paquete is None:
        nombre_paquete = nombre_modulo
    
    try:
        __import__(nombre_modulo)
        print(f"   ✅ {nombre_paquete}")
        return True
    except ImportError:
        print(f"   ❌ {nombre_paquete} - NO INSTALADO")
        return False

def verificar_tesseract():
    """Verifica si Tesseract OCR está instalado"""
    try:
        import pytesseract
        version = pytesseract.get_tesseract_version()
        print(f"   ✅ Tesseract OCR (versión {version})")
        return True
    except Exception as e:
        print(f"   ⚠️  Tesseract OCR - NO DETECTADO")
        print(f"      Descarga desde: https://github.com/UB-Mannheim/tesseract/wiki")
        return False

def verificar_estructura():
    """Verifica la estructura de directorios"""
    directorios = [
        'src',
        'src/services',
        'src/config',
        'data/temp',
        'data/temp/captchas',
        'data/temp/resultados'
    ]
    
    print("\n📁 Estructura de directorios:")
    todos_ok = True
    for directorio in directorios:
        existe = os.path.exists(directorio)
        if existe:
            print(f"   ✅ {directorio}")
        else:
            print(f"   ❌ {directorio} - FALTA")
            os.makedirs(directorio, exist_ok=True)
            print(f"      ✨ Creado automáticamente")
            todos_ok = False
    
    return todos_ok

def main():
    print("=" * 60)
    print("🔍 VERIFICACIÓN DE INSTALACIÓN - CalexaScraping")
    print("=" * 60)
    
    # Python version
    print(f"\n🐍 Python: {sys.version}")
    print(f"📂 Directorio: {os.getcwd()}")
    
    # Módulos principales
    print("\n📦 Módulos Python:")
    modulos = {
        'selenium': 'selenium',
        'webdriver_manager': 'webdriver-manager',
        'easyocr': 'easyocr',
        'pytesseract': 'pytesseract',
        'cv2': 'opencv-python',
        'PIL': 'pillow',
        'requests': 'requests',
        'pandas': 'pandas',
        'numpy': 'numpy'
    }
    
    modulos_ok = []
    modulos_falta = []
    
    for modulo, paquete in modulos.items():
        if verificar_modulo(modulo, paquete):
            modulos_ok.append(paquete)
        else:
            modulos_falta.append(paquete)
    
    # Tesseract
    print("\n🔤 OCR Engines:")
    tesseract_ok = verificar_tesseract()
    
    # Estructura
    estructura_ok = verificar_estructura()
    
    # Resumen
    print("\n" + "=" * 60)
    print("📊 RESUMEN")
    print("=" * 60)
    
    if modulos_falta:
        print(f"\n❌ Faltan {len(modulos_falta)} módulos:")
        for modulo in modulos_falta:
            print(f"   • {modulo}")
        print("\n💡 Instalar con:")
        print("   pip install -r requirements.txt")
    else:
        print(f"\n✅ Todos los módulos Python instalados ({len(modulos_ok)})")
    
    if not tesseract_ok:
        print("\n⚠️  Tesseract OCR no detectado (opcional pero recomendado)")
        print("   Descarga: https://github.com/UB-Mannheim/tesseract/wiki")
        print("   Después de instalar, agrega a PATH o configura:")
        print("   pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'")
    
    if estructura_ok:
        print("\n✅ Estructura de directorios completa")
    
    # Estado final
    print("\n" + "=" * 60)
    if not modulos_falta and estructura_ok:
        print("🎉 ¡TODO LISTO! Puedes ejecutar el scraper")
        print("\n🚀 Próximos pasos:")
        print("   python test_optimizado.py")
    else:
        print("⚠️  Completar instalación antes de continuar")
    print("=" * 60)

if __name__ == "__main__":
    main()