# identificar_elementos.py
from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import os
import time

print("🔍 IDENTIFICANDO ELEMENTOS CEJ")
print("=" * 40)

driver = None
try:
    # Configurar navegador
    chrome_options = Options()
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--window-size=1200,800")
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    # Cargar página CEJ
    driver.get("https://cej.pj.gob.pe/cej/forms/busquedaform.html")
    time.sleep(3)
    
    print("✅ Página cargada")
    print(f"📄 Título: {driver.title}")
    
    # Buscar todos los elementos importantes
    print("\n📋 ELEMENTOS ENCONTRADOS:")
    
    # 1. Todos los inputs de texto
    print("\n1. INPUTS DE TEXTO:")
    inputs = driver.find_elements(By.TAG_NAME, "input")
    for input_elem in inputs:
        input_type = input_elem.get_attribute("type") or "text"
        input_name = input_elem.get_attribute("name") or "sin nombre"
        input_id = input_elem.get_attribute("id") or "sin id"
        
        if input_type in ["text", "number"]:
            print(f"   📍 name='{input_name}' id='{input_id}' type='{input_type}'")
    
    # 2. Todos los selects (combobox)
    print("\n2. SELECTS (COMBOBOX):")
    selects = driver.find_elements(By.TAG_NAME, "select")
    for select in selects:
        select_name = select.get_attribute("name") or "sin nombre"
        select_id = select.get_attribute("id") or "sin id"
        print(f"   📍 name='{select_name}' id='{select_id}'")
    
    # 3. Todas las imágenes (CAPTCHA)
    print("\n3. IMÁGENES (CAPTCHA):")
    images = driver.find_elements(By.TAG_NAME, "img")
    for img in images:
        img_src = img.get_attribute("src") or "sin src"
        img_id = img.get_attribute("id") or "sin id"
        img_alt = img.get_attribute("alt") or "sin alt"
        
        # Filtrar imágenes que podrían ser CAPTCHA
        if "captcha" in img_src.lower() or "captcha" in img_alt.lower():
            print(f"   🎯 CAPTCHA: src='{img_src}' id='{img_id}' alt='{img_alt}'")
        else:
            print(f"   📷 Imagen: src='{img_src}' id='{img_id}'")
    
    # 4. Botones
    print("\n4. BOTONES:")
    buttons = driver.find_elements(By.TAG_NAME, "button")
    inputs_submit = driver.find_elements(By.CSS_SELECTOR, "input[type='submit'], input[type='button']")
    
    for btn in buttons:
        btn_text = btn.text or "sin texto"
        btn_id = btn.get_attribute("id") or "sin id"
        print(f"   🔘 Botón: texto='{btn_text}' id='{btn_id}'")
    
    for inp in inputs_submit:
        inp_value = inp.get_attribute("value") or "sin valor"
        inp_name = inp.get_attribute("name") or "sin nombre"
        print(f"   🔘 Input botón: value='{inp_value}' name='{inp_name}'")
    
    # 5. Tomar screenshot para referencia
    os.makedirs('data/temp', exist_ok=True)
    driver.save_screenshot('data/temp/cej_elementos.png')
    print(f"\n📸 Screenshot guardado: data/temp/cej_elementos.png")
    
    print("\n" + "="*50)
    print("🎯 INSTRUCCIONES:")
    print("1. Mira el screenshot para referencia")
    print("2. Anota los NOMBRES (name=) de los campos importantes:")
    print("   - Campo expediente")
    print("   - Select materia") 
    print("   - Campo CAPTCHA")
    print("   - Botón buscar")
    print("3. También anota el ID o SRC de la imagen CAPTCHA")
    
    input("\n⏹️ Presiona Enter para cerrar...")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    if driver:
        driver.quit()