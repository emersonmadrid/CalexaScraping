#!/usr/bin/env python3
# src/services/captcha_solver_cnn.py
"""
Solver de CAPTCHA usando CNN (Convolutional Neural Network)
Entrena un modelo de deep learning para reconocer CAPTCHAs
"""
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import os
import pickle
import json
from datetime import datetime

# TensorFlow/Keras para el modelo CNN
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_DISPONIBLE = True
except ImportError:
    TENSORFLOW_DISPONIBLE = False
    print("⚠️ TensorFlow no disponible. Instala con: pip install tensorflow")

class CaptchaSolverCNN:
    def __init__(self):
        """
        Solver CNN con arquitectura de deep learning
        """
        print("🔧 Inicializando solver CNN...")
        
        if not TENSORFLOW_DISPONIBLE:
            raise ImportError("TensorFlow es requerido para este solver")
        
        # Configuración
        self.img_width = 200
        self.img_height = 50
        self.max_length = 6  # Longitud máxima del CAPTCHA
        
        # Directorio de entrenamiento
        self.training_dir = "data/training_cnn"
        self.model_path = os.path.join(self.training_dir, "captcha_model.h5")
        self.char_to_num_path = os.path.join(self.training_dir, "char_to_num.pkl")
        
        # Caracteres permitidos
        self.caracteres = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        
        # Mapeos
        self.char_to_num = {char: idx for idx, char in enumerate(self.caracteres)}
        self.num_to_char = {idx: char for idx, char in enumerate(self.caracteres)}
        
        # Modelo
        self.modelo = None
        self.cargar_modelo()
        
        print(f"✅ Solver CNN listo")
    
    def crear_modelo(self):
        """
        Crea la arquitectura del modelo CNN
        """
        print("🏗️ Creando arquitectura del modelo...")
        
        # Input
        input_img = layers.Input(
            shape=(self.img_width, self.img_height, 1), 
            name="image"
        )
        
        # Capas convolucionales
        x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input_img)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        # Reshape para RNN
        new_shape = ((self.img_width // 8), (self.img_height // 8) * 128)
        x = layers.Reshape(target_shape=new_shape)(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.2)(x)
        
        # RNN layers
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)
        
        # Output layer
        x = layers.Dense(
            len(self.caracteres) + 1,  # +1 para blank token en CTC
            activation="softmax",
            name="dense"
        )(x)
        
        # Crear modelo
        modelo = keras.models.Model(inputs=input_img, outputs=x, name="captcha_cnn")
        
        # CTC loss
        modelo.compile(optimizer=keras.optimizers.Adam(), loss=self.ctc_loss)
        
        print(f"✅ Modelo creado: {modelo.count_params():,} parámetros")
        return modelo
    
    def ctc_loss(self, y_true, y_pred):
        """
        CTC (Connectionist Temporal Classification) Loss
        """
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
        
        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        
        loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
        return loss
    
    def cargar_modelo(self):
        """Carga el modelo si existe"""
        if os.path.exists(self.model_path):
            try:
                self.modelo = keras.models.load_model(
                    self.model_path,
                    custom_objects={"ctc_loss": self.ctc_loss}
                )
                print(f"📚 Modelo cargado desde {self.model_path}")
                
                # Cargar mapeos
                if os.path.exists(self.char_to_num_path):
                    with open(self.char_to_num_path, 'rb') as f:
                        self.char_to_num = pickle.load(f)
                    self.num_to_char = {v: k for k, v in self.char_to_num.items()}
                
                return True
            except Exception as e:
                print(f"⚠️ Error cargando modelo: {e}")
                return False
        else:
            print("📝 No hay modelo entrenado")
            return False
    
    def guardar_modelo(self):
        """Guarda el modelo entrenado"""
        if self.modelo is None:
            return
        
        os.makedirs(self.training_dir, exist_ok=True)
        
        self.modelo.save(self.model_path)
        print(f"💾 Modelo guardado: {self.model_path}")
        
        with open(self.char_to_num_path, 'wb') as f:
            pickle.dump(self.char_to_num, f)
        print(f"💾 Mapeos guardados: {self.char_to_num_path}")
    
    def preprocesar_imagen(self, imagen_pil):
        """
        Preprocesa imagen para el modelo
        """
        # Convertir a escala de grises
        if imagen_pil.mode != 'L':
            imagen_pil = imagen_pil.convert('L')
        
        # Redimensionar
        imagen_pil = imagen_pil.resize((self.img_width, self.img_height))
        
        # Convertir a array
        img_array = np.array(imagen_pil)
        
        # Normalizar
        img_array = img_array.astype(np.float32) / 255.0
        
        # Agregar dimensión de canal
        img_array = np.expand_dims(img_array, axis=-1)
        
        return img_array
    
    def encode_labels(self, texto):
        """
        Codifica texto a números
        """
        label = [self.char_to_num.get(char, 0) for char in texto]
        
        # Padding hasta max_length
        label = label + [len(self.caracteres)] * (self.max_length - len(label))
        
        return np.array(label[:self.max_length])
    
    def decode_predictions(self, pred):
        """
        Decodifica predicciones del modelo
        """
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        
        # CTC decode
        results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
        
        # Convertir a numpy
        results = results.numpy()
        
        # Decodificar
        output_text = []
        for res in results:
            chars = []
            for num in res:
                if num >= 0 and num < len(self.caracteres):
                    chars.append(self.num_to_char[num])
            output_text.append(''.join(chars))
        
        return output_text
    
    def obtener_imagen_captcha(self, driver, max_reintentos=3):
        """
        Obtiene la imagen del CAPTCHA con reintentos y validación
        
        Args:
            driver: WebDriver de Selenium
            max_reintentos: Número máximo de intentos
        
        Returns:
            PIL.Image or None: Imagen capturada o None si falla
        """
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        import time
        
        for intento in range(max_reintentos):
            try:
                print(f"   📸 Intento de captura {intento + 1}/{max_reintentos}...")
                
                # 1. Esperar a que el elemento CAPTCHA esté presente
                captcha_img = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.ID, "captcha_image"))
                )
                
                # 2. CRÍTICO: Esperar a que el src esté cargado y no esté vacío
                WebDriverWait(driver, 10).until(
                    lambda d: captcha_img.get_attribute("src") and 
                            len(captcha_img.get_attribute("src")) > 50  # src debe tener contenido
                )
                
                # 3. Esperar a que la imagen esté completamente cargada
                driver.execute_script("""
                    return arguments[0].complete && 
                        arguments[0].naturalHeight > 0 &&
                        arguments[0].naturalWidth > 0;
                """, captcha_img)
                
                # 4. Pequeña espera adicional para asegurar
                time.sleep(0.5)
                
                # 5. Verificar dimensiones antes de capturar
                size = captcha_img.size
                if size['width'] < 10 or size['height'] < 10:
                    print(f"   ⚠️ Dimensiones inválidas: {size}")
                    if intento < max_reintentos - 1:
                        print(f"   🔄 Reintentando en 2 segundos...")
                        time.sleep(2)
                        continue
                    else:
                        return None
                
                print(f"   📏 Dimensiones: {size['width']}x{size['height']}")
                
                # MÉTODO 1: Screenshot directo (más confiable)
                try:
                    png_bytes = captcha_img.screenshot_as_png
                    imagen = Image.open(BytesIO(png_bytes))
                    
                    # Validar que la imagen no esté vacía o corrupta
                    if imagen.size[0] > 0 and imagen.size[1] > 0:
                        # Verificar que no sea una imagen completamente blanca/negra (error común)
                        img_array = np.array(imagen.convert('L'))
                        varianza = img_array.var()
                        
                        if varianza < 10:  # Imagen casi uniforme
                            print(f"   ⚠️ Imagen con varianza baja ({varianza:.2f}), probablemente vacía")
                            if intento < max_reintentos - 1:
                                time.sleep(2)
                                continue
                        
                        print(f"   ✅ Captura exitosa: {imagen.size}, modo: {imagen.mode}, varianza: {varianza:.2f}")
                        return imagen
                        
                except Exception as e:
                    print(f"   ❌ Screenshot falló: {e}")
                
                # MÉTODO 2: Desde atributo src
                try:
                    img_src = captcha_img.get_attribute("src")
                    
                    if img_src and len(img_src) > 50:
                        print(f"   🔗 Intentando desde src ({len(img_src)} chars)...")
                        
                        # Si es data URI
                        if img_src.startswith("data:image"):
                            match = re.search(r'base64,(.+)', img_src)
                            if match:
                                img_data = base64.b64decode(match.group(1))
                                imagen = Image.open(BytesIO(img_data))
                                
                                if imagen.size[0] > 0 and imagen.size[1] > 0:
                                    print(f"   ✅ Captura desde data URI: {imagen.size}")
                                    return imagen
                        
                        # Si es URL HTTP
                        elif img_src.startswith("http"):
                            cookies = {cookie['name']: cookie['value'] 
                                    for cookie in driver.get_cookies()}
                            
                            response = requests.get(img_src, cookies=cookies, timeout=10)
                            if response.status_code == 200 and len(response.content) > 100:
                                imagen = Image.open(BytesIO(response.content))
                                
                                if imagen.size[0] > 0 and imagen.size[1] > 0:
                                    print(f"   ✅ Captura desde URL: {imagen.size}")
                                    return imagen
                                
                except Exception as e:
                    print(f"   ❌ Captura desde src falló: {e}")
                
                # Si llegamos aquí, el intento falló
                if intento < max_reintentos - 1:
                    print(f"   🔄 Reintentando en 2 segundos...")
                    time.sleep(2)
                    
                    # Opcional: recargar el CAPTCHA para siguiente intento
                    try:
                        reload_btn = driver.find_element(By.ID, "btnReload")
                        reload_btn.click()
                        time.sleep(2)
                    except:
                        pass
                
            except Exception as e:
                print(f"   💥 Error en intento {intento + 1}: {e}")
                if intento < max_reintentos - 1:
                    time.sleep(2)
        
        print("   ❌ No se pudo capturar después de todos los intentos")
        return None

    def diagnosticar_captcha(driver):
        """
        Función de diagnóstico para depurar problemas de captura
        """
        from selenium.webdriver.common.by import By
        
        print("\n🔍 DIAGNÓSTICO DE CAPTCHA")
        print("=" * 50)
        
        try:
            # 1. Verificar elemento
            captcha_img = driver.find_element(By.ID, "captcha_image")
            print("✅ Elemento encontrado")
            
            # 2. Atributos
            src = captcha_img.get_attribute("src")
            print(f"📝 Src length: {len(src) if src else 0}")
            if src:
                print(f"   Tipo: {'data URI' if src.startswith('data:') else 'URL' if src.startswith('http') else 'otro'}")
            
            # 3. Propiedades CSS
            display = captcha_img.value_of_css_property("display")
            visibility = captcha_img.value_of_css_property("visibility")
            opacity = captcha_img.value_of_css_property("opacity")
            
            print(f"🎨 CSS: display={display}, visibility={visibility}, opacity={opacity}")
            
            # 4. Dimensiones
            size = captcha_img.size
            location = captcha_img.location
            
            print(f"📏 Size: {size['width']}x{size['height']}")
            print(f"📍 Location: x={location['x']}, y={location['y']}")
            
            # 5. Estado de carga
            is_complete = driver.execute_script("""
                var img = arguments[0];
                return {
                    complete: img.complete,
                    naturalWidth: img.naturalWidth,
                    naturalHeight: img.naturalHeight,
                    currentSrc: img.currentSrc ? img.currentSrc.substring(0, 100) : null
                };
            """, captcha_img)
            
            print(f"🔄 Estado de carga:")
            print(f"   Complete: {is_complete['complete']}")
            print(f"   Natural size: {is_complete['naturalWidth']}x{is_complete['naturalHeight']}")
            
            # 6. Intentar captura
            try:
                png_bytes = captcha_img.screenshot_as_png
                imagen = Image.open(BytesIO(png_bytes))
                
                img_array = np.array(imagen.convert('L'))
                varianza = img_array.var()
                
                print(f"📸 Captura: {imagen.size}, varianza={varianza:.2f}")
                
                # Guardar para inspección
                os.makedirs("data/temp/diagnostico", exist_ok=True)
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = f"data/temp/diagnostico/captcha_diagnostico_{timestamp}.png"
                imagen.save(path)
                print(f"💾 Guardado en: {path}")
                
            except Exception as e:
                print(f"❌ Error en captura: {e}")
            
        except Exception as e:
            print(f"❌ Error general: {e}")
            import traceback
            traceback.print_exc()
        
        print("=" * 50)  


    def predecir(self, imagen_pil):
        """
        Predice el texto del CAPTCHA
        """
        if self.modelo is None:
            print("❌ No hay modelo entrenado")
            return None
        
        # Preprocesar
        img_array = self.preprocesar_imagen(imagen_pil)
        
        # Expandir dimensión batch
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predecir
        pred = self.modelo.predict(img_array, verbose=0)
        
        # Decodificar
        texto = self.decode_predictions(pred)[0]
        
        return texto
    
    def resolver_captcha(self, driver):
        """
        Resuelve CAPTCHA usando el modelo CNN
        """
        print("\n🔍 Resolviendo CAPTCHA con CNN...")
        
        # Capturar imagen
        imagen = self.obtener_imagen_captcha(driver)
        if not imagen:
            print("⚠️ Captura falló, ejecutando diagnóstico...")
            self.diagnosticar_captcha(driver)
            return None
        
        # Guardar para referencia
        os.makedirs("data/temp/captchas", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ruta_temp = f"data/temp/captchas/captcha_cnn_{timestamp}.png"
        imagen.save(ruta_temp)
        print(f"   💾 Guardado: {ruta_temp}")
        
        # Predecir
        texto = self.predecir(imagen)
        
        if texto:
            print(f"   🎯 Predicción: '{texto}'")
        else:
            print("   ❌ No se pudo predecir")
        
        return texto
    
    def recargar_captcha(self, driver):
        """Recarga el CAPTCHA"""
        try:
            from selenium.webdriver.common.by import By
            import time
            
            reload_btn = driver.find_element(By.ID, "btnReload")
            reload_btn.click()
            time.sleep(1.5)
        except Exception as e:
            print(f"   ⚠️ No se pudo recargar: {e}")
    
    def recolectar_datos_entrenamiento(self, driver, num_ejemplos=100):
        """
        Recolecta ejemplos etiquetados para entrenamiento
        """
        print(f"\n{'='*60}")
        print(f"📚 RECOLECCIÓN DE DATOS DE ENTRENAMIENTO")
        print(f"{'='*60}")
        print(f"Objetivo: {num_ejemplos} ejemplos")
        print()
        
        # Directorio
        img_dir = os.path.join(self.training_dir, "images")
        os.makedirs(img_dir, exist_ok=True)
        
        # Cargar ejemplos existentes
        labels_file = os.path.join(self.training_dir, "labels.json")
        if os.path.exists(labels_file):
            with open(labels_file, 'r') as f:
                labels = json.load(f)
        else:
            labels = {}
        
        ejemplos_iniciales = len(labels)
        print(f"📊 Ejemplos existentes: {ejemplos_iniciales}")
        print()
        
        for i in range(num_ejemplos):
            print(f"\n{'='*50}")
            print(f"📋 Ejemplo {i+1}/{num_ejemplos}")
            print(f"{'='*50}")
            
            # Capturar
            imagen = self.obtener_imagen_captcha(driver)
            if not imagen:
                print("❌ Error capturando")
                continue
            
            # Mostrar en pantalla (el navegador debe estar visible)
            print("\n👁️  MIRA EL CAPTCHA EN EL NAVEGADOR")
            print()
            print("💡 Escribe el texto del CAPTCHA (o 'skip' para saltar): ", end='')
            texto = input().upper().strip()
            
            if texto == 'SKIP' or not texto:
                print("⏭️ Saltado")
                self.recargar_captcha(driver)
                continue
            
            # Validar
            if not all(c in self.caracteres for c in texto):
                print(f"⚠️ Caracteres inválidos. Solo: {self.caracteres}")
                continue
            
            if len(texto) > self.max_length:
                print(f"⚠️ Muy largo. Máximo {self.max_length} caracteres")
                continue
            
            # Guardar imagen
            filename = f"{texto}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            filepath = os.path.join(img_dir, filename)
            imagen.save(filepath)
            
            # Guardar label
            labels[filename] = texto
            
            print(f"✅ Guardado: {filename}")
            
            # Guardar labels después de cada ejemplo
            with open(labels_file, 'w') as f:
                json.dump(labels, f, indent=2)
            
            # Recargar
            if i < num_ejemplos - 1:
                self.recargar_captcha(driver)
                import time
                time.sleep(1)
        
        ejemplos_finales = len(labels)
        print(f"\n{'='*60}")
        print(f"✅ RECOLECCIÓN COMPLETADA")
        print(f"{'='*60}")
        print(f"Ejemplos recolectados: {ejemplos_finales - ejemplos_iniciales}")
        print(f"Total en dataset: {ejemplos_finales}")
        print(f"{'='*60}")
    
    def entrenar(self, epochs=50, batch_size=16):
        """
        Entrena el modelo con los datos recolectados
        """
        print(f"\n{'='*60}")
        print(f"🎓 ENTRENAMIENTO DEL MODELO")
        print(f"{'='*60}")
        
        # Cargar datos
        labels_file = os.path.join(self.training_dir, "labels.json")
        if not os.path.exists(labels_file):
            print("❌ No hay datos de entrenamiento")
            return False
        
        with open(labels_file, 'r') as f:
            labels = json.load(f)
        
        print(f"📚 Total de ejemplos: {len(labels)}")
        
        if len(labels) < 50:
            print("⚠️ Se recomienda al menos 50 ejemplos para buen entrenamiento")
            print("   Continuar de todas formas? (s/n): ", end='')
            if input().lower() != 's':
                return False
        
        # Preparar datos
        img_dir = os.path.join(self.training_dir, "images")
        
        X = []
        y = []
        
        print("\n📦 Cargando imágenes...")
        for filename, texto in labels.items():
            filepath = os.path.join(img_dir, filename)
            if not os.path.exists(filepath):
                continue
            
            try:
                img = Image.open(filepath)
                img_array = self.preprocesar_imagen(img)
                label = self.encode_labels(texto)
                
                X.append(img_array)
                y.append(label)
            except Exception as e:
                print(f"⚠️ Error con {filename}: {e}")
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"✅ Datos preparados: {X.shape}, {y.shape}")
        
        # Crear modelo si no existe
        if self.modelo is None:
            self.modelo = self.crear_modelo()
        
        # Split train/validation
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"\n📊 Train: {len(X_train)}, Validation: {len(X_val)}")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5
            )
        ]
        
        # Entrenar
        print(f"\n🎯 Iniciando entrenamiento ({epochs} epochs)...")
        history = self.modelo.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Guardar
        self.guardar_modelo()
        
        print(f"\n{'='*60}")
        print(f"🎉 ENTRENAMIENTO COMPLETADO")
        print(f"{'='*60}")
        
        return True


def entrenar_modelo_completo():
    """
    Pipeline completo de entrenamiento
    """
    print("=" * 60)
    print("🤖 ENTRENAMIENTO DE MODELO CNN PARA CAPTCHA")
    print("=" * 60)
    print()
    print("Este proceso tiene 2 etapas:")
    print("1. Recolectar ejemplos etiquetados")
    print("2. Entrenar el modelo con los ejemplos")
    print()
    
    # Importar aquí para evitar errores de módulo
    import sys
    import os
    # Asegurar que el path incluya el directorio raíz
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    
    from src.services.browser_manager import BrowserManager
    
    driver = None
    
    try:
        # Crear solver
        solver = CaptchaSolverCNN()
        
        # Etapa 1: Recolección
        print("¿Cuántos ejemplos recolectar? (recomendado: 100+): ", end='')
        try:
            num_ejemplos = int(input())
        except:
            num_ejemplos = 100
        
        print("\n🌐 Iniciando navegador...")
        browser = BrowserManager()
        driver = browser.iniciar_navegador(headless=False)
        
        print("🔗 Navegando a CEJ...")
        driver.get("https://cej.pj.gob.pe/cej/forms/busquedaform.html")
        import time
        time.sleep(3)
        driver.maximize_window()
        
        # Recolectar
        solver.recolectar_datos_entrenamiento(driver, num_ejemplos)
        
        # Cerrar navegador
        driver.quit()
        driver = None
        
        # Etapa 2: Entrenamiento
        print("\n¿Continuar con entrenamiento? (s/n): ", end='')
        if input().lower() == 's':
            solver.entrenar(epochs=50, batch_size=16)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Cancelado por el usuario")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if driver:
            driver.quit()


if __name__ == "__main__":
    entrenar_modelo_completo()