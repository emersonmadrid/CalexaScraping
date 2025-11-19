# src/services/captcha_solver_cnn.py
"""
CAPTCHA Solver usando CNN (Convolutional Neural Network)
Entrenamiento propio, sin costos, alta precisión después de entrenar
"""
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import os
import pickle
import json
from datetime import datetime
from selenium.webdriver.common.by import By

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_DISPONIBLE = True
except ImportError:
    TENSORFLOW_DISPONIBLE = False
    print("⚠️ TensorFlow no instalado. Instala con: pip install tensorflow")


class CaptchaSolverCNN:
    def __init__(self, modelo_path="data/training/captcha_model"):
        """
        Solver con CNN para máxima precisión
        
        Args:
            modelo_path: Ruta donde se guarda/carga el modelo
        """
        if not TENSORFLOW_DISPONIBLE:
            raise ImportError("TensorFlow es necesario. Instala: pip install tensorflow")
        
        self.modelo_path = modelo_path
        self.training_dir = "data/training"
        self.captcha_width = 150
        self.captcha_height = 50
        self.max_length = 6  # Longitud máxima del CAPTCHA
        self.caracteres = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        self.char_to_num = {char: idx for idx, char in enumerate(self.caracteres)}
        self.num_to_char = {idx: char for idx, char in enumerate(self.caracteres)}
        
        # Cargar o crear modelo
        self.modelo = self.cargar_o_crear_modelo()
        
        print(f"✅ CNN Solver listo")
        print(f"   Caracteres: {len(self.caracteres)}")
        print(f"   Tamaño entrada: {self.captcha_width}x{self.captcha_height}")
    
    def cargar_o_crear_modelo(self):
        """Carga modelo existente o crea uno nuevo"""
        if os.path.exists(f"{self.modelo_path}.keras"):
            try:
                print("📦 Cargando modelo entrenado...")
                modelo = keras.models.load_model(f"{self.modelo_path}.keras")
                print("✅ Modelo cargado")
                return modelo
            except Exception as e:
                print(f"⚠️ Error cargando modelo: {e}")
                print("   Creando modelo nuevo...")
        
        return self.crear_modelo()
    
    def crear_modelo(self):
        """
        Crea arquitectura CNN para reconocimiento de CAPTCHA
        Inspirada en CRNN (CNN + RNN) para secuencias
        """
        print("🏗️ Creando arquitectura CNN...")
        
        # Input layer
        input_img = layers.Input(
            shape=(self.captcha_height, self.captcha_width, 1),
            name="image",
            dtype="float32"
        )
        
        # Capas convolucionales para extracción de características
        x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input_img)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        # Reshape para RNN
        new_shape = ((self.captcha_height // 8) * 128, self.captcha_width // 8)
        x = layers.Reshape(target_shape=new_shape)(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.2)(x)
        
        # Capas recurrentes para secuencia
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
        
        # Output layer - múltiples caracteres
        outputs = []
        for i in range(self.max_length):
            out = layers.Dense(len(self.caracteres) + 1, activation="softmax", 
                             name=f"char_{i}")(x[:, -1, :])
            outputs.append(out)
        
        modelo = keras.Model(inputs=input_img, outputs=outputs)
        modelo.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ Modelo creado")
        return modelo
    
    def preprocesar_imagen(self, imagen_pil):
        """
        Preprocesa imagen para el modelo
        
        Returns:
            numpy array: Imagen normalizada
        """
        # Convertir a escala de grises
        img = imagen_pil.convert('L')
        
        # Redimensionar
        img = img.resize((self.captcha_width, self.captcha_height))
        
        # Convertir a numpy y normalizar
        img_array = np.array(img)
        img_array = img_array.astype('float32') / 255.0
        
        # Mejorar contraste
        img_array = cv2.equalizeHist((img_array * 255).astype('uint8')).astype('float32') / 255.0
        
        # Añadir dimensión de canal
        img_array = np.expand_dims(img_array, axis=-1)
        
        return img_array
    
    def obtener_imagen_captcha(self, driver):
        """Captura imagen del CAPTCHA"""
        try:
            captcha_img = driver.find_element(By.ID, "captcha_image")
            png_bytes = captcha_img.screenshot_as_png
            imagen = Image.open(BytesIO(png_bytes))
            return imagen
        except Exception as e:
            print(f"❌ Error capturando: {e}")
            return None
    
    def predecir(self, imagen_pil):
        """
        Predice el texto del CAPTCHA
        
        Args:
            imagen_pil: PIL Image
        
        Returns:
            str: Texto predicho
        """
        # Preprocesar
        img_array = self.preprocesar_imagen(imagen_pil)
        img_array = np.expand_dims(img_array, axis=0)  # Batch dimension
        
        # Predecir
        predicciones = self.modelo.predict(img_array, verbose=0)
        
        # Decodificar
        texto = ""
        for pred in predicciones:
            char_idx = np.argmax(pred[0])
            if char_idx < len(self.caracteres):  # Ignorar "blank"
                texto += self.num_to_char[char_idx]
        
        return texto
    
    def resolver_captcha(self, driver):
        """
        Resuelve el CAPTCHA usando el modelo
        
        Returns:
            str: Texto del CAPTCHA o None
        """
        print("\n🔍 Resolviendo CAPTCHA con CNN...")
        
        # Capturar
        imagen = self.obtener_imagen_captcha(driver)
        if not imagen:
            return None
        
        # Guardar para referencia
        os.makedirs("data/temp/captchas", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        imagen.save(f"data/temp/captchas/captcha_{timestamp}.png")
        
        # Predecir
        texto = self.predecir(imagen)
        
        if texto and 3 <= len(texto) <= 8:
            print(f"   🎯 Predicción: '{texto}'")
            return texto
        
        print(f"   ❌ Predicción inválida: '{texto}'")
        return None
    
    def entrenar(self, datos_entrenamiento, epochs=50, batch_size=32):
        """
        Entrena el modelo con datos etiquetados
        
        Args:
            datos_entrenamiento: Lista de tuplas (imagen_path, texto_correcto)
            epochs: Número de épocas
            batch_size: Tamaño del batch
        """
        print(f"\n🎓 ENTRENANDO MODELO CNN")
        print(f"   Ejemplos: {len(datos_entrenamiento)}")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        
        # Preparar datos
        X = []
        y = [[] for _ in range(self.max_length)]
        
        for img_path, texto in datos_entrenamiento:
            try:
                # Cargar y preprocesar imagen
                img = Image.open(img_path)
                img_array = self.preprocesar_imagen(img)
                X.append(img_array)
                
                # Codificar texto
                texto = texto.upper().ljust(self.max_length)[:self.max_length]
                for i, char in enumerate(texto):
                    if char in self.char_to_num:
                        y[i].append(self.char_to_num[char])
                    else:
                        y[i].append(len(self.caracteres))  # blank
            
            except Exception as e:
                print(f"   ⚠️ Error procesando {img_path}: {e}")
        
        X = np.array(X)
        y = [np.array(y_i) for y_i in y]
        
        print(f"\n   📊 Datos preparados: {X.shape}")
        
        # Dividir en train/val
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train = [y_i[:split_idx] for y_i in y]
        y_val = [y_i[split_idx:] for y_i in y]
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5),
            keras.callbacks.ModelCheckpoint(
                f"{self.modelo_path}_best.keras",
                save_best_only=True,
                monitor='val_loss'
            )
        ]
        
        # Entrenar
        print("\n   🚀 Iniciando entrenamiento...")
        history = self.modelo.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Guardar modelo final
        os.makedirs(os.path.dirname(self.modelo_path), exist_ok=True)
        self.modelo.save(f"{self.modelo_path}.keras")
        print(f"\n   ✅ Modelo guardado en: {self.modelo_path}.keras")
        
        return history
    
    def recolectar_datos_interactivo(self, driver, num_ejemplos=100):
        """
        Modo interactivo para recolectar datos de entrenamiento
        
        Args:
            driver: WebDriver
            num_ejemplos: Número de ejemplos a recolectar
        """
        print(f"\n📚 RECOLECCIÓN DE DATOS")
        print(f"   Objetivo: {num_ejemplos} ejemplos")
        
        datos = []
        os.makedirs(f"{self.training_dir}/images", exist_ok=True)
        
        for i in range(num_ejemplos):
            print(f"\n{'='*50}")
            print(f"Ejemplo {i+1}/{num_ejemplos}")
            print(f"{'='*50}")
            
            # Capturar
            imagen = self.obtener_imagen_captcha(driver)
            if not imagen:
                print("❌ No se pudo capturar")
                continue
            
            # Guardar temporal
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            temp_path = f"{self.training_dir}/temp_{timestamp}.png"
            imagen.save(temp_path)
            
            # Mostrar y pedir etiqueta
            print("👁️  Mira el CAPTCHA en el navegador")
            print("💡 Escribe el texto (o 'skip' para saltar): ", end='')
            texto = input().upper().strip()
            
            if texto and texto != 'SKIP':
                # Guardar con nombre descriptivo
                final_path = f"{self.training_dir}/images/{texto}_{timestamp}.png"
                os.rename(temp_path, final_path)
                datos.append((final_path, texto))
                print(f"✅ Guardado: '{texto}'")
            else:
                os.remove(temp_path)
                print("⏭️ Saltado")
            
            # Recargar CAPTCHA
            if i < num_ejemplos - 1:
                try:
                    reload_btn = driver.find_element(By.ID, "btnReload")
                    reload_btn.click()
                    import time
                    time.sleep(1.5)
                except:
                    pass
        
        # Guardar índice
        with open(f"{self.training_dir}/training_data.json", 'w') as f:
            json.dump(datos, f, indent=2)
        
        print(f"\n✅ {len(datos)} ejemplos recolectados")
        print(f"   Archivo: {self.training_dir}/training_data.json")
        
        return datos
    
    def recargar_captcha(self, driver):
        """Recarga el CAPTCHA"""
        try:
            import time
            reload_btn = driver.find_element(By.ID, "btnReload")
            reload_btn.click()
            time.sleep(1.5)
        except Exception as e:
            print(f"⚠️ No se pudo recargar: {e}")


# ============================================================
# SCRIPT DE ENTRENAMIENTO
# ============================================================

def entrenar_modelo_completo():
    """
    Script completo para entrenar el modelo desde cero
    """
    from services.browser_manager import BrowserManager
    import time
    
    print("=" * 60)
    print("🎓 ENTRENAMIENTO COMPLETO DE MODELO CNN")
    print("=" * 60)
    
    # Inicializar
    solver = CaptchaSolverCNN()
    browser = BrowserManager()
    driver = browser.iniciar_navegador(headless=False)
    
    try:
        # Navegar
        driver.get("https://cej.pj.gob.pe/cej/forms/busquedaform.html")
        time.sleep(3)
        driver.maximize_window()
        
        # Fase 1: Recolectar datos
        print("\n📊 FASE 1: Recolección de datos")
        print("Recomendación: Al menos 200-500 ejemplos para buena precisión")
        print("\n¿Cuántos ejemplos recolectar? (mínimo 50): ", end='')
        try:
            num_ejemplos = int(input())
            if num_ejemplos < 50:
                print("⚠️ Se recomienda al menos 50 ejemplos")
                num_ejemplos = 50
        except:
            num_ejemplos = 100
        
        datos = solver.recolectar_datos_interactivo(driver, num_ejemplos)
        
        if len(datos) < 20:
            print("\n❌ Necesitas al menos 20 ejemplos para entrenar")
            return
        
        # Fase 2: Entrenar
        print("\n🚀 FASE 2: Entrenamiento")
        print("Esto puede tomar varios minutos...")
        print("\n¿Continuar? (s/n): ", end='')
        if input().lower() != 's':
            print("❌ Cancelado")
            return
        
        history = solver.entrenar(datos, epochs=50, batch_size=16)
        
        # Fase 3: Probar
        print("\n✅ ENTRENAMIENTO COMPLETADO")
        print("\n🧪 FASE 3: Prueba")
        print("¿Probar el modelo ahora? (s/n): ", end='')
        if input().lower() == 's':
            probar_modelo(solver, driver, num_pruebas=10)
    
    finally:
        driver.quit()


def probar_modelo(solver, driver, num_pruebas=10):
    """Prueba el modelo entrenado"""
    import time
    
    aciertos = 0
    
    for i in range(num_pruebas):
        print(f"\n{'='*40}")
        print(f"Prueba {i+1}/{num_pruebas}")
        print(f"{'='*40}")
        
        texto_pred = solver.resolver_captcha(driver)
        
        if texto_pred:
            print("👁️  Mira el CAPTCHA en el navegador")
            print(f"❓ ¿Es correcto '{texto_pred}'? (s/n): ", end='')
            if input().lower() == 's':
                aciertos += 1
                print("✅ Correcto")
            else:
                print("❌ Incorrecto")
        
        if i < num_pruebas - 1:
            solver.recargar_captcha(driver)
            time.sleep(1.5)
    
    precision = (aciertos / num_pruebas) * 100
    print(f"\n📊 Precisión: {precision:.1f}% ({aciertos}/{num_pruebas})")


if __name__ == "__main__":
    entrenar_modelo_completo()