"""CNN-based CAPTCHA solver with proper integration."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from src.models.data_models import CaptchaPrediction, CapturedCaptcha
from src.services.captcha_common import BaseCaptchaSolver
from src.utils.helpers import ensure_dir
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    LOGGER.warning("TensorFlow no disponible - solver CNN deshabilitado")


class CaptchaSolverCNN(BaseCaptchaSolver):
    """Deep learning solver using CNN architecture."""
    
    name = "cnn"
    priority = 30
    
    def __init__(self, training_dir: Path | None = None) -> None:
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow requerido: pip install tensorflow")
        
        from src.config.settings import load_settings
        settings = load_settings()
        self.training_dir = training_dir or (settings.training_dir / "cnn")
        ensure_dir(self.training_dir)
        
        # Configuración del modelo
        self.img_width = 200
        self.img_height = 50
        self.max_length = 6
        
        # Caracteres permitidos
        self.caracteres = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        self.char_to_num = {char: idx for idx, char in enumerate(self.caracteres)}
        self.num_to_char = {idx: char for idx, char in enumerate(self.caracteres)}
        
        # Rutas
        self.model_path = self.training_dir / "model.h5"
        self.mappings_path = self.training_dir / "mappings.pkl"
        self.labels_path = self.training_dir / "labels.json"
        
        # Cargar modelo
        self.modelo = self._cargar_modelo()
        
        LOGGER.info("Solver CNN inicializado (modelo cargado: %s)", self.modelo is not None)
    
    def predict(self, capture: CapturedCaptcha) -> CaptchaPrediction | None:
        """Predice el texto del CAPTCHA usando CNN."""
        if self.modelo is None:
            LOGGER.debug("CNN: modelo no entrenado")
            return None
        
        try:
            img_array = self._preprocesar_imagen(capture.image)
            img_array = np.expand_dims(img_array, axis=0)
            
            pred = self.modelo.predict(img_array, verbose=0)
            texto = self._decodificar_prediccion(pred)[0]
            
            if not texto or len(texto) < 3:
                return None
            
            LOGGER.debug("CNN predijo: '%s'", texto)
            return CaptchaPrediction(
                text=texto,
                solver=self.name,
                confidence=0.8,
                image_path=capture.image_path,
                metadata={"model": "cnn"}
            )
            
        except Exception as exc:
            LOGGER.exception("Error en predicción CNN: %s", exc)
            return None
    
    def _preprocesar_imagen(self, imagen_pil: Image.Image) -> np.ndarray:
        """Preprocesa imagen para el modelo."""
        if imagen_pil.mode != 'L':
            imagen_pil = imagen_pil.convert('L')
        
        imagen_pil = imagen_pil.resize((self.img_width, self.img_height))
        img_array = np.array(imagen_pil, dtype=np.float32) / 255.0
        return np.expand_dims(img_array, axis=-1)
    
    def _decodificar_prediccion(self, pred: np.ndarray) -> list[str]:
        """Decodifica las predicciones del modelo usando CTC."""
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        results = keras.backend.ctc_decode(
            pred, 
            input_length=input_len, 
            greedy=True
        )[0][0].numpy()
        
        output_text = []
        for res in results:
            chars = []
            for num in res:
                if 0 <= num < len(self.caracteres):
                    chars.append(self.num_to_char[num])
            output_text.append(''.join(chars))
        
        return output_text
    
    def _crear_modelo(self) -> keras.Model:
        """Crea la arquitectura CNN + RNN."""
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
        
        # Capas recurrentes
        x = layers.Bidirectional(
            layers.LSTM(128, return_sequences=True, dropout=0.25)
        )(x)
        x = layers.Bidirectional(
            layers.LSTM(64, return_sequences=True, dropout=0.25)
        )(x)
        
        # Capa de salida
        x = layers.Dense(
            len(self.caracteres) + 1,
            activation="softmax",
            name="output"
        )(x)
        
        modelo = keras.models.Model(inputs=input_img, outputs=x)
        modelo.compile(optimizer=keras.optimizers.Adam(), loss=self._ctc_loss)
        
        LOGGER.info("Modelo CNN creado: %s parámetros", modelo.count_params())
        return modelo
    
    def _ctc_loss(self, y_true, y_pred):
        """CTC Loss para secuencias."""
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
        
        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        
        return keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    
    def _cargar_modelo(self) -> keras.Model | None:
        """Carga modelo entrenado si existe."""
        if not self.model_path.exists():
            LOGGER.info("No hay modelo CNN entrenado en %s", self.model_path)
            return None
        
        try:
            modelo = keras.models.load_model(
                self.model_path,
                custom_objects={"_ctc_loss": self._ctc_loss}
            )
            
            if self.mappings_path.exists():
                with self.mappings_path.open('rb') as f:
                    mappings = pickle.load(f)
                    self.char_to_num = mappings['char_to_num']
                    self.num_to_char = mappings['num_to_char']
            
            LOGGER.info("Modelo CNN cargado desde %s", self.model_path)
            return modelo
            
        except Exception as exc:
            LOGGER.exception("Error cargando modelo CNN: %s", exc)
            return None
    
    def guardar_modelo(self) -> None:
        """Guarda el modelo entrenado."""
        if self.modelo is None:
            return
        
        self.modelo.save(self.model_path)
        
        mappings = {
            'char_to_num': self.char_to_num,
            'num_to_char': self.num_to_char
        }
        with self.mappings_path.open('wb') as f:
            pickle.dump(mappings, f)
        
        LOGGER.info("Modelo CNN guardado en %s", self.model_path)
    
    def agregar_ejemplo(self, imagen_pil: Image.Image, texto: str) -> None:
        """Agrega ejemplo al conjunto de entrenamiento."""
        ensure_dir(self.training_dir / "images")
        
        # Guardar imagen
        from src.utils.helpers import timestamp
        filename = f"{texto}_{timestamp()}.png"
        filepath = self.training_dir / "images" / filename
        imagen_pil.save(filepath)
        
        # Actualizar labels
        labels = {}
        if self.labels_path.exists():
            with self.labels_path.open('r') as f:
                labels = json.load(f)
        
        labels[filename] = texto
        
        with self.labels_path.open('w') as f:
            json.dump(labels, f, indent=2)
        
        LOGGER.debug("Ejemplo agregado: %s -> %s", filename, texto)
    
    def entrenar(self, epochs: int = 50, batch_size: int = 16) -> bool:
        """Entrena el modelo con los ejemplos recolectados."""
        if not self.labels_path.exists():
            LOGGER.error("No hay datos de entrenamiento en %s", self.labels_path)
            return False
        
        with self.labels_path.open('r') as f:
            labels = json.load(f)
        
        LOGGER.info("Preparando entrenamiento con %s ejemplos", len(labels))
        
        if len(labels) < 50:
            LOGGER.warning("Se recomiendan al menos 50 ejemplos (actual: %s)", len(labels))
        
        # Cargar datos
        X, y = [], []
        images_dir = self.training_dir / "images"
        
        for filename, texto in labels.items():
            filepath = images_dir / filename
            if not filepath.exists():
                continue
            
            try:
                img = Image.open(filepath)
                img_array = self._preprocesar_imagen(img)
                label = self._codificar_label(texto)
                
                X.append(img_array)
                y.append(label)
            except Exception as exc:
                LOGGER.warning("Error procesando %s: %s", filename, exc)
        
        X = np.array(X)
        y = np.array(y)
        
        LOGGER.info("Datos preparados: X=%s, y=%s", X.shape, y.shape)
        
        # Split train/val
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Crear modelo si no existe
        if self.modelo is None:
            self.modelo = self._crear_modelo()
        
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
        LOGGER.info("Iniciando entrenamiento: %s epochs, batch_size=%s", epochs, batch_size)
        
        try:
            self.modelo.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                batch_size=batch_size,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            )
            
            self.guardar_modelo()
            LOGGER.info("Entrenamiento completado exitosamente")
            return True
            
        except Exception as exc:
            LOGGER.exception("Error durante entrenamiento: %s", exc)
            return False
    
    def _codificar_label(self, texto: str) -> np.ndarray:
        """Codifica texto a array numérico."""
        label = [self.char_to_num.get(char, 0) for char in texto]
        label = label + [len(self.caracteres)] * (self.max_length - len(label))
        return np.array(label[:self.max_length])


__all__ = ["CaptchaSolverCNN"]