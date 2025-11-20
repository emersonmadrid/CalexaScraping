"""CNN based CAPTCHA solver and training helper."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from src.config.settings import load_settings
from src.models.data_models import CaptchaPrediction, CapturedCaptcha
from src.services.captcha_common import BaseCaptchaSolver, CaptchaCaptureService
from src.utils.helpers import ensure_dir, timestamp
from src.utils.logger import get_logger

try:  # pragma: no cover - depende de TensorFlow
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    TENSORFLOW_AVAILABLE = True
except Exception:  # pragma: no cover
    TENSORFLOW_AVAILABLE = False
    tf = None  # type: ignore
    keras = None  # type: ignore
    layers = None  # type: ignore

LOGGER = get_logger(__name__)


@dataclass(slots=True)
class CaptchaCNNConfig:
    img_width: int = 200
    img_height: int = 50
    max_length: int = 6
    characters: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


class CaptchaSolverCNN(BaseCaptchaSolver):
    """TensorFlow based solver using a CNN+RNN architecture."""

    name = "cnn"
    priority = 10

    def __init__(self, training_dir: Path | None = None) -> None:
        settings = load_settings()
        self.config = CaptchaCNNConfig()
        default_dir = ensure_dir(settings.training_dir / "cnn")
        self.training_dir = ensure_dir(training_dir or default_dir)
        self.model_path = self.training_dir / "captcha_model.h5"
        self.char_map_path = self.training_dir / "char_to_num.pkl"

        self.char_to_num = {char: idx for idx, char in enumerate(self.config.characters)}
        self.num_to_char = {idx: char for char, idx in self.char_to_num.items()}
        self.modelo = None

        if TENSORFLOW_AVAILABLE:
            self._cargar_modelo()
        else:  # pragma: no cover
            LOGGER.warning("TensorFlow no disponible, solver CNN apagado")

    def predict(self, capture: CapturedCaptcha) -> CaptchaPrediction | None:
        if not TENSORFLOW_AVAILABLE or self.modelo is None:
            return None

        img_array = self._preprocesar_imagen(capture.image)
        img_array = np.expand_dims(img_array, axis=0)

        pred = self.modelo.predict(img_array, verbose=0)
        texto = self._decodificar(pred)
        if not texto:
            return None

        return CaptchaPrediction(
            text=texto,
            solver=self.name,
            confidence=None,
            image_path=capture.image_path,
            metadata={"modelo": "cnn"},
        )

    def _preprocesar_imagen(self, imagen_pil: Image.Image) -> np.ndarray:
        imagen = imagen_pil.convert("L").resize(
            (self.config.img_width, self.config.img_height)
        )
        array = np.array(imagen).astype("float32") / 255.0
        array = np.expand_dims(array, axis=-1)
        return array

    def _decodificar(self, predicciones: np.ndarray) -> str | None:
        if keras is None:
            return None
        input_len = np.ones(predicciones.shape[0]) * predicciones.shape[1]
        results = keras.backend.ctc_decode(
            predicciones, input_length=input_len, greedy=True
        )[0][0]
        results = results.numpy()

        for resultado in results:
            caracteres = [
                self.num_to_char.get(int(num))
                for num in resultado
                if 0 <= int(num) < len(self.config.characters)
            ]
            texto = "".join(c for c in caracteres if c)
            if texto:
                return texto
        return None

    def _crear_modelo(self):  # pragma: no cover - depende de TensorFlow
        if keras is None or layers is None:
            return None

        input_img = layers.Input(
            shape=(self.config.img_width, self.config.img_height, 1), name="image"
        )

        x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input_img)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
        x = layers.MaxPooling2D((2, 2))(x)

        new_shape = ((self.config.img_width // 8), (self.config.img_height // 8) * 128)
        x = layers.Reshape(target_shape=new_shape)(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.2)(x)

        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

        output = layers.Dense(len(self.config.characters) + 1, activation="softmax")(x)

        modelo = keras.models.Model(inputs=input_img, outputs=output, name="captcha_cnn")
        modelo.compile(optimizer=keras.optimizers.Adam(), loss=self.ctc_loss)
        return modelo

    def ctc_loss(self, y_true, y_pred):  # pragma: no cover - depende de TensorFlow
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        return keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)

    def _cargar_modelo(self):  # pragma: no cover - depende de TensorFlow
        if not self.model_path.exists():
            LOGGER.info("No hay modelo CNN entrenado todavía")
            return
        try:
            self.modelo = keras.models.load_model(
                self.model_path,
                custom_objects={"ctc_loss": self.ctc_loss},
            )
            LOGGER.info("Modelo CNN cargado desde %s", self.model_path)
            if self.char_map_path.exists():
                with self.char_map_path.open("rb") as fh:
                    self.char_to_num = pickle.load(fh)
                    self.num_to_char = {v: k for k, v in self.char_to_num.items()}
        except Exception as exc:
            LOGGER.error("Error cargando modelo CNN: %s", exc)
            self.modelo = None

    def guardar_modelo(self):  # pragma: no cover - requiere entrenamiento
        if not self.modelo:
            return
        self.modelo.save(self.model_path)
        with self.char_map_path.open("wb") as fh:
            pickle.dump(self.char_to_num, fh)
        LOGGER.info("Modelo CNN guardado en %s", self.model_path)


class CaptchaCNNTrainer:
    """Utility class to collect training data and fit the CNN model."""

    def __init__(
        self,
        solver: CaptchaSolverCNN,
        capture_service: CaptchaCaptureService | None = None,
    ) -> None:
        if not TENSORFLOW_AVAILABLE:
            raise RuntimeError("TensorFlow es requerido para entrenar el modelo CNN")

        self.solver = solver
        self.capture_service = capture_service
        self.images_dir = ensure_dir(self.solver.training_dir / "images")
        self.labels_path = self.solver.training_dir / "labels.json"
        self.labels = self._cargar_labels()

        if not self.solver.modelo:
            self.solver.modelo = self.solver._crear_modelo()

    def recolectar_ejemplos(self, driver, total: int = 50) -> None:
        if not self.capture_service:
            raise ValueError("Se requiere CaptchaCaptureService para recolectar datos")

        for idx in range(total):
            captura = self.capture_service.capture_image(driver)
            if not captura:
                continue

            print("\n👁️  Observa el CAPTCHA en el navegador.")
            texto = input("Texto del CAPTCHA (o 'skip'): ").strip().upper()
            if texto.lower() == "skip" or not texto:
                self.capture_service.reload(driver)
                continue

            filename = f"captcha_{timestamp()}_{idx}.png"
            path = self.images_dir / filename
            captura.image.save(path)
            self.labels[filename] = texto
            self._guardar_labels()
            LOGGER.info("Ejemplo guardado en %s -> %s", path, texto)
            self.capture_service.reload(driver)

    def entrenar(self, epochs: int = 50, batch_size: int = 16) -> None:
        dataset = self._cargar_dataset()
        if not dataset[0].size:
            raise ValueError("No hay datos de entrenamiento disponibles")

        X, y = dataset
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        split = int(len(X) * 0.8)
        train_idx, val_idx = indices[:split], indices[split:]

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5
            ),
        ]

        self.solver.modelo.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
        )

        self.solver.guardar_modelo()
        LOGGER.info("Entrenamiento finalizado")

    def _cargar_labels(self) -> dict[str, str]:
        if self.labels_path.exists():
            with self.labels_path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        return {}

    def _guardar_labels(self) -> None:
        with self.labels_path.open("w", encoding="utf-8") as fh:
            json.dump(self.labels, fh, ensure_ascii=False, indent=2)

    def _cargar_dataset(self) -> tuple[np.ndarray, np.ndarray]:
        imagenes: list[np.ndarray] = []
        etiquetas: list[np.ndarray] = []

        for filename, texto in self.labels.items():
            path = self.images_dir / filename
            if not path.exists():
                continue

            imagen = Image.open(path)
            imagen_array = self.solver._preprocesar_imagen(imagen)
            label = self._encode_label(texto)
            imagenes.append(imagen_array)
            etiquetas.append(label)

        return np.array(imagenes), np.array(etiquetas)

    def _encode_label(self, texto: str) -> np.ndarray:
        label = [self.solver.char_to_num.get(char, 0) for char in texto]
        fill = [len(self.solver.config.characters)] * (
            self.solver.config.max_length - len(label)
        )
        label.extend(fill)
        return np.array(label[: self.solver.config.max_length])


__all__ = ["CaptchaSolverCNN", "CaptchaCNNTrainer"]
