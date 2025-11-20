"""Pattern-matching solver backed by manual training data."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

try:  # pragma: no cover - depende de easyocr
    import easyocr
except Exception:  # pragma: no cover
    easyocr = None

from src.config.settings import load_settings
from src.models.data_models import CaptchaPrediction, CapturedCaptcha
from src.services.captcha_common import BaseCaptchaSolver
from src.utils.helpers import ensure_dir, timestamp
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


@dataclass(slots=True)
class PatternMatch:
    distancia: int
    texto: str


class CaptchaSolverML(BaseCaptchaSolver):
    """Simple solver that memorises previously labelled CAPTCHAs."""

    name = "pattern_matching"
    priority = 40

    def __init__(self, training_dir: Path | None = None) -> None:
        settings = load_settings()
        self.training_dir = training_dir or settings.training_dir
        self.model_path = self.training_dir / "captcha_patterns.pkl"
        self.reader = self._crear_easyocr()
        self.usar_easyocr = self.reader is not None
        self.patrones = self._cargar_patrones()

    def predict(self, capture: CapturedCaptcha) -> CaptchaPrediction | None:
        candidato = self.buscar_en_patrones(capture.image)
        if candidato:
            return CaptchaPrediction(
                text=candidato,
                solver=self.name,
                confidence=1.0,
                image_path=capture.image_path,
                metadata={"tipo": "pattern"},
            )

        texto = self._ocr_fallback(capture.image)
        if not texto:
            return None

        return CaptchaPrediction(
            text=texto,
            solver=self.name,
            confidence=0.3,
            image_path=capture.image_path,
            metadata={"tipo": "ocr-fallback"},
        )

    # --- Pattern helpers --------------------------------------------
    def _cargar_patrones(self) -> dict[str, str]:
        if self.model_path.exists():
            with self.model_path.open("rb") as fh:
                patrones = pickle.load(fh)
                LOGGER.info("Patrones cargados: %s", len(patrones))
                return patrones
        return {}

    def guardar_patrones(self) -> None:
        ensure_dir(self.training_dir)
        with self.model_path.open("wb") as fh:
            pickle.dump(self.patrones, fh)
        LOGGER.info("Patrones guardados: %s", len(self.patrones))

    def agregar_a_entrenamiento(self, imagen_pil, texto_correcto: str) -> None:
        ensure_dir(self.training_dir / "images")
        ruta = self.training_dir / "images" / f"{texto_correcto}_{timestamp()}.png"
        imagen_pil.save(ruta)
        self.patrones[self.calcular_hash_visual(imagen_pil)] = texto_correcto
        self.guardar_patrones()

    def calcular_hash_visual(self, imagen_pil) -> str:
        imagen = imagen_pil.resize((32, 32))
        gray = imagen.convert("L")
        pixels = np.array(gray).flatten()
        avg = pixels.mean()
        return "".join("1" if p > avg else "0" for p in pixels)

    def distancia_hamming(self, hash1: str, hash2: str) -> int:
        if len(hash1) != len(hash2):
            return 100
        return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

    def buscar_en_patrones(
        self, imagen_pil, umbral: int = 65
    ) -> str | None:  # pragma: no cover - acceso a pickle
        if not self.patrones:
            return None

        hash_actual = self.calcular_hash_visual(imagen_pil)
        coincidencias: list[PatternMatch] = []

        for hash_guardado, texto in self.patrones.items():
            distancia = self.distancia_hamming(hash_actual, hash_guardado)
            if distancia < umbral:
                coincidencias.append(PatternMatch(distancia, texto))

        if not coincidencias:
            return None

        coincidencias.sort(key=lambda c: c.distancia)
        mejor = coincidencias[0]
        LOGGER.debug("Patrón conocido encontrado (%s)", mejor.distancia)
        return mejor.texto

    # --- OCR fallback -----------------------------------------------
    def _ocr_fallback(self, imagen_pil):
        if not self.usar_easyocr or not self.reader:
            return None
        try:
            procesada = self._preprocesar_imagen(imagen_pil)
            resultados = self.reader.readtext(
                procesada,
                detail=0,
                paragraph=False,
                text_threshold=0.45,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            )
            texto = "".join(resultados).strip() if resultados else ""
            texto = "".join(c for c in texto if c.isalnum()).upper()
            return texto if 3 <= len(texto) <= 8 else None
        except Exception as exc:  # pragma: no cover
            LOGGER.debug("EasyOCR fallback falló: %s", exc)
            return None

    def _preprocesar_imagen(self, imagen_pil):
        img = np.array(imagen_pil)
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img

        scale = 5
        width = int(gray.shape[1] * scale)
        height = int(gray.shape[0] * scale)
        large = cv2.resize(gray, (width, height), interpolation=cv2.INTER_CUBIC)

        denoised = cv2.fastNlMeansDenoising(large, None, 15, 7, 21)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        binary = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            15,
            2,
        )
        return binary

    def _crear_easyocr(self):
        if easyocr is None:
            return None
        try:
            return easyocr.Reader(["en"], gpu=False, verbose=False)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("EasyOCR no disponible: %s", exc)
            return None


__all__ = ["CaptchaSolverML"]
