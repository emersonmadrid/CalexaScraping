"""OCR based solver used as fallback."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import cv2
import numpy as np
import pytesseract

try:  # pragma: no cover - optional dependency
    import easyocr
except Exception:  # pragma: no cover
    easyocr = None

from src.models.data_models import CaptchaPrediction, CapturedCaptcha
from src.services.captcha_common import BaseCaptchaSolver
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


@dataclass(slots=True)
class OCRResult:
    text: str
    engine: str
    variant: str


class CaptchaSolverOptimizado(BaseCaptchaSolver):
    """Ensemble using Tesseract + EasyOCR with aggressive preprocessing."""

    name = "ocr_ensemble"
    priority = 50

    def __init__(self) -> None:
        self.usar_tesseract = self._tesseract_disponible()
        self.reader = self._crear_easyocr()
        self.usar_easyocr = self.reader is not None

    def predict(self, capture: CapturedCaptcha) -> CaptchaPrediction | None:
        variantes = self.preprocesar_imagen(capture.image)
        hallazgos: list[OCRResult] = []

        for nombre, imagen_cv in variantes:
            texto_tess = self._ocr_tesseract(imagen_cv)
            if texto_tess:
                hallazgos.append(OCRResult(texto_tess, "tesseract", nombre))

            texto_easy = self._ocr_easyocr(imagen_cv)
            if texto_easy:
                hallazgos.append(OCRResult(texto_easy, "easyocr", nombre))

        if not hallazgos:
            return None

        candidato = self._seleccionar(hallazgos)
        if not candidato:
            return None

        LOGGER.debug(
            "OCR ensemble eligió '%s' usando %s (%s)",
            candidato.text,
            candidato.engine,
            candidato.variant,
        )
        return CaptchaPrediction(
            text=candidato.text,
            solver=self.name,
            confidence=None,
            image_path=capture.image_path,
            metadata={"engine": candidato.engine, "variant": candidato.variant},
        )

    # --- OCR helpers -------------------------------------------------
    def preprocesar_imagen(self, imagen_pil):
        """Return multiple OpenCV matrices for OCR."""
        img = np.array(imagen_pil)

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        scale = 4
        width = int(gray.shape[1] * scale)
        height = int(gray.shape[0] * scale)
        gray_large = cv2.resize(gray, (width, height), interpolation=cv2.INTER_CUBIC)

        denoised = cv2.fastNlMeansDenoising(gray_large, None, 10, 7, 21)

        resultados = []

        for umbral in (100, 127, 150):
            _, thresh = cv2.threshold(denoised, umbral, 255, cv2.THRESH_BINARY)
            resultados.append((f"thresh_{umbral}", thresh))

        _, thresh_otsu = cv2.threshold(
            denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        resultados.append(("thresh_otsu", thresh_otsu))

        for block_size in (11, 15, 19):
            thresh_adapt = cv2.adaptiveThreshold(
                denoised,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                block_size,
                2,
            )
            resultados.append((f"adapt_{block_size}", thresh_adapt))

        kernel = np.ones((2, 2), np.uint8)
        morph = cv2.morphologyEx(thresh_otsu, cv2.MORPH_CLOSE, kernel)
        resultados.append(("morph", morph))

        return resultados

    def _limpiar(self, texto: str | None) -> str | None:
        if not texto:
            return None
        texto = "".join(c for c in texto if c.isalnum()).upper()
        if not (3 <= len(texto) <= 8):
            return None
        return texto

    def _ocr_tesseract(self, matriz) -> str | None:
        if not self.usar_tesseract:
            return None
        try:
            config = (
                "--psm 8 --oem 3 "
                "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            )
            texto = pytesseract.image_to_string(matriz, config=config)
            return self._limpiar(texto.strip())
        except Exception as exc:  # pragma: no cover - dependencia externa
            LOGGER.debug("Tesseract falló: %s", exc)
            return None

    def _ocr_easyocr(self, matriz) -> str | None:
        if not self.usar_easyocr or not self.reader:
            return None
        try:
            resultados = self.reader.readtext(
                matriz,
                detail=0,
                paragraph=False,
                text_threshold=0.5,
                low_text=0.4,
                link_threshold=0.4,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            )
            texto = "".join(resultados).strip() if resultados else None
            return self._limpiar(texto)
        except Exception as exc:  # pragma: no cover
            LOGGER.debug("EasyOCR falló: %s", exc)
            return None

    def _seleccionar(self, resultados: Sequence[OCRResult]) -> OCRResult | None:
        textos = [resultado.text for resultado in resultados]
        contador = Counter(textos)

        if not contador:
            return None

        if len(contador) == 1:
            unico = textos[0]
            return next(result for result in resultados if result.text == unico)

        total = sum(contador.values())
        mas_comun, freq = contador.most_common(1)[0]

        if freq / total > 0.5:
            return next(result for result in resultados if result.text == mas_comun)

        consenso = self._consenso_por_posicion(textos)
        if consenso:
            for resultado in resultados:
                if resultado.text == consenso:
                    return resultado
            return OCRResult(consenso, "consenso", "consenso")

        return next(result for result in resultados if result.text == mas_comun)

    def _consenso_por_posicion(self, textos: Sequence[str]) -> str | None:
        longitudes = Counter(len(t) for t in textos)
        if not longitudes:
            return None
        longitud_comun = longitudes.most_common(1)[0][0]
        candidatos = [t for t in textos if len(t) == longitud_comun]
        if not candidatos:
            return None

        resultado = []
        for idx in range(longitud_comun):
            letras = [t[idx] for t in candidatos]
            letra = Counter(letras).most_common(1)[0][0]
            resultado.append(letra)
        return "".join(resultado)

    # --- inicialización ---------------------------------------------
    def _tesseract_disponible(self) -> bool:
        try:
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            LOGGER.warning("Tesseract no disponible")
            return False

    def _crear_easyocr(self):
        if easyocr is None:
            LOGGER.warning("EasyOCR no disponible")
            return None
        try:
            return easyocr.Reader(["en"], gpu=False, verbose=False)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("EasyOCR no se pudo inicializar: %s", exc)
            return None


__all__ = ["CaptchaSolverOptimizado"]
