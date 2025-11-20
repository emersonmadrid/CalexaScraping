"""Interactive training helpers for collecting labelled CAPTCHAs."""

from __future__ import annotations

import time
from dataclasses import dataclass

from src.services.captcha_common import CaptchaCaptureService
from src.services.captcha_solver_ml import CaptchaSolverML


@dataclass(slots=True)
class TrainingStats:
    processed: int = 0
    saved: int = 0
    skipped: int = 0
    corrected: int = 0


class ManualTrainingSession:
    """Captura CAPTCHAs y solicita al usuario que ingrese el texto manualmente."""

    def __init__(
        self,
        solver: CaptchaSolverML,
        capture_service: CaptchaCaptureService,
    ) -> None:
        self.solver = solver
        self.capture_service = capture_service

    def run(self, driver, total: int) -> TrainingStats:
        stats = TrainingStats()

        for idx in range(total):
            print(f"\n{'─' * 60}")
            print(f"📋 Ejemplo {idx + 1}/{total}")
            print("👁️  Observa el CAPTCHA directamente en el navegador.")

            captura = self.capture_service.capture_image(driver)
            if not captura:
                print("❌ No se pudo capturar el CAPTCHA, reintentando...")
                time.sleep(1)
                continue

            texto = input("💬 Texto del CAPTCHA (o 'saltar'): ").strip().upper()
            stats.processed += 1

            if not texto or texto.lower() == "saltar":
                print("⏭️  Ejemplo omitido")
                stats.skipped += 1
                self.capture_service.reload(driver)
                time.sleep(1)
                continue

            texto = "".join(c for c in texto if c.isalnum())
            if not (3 <= len(texto) <= 8):
                print("⚠️  Longitud inesperada, ejemplo descartado")
                stats.skipped += 1
                self.capture_service.reload(driver)
                time.sleep(1)
                continue

            self.solver.agregar_a_entrenamiento(captura.image, texto)
            stats.saved += 1
            print(f"✅ Guardado: {texto}")

            self.capture_service.reload(driver)
            time.sleep(1)

        return stats


class PatternTrainingSession:
    """Sugiere una predicción usando el solver y solicita confirmación."""

    def __init__(
        self,
        solver: CaptchaSolverML,
        capture_service: CaptchaCaptureService,
    ) -> None:
        self.solver = solver
        self.capture_service = capture_service

    def run(self, driver, total: int) -> TrainingStats:
        stats = TrainingStats()

        for idx in range(total):
            print(f"\n{'=' * 60}")
            print(f"🔁 Capturando ejemplo {idx + 1}/{total}")

            captura = self.capture_service.capture_image(driver)
            if not captura:
                print("❌ No se pudo capturar el CAPTCHA, reintentando...")
                time.sleep(1)
                continue

            stats.processed += 1
            prediction = self.solver.predict(captura)

            texto = None
            if prediction:
                print(f"🤖 Predicción ({prediction.solver}): '{prediction.text}'")
                respuesta = input("¿Es correcto? [s/n/c=corregir/skip]: ").strip().lower()

                if respuesta == "s":
                    texto = prediction.text
                elif respuesta == "c":
                    texto = self._ingresar_texto_manual()
                    if texto:
                        stats.corrected += 1
                elif respuesta == "n":
                    texto = self._ingresar_texto_manual()
                    if texto:
                        stats.corrected += 1
                else:
                    stats.skipped += 1
            else:
                print("⚠️  No se pudo predecir automáticamente")
                texto = self._ingresar_texto_manual()
                if not texto:
                    stats.skipped += 1

            if texto:
                self.solver.agregar_a_entrenamiento(captura.image, texto)
                stats.saved += 1
                print(f"💾 Ejemplo almacenado ('{texto}')")

            self.capture_service.reload(driver)
            time.sleep(1)

        return stats

    def _ingresar_texto_manual(self) -> str | None:
        texto = input("Texto correcto (Enter para omitir): ").strip().upper()
        texto = "".join(c for c in texto if c.isalnum())
        return texto if 3 <= len(texto) <= 8 else None


__all__ = [
    "ManualTrainingSession",
    "PatternTrainingSession",
    "TrainingStats",
]
