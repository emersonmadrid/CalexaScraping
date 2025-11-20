"""CNN training session for CAPTCHA solving."""

from __future__ import annotations

import time
from dataclasses import dataclass

from src.services.captcha_common import CaptchaCaptureService
from src.services.captcha_solver_cnn import CaptchaSolverCNN
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


@dataclass(slots=True)
class CNNTrainingStats:
    """Statistics for CNN training session."""
    captured: int = 0
    saved: int = 0
    skipped: int = 0
    errors: int = 0


class CNNTrainingSession:
    """Interactive session to collect labeled CAPTCHAs for CNN training."""
    
    def __init__(
        self,
        solver: CaptchaSolverCNN,
        capture_service: CaptchaCaptureService,
    ) -> None:
        self.solver = solver
        self.capture_service = capture_service
    
    def recolectar_ejemplos(self, driver, num_ejemplos: int) -> CNNTrainingStats:
        """Recolecta ejemplos etiquetados manualmente."""
        stats = CNNTrainingStats()
        
        print(f"\n{'='*60}")
        print(f"📚 RECOLECCIÓN DE EJEMPLOS PARA CNN")
        print(f"{'='*60}")
        print(f"Objetivo: {num_ejemplos} ejemplos")
        print(f"Caracteres válidos: {self.solver.caracteres}")
        print(f"Longitud: 3-{self.solver.max_length} caracteres")
        print()
        
        for idx in range(num_ejemplos):
            print(f"\n{'─'*60}")
            print(f"📋 Ejemplo {idx + 1}/{num_ejemplos}")
            print("─"*60)
            
            # Capturar imagen
            captura = self.capture_service.capture_image(driver)
            if not captura:
                print("❌ Error capturando CAPTCHA")
                stats.errors += 1
                time.sleep(1)
                continue
            
            stats.captured += 1
            
            # Mostrar estadísticas de la imagen
            print(f"📸 Imagen capturada: {captura.image.size}")
            if captura.variance:
                print(f"   Varianza: {captura.variance:.2f}")
            
            # Solicitar etiqueta
            print()
            print("👁️  MIRA EL CAPTCHA EN EL NAVEGADOR")
            print()
            texto = input("💬 Texto del CAPTCHA (o 'skip'/'saltar' para omitir): ").strip().upper()
            
            # Validar entrada
            if not texto or texto.lower() in ('skip', 'saltar'):
                print("⏭️  Ejemplo omitido")
                stats.skipped += 1
                self.capture_service.reload(driver)
                time.sleep(1)
                continue
            
            # Limpiar y validar
            texto = ''.join(c for c in texto if c.isalnum())
            
            if not (3 <= len(texto) <= self.solver.max_length):
                print(f"⚠️  Longitud inválida (debe ser 3-{self.solver.max_length})")
                stats.skipped += 1
                self.capture_service.reload(driver)
                time.sleep(1)
                continue
            
            if not all(c in self.solver.caracteres for c in texto):
                print(f"⚠️  Caracteres inválidos. Solo se permiten: {self.solver.caracteres}")
                stats.skipped += 1
                self.capture_service.reload(driver)
                time.sleep(1)
                continue
            
            # Guardar ejemplo
            try:
                self.solver.agregar_ejemplo(captura.image, texto)
                stats.saved += 1
                print(f"✅ Guardado: '{texto}'")
            except Exception as exc:
                LOGGER.exception("Error guardando ejemplo: %s", exc)
                stats.errors += 1
            
            # Recargar para siguiente
            if idx < num_ejemplos - 1:
                self.capture_service.reload(driver)
                time.sleep(1)
        
        return stats
    
    def entrenar_modelo(
        self,
        epochs: int = 50,
        batch_size: int = 16
    ) -> bool:
        """Entrena el modelo CNN con los ejemplos recolectados."""
        print(f"\n{'='*60}")
        print(f"🎓 ENTRENAMIENTO DEL MODELO CNN")
        print(f"{'='*60}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print()
        
        return self.solver.entrenar(epochs=epochs, batch_size=batch_size)


def entrenar_cnn_completo(
    driver,
    num_ejemplos: int = 100,
    epochs: int = 50,
    batch_size: int = 16
) -> bool:
    """Pipeline completo: recolectar + entrenar."""
    from src.config.settings import load_settings
    
    settings = load_settings()
    
    # Crear servicios
    capture_service = CaptchaCaptureService(settings.temp_dir)
    solver = CaptchaSolverCNN(settings.training_dir / "cnn")
    session = CNNTrainingSession(solver, capture_service)
    
    # Etapa 1: Recolectar
    print("\n" + "="*60)
    print("ETAPA 1: RECOLECCIÓN DE EJEMPLOS")
    print("="*60)
    
    stats = session.recolectar_ejemplos(driver, num_ejemplos)
    
    print(f"\n{'='*60}")
    print(f"📊 RESUMEN DE RECOLECCIÓN")
    print(f"{'='*60}")
    print(f"Capturados:  {stats.captured}")
    print(f"Guardados:   {stats.saved}")
    print(f"Omitidos:    {stats.skipped}")
    print(f"Errores:     {stats.errors}")
    print(f"{'='*60}")
    
    if stats.saved < 20:
        print("\n⚠️  ADVERTENCIA: Muy pocos ejemplos guardados")
        print("   Se recomiendan al menos 50 ejemplos para buen entrenamiento")
        print()
        respuesta = input("¿Continuar de todas formas? (s/n): ").strip().lower()
        if respuesta != 's':
            print("❌ Entrenamiento cancelado")
            return False
    
    # Etapa 2: Entrenar
    print("\n" + "="*60)
    print("ETAPA 2: ENTRENAMIENTO")
    print("="*60)
    print()
    respuesta = input("¿Iniciar entrenamiento ahora? (s/n): ").strip().lower()
    
    if respuesta != 's':
        print("⏸️  Entrenamiento pospuesto")
        print(f"   Puedes entrenar después con: python -m src.cli train-cnn --train-only")
        return False
    
    return session.entrenar_modelo(epochs=epochs, batch_size=batch_size)


__all__ = [
    "CNNTrainingSession",
    "CNNTrainingStats",
    "entrenar_cnn_completo"
]