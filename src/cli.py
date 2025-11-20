"""Auxiliary command-line utilities for CalexaScraping."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

from src.config.settings import PathSettings, load_settings
from src.main import build_captcha_manager, run_pipeline
from src.services.browser_manager import BrowserManager
from src.services.captcha_common import CaptchaCaptureService
from src.services.captcha_solver_ml import CaptchaSolverML
from src.services.training import ManualTrainingSession, PatternTrainingSession
from src.utils.helpers import timestamp
from src.utils.logger import configure_logging, get_logger

LOGGER = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.cli",
        description="Herramientas auxiliares para CalexaScraping",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Comando: run
    run_parser = subparsers.add_parser("run", help="Ejecuta el flujo principal de scraping")
    run_parser.add_argument(
        "--headless",
        dest="headless",
        action="store_const",
        const=True,
        help="Forzar ejecución en modo headless",
    )
    run_parser.add_argument(
        "--ui",
        dest="headless",
        action="store_const",
        const=False,
        help="Forzar ejecución con navegador visible",
    )
    run_parser.set_defaults(func=command_run, headless=None)

    # Comando: train-manual
    manual_parser = subparsers.add_parser(
        "train-manual", help="Sesión guiada para ingresar CAPTCHAs manualmente"
    )
    manual_parser.add_argument("-n", "--samples", type=int, default=50, help="Ejemplos a recolectar")
    manual_parser.set_defaults(func=command_train_manual)

    # Comando: train-assisted
    pattern_parser = subparsers.add_parser(
        "train-assisted",
        help="Sesión asistida: el solver propone respuestas y tú las confirmas",
    )
    pattern_parser.add_argument(
        "-n", "--samples", type=int, default=50, help="Ejemplos a recolectar"
    )
    pattern_parser.set_defaults(func=command_train_assisted)

    # Comando: train-cnn
    cnn_parser = subparsers.add_parser(
        "train-cnn",
        help="Entrena el modelo CNN para reconocimiento de CAPTCHAs"
    )
    cnn_parser.add_argument(
        "-n", "--samples", 
        type=int, 
        default=100, 
        help="Ejemplos a recolectar"
    )
    cnn_parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Número de epochs para entrenamiento"
    )
    cnn_parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Tamaño del batch"
    )
    cnn_parser.add_argument(
        "--train-only",
        action="store_true",
        help="Solo entrenar con datos existentes (no recolectar)"
    )
    cnn_parser.set_defaults(func=command_train_cnn)

    # Comando: captcha-test
    test_parser = subparsers.add_parser(
        "captcha-test", help="Ejecuta pruebas rápidas del solver en vivo"
    )
    test_parser.add_argument(
        "-n", "--attempts", type=int, default=5, help="Número de intentos de prueba"
    )
    test_parser.set_defaults(func=command_captcha_test)

    # Comando: diagnostics
    diag_parser = subparsers.add_parser(
        "diagnostics", help="Verifica dependencias, estructura y configuración"
    )
    diag_parser.set_defaults(func=command_diagnostics)

    return parser


def command_run(args: argparse.Namespace) -> int:
    return run_pipeline(headless=args.headless)


def command_train_manual(args: argparse.Namespace) -> int:
    settings = load_settings()
    settings.headless = False
    configure_logging(log_dir=settings.logs_dir)

    capture_service = CaptchaCaptureService(settings.temp_dir)
    solver = CaptchaSolverML(settings.training_dir)
    session = ManualTrainingSession(solver, capture_service)

    print("\n🧑‍🏫 ENTRENAMIENTO MANUAL")
    print("=" * 60)
    print(f"Objetivo: {args.samples} ejemplos")
    print("⚠️  IMPORTANTE: Los CAPTCHAs tienen 4 caracteres")
    print()

    with BrowserManager(settings) as driver:
        driver.get(settings.base_url)
        stats = session.run(driver, args.samples)

    print("\n📊 RESUMEN")
    print("=" * 60)
    print(f"Procesados: {stats.processed}")
    print(f"Guardados: {stats.saved}")
    print(f"Omitidos: {stats.skipped}")
    print(f"Total acumulado: {len(solver.patrones)} patrones\n")
    return 0


def command_train_assisted(args: argparse.Namespace) -> int:
    settings = load_settings()
    settings.headless = False
    configure_logging(log_dir=settings.logs_dir)

    capture_service = CaptchaCaptureService(settings.temp_dir)
    solver = CaptchaSolverML(settings.training_dir)
    session = PatternTrainingSession(solver, capture_service)

    print("\n🤖 ENTRENAMIENTO ASISTIDO")
    print("=" * 60)
    print(f"Objetivo: {args.samples} ejemplos")
    print("⚠️  IMPORTANTE: Los CAPTCHAs tienen 4 caracteres")
    print()

    with BrowserManager(settings) as driver:
        driver.get(settings.base_url)
        stats = session.run(driver, args.samples)

    print("\n📊 RESUMEN")
    print("=" * 60)
    print(f"Procesados: {stats.processed}")
    print(f"Confirmados: {stats.saved}")
    print(f"Correcciones: {stats.corrected}")
    print(f"Omitidos: {stats.skipped}")
    print(f"Total en base: {len(solver.patrones)} patrones\n")
    return 0


def command_train_cnn(args: argparse.Namespace) -> int:
    """Entrena el modelo CNN para CAPTCHAs."""
    try:
        from src.services.captcha_solver_cnn import CaptchaSolverCNN
        from src.services.training_cnn import CNNTrainingSession, entrenar_cnn_completo
    except ImportError as exc:
        print(f"\n❌ Error importando módulos CNN: {exc}")
        print("\n💡 Asegúrate de tener TensorFlow instalado:")
        print("   pip install tensorflow scikit-learn")
        return 1
    
    settings = load_settings()
    settings.headless = False
    configure_logging(log_dir=settings.logs_dir)
    
    # Si es solo entrenamiento
    if args.train_only:
        print("\n🎓 ENTRENAR MODELO CNN CON DATOS EXISTENTES")
        print("=" * 60)
        
        solver = CaptchaSolverCNN(settings.training_dir / "cnn")
        
        if not solver.labels_path.exists():
            print(f"\n❌ No hay datos de entrenamiento en {solver.labels_path}")
            print("   Primero recolecta ejemplos con: python -m src.cli train-cnn -n 100")
            return 1
        
        exito = solver.entrenar(epochs=args.epochs, batch_size=args.batch_size)
        return 0 if exito else 1
    
    # Flujo completo: recolectar + entrenar
    print("\n🤖 ENTRENAMIENTO CNN - FLUJO COMPLETO")
    print("=" * 60)
    print(f"Ejemplos a recolectar: {args.samples}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print()
    
    with BrowserManager(settings) as driver:
        driver.get(settings.base_url)
        time.sleep(2)
        
        exito = entrenar_cnn_completo(
            driver,
            num_ejemplos=args.samples,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
    
    return 0 if exito else 1


def command_captcha_test(args: argparse.Namespace) -> int:
    """Prueba visual mejorada con verificación manual."""
    settings = load_settings()
    settings.headless = False
    configure_logging(log_dir=settings.logs_dir)

    manager = build_captcha_manager(settings)

    print("\n🔬 PRUEBA VISUAL DE CAPTCHAS")
    print("=" * 60)
    print(f"Intentos: {args.attempts}")
    print(f"Solvers disponibles: {', '.join(manager.solver_names)}")
    print()
    print("💡 INSTRUCCIONES:")
    print("   - Mira el CAPTCHA en el navegador")
    print("   - Compara con la predicción del modelo")
    print("   - Verifica la imagen guardada en data/temp/captchas/")
    print("   - Presiona Enter para continuar al siguiente")
    print()

    exitos = 0
    exitos_verificados = 0
    resultados_por_solver = {}
    resultados_detallados = []

    with BrowserManager(settings) as driver:
        driver.get(settings.base_url)
        
        for intento in range(1, args.attempts + 1):
            print(f"\n{'='*60}")
            print(f"🎯 INTENTO {intento}/{args.attempts}")
            print(f"{'='*60}")
            
            # Capturar y predecir
            prediccion = manager.solve(driver)
            
            if prediccion:
                exitos += 1
                solver_name = prediccion.solver
                resultados_por_solver[solver_name] = resultados_por_solver.get(solver_name, 0) + 1
                
                # Mostrar resultado
                print(f"\n📊 PREDICCIÓN DEL MODELO:")
                print(f"   Texto: '{prediccion.text}'")
                print(f"   Longitud: {len(prediccion.text)} caracteres")
                print(f"   Solver: {solver_name}")
                if prediccion.image_path:
                    print(f"   Imagen: {prediccion.image_path}")
                
                # Solicitar verificación manual
                print(f"\n👁️  Mira el CAPTCHA en el navegador y compara")
                respuesta = input("¿Es correcto? (s/n/Enter para continuar): ").strip().lower()
                
                es_correcto = None
                texto_real = None
                
                if respuesta == 's':
                    es_correcto = True
                    exitos_verificados += 1
                    print("   ✅ Marcado como CORRECTO")
                elif respuesta == 'n':
                    es_correcto = False
                    texto_real = input("   ¿Cuál era el texto real? (4 caracteres): ").strip().upper()
                    print(f"   ❌ Marcado como INCORRECTO (real: {texto_real})")
                
                resultados_detallados.append({
                    'intento': intento,
                    'prediccion': prediccion.text,
                    'longitud': len(prediccion.text),
                    'solver': solver_name,
                    'verificado': es_correcto,
                    'texto_real': texto_real,
                    'imagen': str(prediccion.image_path) if prediccion.image_path else None
                })
                
            else:
                print("❌ No se pudo resolver el CAPTCHA")
                resultados_detallados.append({
                    'intento': intento,
                    'prediccion': None,
                    'longitud': 0,
                    'solver': None,
                    'verificado': False,
                    'texto_real': None,
                    'imagen': None
                })
            
            # Recargar para siguiente
            if intento < args.attempts:
                manager.capture_service.reload(driver)
                time.sleep(1)

    # Resumen final
    tasa = (exitos / args.attempts) * 100 if args.attempts else 0
    
    print("\n" + "="*60)
    print("📊 RESUMEN DETALLADO")
    print("="*60)
    print(f"Total intentos:     {args.attempts}")
    print(f"Predicciones:       {exitos}/{args.attempts} ({tasa:.1f}%)")
    
    if exitos_verificados > 0:
        tasa_verificada = (exitos_verificados / exitos) * 100
        print(f"Verificados OK:     {exitos_verificados}/{exitos} ({tasa_verificada:.1f}%)")
    
    print("\n📈 Por solver:")
    for solver, count in sorted(resultados_por_solver.items(), key=lambda x: -x[1]):
        print(f"   {solver}: {count}")
    
    # Análisis de longitudes
    longitudes = [r['longitud'] for r in resultados_detallados if r['longitud'] > 0]
    if longitudes:
        from collections import Counter
        contador_long = Counter(longitudes)
        print("\n📏 Distribución de longitudes detectadas:")
        for long, count in sorted(contador_long.items()):
            print(f"   {long} caracteres: {count} veces")
    
    # Errores detectados
    errores = [r for r in resultados_detallados if r['verificado'] is False]
    if errores:
        print(f"\n❌ Errores detectados manualmente: {len(errores)}")
        for error in errores:
            if error['texto_real']:
                print(f"   Intento {error['intento']}: '{error['prediccion']}' → '{error['texto_real']}'")
    
    # Guardar reporte
    reporte_path = Path("data/temp") / f"test_report_{timestamp()}.json"
    with open(reporte_path, 'w', encoding='utf-8') as f:
        json.dump({
            'total': args.attempts,
            'exitos': exitos,
            'exitos_verificados': exitos_verificados,
            'tasa_exito': tasa,
            'por_solver': resultados_por_solver,
            'resultados': resultados_detallados
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Reporte guardado en: {reporte_path}")
    print("="*60)
    
    return 0


def command_diagnostics(_: argparse.Namespace) -> int:
    settings = load_settings()
    configure_logging(log_dir=settings.logs_dir)

    print("\n🔍 DIAGNÓSTICO DEL ENTORNO")
    print("=" * 60)
    print(f"Python: {sys.version}")
    print(f"Directorio: {PathSettings().root}\n")

    modules = {
        "selenium": "selenium",
        "webdriver_manager": "webdriver-manager",
        "easyocr": "easyocr",
        "pytesseract": "pytesseract",
        "cv2": "opencv-python",
        "PIL": "pillow",
        "requests": "requests",
        "pandas": "pandas",
        "numpy": "numpy",
        "tensorflow": "tensorflow (opcional, para CNN)",
        "sklearn": "scikit-learn (opcional, para CNN)",
    }

    missing = []
    optional_missing = []
    
    for module, package in modules.items():
        is_optional = "(opcional" in package.lower()
        if importlib.util.find_spec(module):
            print(f"   ✅ {package}")
        else:
            print(f"   {'⚠️ ' if is_optional else '❌'} {package}")
            if is_optional:
                optional_missing.append(package)
            else:
                missing.append(package)

    print("\n🔤 Tesseract OCR:")
    try:
        import pytesseract
        version = pytesseract.get_tesseract_version()
        print(f"   ✅ Detectado (versión {version})")
        tesseract_ok = True
    except Exception:
        print("   ⚠️  No detectado (opcional)")
        tesseract_ok = False

    print("\n📁 Directorios clave:")
    paths = PathSettings()
    folders = [
        paths.root / "src",
        paths.data_dir,
        paths.inputs_dir,
        paths.temp_dir,
        paths.logs_dir,
        paths.training_dir,
        paths.drivers_dir,
    ]
    structure_ok = True
    for folder in folders:
        if folder.exists():
            print(f"   ✅ {folder}")
        else:
            print(f"   ❌ {folder}")
            structure_ok = False

    status_ok = not missing and structure_ok

    print("\n📦 Resumen")
    print("=" * 60)
    if missing:
        print(f"❌ Faltan módulos obligatorios: {', '.join(missing)}")
    else:
        print("✅ Dependencias principales instaladas")

    if optional_missing:
        print(f"⚠️  Módulos opcionales faltantes: {', '.join(optional_missing)}")
        print("   (Para usar CNN, instala: pip install tensorflow scikit-learn)")

    if not tesseract_ok:
        print("⚠️  Tesseract no detectado (recomendado)")

    if structure_ok:
        print("✅ Estructura de carpetas completa")
    else:
        print("⚠️  Falta crear algunas carpetas")

    print(f"\nChromeDriver esperado en: {settings.chromedriver_path}")

    return 0 if status_ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())