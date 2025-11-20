"""Auxiliary command-line utilities for CalexaScraping."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time

from src.config.settings import PathSettings, load_settings
from src.main import build_captcha_manager, run_pipeline
from src.services.browser_manager import BrowserManager
from src.services.captcha_common import CaptchaCaptureService
from src.services.captcha_solver_ml import CaptchaSolverML
from src.services.training import ManualTrainingSession, PatternTrainingSession
from src.utils.logger import configure_logging, get_logger

LOGGER = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.cli",
        description="Herramientas auxiliares para CalexaScraping",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

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

    manual_parser = subparsers.add_parser(
        "train-manual", help="Sesion guiada para ingresar CAPTCHAs manualmente"
    )
    manual_parser.add_argument("-n", "--samples", type=int, default=50, help="Ejemplos a recolectar")
    manual_parser.set_defaults(func=command_train_manual)

    pattern_parser = subparsers.add_parser(
        "train-assisted",
        help="Sesion asistida: el solver propone respuestas y tú las confirmas",
    )
    pattern_parser.add_argument(
        "-n", "--samples", type=int, default=50, help="Ejemplos a recolectar"
    )
    pattern_parser.set_defaults(func=command_train_assisted)

    test_parser = subparsers.add_parser(
        "captcha-test", help="Ejecuta pruebas rápidas del solver en vivo"
    )
    test_parser.add_argument(
        "-n", "--attempts", type=int, default=5, help="Número de intentos de prueba"
    )
    test_parser.set_defaults(func=command_captcha_test)

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
    print(f"Objetivo: {args.samples} ejemplos\n")

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
    print(f"Objetivo: {args.samples} ejemplos\n")

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


def command_captcha_test(args: argparse.Namespace) -> int:
    settings = load_settings()
    settings.headless = False
    configure_logging(log_dir=settings.logs_dir)

    manager = build_captcha_manager(settings)

    print("\n🔬 PRUEBA RÁPIDA DE CAPTCHAS")
    print("=" * 60)
    print(f"Intentos: {args.attempts}\n")

    exitos = 0

    with BrowserManager(settings) as driver:
        driver.get(settings.base_url)
        for intento in range(1, args.attempts + 1):
            print(f"\n--- Intento {intento}/{args.attempts} ---")
            prediccion = manager.solve(driver)
            if prediccion:
                exitos += 1
                print(f"✅ Resultado: '{prediccion.text}' (solver: {prediccion.solver})")
            else:
                print("❌ No se pudo resolver el CAPTCHA")
            manager.capture_service.reload(driver)
            time.sleep(1)

    tasa = (exitos / args.attempts) * 100 if args.attempts else 0
    print("\n📊 RESUMEN")
    print("=" * 60)
    print(f"Éxitos: {exitos}/{args.attempts} ({tasa:.1f}%)\n")
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
    }

    missing = []
    for module, package in modules.items():
        if importlib.util.find_spec(module):
            print(f"   ✅ {package}")
        else:
            print(f"   ❌ {package}")
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
        print(f"❌ Faltan módulos: {', '.join(missing)}")
    else:
        print("✅ Dependencias principales instaladas")

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
