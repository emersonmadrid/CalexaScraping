#!/usr/bin/env python3
"""Muestra los selectores configurados del formulario CEJ."""

from __future__ import annotations

from src.config import selectors


def main() -> None:
    print("🔍 SELECTORES CONFIGURADOS")
    print("=" * 40)

    campos = {
        "EXPEDIENTE_INPUT": selectors.EXPEDIENTE_INPUT,
        "CAPTCHA_INPUT": selectors.CAPTCHA_INPUT,
        "CAPTCHA_IMAGE": selectors.CAPTCHA_IMAGE,
        "RELOAD_CAPTCHA_BUTTON": selectors.RELOAD_CAPTCHA_BUTTON,
        "BUSCAR_BUTTON": selectors.BUSCAR_BUTTON,
        "PARTE_INPUT": selectors.PARTE_INPUT,
        "DISTRITO_SELECT": selectors.DISTRITO_SELECT,
        "ORGANO_SELECT": selectors.ORGANO_SELECT,
        "ESPECIALIDAD_SELECT": selectors.ESPECIALIDAD_SELECT,
        "ANIO_SELECT": selectors.ANIO_SELECT,
    }

    for nombre, valor in campos.items():
        print(f"{nombre:<25} -> {valor}")

    print("
💡 Edita src/config/selectors.py para actualizar estos valores.")


if __name__ == "__main__":
    main()
