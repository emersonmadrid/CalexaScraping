#!/usr/bin/env python3
"""Compatibilidad con versiones anteriores: usa src.cli train-manual."""

from __future__ import annotations

from src import cli as cli_module


def main() -> int:
    return cli_module.main(["train-manual"])


if __name__ == "__main__":
    raise SystemExit(main())
