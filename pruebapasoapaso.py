#!/usr/bin/env python3
"""Compatibilidad con versiones anteriores: usa src.cli run --ui."""

from __future__ import annotations

from src import cli as cli_module


def main() -> int:
    return cli_module.main(["run", "--ui"])


if __name__ == "__main__":
    raise SystemExit(main())
