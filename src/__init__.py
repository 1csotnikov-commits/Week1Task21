"""Пакет src: пайплайн индексации документов (Week 1 Task 21)."""

from __future__ import annotations

import sys

__version__ = "0.1.0"

# На Windows консоль часто использует cp1252/cp866, из-за чего русский текст
# в print() падает с UnicodeEncodeError. Переключаем stdout/stderr на UTF-8.
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
