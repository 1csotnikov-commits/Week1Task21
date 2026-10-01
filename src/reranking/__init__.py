"""Reranking (переранжирование) — задел на День 23.

Интерфейс-заглушка: на следующих этапах сюда добавляется логика переранжирования.
"""

from __future__ import annotations

from typing import List

from src.models import ChunkWithScore


def rerank(chunks: List[ChunkWithScore], query: str) -> List[ChunkWithScore]:
    """Переранжировать чанки по запросу.

    Заглушка для Дня 23 — реализация будет добавлена позже.
    """
    raise NotImplementedError(
        "Переранжирование (reranking) реализуется на Дне 23."
    )


__all__ = ["rerank"]
