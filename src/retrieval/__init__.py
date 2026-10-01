"""Retrieval (поиск) — задел на День 22.

Интерфейс-заглушка: на следующих этапах сюда добавляется логика поиска.
"""

from __future__ import annotations

from typing import List

from src.models import ChunkWithScore


def search(query: str, top_k: int = 5) -> List[ChunkWithScore]:
    """Найти наиболее релевантные чанки по запросу.

    Заглушка для Дня 22 — реализация будет добавлена позже.
    """
    raise NotImplementedError(
        "Поиск (retrieval) реализуется на Дне 22. "
        "Сейчас доступен низкоуровневый поиск через IndexStore.search()."
    )


__all__ = ["search"]
