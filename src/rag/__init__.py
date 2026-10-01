"""RAG (генерация ответа) — задел на День 24.

Интерфейс-заглушка: на следующих этапах сюда добавляется логика генерации.
"""

from __future__ import annotations

from typing import List

from src.models import Chunk, Citation


def build_prompt(question: str, chunks: List[Chunk]) -> str:
    """Собрать промпт для LLM из вопроса и найденных чанков.

    Заглушка для Дня 24 — реализация будет добавлена позже.
    """
    raise NotImplementedError(
        "Генерация промпта (RAG) реализуется на Дне 24."
    )


def extract_citations(answer: str, chunks: List[Chunk]) -> List[Citation]:
    """Извлечь цитаты/ссылки на источники из ответа.

    Заглушка для Дня 24 — реализация будет добавлена позже.
    """
    raise NotImplementedError(
        "Извлечение цитат (RAG) реализуется на Дне 24."
    )


__all__ = ["build_prompt", "extract_citations"]
