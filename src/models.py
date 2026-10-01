"""Общие структуры данных пайплайна индексации.

Все сущности (Document, Chunk, ChunkWithScore, Citation) собраны здесь,
чтобы модули ingestion / chunking / indexing / retrieval / rag не зависели
друг от друга циклическими импортами.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


def estimate_tokens(text: str) -> int:
    """Эвристическая оценка количества токенов (~4 символа на токен).

    Точная токенизация не требуется по заданию, используется простая оценка.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


@dataclass
class Document:
    """Документ, загруженный из файла."""

    source: str          # относительный путь к файлу (относительно корня documents)
    title: str           # имя файла без расширения
    text: str            # полный текст документа
    format: str          # расширение без точки: md, txt, py, js, json
    char_count: int = 0
    token_count: int = 0

    def __post_init__(self) -> None:
        self.format = self.format.lower().lstrip(".")
        if not self.char_count:
            self.char_count = len(self.text)
        if not self.token_count:
            self.token_count = estimate_tokens(self.text)


@dataclass
class Chunk:
    """Фрагмент документа (чанк) с метаданными."""

    chunk_id: str
    source: str
    title: str
    text: str
    start_char: int
    end_char: int      # конец (исключительно), как в срезе text[start_char:end_char]
    strategy: str
    section: str = ""
    char_count: int = 0
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.char_count:
            self.char_count = len(self.text)
        if not self.token_count:
            self.token_count = estimate_tokens(self.text)


@dataclass
class ChunkWithScore:
    """Чанк с оценкой релевантности (используется retrieval / reranking)."""

    chunk: Chunk
    score: float


@dataclass
class Citation:
    """Ссылка на источник, используемая в RAG-ответе."""

    chunk_id: str
    source: str
    title: str
    section: str
    text: str
    score: float = 0.0
