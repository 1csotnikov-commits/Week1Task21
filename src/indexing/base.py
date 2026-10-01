"""Абстракция хранилища индексов."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from src.models import Chunk, ChunkWithScore


class IndexStore(ABC):
    """Интерфейс хранилища индексов.

    Позволяет добавлять альтернативные реализации (JSON, чистый SQLite и т.п.)
    без изменения остального кода.
    """

    @abstractmethod
    def begin(self, index_name: str, strategy: str, model: str, provider: str, dim: int) -> None:
        """Начать построение нового индекса."""

    @abstractmethod
    def add(self, chunk: Chunk, embedding: np.ndarray) -> None:
        """Добавить один чанк с эмбеддингом."""

    @abstractmethod
    def finalize(self, num_documents: int) -> None:
        """Завершить построение индекса и записать его на диск."""

    @abstractmethod
    def load(self, index_name: str) -> None:
        """Загрузить существующий индекс."""

    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[ChunkWithScore]:
        """Найти ближайшие чанки по косинусной близости."""

    @abstractmethod
    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        """Получить чанк по идентификатору."""

    @abstractmethod
    def get_embedding(self, chunk_id: str) -> Optional[np.ndarray]:
        """Получить эмбеддинг чанка (для отладки)."""

    @abstractmethod
    def list_chunks(self, source: Optional[str] = None, section: Optional[str] = None, limit: Optional[int] = None) -> List[Chunk]:
        """Список чанков с фильтрами."""

    @abstractmethod
    def info(self, index_name: str) -> dict:
        """Метаданные индекса."""

    @abstractmethod
    def list_indices(self) -> List[dict]:
        """Список доступных индексов."""

    @abstractmethod
    def delete(self, index_name: str) -> None:
        """Удалить индекс."""
