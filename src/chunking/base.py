"""Абстракция чанкера."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from src.models import Chunk, Document


class BaseChunker(ABC):
    """Интерфейс стратегии чанкинга."""

    strategy: str = "base"

    @abstractmethod
    def chunk(self, document: Document) -> List[Chunk]:
        """Разбить документ на чанки."""
        raise NotImplementedError
