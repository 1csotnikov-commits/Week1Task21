"""Абстракция генератора эмбеддингов."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np


class EmbeddingProvider(ABC):
    """Интерфейс генератора эмбеддингов.

    Реализации возвращают нормализованные векторы (L2-норма == 1),
    что позволяет использовать IndexFlatIP (inner product) как косинусную близость.
    """

    name: str = "base"
    dim: Optional[int] = None

    @abstractmethod
    def encode(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        """Вернуть матрицу эмбеддингов формы (len(texts), dim)."""
        raise NotImplementedError
