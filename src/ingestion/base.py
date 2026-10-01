"""Абстракция загрузчика документов."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

from src.models import Document


class DocumentLoader(ABC):
    """Интерфейс загрузчика документов.

    Реализации должны возвращать список объектов Document.
    """

    @abstractmethod
    def load(self, path: str | Path, extensions: Optional[List[str]] = None) -> List[Document]:
        """Загрузить документы из файла или папки.

        :param path: путь к файлу или каталогу.
        :param extensions: допустимые расширения (например, ['.md', '.txt']).
        :return: список загруженных документов.
        """
        raise NotImplementedError
