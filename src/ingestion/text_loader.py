"""Загрузчик текстовых документов (.md, .txt, .py, .js, .json и т.п.)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from src.ingestion.base import DocumentLoader
from src.models import Document

DEFAULT_EXTENSIONS = [".md", ".txt", ".py", ".js", ".json"]

# Порядок попыток декодирования (учитывает русскоязычные тексты на Windows).
_ENCODINGS = ("utf-8-sig", "utf-8", "cp1251")


class TextDocumentLoader(DocumentLoader):
    """Загружает только текстовые форматы. PDF и бинарные файлы игнорируются."""

    def __init__(self, extensions: Optional[List[str]] = None):
        self.extensions = extensions or DEFAULT_EXTENSIONS
        self._normalized_extensions = {
            ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in self.extensions
        }

    def load(self, path: str | Path, extensions: Optional[List[str]] = None) -> List[Document]:
        if extensions is not None:
            self.extensions = extensions
            self._normalized_extensions = {
                ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions
            }

        root = Path(path)
        if not root.exists():
            raise FileNotFoundError(f"Путь не найден: {root}")

        documents: List[Document] = []
        if root.is_file():
            doc = self._load_file(root, root.parent)
            if doc is not None:
                documents.append(doc)
        else:
            for file_path in sorted(root.rglob("*")):
                if not file_path.is_file():
                    continue
                doc = self._load_file(file_path, root)
                if doc is not None:
                    documents.append(doc)

        return documents

    def _load_file(self, file_path: Path, base_dir: Path) -> Optional[Document]:
        ext = file_path.suffix.lower()
        if ext not in self._normalized_extensions:
            return None

        text = self._read_text(file_path)
        if text is None:
            return None

        rel = file_path.relative_to(base_dir).as_posix() if file_path.is_relative_to(base_dir) else file_path.name
        return Document(
            source=rel,
            title=file_path.stem,
            text=text,
            format=ext.lstrip("."),
        )

    def _read_text(self, file_path: Path) -> Optional[str]:
        raw = file_path.read_bytes()
        # Отсекаем бинарные файлы с нулевыми байтами.
        if b"\x00" in raw:
            return None

        last_error: Optional[Exception] = None
        for enc in _ENCODINGS:
            try:
                return raw.decode(enc)
            except UnicodeDecodeError as exc:  # noqa: PERF203
                last_error = exc
        raise ValueError(
            f"Не удалось декодировать файл {file_path}: не найден подходящий текстовый кодек. "
            f"Последняя ошибка: {last_error}"
        )
