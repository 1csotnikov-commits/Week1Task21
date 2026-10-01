"""Стратегия чанкинга A — fixed_size (фиксированный размер с перекрытием)."""

from __future__ import annotations

from typing import List

from src.chunking.base import BaseChunker
from src.chunking.utils import fixed_size_spans, is_code_format, trim_span
from src.models import Chunk, Document


class FixedSizeChunker(BaseChunker):
    """Режет текст на чанки примерно chunk_size символов с перекрытием overlap.

    Рез выполняется по границам предложений/строк, чтобы не рвать слова.
    """

    strategy = "fixed_size"

    def __init__(self, chunk_size: int = 800, overlap: int = 150):
        if chunk_size <= 0:
            raise ValueError("chunk_size должен быть положительным числом")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap должен быть в диапазоне [0, chunk_size)")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: Document) -> List[Chunk]:
        prefer_lines = is_code_format(document.format)
        spans = fixed_size_spans(
            document.text,
            self.chunk_size,
            self.overlap,
            prefer_lines=prefer_lines,
        )

        chunks: List[Chunk] = []
        for idx, (s, e) in enumerate(spans):
            text, s, e = trim_span(document.text, s, e)
            if not text:
                continue
            chunks.append(
                Chunk(
                    chunk_id=self._chunk_id(document.title, idx),
                    source=document.source,
                    title=document.title,
                    text=text,
                    start_char=s,
                    end_char=e,
                    strategy=self.strategy,
                    section="",
                    metadata={"format": document.format},
                )
            )
        return chunks

    @staticmethod
    def _chunk_id(title: str, idx: int) -> str:
        return f"{title}-fixed_size-{idx:04d}"
