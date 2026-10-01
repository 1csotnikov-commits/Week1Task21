"""Стратегии чанкинга документов."""

from __future__ import annotations

from src.chunking.base import BaseChunker
from src.chunking.fixed_size import FixedSizeChunker
from src.chunking.structural import StructuralChunker

STRATEGIES = ("fixed", "structural")


def make_chunker(strategy: str, config=None) -> BaseChunker:
    """Создать чанкер по имени стратегии с параметрами из конфига."""
    strategy = (strategy or "structural").lower()
    if strategy == "fixed":
        chunk_size = 800
        overlap = 150
        if config is not None:
            chunk_size = int(config.get("chunking.fixed_size.chunk_size", 800))
            overlap = int(config.get("chunking.fixed_size.overlap", 150))
        return FixedSizeChunker(chunk_size=chunk_size, overlap=overlap)
    if strategy == "structural":
        max_chunk_size = 1500
        split_by = "heading"
        if config is not None:
            max_chunk_size = int(config.get("chunking.structural.max_chunk_size", 1500))
            split_by = config.get("chunking.structural.split_by", "heading")
        return StructuralChunker(max_chunk_size=max_chunk_size, split_by=split_by)
    raise ValueError(f"Неизвестная стратегия чанкинга: {strategy!r}. Доступно: {', '.join(STRATEGIES)}")


__all__ = ["BaseChunker", "FixedSizeChunker", "StructuralChunker", "make_chunker", "STRATEGIES"]
