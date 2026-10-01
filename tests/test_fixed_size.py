"""Тесты стратегии fixed_size."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunking.fixed_size import FixedSizeChunker  # noqa: E402
from src.models import Document  # noqa: E402


def make_doc(text: str, fmt: str = "txt") -> Document:
    return Document(source=f"t.{fmt}", title="t", text=text, format=fmt)


class TestFixedSize(unittest.TestCase):
    def test_offsets_match_text(self):
        text = "Первое предложение. " * 20
        chunker = FixedSizeChunker(chunk_size=300, overlap=50)
        chunks = chunker.chunk(make_doc(text))
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertEqual(c.text, text[c.start_char:c.end_char])
            self.assertEqual(c.strategy, "fixed_size")

    def test_overlap_present(self):
        text = " ".join(f"Предложение номер {i}." for i in range(80))
        chunker = FixedSizeChunker(chunk_size=200, overlap=80)
        chunks = chunker.chunk(make_doc(text))
        overlaps = [a.end_char - b.start_char for a, b in zip(chunks, chunks[1:])]
        self.assertTrue(any(o > 0 for o in overlaps))

    def test_no_word_split(self):
        text = "Это тестовый текст с длинными словами. " * 40
        chunker = FixedSizeChunker(chunk_size=120, overlap=30)
        for c in chunker.chunk(make_doc(text)):
            s, e = c.start_char, c.end_char
            if s > 0 and text[s - 1].isalnum() and text[s].isalnum():
                self.fail(f"чанк начинается внутри слова: {c.text[:20]!r}")
            if e < len(text) and text[e - 1].isalnum() and text[e].isalnum():
                self.fail(f"чанк заканчивается внутри слова: {c.text[-20:]!r}")

    def test_short_text_single_chunk(self):
        text = "Короткий текст."
        chunks = FixedSizeChunker(chunk_size=800, overlap=150).chunk(make_doc(text))
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, text)

    def test_invalid_params(self):
        with self.assertRaises(ValueError):
            FixedSizeChunker(chunk_size=0)
        with self.assertRaises(ValueError):
            FixedSizeChunker(chunk_size=100, overlap=100)


if __name__ == "__main__":
    unittest.main()
