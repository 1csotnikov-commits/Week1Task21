"""Тесты статистики чанкинга."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import Chunk, Document  # noqa: E402
from src.stats.chunk_stats import compute_stats, render, render_compare  # noqa: E402


class TestStats(unittest.TestCase):
    def _make(self):
        docs = [Document(source="a.txt", title="a", text="x" * 100, format="txt")]
        chunks = [
            Chunk(chunk_id="c0", source="a.txt", title="a", text="Абзац один.", start_char=0, end_char=10, strategy="fixed"),
            Chunk(chunk_id="c1", source="a.txt", title="a", text="Второй абзац.", start_char=5, end_char=20, strategy="fixed"),
        ]
        return docs, chunks

    def test_basic_metrics(self):
        docs, chunks = self._make()
        stats = compute_stats(docs, chunks, 1.5, "fixed", "idx")
        self.assertEqual(stats["num_documents"], 1)
        self.assertEqual(stats["num_chunks"], 2)
        self.assertEqual(stats["index_name"], "idx")
        self.assertEqual(stats["chunk_length"]["min"], min(len(c.text) for c in chunks))
        self.assertEqual(stats["chunk_length"]["max"], max(len(c.text) for c in chunks))

    def test_overlap_fixed(self):
        docs, chunks = self._make()
        stats = compute_stats(docs, chunks, 0.0, "fixed", "idx")
        self.assertEqual(stats["overlap"]["overlapping_chunks"], 1)
        self.assertEqual(stats["overlap"]["avg_overlap"], 5.0)

    def test_structural_sections(self):
        docs = [Document(source="a.md", title="a", text="x", format="md")]
        chunks = [
            Chunk(chunk_id="c0", source="a.md", title="a", text="Текст", start_char=0, end_char=4, strategy="structural", section="A", metadata={"section_count": 1}),
            Chunk(chunk_id="c1", source="a.md", title="a", text="Ещё", start_char=4, end_char=7, strategy="structural", section="B", metadata={"section_count": 2}),
        ]
        stats = compute_stats(docs, chunks, 0.0, "structural", "idx")
        self.assertIn("single_section_ratio", stats["semantic_integrity"])
        self.assertEqual(stats["semantic_integrity"]["single_section"], 1)
        self.assertEqual(stats["semantic_integrity"]["multi_section"], 1)

    def test_render_formats(self):
        docs, chunks = self._make()
        stats = compute_stats(docs, chunks, 1.0, "fixed", "idx")
        for fmt in ("console", "json", "csv", "markdown"):
            out = render(stats, fmt)
            self.assertIsInstance(out, str)
            self.assertTrue(out.strip())

    def test_render_compare(self):
        docs, chunks = self._make()
        a = compute_stats(docs, chunks, 1.0, "fixed", "a")
        b = compute_stats(docs, chunks, 1.0, "structural", "b")
        for fmt in ("console", "json", "csv", "markdown"):
            out = render_compare(a, b, fmt)
            self.assertTrue(out.strip())

    def test_unknown_format(self):
        docs, chunks = self._make()
        stats = compute_stats(docs, chunks, 1.0, "fixed", "idx")
        with self.assertRaises(ValueError):
            render(stats, "xml")


if __name__ == "__main__":
    unittest.main()
