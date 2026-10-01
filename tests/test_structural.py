"""Тесты стратегии structural."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunking.structural import StructuralChunker  # noqa: E402
from src.models import Document  # noqa: E402


def make_doc(text: str, fmt: str) -> Document:
    return Document(source=f"t.{fmt}", title="t", text=text, format=fmt)


class TestStructural(unittest.TestCase):
    def test_markdown_headings_split(self):
        md = (
            "# Введение\nТекст.\n\n"
            "## Раздел 1\nТекст.\n\n"
            "### Подраздел\nТекст.\n"
        )
        chunker = StructuralChunker(max_chunk_size=1000)
        titles = [s.title for s in chunker._extract_sections(md, "md")]
        self.assertEqual(titles, ["Введение", "Раздел 1", "Подраздел"])

    def test_markdown_merges_small_sections(self):
        md = "# A\nодин\n\n# B\nдва\n"
        chunks = StructuralChunker(max_chunk_size=1000).chunk(make_doc(md, "md"))
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].metadata["section_count"], 2)
        self.assertEqual(chunks[0].section, "A")

    def test_python_ast(self):
        code = (
            "import os\n\n"
            "def foo():\n    return 1\n\n"
            "class Bar:\n    def baz(self):\n        pass\n"
        )
        titles = [s.title for s in StructuralChunker()._extract_sections(code, "py")]
        self.assertTrue(any("def foo" in t for t in titles))
        self.assertTrue(any("class Bar" in t for t in titles))

    def test_text_paragraphs(self):
        text = "Первый абзац.\n\nВторой абзац.\n\nТретий абзац."
        sections = StructuralChunker()._extract_sections(text, "txt")
        self.assertGreaterEqual(len(sections), 3)

    def test_large_section_split(self):
        md = "# Большой раздел\n" + ("Очень длинный текст. " * 200) + "\n"
        chunks = StructuralChunker(max_chunk_size=500).chunk(make_doc(md, "md"))
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(c.char_count, 500 + 60)

    def test_metadata_section_count(self):
        md = "# A\nкоротко\n\n# B\nещё\n"
        chunks = StructuralChunker(max_chunk_size=1000).chunk(make_doc(md, "md"))
        self.assertGreaterEqual(len(chunks), 1)
        for c in chunks:
            self.assertIn("section_count", c.metadata)


if __name__ == "__main__":
    unittest.main()
