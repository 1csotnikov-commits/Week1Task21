"""Тесты загрузчика документов."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.text_loader import TextDocumentLoader  # noqa: E402


class TestLoader(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, rel: str, content: str) -> Path:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return p

    def test_loads_multiple_formats(self):
        self._write("a.md", "# Заголовок\nТекст")
        self._write("b.txt", "Обычный текст")
        self._write("c.py", "def f():\n    return 1")
        self._write("sub/d.json", '{"k": "v"}')
        self._write("ignore.pdf", "бинарный")
        loader = TextDocumentLoader()
        docs = loader.load(self.root)
        self.assertEqual(len(docs), 4)
        self.assertEqual(sorted(d.title for d in docs), ["a", "b", "c", "d"])

    def test_format_normalized(self):
        self._write("x.MD", "text")
        docs = TextDocumentLoader().load(self.root)
        self.assertEqual(docs[0].format, "md")

    def test_recursive(self):
        self._write("nested/deep/file.txt", "text")
        docs = TextDocumentLoader().load(self.root)
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].source, "nested/deep/file.txt")

    def test_empty_dir(self):
        self.assertEqual(TextDocumentLoader().load(self.root), [])

    def test_missing_path_raises(self):
        with self.assertRaises(FileNotFoundError):
            TextDocumentLoader().load(self.root / "nope")


if __name__ == "__main__":
    unittest.main()
