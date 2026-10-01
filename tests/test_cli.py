"""Тесты CLI-команд (без загрузки тяжёлой модели — с fake-провайдером)."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.cli.commands import build_help  # noqa: E402
from src.cli.main import App, handle_command, parse_args  # noqa: E402
from src.config import Config  # noqa: E402
from src.embeddings.base import EmbeddingProvider  # noqa: E402
from src.indexing.pipeline import build_index  # noqa: E402


class FakeProvider(EmbeddingProvider):
    name = "fake"
    dim = 768

    def encode(self, texts, batch_size=None):
        arr = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            arr[i, hash(t) % self.dim] = 1.0
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms


class TestCLI(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.docs = self.root / "documents"
        self.index = self.root / "index"
        self.docs.mkdir(parents=True)
        (self.docs / "a.md").write_text("# Заголовок\nТекст документа.\n\n## Раздел\nЕщё текст.", encoding="utf-8")
        (self.docs / "b.txt").write_text("Первый абзац.\n\nВторой абзац с более длинным содержанием.", encoding="utf-8")

        self.config = Config({
            "documents": {"path": str(self.docs), "extensions": [".md", ".txt"]},
            "index": {"path": str(self.index)},
            "embedding": {"provider": "nomic", "model": "fake-model", "device": "cpu", "batch_size": 16},
            "chunking": {
                "default_strategy": "fixed",
                "fixed_size": {"chunk_size": 200, "overlap": 40},
                "structural": {"max_chunk_size": 300},
            },
        })
        self.app = App(self.config)

        # Строим два индекса напрямую с fake-провайдером.
        build_index(self.docs, "fixed", "fixed_idx", self.config, provider=FakeProvider())
        build_index(self.docs, "structural", "structural_idx", self.config, provider=FakeProvider())

    def tearDown(self):
        self._tmp.cleanup()

    def test_parse_args(self):
        pos, flags = parse_args(["doc", "--strategy", "fixed", "--limit=5", "--flag"])
        self.assertEqual(pos, ["doc"])
        self.assertEqual(flags["strategy"], "fixed")
        self.assertEqual(flags["limit"], "5")
        self.assertIs(flags["flag"], True)

    def test_help(self):
        out = build_help()
        self.assertIn("/index", out)
        self.assertIn("/exit", out)
        self.assertIn("build", build_help("/index"))
        self.assertIn("fixed", build_help("/index build"))

    def test_handle_exit_and_debug(self):
        should_exit, out = handle_command("/exit", self.app)
        self.assertTrue(should_exit)
        _, out = handle_command("/debug on", self.app)
        self.assertIn("включена", out)
        _, out = handle_command("/debug off", self.app)
        self.assertIn("выключена", out)

    def test_unknown_command(self):
        should_exit, out = handle_command("/foo", self.app)
        self.assertFalse(should_exit)
        self.assertIn("Неизвестная команда", out)

    def test_index_list_and_info(self):
        _, out = handle_command("/index list", self.app)
        self.assertIn("fixed_idx", out)
        self.assertIn("structural_idx", out)

        _, out = handle_command("/index info fixed_idx", self.app)
        self.assertIn("fixed", out)
        self.assertIn("strategy", out)

    def test_stats_formats(self):
        for fmt in ("console", "json", "csv", "markdown"):
            _, out = handle_command(f"/index stats fixed_idx --format {fmt}", self.app)
            self.assertTrue(out.strip())

    def test_compare(self):
        _, out = handle_command("/index compare fixed_idx structural_idx --format console", self.app)
        self.assertIn("Сравнение", out)

    def test_show_chunk(self):
        app = self.app
        app.store.load("fixed_idx")
        chunk_id = app.store.list_chunks(limit=1)[0].chunk_id
        _, out = handle_command(f"/index show-chunk fixed_idx {chunk_id}", self.app)
        self.assertIn("--- Текст чанка ---", out)

    def test_list_chunks_with_filter(self):
        _, out = handle_command("/index list-chunks fixed_idx --limit 2", self.app)
        self.assertIn("Чанков:", out)

    def test_delete(self):
        _, out = handle_command("/index delete structural_idx", self.app)
        self.assertIn("удалён", out)
        self.assertFalse((self.index / "structural_idx").exists())


if __name__ == "__main__":
    unittest.main()
