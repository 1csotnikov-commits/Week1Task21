"""Тесты хранилища FAISS + SQLite (с fallback на numpy)."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.indexing.faiss_sqlite_store import FAISSSQLiteStore  # noqa: E402
from src.models import Chunk, Document  # noqa: E402

DIM = 8


def _unit_vec() -> np.ndarray:
    v = np.ones(DIM, dtype=np.float32)
    return v / np.linalg.norm(v)


class TestStore(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _build(self, name: str = "idx") -> FAISSSQLiteStore:
        store = FAISSSQLiteStore(self.root)
        store.begin(name, "fixed", "model", "nomic", DIM)
        doc = Document(source="a.txt", title="a", text="hello world", format="txt")
        store.add_document(doc)
        for i in range(3):
            chunk = Chunk(
                chunk_id=f"c{i}",
                source="a.txt",
                title="a",
                text=f"текст {i}",
                start_char=0,
                end_char=5,
                strategy="fixed",
                section="s",
            )
            store.add(chunk, _unit_vec())
        store.finalize(num_documents=1)
        return store

    def test_build_load_search(self):
        store = self._build()
        store.load("idx")
        self.assertEqual(len(store.list_chunks()), 3)
        self.assertEqual(store.get_chunk("c1").text, "текст 1")

        results = store.search(_unit_vec(), top_k=2)
        self.assertEqual(len(results), 2)
        self.assertGreater(results[0].score, 0.9)

    def test_info_and_list(self):
        self._build()
        store = FAISSSQLiteStore(self.root)
        info = store.info("idx")
        self.assertEqual(info["num_chunks"], 3)
        self.assertEqual(info["num_documents"], 1)
        self.assertEqual(info["strategy"], "fixed")
        self.assertEqual(len(store.list_indices()), 1)

    def test_get_embedding(self):
        store = self._build()
        store.load("idx")
        emb = store.get_embedding("c0")
        self.assertEqual(emb.shape, (DIM,))

    def test_delete(self):
        self._build()
        store = FAISSSQLiteStore(self.root)
        store.delete("idx")
        self.assertFalse((self.root / "idx").exists())

    def test_load_missing(self):
        store = FAISSSQLiteStore(self.root)
        with self.assertRaises(ValueError):
            store.load("missing")


if __name__ == "__main__":
    unittest.main()
