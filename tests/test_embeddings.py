"""Тесты генерации эмбеддингов (nomic-embed-text-v2-moe).

Тест пропускается, если модель недоступна (нет сети / не скачалась).
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.embeddings.nomic_provider import NomicProvider  # noqa: E402


class TestEmbeddings(unittest.TestCase):
    def test_nomic_provider_dim_and_normalization(self):
        try:
            import sentence_transformers  # noqa: F401
        except Exception:  # noqa: BLE001 - нет пакета или не грузится DLL torch
            self.skipTest("sentence-transformers недоступен")

        provider = NomicProvider()
        try:
            vecs = provider.encode(["Привет, мир", "Второй русский текст"])
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"Модель не загрузилась: {exc}")

        self.assertEqual(vecs.shape, (2, 768))
        norms = np.linalg.norm(vecs, axis=1)
        np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-4)


if __name__ == "__main__":
    unittest.main()
