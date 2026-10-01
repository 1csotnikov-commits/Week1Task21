"""Реализация EmbeddingProvider на базе nomic-ai/nomic-embed-text-v2-moe.

Единственная разрешённая модель эмбеддингов (768 измерений, Apache 2.0, русский язык).
Модель скачивается автоматически при первом запуске (~1–2 ГБ).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from src.embeddings.base import EmbeddingProvider

DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v2-moe"


class NomicProvider(EmbeddingProvider):
    """Эмбеддинги через локальную модель nomic-embed-text-v2-moe."""

    name = "nomic"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
        batch_size: int = 16,
        trust_remote_code: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.trust_remote_code = trust_remote_code
        self._model = None
        self.dim = 768

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        print(
            "Загрузка модели эмбеддингов... "
            "При первом запуске модель скачивается автоматически (~1–2 ГБ) и это может занять время."
        )
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "Пакет sentence-transformers не установлен. "
                "Выполните: pip install sentence-transformers einops"
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Не удалось импортировать sentence-transformers. "
                "На Windows это часто связано с отсутствием Microsoft Visual C++ Redistributable "
                "(не загружается DLL библиотеки torch). "
                f"Техническая ошибка: {exc}"
            ) from exc

        try:
            self._model = SentenceTransformer(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
                device=self.device,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Не удалось загрузить модель {self.model_name}. "
                "Проверьте подключение к интернету и повторите попытку. "
                f"Техническая ошибка: {exc}"
            ) from exc

        try:
            self.dim = int(self._model.get_sentence_embedding_dimension())
        except Exception:  # noqa: BLE001
            self.dim = 768

        print(f"Модель {self.model_name} загружена (размерность: {self.dim}).")

    def encode(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        self._ensure_loaded()
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        bs = batch_size or self.batch_size
        vectors = self._model.encode(
            texts,
            batch_size=bs,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)
