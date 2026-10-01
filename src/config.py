"""Загрузка и работа с конфигурацией (config/config.yaml)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


class Config:
    """Обёртка над YAML-конфигом с доступом по точечному пути (dotted path)."""

    def __init__(self, data: Dict[str, Any], path: Path = DEFAULT_CONFIG_PATH):
        self.data = data or {}
        self.path = path

    @classmethod
    def load(cls, path: Optional[str | Path] = None) -> "Config":
        config_path = Path(path) if path else DEFAULT_CONFIG_PATH
        if not config_path.exists():
            raise FileNotFoundError(
                f"Файл конфигурации не найден: {config_path}. "
                "Создайте config/config.yaml или укажите путь через --config."
            )
        with open(config_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls(data, config_path)

    def get(self, dotted: str, default: Any = None) -> Any:
        """Получить значение по точечному пути, например 'chunking.fixed_size.chunk_size'."""
        current: Any = self.data
        for part in dotted.split("."):
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        return current

    def resolve_path(self, value: str | Path) -> Path:
        """Преобразовать путь из конфига в абсолютный (относительно корня проекта)."""
        p = Path(value)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    @property
    def documents_path(self) -> Path:
        return self.resolve_path(self.get("documents.path", "data/documents"))

    @property
    def index_path(self) -> Path:
        return self.resolve_path(self.get("index.path", "data/index"))

    @property
    def embedding_model(self) -> str:
        return self.get("embedding.model", "nomic-ai/nomic-embed-text-v2-moe")

    @property
    def embedding_provider(self) -> str:
        return self.get("embedding.provider", "nomic")

    @property
    def embedding_batch_size(self) -> int:
        return int(self.get("embedding.batch_size", 16))

    @property
    def default_strategy(self) -> str:
        return self.get("chunking.default_strategy", "structural")

    @property
    def document_extensions(self) -> list:
        return list(self.get("documents.extensions", [".md", ".txt", ".py", ".js", ".json"]))
