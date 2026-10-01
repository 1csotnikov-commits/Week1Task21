"""Пример модуля для структурного чанкинга Python."""

import json
from pathlib import Path


def load_config(path):
    """Загрузить конфигурацию из JSON-файла."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_config(data, path):
    """Сохранить конфигурацию в JSON-файл."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


class DocumentStore:
    """Простое хранилище документов в памяти."""

    def __init__(self):
        self._items = {}

    def add(self, key, value):
        self._items[key] = value

    def get(self, key, default=None):
        return self._items.get(key, default)

    def all(self):
        return list(self._items.values())


def main():
    store = DocumentStore()
    store.add("example", {"title": "Пример", "text": "Текст документа"})
    print(store.get("example"))


if __name__ == "__main__":
    main()
