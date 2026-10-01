"""Единый источник описаний команд для /help и парсера.

Все описания, usage-строки и допустимые значения флагов задаются здесь,
чтобы /help и разбор команд не расходились.
"""

from __future__ import annotations

from typing import Any, Dict, List

STRATEGY_CHOICES = ("fixed", "structural")
FORMAT_CHOICES = ("console", "json", "csv", "markdown")

# Структура команд. Используется и для /help, и для валидации в парсере.
COMMANDS: Dict[str, Dict[str, Any]] = {
    "/help": {
        "description": "Показать справку по всем командам или по конкретной команде",
        "usage": "/help [command]",
        "example": "/help /index build",
        "arguments": [
            {"name": "command", "required": False, "description": "команда, для которой показать справку"},
        ],
    },
    "/index": {
        "description": "Управление индексами документов",
        "usage": "/index <subcommand> [options]",
        "example": "/index build data/documents --strategy structural --name my_index",
        "subcommands": {
            "build": {
                "description": "Построить индекс из документов (загрузка -> чанкинг -> эмбеддинги -> FAISS+SQLite)",
                "usage": "/index build <path> [--strategy fixed|structural] [--name <index_name>] [--config <path>]",
                "example": "/index build data/documents --strategy fixed --name my_index",
                "arguments": [
                    {"name": "path", "required": False, "description": "путь к файлу/папке с документами (по умолчанию из конфига)"},
                ],
                "flags": [
                    {"name": "strategy", "choices": STRATEGY_CHOICES, "description": "стратегия чанкинга"},
                    {"name": "name", "description": "имя индекса (по умолчанию генерируется)"},
                    {"name": "config", "description": "путь к YAML-конфигу"},
                ],
            },
            "list": {
                "description": "Показать список доступных индексов",
                "usage": "/index list",
                "example": "/index list",
            },
            "info": {
                "description": "Показать метаданные индекса",
                "usage": "/index info <index_name>",
                "example": "/index info my_index",
                "arguments": [{"name": "index_name", "required": True, "description": "имя индекса"}],
            },
            "stats": {
                "description": "Вывести статистику по индексу",
                "usage": "/index stats <index_name> [--format console|json|csv|markdown]",
                "example": "/index stats my_index --format markdown",
                "arguments": [{"name": "index_name", "required": True, "description": "имя индекса"}],
                "flags": [{"name": "format", "choices": FORMAT_CHOICES, "description": "формат вывода"}],
            },
            "compare": {
                "description": "Сравнить статистику двух индексов (стратегий)",
                "usage": "/index compare <index1> <index2> [--format console|json|csv|markdown]",
                "example": "/index compare fixed_idx structural_idx --format markdown",
                "arguments": [
                    {"name": "index1", "required": True, "description": "первый индекс"},
                    {"name": "index2", "required": True, "description": "второй индекс"},
                ],
                "flags": [{"name": "format", "choices": FORMAT_CHOICES, "description": "формат вывода"}],
            },
            "show-chunk": {
                "description": "Показать полный текст и метаданные чанка",
                "usage": "/index show-chunk <index_name> <chunk_id>",
                "example": "/index show-chunk my_index my_index-structural-0000",
                "arguments": [
                    {"name": "index_name", "required": True, "description": "имя индекса"},
                    {"name": "chunk_id", "required": True, "description": "идентификатор чанка"},
                ],
            },
            "list-chunks": {
                "description": "Список чанков с фильтрами",
                "usage": "/index list-chunks <index_name> [--source <source>] [--section <section>] [--limit N]",
                "example": "/index list-chunks my_index --source intro.txt --limit 10",
                "arguments": [{"name": "index_name", "required": True, "description": "имя индекса"}],
                "flags": [
                    {"name": "source", "description": "фильтр по source"},
                    {"name": "section", "description": "фильтр по section"},
                    {"name": "limit", "description": "максимальное число чанков"},
                ],
            },
            "delete": {
                "description": "Удалить индекс",
                "usage": "/index delete <index_name>",
                "example": "/index delete my_index",
                "arguments": [{"name": "index_name", "required": True, "description": "имя индекса"}],
            },
        },
    },
    "/debug": {
        "description": "Включить/выключить отладочный режим (показывает эмбеддинги и стектрейсы ошибок)",
        "usage": "/debug on|off",
        "example": "/debug on",
        "arguments": [{"name": "mode", "required": True, "description": "on или off"}],
    },
    "/exit": {
        "description": "Завершить работу",
        "usage": "/exit",
        "example": "/exit",
    },
}


def list_index_subcommands() -> List[str]:
    return list(COMMANDS["/index"]["subcommands"].keys())


def _examples_block() -> List[str]:
    """Собрать список примеров для каждой команды (для общей справки)."""
    lines: List[str] = []
    entries: List[tuple] = [("/help", COMMANDS["/help"]), ("/index", COMMANDS["/index"])]
    for sub, spec in COMMANDS["/index"]["subcommands"].items():
        entries.append((f"/index {sub}", spec))
    entries.append(("/debug", COMMANDS["/debug"]))
    entries.append(("/exit", COMMANDS["/exit"]))
    for name, spec in entries:
        example = spec.get("example")
        if example:
            lines.append(f"  {name:<18} {example}")
    return lines


def build_help(command: str = None) -> str:
    """Сформировать текст справки из COMMANDS."""
    lines: List[str] = []

    if command is None:
        lines.append("Доступные команды:")
        lines.append("")
        for name in COMMANDS:
            spec = COMMANDS[name]
            lines.append(f"  {name:<20} {spec['description']}")
            lines.append(f"  {'':<20} {spec['usage']}")
        lines.append("")
        lines.append("Подкоманды /index:")
        for sub in COMMANDS["/index"]["subcommands"]:
            lines.append(f"  /index {sub:<14} {COMMANDS['/index']['subcommands'][sub]['description']}")
        lines.append("")
        lines.append("Примеры использования:")
        lines.extend(_examples_block())
        return "\n".join(lines)

    # Справка по конкретной команде.
    if command == "/index":
        lines.append(f"/index — {COMMANDS['/index']['description']}")
        lines.append(f"Использование: {COMMANDS['/index']['usage']}")
        lines.append("")
        lines.append("Подкоманды:")
        for sub, spec in COMMANDS["/index"]["subcommands"].items():
            lines.append(f"  {spec['usage']}")
            lines.append(f"      {spec['description']}")
        lines.append("")
        lines.append("Примеры:")
        for sub, spec in COMMANDS["/index"]["subcommands"].items():
            example = spec.get("example")
            if example:
                lines.append(f"  /index {sub:<12} {example}")
        return "\n".join(lines)

    if command.startswith("/index "):
        sub = command.split(" ", 1)[1].strip()
        subs = COMMANDS["/index"]["subcommands"]
        if sub in subs:
            spec = subs[sub]
            lines.append(f"/index {sub} — {spec['description']}")
            lines.append(f"Использование: {spec['usage']}")
            for arg in spec.get("arguments", []):
                req = "обязательный" if arg.get("required") else "опциональный"
                lines.append(f"  {arg['name']}: {arg.get('description', '')} ({req})")
            for flag in spec.get("flags", []):
                choices = f" [{', '.join(flag['choices'])}]" if "choices" in flag else ""
                lines.append(f"  --{flag['name']}{choices}: {flag.get('description', '')}")
            example = spec.get("example")
            if example:
                lines.append(f"Пример: {example}")
            return "\n".join(lines)
        return f"Неизвестная подкоманда /index: {sub!r}\n" + build_help("/index")

    if command in COMMANDS:
        spec = COMMANDS[command]
        lines.append(f"{command} — {spec['description']}")
        lines.append(f"Использование: {spec['usage']}")
        for arg in spec.get("arguments", []):
            req = "обязательный" if arg.get("required") else "опциональный"
            lines.append(f"  {arg['name']}: {arg.get('description', '')} ({req})")
        example = spec.get("example")
        if example:
            lines.append(f"Пример: {example}")
        return "\n".join(lines)

    return f"Неизвестная команда: {command!r}\n" + build_help()
