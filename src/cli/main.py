"""Точка входа CLI (интерактивный REPL) пайплайна индексации."""

from __future__ import annotations

import shlex
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from src.cli.commands import (
    FORMAT_CHOICES,
    STRATEGY_CHOICES,
    build_help,
    list_index_subcommands,
)
from src.config import Config
from src.indexing.faiss_sqlite_store import FAISSSQLiteStore
from src.indexing.pipeline import build_index
from src.stats.chunk_stats import compute_stats, render, render_compare

BANNER = "Пайплайн индексации документов. Введите /help для списка команд, /exit для выхода."


def parse_args(tokens: List[str]) -> Tuple[List[str], dict]:
    """Разобрать аргументы: позиционные + флаги --flag value / --flag=value."""
    positional: List[str] = []
    flags: dict = {}
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.startswith("--"):
            key = token[2:]
            if "=" in key:
                k, v = key.split("=", 1)
                flags[k] = v
            elif i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                flags[key] = tokens[i + 1]
                i += 1
            else:
                flags[key] = True
        else:
            positional.append(token)
        i += 1
    return positional, flags


class App:
    """Состояние CLI-приложения."""

    def __init__(self, config: Config):
        self.config = config
        self.debug = False
        self.store = FAISSSQLiteStore(config.index_path)


def _require_positional(positional: List[str], n: int, usage: str) -> None:
    if len(positional) < n:
        raise ValueError(f"Недостаточно аргументов. Использование: {usage}")


def _load_stats(app: App, index_name: str) -> dict:
    info = app.store.info(index_name)
    app.store.load(index_name)
    chunks = app.store.list_chunks()
    documents = app.store.get_documents()
    strategy = info.get("strategy") or app.store._strategy or ""
    elapsed = float(info.get("indexing_time_seconds") or 0.0)
    return compute_stats(documents, chunks, elapsed, strategy, index_name)


# ---------------------------------------------------------------------- #
# Обработчики команд
# ---------------------------------------------------------------------- #
def _cmd_help(args: List[str]) -> str:
    if args:
        return build_help(args[0])
    return build_help()


def _cmd_debug(app: App, args: List[str]) -> str:
    _require_positional(args, 1, "/debug on|off")
    mode = args[0].lower()
    if mode == "on":
        app.debug = True
    elif mode == "off":
        app.debug = False
    else:
        raise ValueError(f"Режим отладки: on или off, получено {mode!r}")
    return f"Отладка: {'включена' if app.debug else 'выключена'}"


def _cmd_index(app: App, args: List[str]) -> str:
    if not args:
        return build_help("/index")
    sub = args[0]
    rest = args[1:]

    if sub == "build":
        return _cmd_build(app, rest)
    if sub == "list":
        return _cmd_list(app)
    if sub == "info":
        _require_positional(rest, 1, "/index info <index_name>")
        return _cmd_info(app, rest[0])
    if sub == "stats":
        _require_positional(rest, 1, "/index stats <index_name> [--format ...]")
        pos, flags = parse_args(rest)
        return _cmd_stats(app, pos[0], flags)
    if sub == "compare":
        _require_positional(rest, 2, "/index compare <index1> <index2> [--format ...]")
        pos, flags = parse_args(rest)
        return _cmd_compare(app, pos[0], pos[1], flags)
    if sub == "show-chunk":
        _require_positional(rest, 2, "/index show-chunk <index_name> <chunk_id>")
        return _cmd_show_chunk(app, rest[0], rest[1])
    if sub == "list-chunks":
        _require_positional(rest, 1, "/index list-chunks <index_name> [--source ...] [--section ...] [--limit N]")
        pos, flags = parse_args(rest)
        return _cmd_list_chunks(app, pos[0], flags)
    if sub == "delete":
        _require_positional(rest, 1, "/index delete <index_name>")
        return _cmd_delete(app, rest[0])

    raise ValueError(f"Неизвестная подкоманда /index: {sub!r}. Доступно: {', '.join(list_index_subcommands())}")


def _cmd_build(app: App, args: List[str]) -> str:
    pos, flags = parse_args(args)
    config = app.config
    if flags.get("config"):
        config = Config.load(flags["config"])

    documents_path = Path(pos[0]) if pos else config.documents_path
    strategy = flags.get("strategy") or config.default_strategy
    if strategy not in STRATEGY_CHOICES:
        raise ValueError(f"Неизвестная стратегия: {strategy!r}. Доступно: {', '.join(STRATEGY_CHOICES)}")

    index_name = flags.get("name") or f"{strategy}-{datetime.now():%Y%m%d-%H%M%S}"

    result = build_index(documents_path, strategy, index_name, config)
    stats = compute_stats(result.documents, result.chunks, result.elapsed, strategy, index_name)
    return render(stats, "console")


def _cmd_list(app: App) -> str:
    indices = app.store.list_indices()
    if not indices:
        return "Индексов нет. Постройте индекс: /index build <path>"
    lines = ["Доступные индексы:"]
    for info in indices:
        lines.append(
            f"  {info['name']:<25} strategy={info['strategy']:<10} "
            f"docs={info['num_documents']} chunks={info['num_chunks']} store={info['store_type']}"
        )
    return "\n".join(lines)


def _cmd_info(app: App, index_name: str) -> str:
    info = app.store.info(index_name)
    lines = [f"Индекс: {info['name']}"]
    for key in ("strategy", "model", "provider", "dim", "store_type", "num_documents", "num_chunks", "indexing_time_seconds", "created_at", "path"):
        lines.append(f"  {key}: {info.get(key, '')}")
    return "\n".join(lines)


def _cmd_stats(app: App, index_name: str, flags: dict) -> str:
    fmt = flags.get("format") or "console"
    if fmt not in FORMAT_CHOICES:
        raise ValueError(f"Неизвестный формат: {fmt!r}. Доступно: {', '.join(FORMAT_CHOICES)}")

    stats = _load_stats(app, index_name)
    output = render(stats, fmt)
    if fmt in ("json", "csv", "markdown"):
        ext = {"json": "json", "csv": "csv", "markdown": "md"}[fmt]
        out_path = app.store._index_dir(index_name) / f"stats.{ext}"
        out_path.write_text(output, encoding="utf-8")
        return output + f"\n\nОтчёт сохранён: {out_path}"
    return output


def _cmd_compare(app: App, a: str, b: str, flags: dict) -> str:
    fmt = flags.get("format") or "console"
    if fmt not in FORMAT_CHOICES:
        raise ValueError(f"Неизвестный формат: {fmt!r}. Доступно: {', '.join(FORMAT_CHOICES)}")

    stats_a = _load_stats(app, a)
    stats_b = _load_stats(app, b)
    output = render_compare(stats_a, stats_b, fmt)
    if fmt in ("json", "csv", "markdown"):
        ext = {"json": "json", "csv": "csv", "markdown": "md"}[fmt]
        out_path = app.config.index_path / f"compare_{a}_vs_{b}.{ext}"
        out_path.write_text(output, encoding="utf-8")
        return output + f"\n\nОтчёт сохранён: {out_path}"
    return output


def _cmd_show_chunk(app: App, index_name: str, chunk_id: str) -> str:
    app.store.load(index_name)
    chunk = app.store.get_chunk(chunk_id)
    if chunk is None:
        raise ValueError(f"Чанк '{chunk_id}' не найден в индексе '{index_name}'")

    lines = ["--- Текст чанка ---", chunk.text, "", "--- Метаданные ---"]
    lines.append(f"chunk_id: {chunk.chunk_id}")
    lines.append(f"source: {chunk.source}")
    lines.append(f"title: {chunk.title}")
    lines.append(f"section: {chunk.section}")
    lines.append(f"strategy: {chunk.strategy}")
    lines.append(f"char_count: {chunk.char_count}")
    lines.append(f"token_count (эвристика): {chunk.token_count}")
    lines.append(f"start_char: {chunk.start_char}")
    lines.append(f"end_char: {chunk.end_char}")
    if chunk.metadata:
        lines.append(f"metadata: {chunk.metadata}")

    if app.debug:
        emb = app.store.get_embedding(chunk_id)
        if emb is not None:
            n = 10
            lines.append("")
            lines.append(f"Эмбеддинг (первые {n} значений): {[round(float(x), 4) for x in emb[:n]]}")
    return "\n".join(lines)


def _cmd_list_chunks(app: App, index_name: str, flags: dict) -> str:
    app.store.load(index_name)
    source = flags.get("source")
    section = flags.get("section")
    limit = None
    if flags.get("limit"):
        limit = int(flags["limit"])

    chunks = app.store.list_chunks(source=source, section=section, limit=limit)
    if not chunks:
        return "Чанков не найдено."
    lines = [f"Чанков: {len(chunks)}"]
    for c in chunks:
        lines.append(
            f"  {c.chunk_id:<40} source={c.source:<20} section={c.section or '-':<20} chars={c.char_count}"
        )
    return "\n".join(lines)


def _cmd_delete(app: App, index_name: str) -> str:
    app.store.delete(index_name)
    return f"Индекс '{index_name}' удалён"


# ---------------------------------------------------------------------- #
# Диспетчер и точка входа
# ---------------------------------------------------------------------- #
def handle_command(line: str, app: App) -> Tuple[bool, str]:
    """Разобрать и выполнить одну команду. Возвращает (выход, вывод)."""
    tokens = shlex.split(line)
    if not tokens:
        return False, ""

    cmd = tokens[0]
    args = tokens[1:]

    if cmd == "/help":
        return False, _cmd_help(args)
    if cmd == "/exit":
        return True, "До свидания!"
    if cmd == "/debug":
        return False, _cmd_debug(app, args)
    if cmd == "/index":
        return False, _cmd_index(app, args)

    return False, f"Неизвестная команда: {cmd!r}. Введите /help для списка команд."


def run(app: App) -> None:
    """Интерактивный цикл REPL."""
    print(BANNER)
    while True:
        try:
            raw = input("> ")
        except (EOFError, KeyboardInterrupt):
            print("\n/exit")
            break

        line = raw.strip()
        if not line:
            continue
        if not line.startswith("/"):
            print("Команды начинаются с '/'. Введите /help для списка команд.")
            continue

        try:
            should_exit, output = handle_command(line, app)
            if output:
                print(output)
            if should_exit:
                break
        except Exception as exc:  # noqa: BLE001
            print(f"Ошибка: {exc}")
            if app.debug:
                traceback.print_exc()


def main() -> None:
    config = Config.load()
    app = App(config)
    run(app)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
