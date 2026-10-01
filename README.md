# Week1Task21 — Локальный пайплайн индексации документов

Локальный пайплайн индексации русскоязычных документов:

1. Загрузка текстовых документов (`.md`, `.txt`, `.py`, `.js`, `.json` и т.п.);
2. Чанкинг двумя стратегиями — `fixed_size` и `structural`;
3. Генерация эмбеддингов локальной моделью `nomic-ai/nomic-embed-text-v2-moe`;
4. Сохранение индекса в **FAISS + SQLite**;
5. Вывод статистики для сравнения стратегий чанкинга.

Интерфейс — только CLI (интерактивный REPL), без web. PDF не поддерживаются.

## Стек

- ОС: Windows Server 2022 (совместимо с Linux/macOS)
- Python 3.10+
- `sentence-transformers` + `einops` (эмбеддинги)
- `faiss-cpu` (векторный поиск, с fallback на `numpy`)
- SQLite (метаданные и текст чанков)

## Архитектура

```
C:\II\Week 1 Task 21\
├── config/
│   └── config.yaml          # конфигурация (эмбеддинги, чанкинг, пути)
├── data/
│   ├── documents/           # пользователь кладёт сюда документы
│   └── index/               # сгенерированные индексы (faiss.index + metadata.db)
├── src/
│   ├── ingestion/           # DocumentLoader: text_loader (.md/.txt/.py/.js/.json)
│   ├── chunking/            # BaseChunker, fixed_size, structural
│   ├── embeddings/          # EmbeddingProvider, nomic_provider
│   ├── indexing/            # IndexStore, faiss_sqlite_store, pipeline
│   ├── stats/               # chunk_stats (метрики + форматы console/json/csv/markdown)
│   ├── retrieval/           # заглушка (задел на День 22)
│   ├── reranking/           # заглушка (задел на День 23)
│   ├── rag/                 # заглушка (задел на День 24)
│   └── cli/                 # main.py (REPL), commands.py (единый источник описаний)
├── tests/                   # unittest-тесты
├── requirements.txt
└── README.md
```

## Установка

```bash
pip install -r requirements.txt
```

или вручную (основные зависимости):

```bash
pip install "sentence-transformers>=3.0,<4.0" "transformers>=4.40,<5.0" einops
```

> **Важно:** при первом запуске модель `nomic-ai/nomic-embed-text-v2-moe`
> скачается автоматически (**~1–2 ГБ**). Убедитесь, что есть доступ в интернет.

> **Совместимость версий:** не ставьте `sentence-transformers 6.x` / `transformers 5.x` —
> кастомный код модели nomic (`modeling_hf_nomic_bert.py`) несовместим с transformers 5.x
> (отсутствует метод `get_extended_attention_mask`). Используйте версии из `requirements.txt`.

### Windows Server 2022: возможные проблемы

- **`faiss-cpu` не устанавливается** — это не критично: при ошибке импорта
  автоматически включается fallback на `numpy` + перебор (для 20–30 страниц
  скорость приемлема). В консоли появится предупреждение.
- **`OSError: [WinError 1114] ... c10.dll`** при импорте `torch` — обычно
  означает отсутствие Microsoft Visual C++ Redistributable. Установите
  актуальный **VC++ Redistributable (x64)**, например:
  `https://aka.ms/vs/17/release/vc_redist.x64.exe` (тихая установка:
  `vc_redist.x64.exe /install /quiet /norestart`).

## Модель эмбеддингов

Используется только локальная модель:

```
nomic-ai/nomic-embed-text-v2-moe
```

- 768 измерений;
- поддержка русского языка;
- лицензия Apache 2.0;
- нормализованные векторы (L2-норма = 1), поэтому `IndexFlatIP`
  (inner product) эквивалентен косинусной близости.

## Запуск CLI

```bash
python -m src.cli.main
```

Откроется интерактивная консоль. Список команд — `/help`.

### Команды

| Команда | Описание |
| --- | --- |
| `/help [command]` | Справка по командам (описания — из единого источника `commands.py`) |
| `/index build <path> [--strategy fixed\|structural] [--name <name>] [--config <path>]` | Построить индекс |
| `/index list` | Список индексов |
| `/index info <name>` | Метаданные индекса |
| `/index stats <name> [--format console\|json\|csv\|markdown]` | Статистика по индексу |
| `/index compare <a> <b> [--format ...]` | Сравнить два индекса |
| `/index show-chunk <name> <chunk_id>` | Полный текст + метаданные чанка |
| `/index list-chunks <name> [--source ...] [--section ...] [--limit N]` | Список чанков с фильтрами |
| `/index delete <name>` | Удалить индекс |
| `/debug on\|off` | Отладочный режим (показывает эмбеддинги и стектрейсы) |
| `/exit` | Выход |

### Пример использования

```bash
python -m src.cli.main
> /index build data/documents --strategy fixed --name fixed_idx
> /index build data/documents --strategy structural --name structural_idx
> /index list
> /index stats fixed_idx
> /index stats fixed_idx --format markdown
> /index compare fixed_idx structural_idx --format console
> /index show-chunk fixed_idx intro-fixed_size-0000
> /index list-chunks structural_idx --section "Введение" --limit 5
> /index delete structural_idx
> /exit
```

## Стратегии чанкинга

### A — `fixed_size`

Параметры: `chunk_size` (по умолчанию 800 символов), `overlap` (150).
Рез — по границам предложений/строк (не рвёт слова). Для кода — по строкам.

### B — `structural`

- **Markdown/README** — по заголовкам `#`, `##`, `###`; крупные разделы дробятся `fixed_size`;
- **Python** — по функциям/классам через AST;
- **Прочий код** — эвристика по отступам / `def` / `class` / blank-строкам;
- **Текст** — по абзацам с эвристикой заголовков (пустые строки, ALL CAPS).

Метаданные каждого чанка: `source`, `title`, `section`, `chunk_id`, `strategy`,
`char_count`, `token_count` (эвристика `len(text) // 4`).

## Статистика для сравнения стратегий

Метрики:

- количество документов, чанков, время индексации;
- размеры чанков: min / max / avg / median + гистограмма (ASCII в консоли);
- метаданные: чанков с `section`, с `title`, число уникальных `source`;
- перекрытие (`fixed_size`): перекрывающихся чанков, средний overlap;
- покрытие: суммарная длина чанков vs длина исходных текстов;
- семантическая целостность: доля чанков, начинающихся/заканчивающихся в середине
  предложения, доля с незакрытыми скобками/кавычками; для `structural` — доля
  чанков с ровно одним разделом vs с несколькими разделами.

### Пример (фрагмент, console)

```
===== Статистика индекса: fixed_idx =====
Стратегия: fixed
Документов: 5
Чанков: 23
Время индексации: 12.34 сек.

--- Размеры чанков (символы) ---
min: 120
max: 812
avg: 640.5
median: 655.0
Гистограмма распределения:
  [   120-    189] #####                                    5
  [   189-    258] ##########                              10
  ...
```

### JSON / CSV / Markdown

```bash
> /index stats fixed_idx --format json
> /index stats fixed_idx --format csv
> /index stats fixed_idx --format markdown
```

Файлы отчётов сохраняются рядом с индексом:
`data/index/<index_name>/stats.json|csv|md`. Сравнение — в
`data/index/compare_<a>_vs_<b>.<ext>`.

## Тесты

```bash
python -m unittest discover -s tests -t .
```

Покрытие: загрузка документов, `fixed_size`, `structural`, эмбеддинги
(размерность 768 и нормализация; пропускается, если модель недоступна),
сохранение/загрузка FAISS + SQLite, статистика, CLI-команды.

## Задел на Дни 22–24

- `src/retrieval` — `search(query, top_k) -> List[ChunkWithScore]`;
- `src/reranking` — `rerank(chunks, query) -> List[ChunkWithScore]`;
- `src/rag` — `build_prompt(question, chunks) -> str`, `extract_citations(answer, chunks) -> List[Citation]`.

Модули — заглушки с интерфейсами, которые заполняются логикой на следующих этапах.
