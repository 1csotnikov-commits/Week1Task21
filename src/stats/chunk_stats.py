"""Статистика для сравнения стратегий чанкинга."""

from __future__ import annotations

import csv
import io
import json
import statistics
from typing import Any, Dict, List, Optional

from src.models import Chunk, Document

# Символы, которыми предложение может считаться "закрытым".
_END_CHARS = set(".!?:;") | set(")]}\u00bb\"'`*_")

_OPEN_TO_CLOSE = {"(": ")", "[": "]", "{": "}", "«": "»"}


def _starts_mid_sentence(text: str) -> bool:
    """Чанк начинается с середины предложения (первый символ — строчная буква)."""
    t = text.lstrip()
    return bool(t) and t[0].islower()


def _ends_mid_sentence(text: str) -> bool:
    """Чанк заканчивается на середине предложения."""
    t = text.rstrip()
    if not t:
        return False
    return t[-1] not in _END_CHARS


def _has_unclosed(text: str) -> bool:
    """Есть ли незакрытые скобки или нечётное число кавычек."""
    stack: List[str] = []
    for ch in text:
        if ch in _OPEN_TO_CLOSE:
            stack.append(_OPEN_TO_CLOSE[ch])
        elif ch in ")]}\u00bb":
            if stack and stack[-1] == ch:
                stack.pop()
    if stack:
        return True
    return text.count('"') % 2 != 0 or text.count("'") % 2 != 0


def _histogram(lengths: List[int], bins: int = 10) -> Dict[str, Any]:
    if not lengths:
        return {"bin_edges": [], "counts": []}
    mn = min(lengths)
    mx = max(lengths)
    if mx <= mn:
        mx = mn + 1
    edges = [mn + (mx - mn) * i / bins for i in range(bins + 1)]
    counts = [0] * bins
    span = mx - mn
    for length in lengths:
        idx = min(bins - 1, int((length - mn) / span * bins))
        counts[idx] += 1
    return {
        "bin_edges": [round(e, 1) for e in edges],
        "counts": counts,
    }


def _overlap_stats(chunks: List[Chunk]) -> Dict[str, Any]:
    by_source: Dict[str, List[Chunk]] = {}
    for c in chunks:
        by_source.setdefault(c.source, []).append(c)

    overlapping = 0
    total_overlap = 0
    for cs in by_source.values():
        ordered = sorted(cs, key=lambda c: c.start_char)
        for i in range(1, len(ordered)):
            ov = ordered[i - 1].end_char - ordered[i].start_char
            if ov > 0:
                overlapping += 1
                total_overlap += ov
    avg = total_overlap / overlapping if overlapping else 0.0
    return {"overlapping_chunks": overlapping, "avg_overlap": round(avg, 2)}


def compute_stats(
    documents: List[Document],
    chunks: List[Chunk],
    elapsed: float,
    strategy: str,
    index_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Рассчитать полный набор метрик по чанкам."""
    num_chunks = len(chunks)
    lengths = [c.char_count for c in chunks]

    total_source_chars = sum(d.char_count for d in documents)
    total_chunk_chars = sum(lengths)
    coverage_ratio = (total_chunk_chars / total_source_chars) if total_source_chars else 0.0

    semantic: Dict[str, Any] = {
        "starts_mid_sentence": sum(1 for c in chunks if _starts_mid_sentence(c.text)),
        "ends_mid_sentence": sum(1 for c in chunks if _ends_mid_sentence(c.text)),
        "unclosed_brackets": sum(1 for c in chunks if _has_unclosed(c.text)),
        "starts_mid_sentence_ratio": round(sum(1 for c in chunks if _starts_mid_sentence(c.text)) / num_chunks, 4) if num_chunks else 0.0,
        "ends_mid_sentence_ratio": round(sum(1 for c in chunks if _ends_mid_sentence(c.text)) / num_chunks, 4) if num_chunks else 0.0,
        "unclosed_brackets_ratio": round(sum(1 for c in chunks if _has_unclosed(c.text)) / num_chunks, 4) if num_chunks else 0.0,
    }

    if strategy == "structural":
        single = sum(1 for c in chunks if int(c.metadata.get("section_count", 0)) <= 1)
        multi = num_chunks - single
        semantic["single_section"] = single
        semantic["multi_section"] = multi
        semantic["single_section_ratio"] = round(single / num_chunks, 4) if num_chunks else 0.0
        semantic["multi_section_ratio"] = round(multi / num_chunks, 4) if num_chunks else 0.0

    result: Dict[str, Any] = {
        "index_name": index_name,
        "strategy": strategy,
        "num_documents": len(documents),
        "num_chunks": num_chunks,
        "indexing_time_seconds": round(elapsed, 4),
        "chunk_length": {
            "min": min(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
            "avg": round(statistics.mean(lengths), 2) if lengths else 0.0,
            "median": round(statistics.median(lengths), 2) if lengths else 0.0,
            "histogram": _histogram(lengths),
        },
        "metadata": {
            "with_section": sum(1 for c in chunks if c.section),
            "with_title": sum(1 for c in chunks if c.title),
            "unique_sources": len({c.source for c in chunks}),
        },
        "coverage": {
            "total_chunk_chars": total_chunk_chars,
            "total_source_chars": total_source_chars,
            "coverage_ratio": round(coverage_ratio, 4),
            "duplication_ratio": round(coverage_ratio - 1.0, 4),
        },
        "semantic_integrity": semantic,
    }

    if strategy == "fixed":
        result["overlap"] = _overlap_stats(chunks)
    else:
        result["overlap"] = None

    return result


# ---------------------------------------------------------------------- #
# Форматирование
# ---------------------------------------------------------------------- #
def _ascii_histogram(hist: Dict[str, Any], width: int = 40) -> str:
    edges = hist.get("bin_edges", [])
    counts = hist.get("counts", [])
    if not edges or not counts:
        return "  (нет данных)"
    max_c = max(counts) or 1
    lines = []
    for i, count in enumerate(counts):
        lo, hi = edges[i], edges[i + 1]
        bar = "#" * max(1, round(count / max_c * width)) if count else ""
        lines.append(f"  [{lo:>7.0f}-{hi:>7.0f}] {bar:<{width}} {count}")
    return "\n".join(lines)


def _flatten(data: Dict[str, Any], prefix: str = "") -> List[tuple]:
    rows: List[tuple] = []
    for key, value in data.items():
        full = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            rows.extend(_flatten(value, full))
        elif isinstance(value, list):
            rows.append((full, json.dumps(value, ensure_ascii=False)))
        else:
            rows.append((full, value))
    return rows


def _render_console(stats: Dict[str, Any]) -> str:
    cl = stats["chunk_length"]
    md = stats["metadata"]
    cov = stats["coverage"]
    sem = stats["semantic_integrity"]

    lines: List[str] = []
    lines.append(f"===== Статистика индекса: {stats.get('index_name') or '-'} =====")
    lines.append(f"Стратегия: {stats['strategy']}")
    lines.append(f"Документов: {stats['num_documents']}")
    lines.append(f"Чанков: {stats['num_chunks']}")
    lines.append(f"Время индексации: {stats['indexing_time_seconds']} сек.")
    lines.append("")
    lines.append("--- Размеры чанков (символы) ---")
    lines.append(f"min: {cl['min']}")
    lines.append(f"max: {cl['max']}")
    lines.append(f"avg: {cl['avg']}")
    lines.append(f"median: {cl['median']}")
    lines.append("Гистограмма распределения:")
    lines.append(_ascii_histogram(cl["histogram"]))
    lines.append("")
    lines.append("--- Метаданные ---")
    lines.append(f"Чанков с section: {md['with_section']}")
    lines.append(f"Чанков с title: {md['with_title']}")
    lines.append(f"Уникальных source: {md['unique_sources']}")
    if stats.get("overlap"):
        lines.append("")
        lines.append("--- Перекрытие (fixed_size) ---")
        lines.append(f"Перекрывающихся чанков: {stats['overlap']['overlapping_chunks']}")
        lines.append(f"Средний overlap: {stats['overlap']['avg_overlap']}")
    lines.append("")
    lines.append("--- Покрытие ---")
    lines.append(f"Суммарная длина чанков: {cov['total_chunk_chars']}")
    lines.append(f"Длина исходных текстов: {cov['total_source_chars']}")
    lines.append(f"Покрытие (>= 1 из-за overlap): {cov['coverage_ratio']}")
    lines.append(f"Дублирование сверх overlap: {cov['duplication_ratio']}")
    lines.append("")
    lines.append("--- Семантическая целостность ---")
    lines.append(f"Начинаются с середины предложения: {sem['starts_mid_sentence_ratio']:.2%} ({sem['starts_mid_sentence']})")
    lines.append(f"Заканчиваются на середине предложения: {sem['ends_mid_sentence_ratio']:.2%} ({sem['ends_mid_sentence']})")
    lines.append(f"Незакрытые скобки/кавычки: {sem['unclosed_brackets_ratio']:.2%} ({sem['unclosed_brackets']})")
    if "single_section_ratio" in sem:
        lines.append(f"Ровно один раздел (section): {sem['single_section_ratio']:.2%} ({sem['single_section']})")
        lines.append(f"Несколько разделов: {sem['multi_section_ratio']:.2%} ({sem['multi_section']})")
    return "\n".join(lines)


def _render_markdown(stats: Dict[str, Any]) -> str:
    out = ["## Статистика индекса `" + str(stats.get("index_name") or "-") + "`", ""]
    out.append("| Метрика | Значение |")
    out.append("| --- | --- |")
    out.extend(f"| {k} | {v} |" for k, v in _flatten(stats))
    out.append("")
    out.append("### Гистограмма распределения длин чанков")
    out.append("```")
    out.append(_ascii_histogram(stats["chunk_length"]["histogram"]))
    out.append("```")
    return "\n".join(out)


def render(stats: Dict[str, Any], fmt: str = "console") -> str:
    """Отрендерить статистику одного индекса в заданном формате."""
    fmt = (fmt or "console").lower()
    if fmt == "console":
        return _render_console(stats)
    if fmt == "json":
        return json.dumps(stats, ensure_ascii=False, indent=2)
    if fmt == "markdown":
        return _render_markdown(stats)
    if fmt == "csv":
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["metric", "value"])
        for key, value in _flatten(stats):
            writer.writerow([key, value])
        return buf.getvalue().rstrip("\r\n")
    raise ValueError(f"Неизвестный формат: {fmt!r}. Доступно: console, json, csv, markdown")


def render_compare(a: Dict[str, Any], b: Dict[str, Any], fmt: str = "console") -> str:
    """Сравнить статистику двух индексов."""
    fmt = (fmt or "console").lower()
    if fmt == "json":
        return json.dumps({"index1": a, "index2": b}, ensure_ascii=False, indent=2)

    flat_a = dict(_flatten(a))
    flat_b = dict(_flatten(b))
    keys = list(dict.fromkeys(list(flat_a.keys()) + list(flat_b.keys())))
    name_a = a.get("index_name") or "index1"
    name_b = b.get("index_name") or "index2"

    if fmt == "console":
        lines = [f"===== Сравнение: {name_a} vs {name_b} =====", ""]
        lines.append(f"{'Метрика':<45} {name_a:>20} {name_b:>20}")
        lines.append("-" * 90)
        for key in keys:
            lines.append(f"{key:<45} {str(flat_a.get(key, '-')):>20} {str(flat_b.get(key, '-')):>20}")
        return "\n".join(lines)

    if fmt == "markdown":
        lines = [f"## Сравнение индексов `{name_a}` и `{name_b}`", ""]
        lines.append(f"| Метрика | {name_a} | {name_b} |")
        lines.append("| --- | --- | --- |")
        for key in keys:
            lines.append(f"| {key} | {flat_a.get(key, '-')} | {flat_b.get(key, '-')} |")
        return "\n".join(lines)

    if fmt == "csv":
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["metric", name_a, name_b])
        for key in keys:
            writer.writerow([key, flat_a.get(key, ""), flat_b.get(key, "")])
        return buf.getvalue().rstrip("\r\n")

    raise ValueError(f"Неизвестный формат: {fmt!r}. Доступно: console, json, csv, markdown")
