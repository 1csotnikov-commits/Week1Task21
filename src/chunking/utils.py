"""Общие вспомогательные функции для чанкинга."""

from __future__ import annotations

import re
from typing import List, Tuple

# Символы конца предложения (после которых можно резать).
_SENTENCE_END = re.compile(r"[.!?…]+")

# Форматы, которые считаются "кодом" (чанкуются по строкам, а не по предложениям).
_CODE_FORMATS = {
    "py", "js", "json", "ts", "java", "c", "cpp", "cs", "go", "rb",
    "php", "sh", "sql", "css", "html", "xml", "yaml", "yml",
}

_WHITESPACE = " \t\r\n"


def is_code_format(fmt: str) -> bool:
    return fmt.lower().lstrip(".") in _CODE_FORMATS


def trim_span(text: str, start: int, end: int) -> Tuple[str, int, int]:
    """Обрезать ведущие/хвостовые пробелы у среза, сохранив корректные смещения."""
    s, e = start, end
    while s < e and text[s] in _WHITESPACE:
        s += 1
    while e > s and text[e - 1] in _WHITESPACE:
        e -= 1
    return text[s:e], s, e


def line_spans(text: str, start: int = 0, end: int = None) -> List[Tuple[int, int]]:
    """Разбить текст на строки (смещения (start, end) с включением символа перевода)."""
    end = len(text) if end is None else end
    spans: List[Tuple[int, int]] = []
    s = start
    for i in range(start, end):
        if text[i] == "\n":
            if i > s:
                spans.append((s, i + 1))
            s = i + 1
    if s < end:
        spans.append((s, end))
    return spans


def sentence_spans(text: str, start: int = 0, end: int = None) -> List[Tuple[int, int]]:
    """Разбить текст на предложения по знакам . ! ? …

    Возвращает смещения (start, end) — конец включает пробелы/переводы строки
    после знака препинания, чтобы следующий чанк начинался без ведущих пробелов.
    """
    end = len(text) if end is None else end
    spans: List[Tuple[int, int]] = []
    s = start
    i = start
    n = end
    while i < n:
        m = _SENTENCE_END.search(text, i, end)
        if m is None:
            break
        stop = m.end()
        j = stop
        while j < n and text[j] in " \t":
            j += 1
        if j < n and text[j] in "\r\n":
            j += 1
        if j > s:
            spans.append((s, j))
        s = j
        i = j
    if s < n:
        spans.append((s, n))
    return spans


def atomic_units(text: str, start: int = 0, end: int = None, prefer_lines: bool = False) -> List[Tuple[int, int]]:
    """Разбить текст на атомарные единицы: строки (для кода) или предложения (для текста)."""
    if prefer_lines:
        return line_spans(text, start, end)
    spans = sentence_spans(text, start, end)
    if not spans:
        spans = line_spans(text, start, end)
    return spans


def _hard_split(text: str, start: int, end: int, chunk_size: int) -> List[Tuple[int, int]]:
    """Разбить слишком длинный фрагмент по границам слов (не рвать слова)."""
    seg = text[start:end]
    if len(seg) <= chunk_size:
        return [(start, end)] if seg else []

    result: List[Tuple[int, int]] = []
    pos = 0
    length = len(seg)
    while pos < length:
        if pos + chunk_size >= length:
            result.append((start + pos, end))
            break
        target = pos + chunk_size
        cut = seg.rfind(" ", pos, target)
        if cut <= pos:
            cut = seg.rfind("\n", pos, target)
        if cut <= pos:
            cut = target  # нет пробела — вынужденный разрез внутри слова

        sub_end = cut
        # Следующая позиция — после пробелов.
        nxt = cut
        while nxt < length and seg[nxt] in " \t\r\n":
            nxt += 1
        if nxt == cut:
            nxt = cut  # прогресс гарантирован, т.к. cut >= pos + 1 при cut==target>pos

        result.append((start + pos, start + sub_end))
        pos = nxt
        if pos <= 0:
            break
    return result


def fixed_size_spans(
    text: str,
    chunk_size: int,
    overlap: int,
    start: int = 0,
    end: int = None,
    prefer_lines: bool = False,
) -> List[Tuple[int, int]]:
    """Жадная разбивка на чанки с перекрытием.

    Единицы объединяются до chunk_size символов; перекрытие overlap реализуется
    на уровне атомарных единиц (предложений/строк).
    """
    end = len(text) if end is None else end
    if end - start <= chunk_size:
        return [(start, end)] if end > start else []

    units = atomic_units(text, start, end, prefer_lines=prefer_lines)
    if not units:
        return []

    spans: List[Tuple[int, int]] = []
    i = 0
    n = len(units)
    while i < n:
        j = i
        cur_len = 0
        while j < n:
            unit_len = units[j][1] - units[j][0]
            if j > i and cur_len + unit_len > chunk_size:
                break
            cur_len += unit_len
            j += 1

        chunk_start = units[i][0]
        chunk_end = units[j - 1][1]

        # Единственная единица длиннее chunk_size — дробим жёстко.
        if j - i == 1 and (units[i][1] - units[i][0]) > chunk_size:
            spans.extend(_hard_split(text, units[i][0], units[i][1], chunk_size))
            i = j
            continue

        spans.append((chunk_start, chunk_end))

        if j >= n:
            break

        # Перекрытие: возвращаемся назад, пока влезает в overlap символов.
        overlap_start = chunk_end - overlap
        k = j
        while k > i + 1 and units[k - 1][0] >= overlap_start:
            k -= 1
        i = max(i + 1, k)

    return spans
