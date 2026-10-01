"""Стратегия чанкинга B — structural (структурный чанкинг)."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import List, Tuple

from src.chunking.base import BaseChunker
from src.chunking.utils import fixed_size_spans, is_code_format, trim_span
from src.models import Chunk, Document

_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.*)$", re.MULTILINE)
_BLANK_SEP = re.compile(r"(?:\r?\n)[ \t]*(?:\r?\n)+")
_CODE_DEF_RE = re.compile(r"^(async\s+)?(function|class|def|const|let|var)\b", re.IGNORECASE)


@dataclass
class _Section:
    """Логический раздел документа: заголовок + диапазон символов."""

    title: str
    start: int
    end: int


class StructuralChunker(BaseChunker):
    """Структурный чанкинг по формату документа.

    - Markdown/README: по заголовкам (#, ##, ###).
    - Python: по функциям/классам через AST.
    - Прочий код: эвристика по отступам/blank-строкам/def/class.
    - Текст: по абзацам с эвристикой заголовков (пустые строки, ALL CAPS).
    """

    strategy = "structural"

    def __init__(self, max_chunk_size: int = 1500, split_by: str = "heading"):
        if max_chunk_size <= 0:
            raise ValueError("max_chunk_size должен быть положительным числом")
        self.max_chunk_size = max_chunk_size
        self.split_by = split_by

    # ------------------------------------------------------------------ #
    # Публичный интерфейс
    # ------------------------------------------------------------------ #
    def chunk(self, document: Document) -> List[Chunk]:
        sections = self._extract_sections(document.text, document.format)
        chunks: List[Chunk] = []
        idx = 0
        i = 0
        n = len(sections)

        while i < n:
            section = sections[i]
            section_len = section.end - section.start

            # Слишком большой раздел дробим дальше fixed_size (без перекрытия).
            if section_len > self.max_chunk_size:
                for text, s, e in self._split_large(document.text, section, document.format):
                    chunks.append(self._make_chunk(document, text, s, e, section.title, 1, idx))
                    idx += 1
                i += 1
                continue

            # Объединяем соседние небольшие разделы до max_chunk_size.
            j = i
            total = 0
            titles: List[str] = []
            while j < n and (total + (sections[j].end - sections[j].start) <= self.max_chunk_size or j == i):
                total += sections[j].end - sections[j].start
                titles.append(sections[j].title)
                j += 1

            start = sections[i].start
            end = sections[j - 1].end
            text, s, e = trim_span(document.text, start, end)
            non_empty_titles = [t for t in titles if t]
            section_title = non_empty_titles[0] if non_empty_titles else ""
            section_count = len(non_empty_titles)

            if text:
                chunks.append(self._make_chunk(document, text, s, e, section_title, section_count, idx))
                idx += 1
            i = j

        return chunks

    # ------------------------------------------------------------------ #
    # Вспомогательные конструкторы
    # ------------------------------------------------------------------ #
    def _make_chunk(
        self,
        document: Document,
        text: str,
        s: int,
        e: int,
        section: str,
        section_count: int,
        idx: int,
    ) -> Chunk:
        return Chunk(
            chunk_id=f"{document.title}-structural-{idx:04d}",
            source=document.source,
            title=document.title,
            text=text,
            start_char=s,
            end_char=e,
            strategy=self.strategy,
            section=section,
            metadata={
                "format": document.format,
                "section_count": section_count,
                "section": section,
            },
        )

    def _split_large(self, text: str, section: _Section, fmt: str):
        sub = text[section.start:section.end]
        spans = fixed_size_spans(
            sub,
            self.max_chunk_size,
            0,
            prefer_lines=is_code_format(fmt),
        )
        for s, e in spans:
            t, s, e = trim_span(sub, s, e)
            if t:
                yield t, section.start + s, section.start + e

    # ------------------------------------------------------------------ #
    # Извлечение разделов
    # ------------------------------------------------------------------ #
    def _extract_sections(self, text: str, fmt: str) -> List[_Section]:
        fmt = fmt.lower().lstrip(".")
        if fmt in ("md", "markdown"):
            return self._markdown_sections(text)
        if fmt == "py":
            return self._python_sections(text)
        if is_code_format(fmt):
            return self._code_sections(text)
        return self._text_sections(text)

    def _markdown_sections(self, text: str) -> List[_Section]:
        matches = list(_HEADING_RE.finditer(text))
        if not matches:
            return [self._sec("", 0, len(text))]

        sections: List[_Section] = []
        if matches[0].start() > 0:
            sections.append(self._sec("", 0, matches[0].start()))
        for i, m in enumerate(matches):
            title = m.group(2).strip()
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            sections.append(self._sec(title, start, end))
        return [s for s in sections if s.end > s.start]

    def _python_sections(self, text: str) -> List[_Section]:
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return self._code_sections(text)

        line_starts = self._line_start_offsets(text)
        blocks: List[Tuple[str, int, int]] = []  # (title, start, end)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start_line = node.lineno
                deco_lines = [d.lineno for d in getattr(node, "decorator_list", []) if d.lineno]
                if deco_lines:
                    start_line = min(start_line, min(deco_lines))
                end_line = getattr(node, "end_lineno", None) or node.lineno
                start = line_starts[start_line - 1] if start_line - 1 < len(line_starts) else len(text)
                end = line_starts[end_line] if end_line < len(line_starts) else len(text)
                kind = "class" if isinstance(node, ast.ClassDef) else "def"
                blocks.append((f"{kind} {node.name}", start, end))

        blocks.sort(key=lambda b: b[1])
        sections: List[_Section] = []
        pos = 0
        for title, s, e in blocks:
            if s > pos:
                sections.append(self._sec("", pos, s))
            sections.append(self._sec(title, s, e))
            pos = e
        if pos < len(text):
            sections.append(self._sec("", pos, len(text)))
        return [s for s in sections if s.end > s.start]

    def _code_sections(self, text: str) -> List[_Section]:
        return self._paragraph_sections(text, detect_code_defs=True)

    def _text_sections(self, text: str) -> List[_Section]:
        return self._paragraph_sections(text, detect_caps=True)

    def _paragraph_sections(self, text: str, detect_code_defs: bool = False, detect_caps: bool = False) -> List[_Section]:
        sections: List[_Section] = []
        pos = 0
        for m in _BLANK_SEP.finditer(text):
            s, e = pos, m.start()
            if e > s:
                sections.append(self._section_from_span(text, s, e, detect_code_defs, detect_caps))
            pos = m.end()
        if pos < len(text):
            sections.append(self._section_from_span(text, pos, len(text), detect_code_defs, detect_caps))
        return [s for s in sections if s.end > s.start]

    def _section_from_span(self, text: str, s: int, e: int, detect_code_defs: bool, detect_caps: bool) -> _Section:
        seg = text[s:e]
        first_line_end = seg.find("\n")
        first_line = (seg[:first_line_end] if first_line_end != -1 else seg).strip()

        if not first_line:
            return self._sec("", s, e)

        # Markdown-заголовок в тексте/коде.
        hm = re.match(r"^(#{1,6})[ \t]+(.*)$", first_line)
        if hm:
            return self._sec(hm.group(2).strip(), s, e)

        # Определение функции/класса в коде.
        if detect_code_defs and _CODE_DEF_RE.match(first_line):
            return self._sec(first_line, s, e)

        # Заголовок ALL CAPS в тексте.
        if detect_caps and first_line.isupper() and len(first_line) <= 120 and re.search(r"[А-ЯA-ZЁ]", first_line):
            return self._sec(first_line, s, e)

        return self._sec("", s, e)

    @staticmethod
    def _sec(title: str, start: int, end: int) -> _Section:
        return _Section(title, start, end)

    @staticmethod
    def _line_start_offsets(text: str) -> List[int]:
        starts = [0]
        for i, ch in enumerate(text):
            if ch == "\n":
                starts.append(i + 1)
        return starts
