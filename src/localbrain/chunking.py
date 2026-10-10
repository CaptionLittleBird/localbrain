"""按 Markdown 结构切块：跟踪标题层级，按段落累积到 chunk_size 字符左右。"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .parsing import Section

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_SENTENCE_END = re.compile(r"(?<=[。！？；.!?;\n])")


@dataclass
class Chunk:
    index: int
    text: str
    heading: str  # 标题路径，如 "第二章 方法 > 2.1 样本处理"
    page_start: int | None
    page_end: int | None


def chunk_sections(sections: list[Section], chunk_size: int = 800) -> list[Chunk]:
    chunks: list[Chunk] = []
    headings: list[tuple[int, str]] = []
    heading = ""
    buf: list[str] = []
    buf_len = 0
    pages: list[int] = []

    def flush() -> None:
        nonlocal buf, buf_len, pages
        text = "\n\n".join(buf).strip()
        if text:
            chunks.append(
                Chunk(len(chunks), text, heading, min(pages, default=None), max(pages, default=None))
            )
        buf, buf_len, pages = [], 0, []

    for section in sections:
        for block in _blocks(section.text):
            m = _HEADING.match(block)
            if m:
                # 遇到新标题就截断，保证一个块不跨章节
                flush()
                level = len(m.group(1))
                headings = [h for h in headings if h[0] < level] + [(level, m.group(2))]
                heading = " > ".join(h[1] for h in headings)
                continue
            for piece in _split_long(block, chunk_size):
                if buf and buf_len + len(piece) > chunk_size:
                    flush()
                buf.append(piece)
                buf_len += len(piece)
                if section.page is not None:
                    pages.append(section.page)
    flush()
    return chunks


def _blocks(text: str) -> list[str]:
    """按空行分段，标题行单独成段。"""
    blocks: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        current: list[str] = []
        for line in para.strip("\n").split("\n"):
            if _HEADING.match(line):
                if current:
                    blocks.append("\n".join(current))
                    current = []
                blocks.append(line.strip())
            else:
                current.append(line)
        if current:
            blocks.append("\n".join(current))
    return [b for b in blocks if b.strip()]


def _split_long(block: str, size: int) -> list[str]:
    """超长段落按句子拆；表格按行拆并在每块重复表头。"""
    if len(block) <= size:
        return [block]
    lines = block.split("\n")
    if len(lines) > 2 and lines[0].lstrip().startswith("|"):
        return _split_table(lines, size)
    pieces: list[str] = []
    current = ""
    for sentence in _SENTENCE_END.split(block):
        if current and len(current) + len(sentence) > size:
            pieces.append(current)
            current = ""
        while len(sentence) > size:  # 没有标点的超长文本只能硬切
            pieces.append(sentence[:size])
            sentence = sentence[size:]
        current += sentence
    if current.strip():
        pieces.append(current)
    return pieces


def _split_table(lines: list[str], size: int) -> list[str]:
    header = lines[:2]
    header_len = sum(len(line) + 1 for line in header)
    pieces: list[str] = []
    rows: list[str] = []
    rows_len = header_len
    for row in lines[2:]:
        if rows and rows_len + len(row) > size:
            pieces.append("\n".join(header + rows))
            rows, rows_len = [], header_len
        rows.append(row)
        rows_len += len(row) + 1
    if rows:
        pieces.append("\n".join(header + rows))
    return pieces
