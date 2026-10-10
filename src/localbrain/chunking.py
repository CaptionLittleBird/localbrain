"""按 Markdown 结构切块：跟踪标题层级，按段落累积到 chunk_size 字符左右。"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .parsing import Section

# 只用无歧义的写法：之前的 r"^(#{1,6})\s+(.+?)\s*#*\s*$" 遇到中间有一长串空格的行会回溯到卡死
_HEADING = re.compile(r"(#{1,6})[ \t]+(\S.*)")
_MAX_HEADING_LEN = 200  # 再长就不是标题了，多半是 # 开头的正文或注释
_SENTENCE_END = re.compile(r"(?<=[。！？；.!?;\n])")
_FENCE = ("```", "~~~")


def _heading(line: str) -> tuple[int, str] | None:
    """Markdown 标题行 → (级别, 标题文字)，去掉结尾可选的 # 号。"""
    m = _HEADING.fullmatch(line.rstrip())
    if not m:
        return None
    title = m.group(2).rstrip("#").strip()
    if not title or len(title) > _MAX_HEADING_LEN:
        return None
    return len(m.group(1)), title


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
            h = _heading(block)
            if h:
                # 遇到新标题就截断，保证一个块不跨章节
                flush()
                level, title = h
                headings = [x for x in headings if x[0] < level] + [(level, title)]
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
    """按空行分段，标题行单独成段。代码块整体算一段，里面 # 开头的行（shell 注释、root 提示符）不当标题。"""
    blocks: list[str] = []
    current: list[str] = []
    in_fence = False

    def push() -> None:
        if current:
            blocks.append("\n".join(current))
            current.clear()

    for line in text.split("\n"):
        if line.lstrip().startswith(_FENCE):
            in_fence = not in_fence
            current.append(line)
        elif in_fence:
            current.append(line)
        elif not line.strip():
            push()
        elif _heading(line):
            push()
            blocks.append(line.strip())
        else:
            current.append(line)
    push()
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
