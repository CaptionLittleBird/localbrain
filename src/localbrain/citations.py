"""把回答里的出处 [文件名 p.12] 换成链接，点击直接打开原文（PDF 跳到对应页）。

回答是流式输出的，一个出处可能被拆在前后两段里，所以按字符扫描：遇到 [ 先攒着，等到 ] 再整体判断。
文件名对不上知识库里的文档就原样输出，不会误伤代码里的 [root@study ~]# 之类。
"""
from __future__ import annotations

import re
from urllib.parse import quote

from .store import DocRecord

# 方括号里的一段：文件名，可选的页码 p.12 / p.12-15 / 第12页
_PART = re.compile(r"^(?P<name>.+?)(?:[\s,，]*(?:p\.?|P\.?|第)\s*(?P<page>\d+)(?:\s*[-–~～]\s*\d+)?\s*页?)?$")
_SEPARATORS = re.compile(r"\s*[；;、]\s*")
_NOT_WORD = re.compile(r"[\W_]+")
_MAX_BRACKET = 300  # [ 之后这么多字还没等到 ]，就不是出处


class CitationLinker:
    """用法：每段文本过一遍 feed()，最后调用 flush() 输出攒着没处理完的部分。"""

    def __init__(self, docs: list[DocRecord], base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self._exact: dict[str, str | None] = {}
        self._fuzzy: list[tuple[str, str]] = []
        for d in docs:
            if d.status != "ok":
                continue
            stem = d.filename.rsplit(".", 1)[0]
            for key in {d.path, d.filename, stem}:
                # 同名文件在不同目录里时不知道指的是哪个，不加链接
                self._exact[key] = None if key in self._exact and self._exact[key] != d.path else d.path
            self._fuzzy.append((_norm(stem), d.path))
        self._buf = ""
        self._pending_close = False  # 刚收完一个 [...]，要看下一个字是不是 (，是的话说明本来就是链接

    def feed(self, text: str) -> str:
        out: list[str] = []
        for ch in text:
            if self._pending_close:
                self._pending_close = False
                out.append(self._buf if ch == "(" else self._convert(self._buf))
                self._buf = ""
            if self._buf:
                self._buf += ch
                if ch == "]":
                    self._pending_close = True
                elif ch == "\n" or len(self._buf) > _MAX_BRACKET:
                    out.append(self._buf)
                    self._buf = ""
                elif ch == "[":  # 前面那个 [ 没有闭合，从新的 [ 重新开始
                    out.append(self._buf[:-1])
                    self._buf = "["
            elif ch == "[":
                self._buf = "["
            else:
                out.append(ch)
        return "".join(out)

    def flush(self) -> str:
        text = self._convert(self._buf) if self._pending_close else self._buf
        self._buf, self._pending_close = "", False
        return text

    def _convert(self, bracket: str) -> str:
        """[a.pdf p.3；b.pdf p.5] → 每个出处各自成为一个链接；有一个对不上就整体原样输出。"""
        parts = [p for p in _SEPARATORS.split(bracket[1:-1].strip()) if p]
        links = []
        for part in parts:
            m = _PART.match(part)
            path = self._resolve(m.group("name").strip()) if m else None
            if path is None:
                return bracket
            url = f"{self.base_url}/kb/open?path={quote(path)}"
            if m.group("page"):
                url += f"&page={m.group('page')}"
            links.append(f"[\\[{_escape(part)}\\]]({url})")
        return "".join(links) if links else bracket

    def _resolve(self, name: str) -> str | None:
        if name in self._exact:
            return self._exact[name]
        # 模型常把长文件名写短，比如省掉【高清电子版】前缀；只在唯一匹配时才算
        key = _norm(name.rsplit(".", 1)[0] if "." in name[-6:] else name)
        if len(key) < 4:
            return None
        hits = {path for stem, path in self._fuzzy if key in stem}
        return hits.pop() if len(hits) == 1 else None


_LINK = re.compile(r"\[\\\[(.{1,300}?)\\\]\]\([^)\s]*/kb/open\?[^)\s]*\)")


def strip_links(text: str) -> str:
    """历史消息里的出处链接还原成 [文件名 p.12]：省上下文，也免得模型照着抄链接。"""
    return _LINK.sub(lambda m: "[" + m.group(1).replace("\\[", "[").replace("\\]", "]") + "]", text)


def _norm(text: str) -> str:
    return _NOT_WORD.sub("", text).lower()


def _escape(text: str) -> str:
    return text.replace("[", "\\[").replace("]", "\\]")
