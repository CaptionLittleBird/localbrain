"""多轮对话时把上一轮的检索结果接回去。

Open WebUI 每次只把对话的文字发过来，上一轮 Agent 调了哪些工具、查到了什么都丢了，追问时只能重新检索。
这里在每轮回答结束后，把这一轮的工具调用和结果按"对话指纹"存在内存里；下一轮请求带着同样的历史过来时，
找回来插到对应的回答前面，模型就能直接用上一轮查到的内容。

指纹只取每条消息里的文字和数字，换行、空格、Markdown 格式的差异不影响匹配。服务重启后缓存清空，
找不到就退回原来的做法（重新检索），不影响正确性。
"""
from __future__ import annotations

import hashlib
import re
import threading
from collections import OrderedDict

from langchain_core.messages import AIMessage, BaseMessage

_NOT_WORD = re.compile(r"[\W_]+")
_MAX_ENTRIES = 256
# 只接回最近这么多轮的检索结果：每轮的结果有上万字，全接回去上下文会越来越长
REUSE_TURNS = 1


def _fingerprint(messages: list[BaseMessage]) -> str:
    h = hashlib.sha256()
    for m in messages:
        text = m.content if isinstance(m.content, str) else str(m.content)
        h.update(m.type.encode())
        h.update(_NOT_WORD.sub("", text).encode())
        h.update(b"\0")
    return h.hexdigest()


class ToolHistory:
    def __init__(self) -> None:
        self._cache: OrderedDict[str, list[BaseMessage]] = OrderedDict()
        self._lock = threading.Lock()

    def remember(self, conversation: list[BaseMessage], answer: str, steps: list[BaseMessage]) -> None:
        """conversation 是这一轮收到的历史，answer 是返回给客户端的完整回答，steps 是中间的工具调用和结果。"""
        if not steps:
            return
        key = _fingerprint([*conversation, AIMessage(answer)])
        with self._lock:
            self._cache[key] = steps
            self._cache.move_to_end(key)
            while len(self._cache) > _MAX_ENTRIES:
                self._cache.popitem(last=False)

    def restore(self, messages: list[BaseMessage]) -> tuple[list[BaseMessage], int]:
        """在最近几轮回答前面插回当时的工具调用和结果。返回新的消息列表和接回了几轮。"""
        answers = [i for i, m in enumerate(messages) if isinstance(m, AIMessage)][-REUSE_TURNS:]
        inserts: dict[int, list[BaseMessage]] = {}
        with self._lock:
            for i in answers:
                steps = self._cache.get(_fingerprint(messages[: i + 1]))
                if steps:
                    inserts[i] = steps
        if not inserts:
            return messages, 0
        out: list[BaseMessage] = []
        for i, m in enumerate(messages):
            out.extend(inserts.get(i, []))
            out.append(m)
        return out, len(inserts)
