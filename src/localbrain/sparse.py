"""BM25 稀疏向量：jieba 分词 + 词哈希，IDF 交给 Qdrant 在服务端算（Modifier.IDF）。

和 bge-m3 的稠密向量一起做混合检索，专门补关键词的精确匹配：人名、型号、编号、专业术语等。
"""
from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter

import jieba

jieba.setLogLevel(logging.WARNING)

_HAS_WORD = re.compile(r"\w")  # 至少含一个字母、数字或汉字，过滤掉纯标点
_STOPWORDS = frozenset(
    "的 了 和 是 在 就 都 而 及 与 着 或 也 把 被 让 对 从 这 那 有 一个 没有 我们 你们 他们 "
    "这个 那个 什么 怎么 如何 哪些 吗 呢 吧 啊 "
    "a an the of to and or in on for is are was were be by with as at it this that".split()
)

# BM25 参数；AVG_LEN 是文档平均词数的近似值，不需要精确
K1 = 1.2
B = 0.75
AVG_LEN = 256


def tokenize(text: str) -> list[str]:
    return [
        t
        for t in (w.strip() for w in jieba.cut_for_search(text.lower()))
        if t and t not in _STOPWORDS and _HAS_WORD.search(t)
    ]


def _token_id(token: str) -> int:
    return int.from_bytes(hashlib.md5(token.encode("utf-8")).digest()[:4], "little")


def doc_vector(text: str) -> tuple[list[int], list[float]]:
    """文档侧：BM25 中的词频饱和项。"""
    tokens = tokenize(text)
    tf = Counter(_token_id(t) for t in tokens)
    norm = K1 * (1 - B + B * len(tokens) / AVG_LEN)
    return list(tf), [f * (K1 + 1) / (f + norm) for f in tf.values()]


def query_vector(text: str) -> tuple[list[int], list[float]]:
    """查询侧：每个词权重为 1，Qdrant 会再乘上 IDF。"""
    ids = sorted({_token_id(t) for t in tokenize(text)})
    return ids, [1.0] * len(ids)
