"""检索：混合召回（向量 + BM25）→ 重排。没配向量模型或重排模型时自动跳过对应步骤。"""
from __future__ import annotations

import re

from .config import Settings
from .models import get_embedder, get_reranker
from .sparse import query_vector
from .store import Hit, VectorStore


class Retriever:
    def __init__(self, settings: Settings, store: VectorStore) -> None:
        self.settings = settings
        self.store = store
        self.embedder = get_embedder()
        self.reranker = get_reranker()
        store.ensure_collection(self.embedder.dim if self.embedder else None)

    def search(self, query: str, top_k: int | None = None, doc_id: str | None = None) -> list[Hit]:
        dense = self.embedder.embed([query])[0] if self.embedder else None
        hits = self.store.hybrid_search(dense, query_vector(query), self.settings.candidates, doc_id)
        if hits and self.reranker:
            scores = self.reranker.score(query, [f"{h.heading}\n{h.text}" if h.heading else h.text for h in hits])
            for hit, score in zip(hits, scores):
                hit.score = score
            hits.sort(key=lambda h: h.score, reverse=True)
        return _dedupe(hits)[: top_k or self.settings.top_k]


_NOT_WORD = re.compile(r"[\W_]+")
_DUPLICATE = 0.8  # 两段文字的三字片段重合度超过这个比例就算重复


def _dedupe(hits: list[Hit]) -> list[Hit]:
    """去掉和排在前面的结果几乎一样的片段。同一本书存了 PDF 和 EPUB 两份时，同一段话不会占掉好几个名额。"""
    kept: list[tuple[Hit, set[str]]] = []
    for hit in hits:
        text = _NOT_WORD.sub("", hit.text)
        grams = {text[i : i + 3] for i in range(max(len(text) - 2, 1))}
        if any(len(grams & g) / max(len(grams | g), 1) >= _DUPLICATE for _, g in kept):
            continue
        kept.append((hit, grams))
    return [h for h, _ in kept]


def format_hit(hit: Hit) -> str:
    """给模型看的片段格式：带上出处信息，方便它引用和继续阅读。"""
    meta = [f"doc_id={hit.doc_id}", f"文件：{hit.path}"]
    if hit.page_start is not None:
        pages = f"p.{hit.page_start}" if hit.page_start == hit.page_end else f"p.{hit.page_start}-{hit.page_end}"
        meta.append(pages)
    meta.append(f"第 {hit.chunk_index} 段")
    if hit.heading:
        meta.append(f"章节：{hit.heading}")
    if hit.score:
        meta.append(f"相关度 {hit.score:.2f}")
    return f"[{' | '.join(meta)}]\n{hit.text}"
