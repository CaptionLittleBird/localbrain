"""检索：混合召回（向量 + BM25）→ 重排。没配向量模型或重排模型时自动跳过对应步骤。"""
from __future__ import annotations

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
        return hits[: top_k or self.settings.top_k]


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
