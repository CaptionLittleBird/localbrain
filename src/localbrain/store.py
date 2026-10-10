"""存储：SQLite 记录文档清单和入库状态，Qdrant 存切块的向量和原文。"""
from __future__ import annotations

import atexit
import sqlite3
import threading
import uuid
from dataclasses import astuple, dataclass, fields
from pathlib import Path

from qdrant_client import QdrantClient, models

from .chunking import Chunk
from .config import SetupError, resolve_path

DENSE = "dense"
SPARSE = "bm25"


@dataclass
class DocRecord:
    doc_id: str
    path: str  # 相对文档目录的路径
    filename: str
    size: int
    mtime: float
    sha256: str
    num_chunks: int
    status: str  # ok / error
    error: str
    updated_at: float


@dataclass
class Hit:
    doc_id: str
    path: str
    chunk_index: int
    heading: str
    page_start: int | None
    page_end: int | None
    text: str
    score: float = 0.0


class Registry:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")  # API 进程读、入库进程写，互不阻塞
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                path TEXT NOT NULL,
                filename TEXT NOT NULL,
                size INTEGER NOT NULL,
                mtime REAL NOT NULL,
                sha256 TEXT NOT NULL,
                num_chunks INTEGER NOT NULL,
                status TEXT NOT NULL,
                error TEXT NOT NULL,
                updated_at REAL NOT NULL)"""
        )
        self._conn.commit()

    def _query(self, sql: str, params: tuple = ()) -> list[DocRecord]:
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [DocRecord(*row) for row in rows]

    def get(self, doc_id: str) -> DocRecord | None:
        rows = self._query("SELECT * FROM documents WHERE doc_id = ?", (doc_id,))
        return rows[0] if rows else None

    def all(self) -> list[DocRecord]:
        return self._query("SELECT * FROM documents ORDER BY path")

    def search(self, keyword: str, limit: int = 100) -> list[DocRecord]:
        return self._query(
            "SELECT * FROM documents WHERE status = 'ok' AND path LIKE ? ORDER BY path LIMIT ?",
            (f"%{keyword}%", limit),
        )

    def upsert(self, rec: DocRecord) -> None:
        placeholders = ", ".join("?" for _ in fields(DocRecord))
        with self._lock:
            self._conn.execute(f"INSERT OR REPLACE INTO documents VALUES ({placeholders})", astuple(rec))
            self._conn.commit()

    def delete(self, doc_id: str | None = None) -> None:
        """删除一篇文档的记录；不传 doc_id 则清空全部。"""
        with self._lock:
            if doc_id is None:
                self._conn.execute("DELETE FROM documents")
            else:
                self._conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            self._conn.commit()


class VectorStore:
    def __init__(self, url: str, collection: str) -> None:
        if url.startswith(("http://", "https://")):
            self.client = QdrantClient(url=url, timeout=60)  # 默认 5 秒，大批量写入时偶尔不够
        else:
            self.client = QdrantClient(path=str(resolve_path(url)))
            atexit.register(self.client.close)  # 嵌入模式要在解释器退出前关闭，否则退出时报错
        self.collection = collection

    def ensure_collection(self, dim: int | None) -> None:
        """dim 是向量模型的维度；为 None 表示没有向量模型，只建 BM25 稀疏索引。"""
        if self.client.collection_exists(self.collection):
            vectors = self.client.get_collection(self.collection).config.params.vectors or {}
            existing = vectors[DENSE].size if DENSE in vectors else None
            if existing != dim:
                raise SetupError(
                    f"向量库是按「{_describe_dim(existing)}」建的，当前配置是「{_describe_dim(dim)}」。"
                    "换了向量模型或 EMBED_BACKEND 的话，先运行 localbrain reset --yes 再重新入库。"
                )
            return
        dense_config = {DENSE: models.VectorParams(size=dim, distance=models.Distance.COSINE)} if dim else {}
        self.client.create_collection(
            self.collection,
            vectors_config=dense_config,
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        self.client.create_payload_index(self.collection, "doc_id", models.PayloadSchemaType.KEYWORD)
        self.client.create_payload_index(self.collection, "chunk_index", models.PayloadSchemaType.INTEGER)

    def replace_doc(
        self,
        doc_id: str,
        path: str,
        chunks: list[Chunk],
        dense: list[list[float]] | None,
        sparse: list[tuple[list[int], list[float]]],
    ) -> None:
        self.delete_doc(doc_id)
        points = []
        for i, (chunk, (indices, values)) in enumerate(zip(chunks, sparse)):
            vector: dict = {SPARSE: models.SparseVector(indices=indices, values=values)}
            if dense is not None:
                vector[DENSE] = dense[i]
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}:{chunk.index}")),
                    vector=vector,
                    payload={
                        "doc_id": doc_id,
                        "path": path,
                        "chunk_index": chunk.index,
                        "heading": chunk.heading,
                        "page_start": chunk.page_start,
                        "page_end": chunk.page_end,
                        "text": chunk.text,
                    },
                )
            )
        for i in range(0, len(points), 128):
            self.client.upsert(self.collection, points[i : i + 128])

    def delete_doc(self, doc_id: str) -> None:
        self.client.delete(self.collection, points_selector=models.FilterSelector(filter=_doc_filter(doc_id)))

    def hybrid_search(
        self,
        dense: list[float] | None,
        sparse: tuple[list[int], list[float]],
        limit: int,
        doc_id: str | None = None,
    ) -> list[Hit]:
        """稠密向量和 BM25 各召回 limit 条，用 RRF 融合排序；没有向量时只用 BM25。"""
        flt = _doc_filter(doc_id) if doc_id else None
        prefetch: list[models.Prefetch] = []
        if dense is not None:
            prefetch.append(models.Prefetch(query=dense, using=DENSE, limit=limit, filter=flt))
        indices, values = sparse
        if indices:
            sparse_vec = models.SparseVector(indices=indices, values=values)
            prefetch.append(models.Prefetch(query=sparse_vec, using=SPARSE, limit=limit, filter=flt))
        if not prefetch:
            return []
        if len(prefetch) == 1:
            only = prefetch[0]
            result = self.client.query_points(
                self.collection, query=only.query, using=only.using, query_filter=flt, limit=limit, with_payload=True
            )
        else:
            result = self.client.query_points(
                self.collection,
                prefetch=prefetch,
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
                with_payload=True,
            )
        return [_to_hit(p.payload, p.score) for p in result.points]

    def get_chunks(self, doc_id: str, start: int, count: int) -> list[Hit]:
        flt = models.Filter(
            must=[
                models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)),
                models.FieldCondition(key="chunk_index", range=models.Range(gte=start, lt=start + count)),
            ]
        )
        points, _ = self.client.scroll(self.collection, scroll_filter=flt, limit=count, with_payload=True)
        return sorted((_to_hit(p.payload) for p in points), key=lambda h: h.chunk_index)

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count

    def drop(self) -> None:
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)


def _describe_dim(dim: int | None) -> str:
    return f"{dim} 维向量" if dim else "无向量，只有 BM25"


def _doc_filter(doc_id: str) -> models.Filter:
    return models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))])


def _to_hit(payload: dict, score: float = 0.0) -> Hit:
    return Hit(
        doc_id=payload["doc_id"],
        path=payload["path"],
        chunk_index=payload["chunk_index"],
        heading=payload.get("heading", ""),
        page_start=payload.get("page_start"),
        page_end=payload.get("page_end"),
        text=payload["text"],
        score=score,
    )
