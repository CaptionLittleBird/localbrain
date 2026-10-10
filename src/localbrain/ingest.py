"""入库：扫描文档目录 → 解析 → 切块 → 向量化 → 写入 Qdrant。没变过的文件直接跳过。"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from pathlib import Path

from .chunking import Chunk, chunk_sections
from .config import Settings
from .models import get_embedder
from .parsing import SUPPORTED_SUFFIXES, parse_file
from .sparse import doc_vector
from .store import DocRecord, Registry, VectorStore

log = logging.getLogger(__name__)


class Ingestor:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.docs_dir = settings.docs_dir
        self.registry = Registry(settings.db_path)
        self.store = VectorStore(settings.qdrant_url, settings.collection)
        self.embedder = get_embedder()
        self.store.ensure_collection(self.embedder.dim if self.embedder else None)

    def sync(self, force: bool = False) -> None:
        """让向量库和文档目录保持一致：新增/修改的入库，删掉的移除。"""
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        files = {self._doc_id(p): p for p in self._scan()}
        for doc_id, path in files.items():
            self._sync_file(doc_id, path, force)
        for rec in self.registry.all():
            if rec.doc_id not in files:
                self.store.delete_doc(rec.doc_id)
                self.registry.delete(rec.doc_id)
                log.info("已移除：%s", rec.path)

    async def watch(self) -> None:
        from watchfiles import awatch

        await asyncio.to_thread(self.sync)
        log.info("开始监听 %s", self.docs_dir)
        async for _ in awatch(self.docs_dir):
            await asyncio.to_thread(self.sync)

    def _scan(self):
        for path in sorted(self.docs_dir.rglob("*")):
            if (
                path.is_file()
                and path.suffix.lower() in SUPPORTED_SUFFIXES
                and not path.name.startswith((".", "~$"))  # 隐藏文件、Word 打开时的临时锁文件
            ):
                yield path

    def _doc_id(self, path: Path) -> str:
        rel = path.relative_to(self.docs_dir).as_posix()
        return hashlib.sha1(rel.encode("utf-8")).hexdigest()[:12]

    def _sync_file(self, doc_id: str, path: Path, force: bool) -> None:
        stat = path.stat()
        rec = self.registry.get(doc_id)
        if not force and rec:
            # 失败过的文件也跳过，等文件变了或 ingest --force 再重试
            if rec.size == stat.st_size and rec.mtime == stat.st_mtime:
                return
            sha = _sha256(path)
            if rec.sha256 == sha and rec.status == "ok":
                rec.size, rec.mtime = stat.st_size, stat.st_mtime
                self.registry.upsert(rec)
                return
        else:
            sha = _sha256(path)
        self._ingest(doc_id, path, stat, sha)

    def _ingest(self, doc_id: str, path: Path, stat: os.stat_result, sha: str) -> None:
        rel = path.relative_to(self.docs_dir).as_posix()
        log.info("入库中：%s", rel)
        started = time.time()
        chunks: list[Chunk] = []
        try:
            chunks = chunk_sections(parse_file(path), self.settings.chunk_size)
            texts = [_embed_text(rel, c) for c in chunks]
            dense = self.embedder.embed(texts) if self.embedder and texts else None
            sparse = [doc_vector(t) for t in texts]
            self.store.replace_doc(doc_id, rel, chunks, dense, sparse)
            status, error = "ok", ""
            log.info("已入库：%s，%d 块，用时 %.1fs", rel, len(chunks), time.time() - started)
        except Exception as e:
            log.exception("入库失败：%s", rel)
            chunks, status, error = [], "error", f"{type(e).__name__}: {e}"
        self.registry.upsert(
            DocRecord(
                doc_id, rel, path.name, stat.st_size, stat.st_mtime, sha, len(chunks), status, error, time.time()
            )
        )


def _embed_text(rel_path: str, chunk: Chunk) -> str:
    """向量化时把文件名和章节标题拼在正文前面，检索时能命中"某某文档里的某某章节"。"""
    title = f"《{Path(rel_path).stem}》"
    if chunk.heading:
        title += f" {chunk.heading}"
    return f"{title}\n{chunk.text}"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
