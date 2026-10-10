"""入库：扫描文档目录 → 解析 → 切块 → 向量化 → 写入 Qdrant。没变过的文件直接跳过。

解析（CPU 和 GPU 都吃）在子进程里并行跑，见 workers.py；向量化和写库在主进程里，
和后面文件的解析同时进行。

出错分两种：
- 文档本身的问题（格式坏了、加密、解析超时）：记为失败，文件不变就不再重试，免得反复卡在同一个文件上；
- 外部服务的问题（oMLX、Qdrant 连不上）：不记失败，中止这次同步，等服务恢复后自动重试。
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .chunking import Chunk, chunk_sections
from .config import SetupError, Settings, get_settings
from .models import ModelInputError, get_embedder
from .parsing import SUPPORTED_SUFFIXES, parse_file
from .sparse import doc_vector
from .store import DocRecord, Registry, VectorStore
from .workers import Cancelled, ParsePool

log = logging.getLogger(__name__)

# 文件最近这么多秒内还在变化（可能还没拷完）就先不入库，下一轮再看
SETTLE_SECONDS = 5
# 同步失败（通常是 oMLX 或 Qdrant 连不上）后多久重试
RETRY_SECONDS = 60


@dataclass
class Job:
    doc_id: str
    path: Path
    rel: str  # 相对文档目录的路径
    size: int
    mtime: float
    sha256: str


@dataclass
class Parsed:
    chunks: list[Chunk]
    sparse: list[tuple[list[int], list[float]]]
    sections: int
    seconds: float


def prepare(job: Job) -> Parsed:
    """在解析子进程里运行：解析、切块、算 BM25 稀疏向量，都是吃 CPU 的活。"""
    log.info("解析中：%s", job.rel)
    started = time.time()
    sections = parse_file(job.path)
    chunks = chunk_sections(sections, get_settings().chunk_size)
    sparse = [doc_vector(_embed_text(job.rel, c)) for c in chunks]
    return Parsed(chunks, sparse, len(sections), time.time() - started)


class Ingestor:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.docs_dir = settings.docs_dir
        self.registry = Registry(settings.db_path)
        self.store = VectorStore(settings.qdrant_url, settings.collection)
        self.embedder = get_embedder()
        self.store.ensure_collection(self.embedder.dim if self.embedder else None)
        self._stop = threading.Event()

    def sync(self, force: bool = False, settle: float = 0, wait: bool = True) -> bool:
        """让向量库和文档目录保持一致：新增/修改的入库，删掉的移除。

        settle > 0 时跳过最近 settle 秒内还在变化的文件。返回 True 表示有文件留到了下一轮。
        wait=False 时如果别的进程正在同步就直接返回 True（留到下一轮），不等待。
        """
        with self._sync_lock(wait) as locked:
            if not locked:
                return True
            return self._sync(force, settle)

    @contextlib.contextmanager
    def _sync_lock(self, wait: bool):
        """同一时间只让一个进程同步：watch 服务和手动 ingest 同时跑的话，会抢着处理同一篇文档。"""
        try:
            import fcntl
        except ImportError:  # Windows 上开发用，不加锁
            yield True
            return
        with open(self.settings.db_path.with_suffix(".lock"), "w") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                if not wait:
                    yield False
                    return
                log.info("另一个进程正在入库（通常是 watch 服务），等它这一轮做完")
                fcntl.flock(f, fcntl.LOCK_EX)
            yield True  # 文件关闭时锁自动释放，进程被杀也不会留下死锁

    def _sync(self, force: bool, settle: float) -> bool:
        records = self._check_consistency(self.registry.all())
        self._ensure_docs_dir(bool(records))
        files = {self._doc_id(p): p for p in scan_docs(self.docs_dir)}
        if records and not files:
            # 多半是目录没挂载上或者被清空了，不能当成"文档全删了"，否则整个知识库要重新入库
            log.warning(
                "文档目录里一个文件都没有，为防止误删，保留已有的 %d 篇入库记录。确实要清空请运行 localbrain reset --yes",
                len(records),
            )
        for rec in records if files else []:
            if rec.doc_id not in files:
                self.store.delete_doc(rec.doc_id)
                self.registry.delete(rec.doc_id)
                log.info("已移除：%s", rec.path)

        jobs: list[Job] = []
        deferred = False
        for doc_id, path in files.items():
            try:
                stat = path.stat()
                if settle and time.time() - max(stat.st_mtime, stat.st_ctime) < settle:
                    # 用 ctime 而不只看 mtime：Finder 拷贝会保留原文件的修改时间，拷贝过程中 mtime 是旧的
                    deferred = True
                    continue
                job = self._job_if_changed(doc_id, path, stat, force)
            except FileNotFoundError:  # 扫描之后被删了，下一轮会移除记录
                continue
            if job:
                jobs.append(job)
        jobs = self._skip_duplicates(jobs)
        if jobs:
            self._ingest_all(jobs)
        return deferred

    async def watch(self) -> None:
        from watchfiles import awatch

        self._ensure_docs_dir(bool(self.registry.all()))
        log.info("开始监听 %s", self.docs_dir)
        dirty, retry_at = True, 0.0
        try:
            # 没有文件变化时每 5 秒也醒一次：处理上一轮没拷完的文件、重试失败的同步
            async for changes in awatch(self.docs_dir, rust_timeout=5000, yield_on_timeout=True):
                dirty = dirty or bool(changes)
                if not dirty or time.monotonic() < retry_at:
                    continue
                try:
                    dirty = await asyncio.to_thread(self.sync, settle=SETTLE_SECONDS, wait=False)
                except Exception:
                    log.exception("同步中断，%d 秒后重试", RETRY_SECONDS)
                    retry_at = time.monotonic() + RETRY_SECONDS
        finally:
            self._stop.set()  # 通知后台的同步线程尽快停下，并结束解析子进程

    def _check_consistency(self, records: list[DocRecord]) -> list[DocRecord]:
        """向量库被清空或重建过（比如删了 Qdrant 的数据卷），入库记录却还在：不处理的话所有文件都被当成
        "没变过"跳过，向量库一直是空的。这种情况清掉记录，全部重新入库。"""
        indexed = [r for r in records if r.status == "ok" and r.num_chunks]
        if indexed and self.store.count() == 0:
            log.warning("向量库是空的，但有 %d 篇文档记录为已入库，可能向量库被重建过，全部重新入库", len(indexed))
            self.registry.delete()
            return []
        return records

    def _ensure_docs_dir(self, has_records: bool) -> None:
        if self.docs_dir.is_dir():
            return
        if has_records:  # 入过库的目录不见了：外接硬盘没挂载、路径改了，不能自己建个空目录了事
            raise SetupError(f"文档目录 {self.docs_dir} 不存在（外接硬盘没挂载？DOCS_DIR 写错了？）")
        self.docs_dir.mkdir(parents=True)

    def _doc_id(self, path: Path) -> str:
        rel = path.relative_to(self.docs_dir).as_posix()
        return hashlib.sha1(rel.encode("utf-8")).hexdigest()[:12]

    def _job_if_changed(self, doc_id: str, path: Path, stat, force: bool) -> Job | None:
        rec = self.registry.get(doc_id)
        # 状态是 indexing 说明上次写到一半被打断了，要重做
        if not force and rec and rec.status != "indexing":
            # 失败过的文件也跳过，等文件变了或 ingest --force 再重试
            if rec.size == stat.st_size and rec.mtime == stat.st_mtime:
                # 重复的文件要看原件还在不在：原件删了或改了，就轮到它入库
                if rec.status != "duplicate" or self._original(rec.sha256, doc_id):
                    return None
            sha = _sha256(path)
            if rec.sha256 == sha and rec.status == "ok":
                rec.size, rec.mtime = stat.st_size, stat.st_mtime
                self.registry.upsert(rec)
                return None
        else:
            sha = _sha256(path)
        rel = path.relative_to(self.docs_dir).as_posix()
        return Job(doc_id, path, rel, stat.st_size, stat.st_mtime, sha)

    def _original(self, sha256: str, doc_id: str) -> str | None:
        """内容完全相同、已经入库的另一篇文档的路径。"""
        return next((r.path for r in self.registry.by_sha(sha256) if r.status == "ok" and r.doc_id != doc_id), None)

    def _skip_duplicates(self, jobs: list[Job]) -> list[Job]:
        """内容和已入库文档（或这批里排在前面的文件）完全相同的不再入库：省时间，检索结果也不会重复。"""
        kept: list[Job] = []
        seen: dict[str, str] = {}
        for job in jobs:
            original = self._original(job.sha256, job.doc_id) or seen.get(job.sha256)
            if original is None:
                seen[job.sha256] = job.rel
                kept.append(job)
                continue
            log.info("跳过重复文件：%s 和 %s 内容完全相同", job.rel, original)
            self.store.delete_doc(job.doc_id)
            self.registry.upsert(self._record(job, 0, "duplicate", f"和「{original}」内容完全相同，没有重复入库"))
        return kept

    def _ingest_all(self, jobs: list[Job]) -> None:
        workers = min(self.settings.ingest_workers, len(jobs))
        log.info("待入库 %d 个文件，用 %d 个进程并行解析", len(jobs), workers)
        with ParsePool(prepare, workers, self.settings.parse_timeout, self._stop) as pool:
            for done, (job, ok, result) in enumerate(pool.imap_unordered(jobs), start=1):
                if ok and not result.chunks:
                    ok, result = False, "没有提取到文字（如果是扫描件或图片，检查 .env 里的 OCR 配置）"
                if ok:
                    self._store(job, result)
                else:
                    self._fail(job, result)
                if len(jobs) > 1:
                    log.info("入库进度 %d/%d", done, len(jobs))

    def _store(self, job: Job, parsed: Parsed) -> None:
        log.info(
            "解析完成：%s，%d 页/段 → %d 块，用时 %.1fs",
            job.rel, parsed.sections, len(parsed.chunks), parsed.seconds,
        )
        started = time.time()
        # 先标记为 indexing：写到一半被打断的话，下一轮能认出来重做
        self.registry.upsert(self._record(job, 0, "indexing", ""))
        texts = [_embed_text(job.rel, c) for c in parsed.chunks]
        try:
            dense = self._embed(texts) if self.embedder else None
        except ModelInputError as e:  # 模型服务拒绝了这篇文档的内容，不是服务本身的问题
            self._fail(job, str(e))
            return
        self.store.replace_doc(job.doc_id, job.rel, parsed.chunks, dense, parsed.sparse)
        self.registry.upsert(self._record(job, len(parsed.chunks), "ok", ""))
        log.info("已入库：%s，%d 块，向量化和写入用时 %.1fs", job.rel, len(parsed.chunks), time.time() - started)

    def _fail(self, job: Job, error: str) -> None:
        log.error("入库失败：%s：%s", job.rel, error)
        # 旧版本的片段也删掉，免得状态显示失败、检索却还能搜到旧内容
        self.store.delete_doc(job.doc_id)
        self.registry.upsert(self._record(job, 0, "error", error))

    def _record(self, job: Job, num_chunks: int, status: str, error: str) -> DocRecord:
        return DocRecord(
            job.doc_id, job.rel, job.path.name, job.size, job.mtime, job.sha256, num_chunks, status, error, time.time()
        )

    def _embed(self, texts: list[str], step: int = 256) -> list[list[float]]:
        """分批向量化；大文档每批打一行进度，免得看起来像卡住了。"""
        assert self.embedder is not None
        vectors: list[list[float]] = []
        for i in range(0, len(texts), step):
            if self._stop.is_set():
                raise Cancelled
            vectors.extend(self.embedder.embed(texts[i : i + step]))
            if len(texts) > step:
                log.info("向量化 %d/%d", len(vectors), len(texts))
        return vectors


def scan_docs(docs_dir: Path):
    """文档目录下所有要入库的文件。"""
    for path in sorted(docs_dir.rglob("*")):
        if (
            path.suffix.lower() in SUPPORTED_SUFFIXES
            and not is_ignored_name(path.name)
            and not any(part.startswith(".") for part in path.relative_to(docs_dir).parent.parts)
            and path.is_file()
        ):
            yield path


def is_ignored_name(name: str) -> bool:
    """隐藏文件（包括上传中的临时文件）、Office 打开文档时生成的临时锁文件。"""
    return name.startswith((".", "~$"))


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
