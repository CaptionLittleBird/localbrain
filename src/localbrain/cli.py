"""命令行入口：localbrain serve | watch | ingest | status | search | ask | eval | reset"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from .config import SetupError, Settings, get_settings


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="localbrain", description="个人本地知识库 + Agent 问答")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("serve", help="启动 API 服务（给 Open WebUI 用）")
    sub.add_parser("watch", help="监听文档目录，有变化就自动入库")
    p = sub.add_parser("ingest", help="扫描文档目录，入库一次")
    p.add_argument("--force", action="store_true", help="全部重新入库（包括之前失败的）")
    sub.add_parser("status", help="查看入库情况和失败原因")
    p = sub.add_parser("search", help="只测检索效果，不调用大模型")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=None, help="返回条数")
    p = sub.add_parser("ask", help="在终端里提问")
    p.add_argument("question")
    p = sub.add_parser("eval", help="用问题集评估检索命中率")
    p.add_argument("file", help='jsonl 文件，每行 {"question": "...", "expected": "文件名里的关键字"}')
    p.add_argument("-k", type=int, default=5)
    p = sub.add_parser("reset", help="清空向量库和入库记录（文档本身不动）")
    p.add_argument("--yes", action="store_true", help="确认执行")
    args = parser.parse_args(argv)

    setup_logging()
    settings = get_settings()
    try:
        _run(args, settings)
    except SetupError as e:
        sys.exit(f"错误：{e}")
    except KeyboardInterrupt:
        pass


def setup_logging() -> None:
    """主进程和解析子进程共用的日志格式。"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpcore", "openai", "docling"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # Docling 表格识别的对齐提示，量大没用；它每次创建对象都会重设日志级别，只能在输出端过滤
    for handler in logging.getLogger().handlers:
        handler.addFilter(lambda r: r.name != "MatchingPostProcessor" or r.levelno >= logging.ERROR)


def _run(args: argparse.Namespace, settings: Settings) -> None:
    match args.command:
        case "serve":
            import uvicorn

            uvicorn.run("localbrain.server:app", host=settings.api_host, port=settings.api_port)
        case "watch":
            from .ingest import Ingestor

            asyncio.run(Ingestor(settings).watch())
        case "ingest":
            from .ingest import Ingestor

            Ingestor(settings).sync(force=args.force)
        case "status":
            _status(settings)
        case "search":
            from .retrieval import format_hit

            for hit in _retriever(settings).search(args.query, top_k=args.k):
                text = format_hit(hit)
                print(text if len(text) <= 500 else text[:500] + " …", end="\n\n")
        case "ask":
            asyncio.run(_ask(settings, args.question))
        case "eval":
            _eval(settings, Path(args.file), args.k)
        case "reset":
            _reset(settings, args.yes)


def _retriever(settings: Settings):
    from .retrieval import Retriever
    from .store import VectorStore

    return Retriever(settings, VectorStore(settings.qdrant_url, settings.collection))


def _status(settings: Settings) -> None:
    from .store import Registry

    labels = {"ok": "正常", "error": "失败", "indexing": "入库中", "duplicate": "重复"}
    docs = Registry(settings.db_path).all()
    failed = [d for d in docs if d.status == "error"]
    for d in docs:
        line = f"[{labels.get(d.status, d.status)}] {d.path}  {d.num_chunks} 段"
        print(line + (f"  {d.error}" if d.error else ""))
    print(f"\n共 {len(docs)} 篇，失败 {len(failed)} 篇。文档目录：{settings.docs_dir}")


async def _ask(settings: Settings, question: str) -> None:
    from langchain_core.messages import HumanMessage

    from .agent import build_agent, stream_answer
    from .store import Registry

    retriever = _retriever(settings)
    agent = build_agent(settings, retriever, Registry(settings.db_path), retriever.store)
    async for text in stream_answer(agent, [HumanMessage(question)]):
        print(text, end="", flush=True)
    print()


def _eval(settings: Settings, file: Path, k: int) -> None:
    """每行 {"question": ..., "expected": "文件名里的关键字", "pages": [起始页, 结束页]}，pages 可省略。

    检索结果的文件名含 expected、页码和 pages 有重叠（前后放宽 1 页）就算命中。
    """
    import time

    rows = [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines() if line.strip()]
    retriever = _retriever(settings)
    ranks: list[int | None] = []
    seconds = 0.0
    for row in rows:
        started = time.time()
        hits = retriever.search(row["question"], top_k=k)
        seconds += time.time() - started
        rank = next((i for i, h in enumerate(hits, start=1) if _eval_hit(row, h)), None)
        ranks.append(rank)
        if rank is None:
            got = [f"{h.path} p.{h.page_start}" if h.page_start else h.path for h in hits[:3]]
            print(f"未命中：{row['question']}\n  期望：{row['expected']} {row.get('pages', '')}\n  实际：{got}")
    n = max(len(rows), 1)
    recall = {c: sum(1 for r in ranks if r and r <= c) / n for c in sorted({1, 3, k})}
    mrr = sum(1 / r for r in ranks if r) / n
    print("\n" + "  ".join(f"recall@{c} = {v:.0%}" for c, v in recall.items()) + f"  MRR = {mrr:.3f}")
    print(f"共 {len(rows)} 题，平均每次检索 {seconds / n:.2f}s（召回 {settings.candidates} 条）")


def _eval_hit(row: dict, hit) -> bool:
    if row["expected"] not in hit.path:
        return False
    pages = row.get("pages")
    if not pages or hit.page_start is None:
        return True
    return hit.page_start <= pages[1] + 1 and hit.page_end >= pages[0] - 1


def _reset(settings: Settings, confirmed: bool) -> None:
    from .store import Registry, VectorStore

    if not confirmed:
        print(f"将清空向量库 {settings.collection} 和入库记录，文档目录不受影响。确认请加 --yes")
        return
    VectorStore(settings.qdrant_url, settings.collection).drop()
    Registry(settings.db_path).delete()
    print("已清空。运行 localbrain ingest 重新入库。")
