"""解析进程池：文档解析放在独立的子进程里，多个文件并行解析，单个文件卡死或崩溃不影响其他文件。

没有直接用 concurrent.futures.ProcessPoolExecutor，因为：
- 它不能单独终止某个任务，超时后只能不再等它，子进程还在后台占着 CPU 和内存；
- 一个子进程崩溃（比如内存不足被系统杀掉），整个池子就不能用了，也分不清是哪个文件引起的。
这里每个子进程同一时间只处理一个文件，超时或崩溃时只终止这一个进程，下一个文件换新进程。
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue
import signal
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from multiprocessing.connection import wait
from typing import Any

log = logging.getLogger(__name__)

# torch、Docling 在 fork 出来的子进程里会出问题，统一用 spawn
_ctx = mp.get_context("spawn")
# 一个子进程处理这么多文件后换新的：Docling 长时间运行内存会慢慢涨上去
_TASKS_PER_WORKER = 20


class Cancelled(Exception):
    """入库被中止（Ctrl+C 或服务退出）。"""


def _worker_main(conn, fn: Callable[[Any], Any], parent_pid: int) -> None:
    from .cli import setup_logging

    setup_logging()
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl+C 由主进程统一处理
    threading.Thread(target=_exit_with_parent, args=(parent_pid,), daemon=True).start()
    while True:
        try:
            task = conn.recv()
        except (EOFError, OSError):
            return
        if task is None:
            return
        try:
            result = (True, fn(task))
        except Exception as e:
            log.debug("解析出错", exc_info=True)
            result = (False, f"{type(e).__name__}: {e}")
        try:
            conn.send(result)
        except (EOFError, OSError):
            return


def _exit_with_parent(parent_pid: int) -> None:
    """主进程被强制结束时（比如 launchd 停止服务）跟着退出，不留下孤儿进程。"""
    while True:
        time.sleep(2)
        if os.getppid() != parent_pid:
            os._exit(1)


class _Worker:
    def __init__(self, fn: Callable[[Any], Any]) -> None:
        self.conn, child = _ctx.Pipe()
        self.proc = _ctx.Process(target=_worker_main, args=(child, fn, os.getpid()), daemon=True)
        self.proc.start()
        child.close()
        self.task: Any = None
        self.deadline = 0.0
        self.done = 0

    def stop(self) -> None:
        try:
            self.conn.send(None)
        except OSError:
            pass
        self.proc.join(5)
        self.kill()

    def kill(self) -> None:
        if self.proc.is_alive():
            self.proc.kill()
            self.proc.join(5)
        self.conn.close()


class ParsePool:
    """用法：

        with ParsePool(fn, workers=2, timeout=3600) as pool:
            for task, ok, result in pool.imap_unordered(tasks):
                ...  # ok 为 False 时 result 是错误信息

    fn 必须是模块顶层的函数（子进程要能 import 到）。调度在后台线程里进行，
    调用方处理上一个结果（比如向量化）的同时，子进程已经在解析后面的文件了。
    """

    def __init__(self, fn: Callable[[Any], Any], workers: int, timeout: float, stop: threading.Event | None = None):
        self.fn = fn
        self.workers = max(1, workers)
        self.timeout = timeout
        self.stop = stop or threading.Event()
        self._closing = threading.Event()
        self._idle: list[_Worker] = []
        self._busy: list[_Worker] = []
        self._thread: threading.Thread | None = None

    def __enter__(self) -> ParsePool:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def imap_unordered(self, tasks: list) -> Iterator[tuple[Any, bool, Any]]:
        results: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._schedule, args=(deque(tasks), results), daemon=True)
        self._thread.start()
        for _ in range(len(tasks)):
            while True:
                if self.stop.is_set():
                    raise Cancelled
                try:
                    item = results.get(timeout=0.5)
                    break
                except queue.Empty:
                    continue
            if isinstance(item, BaseException):
                raise item
            yield item

    def close(self) -> None:
        self._closing.set()
        if self._thread is not None:
            self._thread.join(10)
        for w in self._busy:
            w.kill()
        for w in self._idle:
            w.stop()
        self._busy.clear()
        self._idle.clear()

    def _schedule(self, pending: deque, results: queue.Queue) -> None:
        try:
            while (pending or self._busy) and not self._closing.is_set():
                self._dispatch(pending, results)
                if self._busy:
                    self._collect(results)
                else:  # 结果积压、在等调用方取走，别空转
                    time.sleep(0.2)
        except BaseException as e:  # 调度线程出错要告诉调用方，不能让它一直等
            results.put(e)

    def _dispatch(self, pending: deque, results: queue.Queue) -> None:
        # 调用方还没取走的结果太多（解析比向量化快）就先不派新任务，免得解析结果堆在内存里
        while pending and len(self._busy) < self.workers and results.qsize() < self.workers:
            task = pending.popleft()
            worker = self._idle.pop() if self._idle else _Worker(self.fn)
            try:
                worker.conn.send(task)
            except OSError:
                worker.kill()
                results.put((task, False, "解析进程启动失败"))
                continue
            worker.task, worker.deadline = task, time.monotonic() + self.timeout
            self._busy.append(worker)

    def _collect(self, results: queue.Queue) -> None:
        # 最多等 1 秒就检查一遍超时和关闭信号
        wait_for = min(1.0, max(0.0, min(w.deadline for w in self._busy) - time.monotonic()))
        wait([w.conn for w in self._busy] + [w.proc.sentinel for w in self._busy], wait_for)
        for w in list(self._busy):
            reusable = False
            if w.conn.poll():  # 有结果，或者管道已断（子进程退出了）
                try:
                    ok, payload = w.conn.recv()
                    reusable = True
                except (EOFError, OSError):
                    ok, payload = False, self._died(w)
            elif not w.proc.is_alive():
                ok, payload = False, self._died(w)
            elif time.monotonic() > w.deadline:
                ok, payload = False, f"解析超时（超过 {self.timeout:.0f} 秒），已终止。可以调大 PARSE_TIMEOUT"
            else:
                continue
            self._busy.remove(w)
            w.done += 1
            if not reusable:
                w.kill()
            elif w.done >= _TASKS_PER_WORKER:
                w.stop()
            else:
                self._idle.append(w)
            results.put((w.task, ok, payload))

    @staticmethod
    def _died(w: _Worker) -> str:
        w.proc.join(1)
        code = w.proc.exitcode
        reason = "可能是内存不足被系统终止" if code is not None and code < 0 else "解析库崩溃"
        return f"解析进程异常退出（退出码 {code}，{reason}）"
