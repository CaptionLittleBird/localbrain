import sys
import threading
import time

import pytest

from localbrain import workers
from localbrain.workers import Cancelled, ParsePool

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="用到 SIGKILL")


def collect(tasks, **kw):
    from pool_tasks import run

    with ParsePool(run, **kw) as pool:
        return {task: (ok, result) for task, ok, result in pool.imap_unordered(tasks)}


def test_timeout_crash_and_error_are_isolated():
    tasks = [("ok", 0.1), ("hang", 0), ("raise", 0), ("crash", 0), ("ok", 0.2)]
    started = time.time()
    got = collect(tasks, workers=3, timeout=3)
    assert got[("ok", 0.1)][0] and got[("ok", 0.2)][0]
    assert "超时" in got[("hang", 0)][1]
    assert "坏文件" in got[("raise", 0)][1]
    assert "异常退出" in got[("crash", 0)][1]
    assert time.time() - started < 15


def test_workers_are_recycled(monkeypatch):
    from pool_tasks import run

    monkeypatch.setattr(workers, "_TASKS_PER_WORKER", 2)
    with ParsePool(run, workers=1, timeout=10) as pool:
        pids = {pid for _, ok, pid in pool.imap_unordered([("ok", 0.01)] * 6) if ok}
    assert len(pids) == 3  # 每个进程处理 2 个后换新


def test_cancel_stops_quickly():
    from pool_tasks import run

    stop = threading.Event()
    threading.Timer(0.5, stop.set).start()
    started = time.time()
    with pytest.raises(Cancelled):
        with ParsePool(run, workers=2, timeout=100, stop=stop) as pool:
            list(pool.imap_unordered([("hang", 0), ("hang", 0)]))
    assert time.time() - started < 10
