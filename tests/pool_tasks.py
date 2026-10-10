"""进程池测试用的任务函数，放在单独的模块里，spawn 出来的子进程才能导入。"""
import os
import signal
import time


def run(task):
    kind, arg = task
    if kind == "ok":
        time.sleep(arg)
        return os.getpid()
    if kind == "raise":
        raise ValueError("坏文件")
    if kind == "hang":
        while True:
            pass
    if kind == "crash":
        os.kill(os.getpid(), signal.SIGKILL)
