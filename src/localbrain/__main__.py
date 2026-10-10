"""支持 python -m localbrain ... 的调用方式，等同于 localbrain 命令。"""
from .cli import main

# 解析子进程用 spawn 启动时会重新导入主模块，没有这个判断就会再跑一遍命令
if __name__ == "__main__":
    main()
