"""localbrain：个人本地知识库 + Agent 问答。"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]

# 尽早加载 .env：HF_ENDPOINT、代理等变量要在 huggingface_hub / torch 导入前生效
load_dotenv(ROOT / ".env")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # MPS 不支持的算子自动退回 CPU
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# 访问本机服务（Qdrant、oMLX）不走系统代理
_no_proxy = [p for p in os.environ.get("NO_PROXY", os.environ.get("no_proxy", "")).split(",") if p]
for _host in ("localhost", "127.0.0.1", "::1"):
    if _host not in _no_proxy:
        _no_proxy.append(_host)
os.environ["NO_PROXY"] = os.environ["no_proxy"] = ",".join(_no_proxy)
