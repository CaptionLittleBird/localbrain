"""配置：读取项目根目录的 .env 和环境变量（环境变量优先）。"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings

from . import ROOT


class Settings(BaseSettings):
    # 云端大模型，任何 OpenAI 兼容接口都行（DeepSeek / 通义 / Kimi …）
    llm_base_url: str = "https://api.deepseek.com"
    llm_api_key: str = ""
    llm_model: str = "deepseek-chat"
    llm_temperature: float = 0.3

    # 本服务的 HTTP 接口（oMLX 默认占用 8000，这里避开）
    localbrain_api_key: str = "change-me"
    api_host: str = "127.0.0.1"
    api_port: int = 8800
    web_password: str = ""  # 文档管理页面 /kb 的登录密码；不设置则页面不可用
    # 回答里出处链接的地址前缀，浏览器要能打开它。留空则按 API_HOST 推断：
    # 只监听本机时是 http://127.0.0.1:端口，监听局域网时是 http://<本机名>.local:端口
    public_url: str = ""

    # 知识库存储
    docs_dir: Path = Path("data/docs")
    db_path: Path = Path("data/localbrain.db")
    qdrant_port: int = 6333
    # 留空则连 http://127.0.0.1:{qdrant_port}；
    # 也可以填一个本地目录，走 Qdrant 嵌入模式：不用起 Docker，但只能单进程访问，适合开发调试
    qdrant_url: str = ""
    collection: str = "localbrain"

    # 向量 / 重排模型的后端：
    #   omlx  调用 oMLX（或其他 OpenAI 兼容服务）的 /v1/embeddings、/v1/rerank
    #   local 本进程用 sentence-transformers 加载，模型名填 HuggingFace ID
    #   none  不用：没有向量模型就只做 BM25 关键词检索，没有重排模型就直接用召回顺序
    embed_backend: Literal["omlx", "local", "none"] = "omlx"
    rerank_backend: Literal["omlx", "local", "none"] = "omlx"
    omlx_base_url: str = "http://127.0.0.1:8000/v1"
    omlx_api_key: str = ""
    embed_model: str = "bge-m3"
    rerank_model: str = "bge-reranker-v2-m3"
    device: str = "auto"  # 仅 local 后端使用：auto / mps / cuda / cpu

    # 文档解析
    pdf_parser: Literal["docling", "simple"] = "docling"  # simple 只提取文字，不需要模型
    docling_artifacts_path: str = ""  # Docling 模型的本地目录；留空则首次解析 PDF 时自动下载
    # 扫描版 PDF 和图片的 OCR，仅 macOS 生效（用系统自带的 Vision 框架）：
    #   auto   PDF 有文字层就不 OCR（OCR 占解析时间的八成），缺文字层的页较多才开；图片总是 OCR
    #   always PDF 里的插图、截图也 OCR，慢 4 倍左右
    #   never  不 OCR
    ocr: Literal["auto", "always", "never"] = "auto"

    # 入库
    ingest_workers: int = 2  # 并行解析的进程数；每个进程占 2～4GB 内存
    parse_timeout: int = 3600  # 单个文件解析超过这么多秒就放弃，防止卡住整个入库

    # 切块与检索
    chunk_size: int = 800  # 每块大约多少字符
    # 混合检索召回多少条交给重排。oMLX 重排是逐条算的，耗时和条数成正比（每条约 0.14 秒）；
    # 评估集上 15 条和 30 条命中率一样，速度快一倍。文档很多时可以适当调大
    candidates: int = 15
    top_k: int = 6  # 重排后返回给 Agent 的条数

    @field_validator("ocr", mode="before")
    @classmethod
    def _legacy_ocr(cls, v: Any) -> Any:
        """兼容旧配置 OCR=true / false。"""
        legacy = {"true": "auto", "1": "auto", "yes": "auto", "on": "auto",
                  "false": "never", "0": "never", "no": "never", "off": "never"}
        return legacy.get(str(v).strip().lower(), v)

    def model_post_init(self, context: Any, /) -> None:
        self.docs_dir = resolve_path(self.docs_dir)
        self.db_path = resolve_path(self.db_path)
        if not self.qdrant_url:
            self.qdrant_url = f"http://127.0.0.1:{self.qdrant_port}"
        if not self.public_url:
            host = "127.0.0.1" if self.api_host in ("127.0.0.1", "localhost", "::1") else _local_hostname()
            self.public_url = f"http://{host}:{self.api_port}"


class SetupError(RuntimeError):
    """配置或外部服务的问题，命令行只显示提示信息，不打印调用栈。"""


def _local_hostname() -> str:
    """局域网里用 Bonjour 名访问本机，比 IP 稳定（DHCP 换了 IP 也不受影响）。"""
    import socket

    name = socket.gethostname()
    return name if name.endswith(".local") or "." in name else f"{name}.local"


def resolve_path(p: str | Path) -> Path:
    """相对路径一律以项目根目录为准，不受启动目录影响。"""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


@lru_cache
def get_settings() -> Settings:
    return Settings()
