"""向量模型和重排模型，后端由 EMBED_BACKEND / RERANK_BACKEND 决定（见 config.py）。

- omlx：调用 oMLX 的 /v1/embeddings 和 /v1/rerank，模型由 oMLX 下载和管理（Mac 上的默认方案）
- local：本进程用 sentence-transformers 加载，Mac 上走 MPS（GPU）
- none：不使用，get_embedder() / get_reranker() 返回 None
"""
from __future__ import annotations

import logging
import threading
from functools import lru_cache
from typing import Protocol

import httpx

from .config import SetupError, get_settings

log = logging.getLogger(__name__)


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]: ...


@lru_cache
def get_embedder() -> Embedder | None:
    s = get_settings()
    match s.embed_backend:
        case "omlx":
            log.info("向量模型：%s（oMLX %s）", s.embed_model, s.omlx_base_url)
            return ApiEmbedder(_omlx_client(), s.embed_model)
        case "local":
            device = pick_device(s.device)
            log.info("向量模型：%s（本地，%s）", s.embed_model, device)
            return LocalEmbedder(s.embed_model, device)
        case _:
            log.info("未启用向量模型，只做 BM25 关键词检索")
            return None


@lru_cache
def get_reranker() -> Reranker | None:
    s = get_settings()
    match s.rerank_backend:
        case "omlx":
            log.info("重排模型：%s（oMLX %s）", s.rerank_model, s.omlx_base_url)
            return ApiReranker(_omlx_client(), s.rerank_model)
        case "local":
            device = pick_device(s.device)
            log.info("重排模型：%s（本地，%s）", s.rerank_model, device)
            return LocalReranker(s.rerank_model, device)
        case _:
            log.info("未启用重排模型，直接使用混合检索的排序")
            return None


# ---------- oMLX（OpenAI 兼容接口）----------


class ApiClient:
    def __init__(self, base_url: str, api_key: str) -> None:
        self.base_url = base_url.rstrip("/")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = httpx.Client(base_url=self.base_url, headers=headers, timeout=120)

    def post(self, path: str, payload: dict) -> dict:
        try:
            resp = self._http.post(path, json=payload)
        except httpx.TransportError as e:
            raise SetupError(
                f"连不上模型服务 {self.base_url}（{e}）。请确认 oMLX 已启动、OMLX_BASE_URL 配置正确。"
            ) from e
        if resp.status_code >= 400:
            raise SetupError(
                f"模型服务返回 {resp.status_code}（{path}，模型 {payload.get('model')}）：{resp.text[:300]}"
            )
        return resp.json()


@lru_cache
def _omlx_client() -> ApiClient:
    s = get_settings()
    return ApiClient(s.omlx_base_url, s.omlx_api_key)


class ApiEmbedder:
    def __init__(self, client: ApiClient, model: str, batch_size: int = 32) -> None:
        self.client = client
        self.model = model
        self.batch_size = batch_size
        self.dim = len(self.embed(["dim"])[0])  # 请求一次，顺便确认服务可用并拿到向量维度

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            data = self.client.post("/embeddings", {"model": self.model, "input": batch})["data"]
            vectors.extend(d["embedding"] for d in sorted(data, key=lambda d: d["index"]))
        return vectors


class ApiReranker:
    def __init__(self, client: ApiClient, model: str) -> None:
        self.client = client
        self.model = model

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        results = self.client.post(
            "/rerank", {"model": self.model, "query": query, "documents": texts}
        )["results"]
        scores = [0.0] * len(texts)
        for r in results:
            scores[r["index"]] = float(r["relevance_score"])
        return scores


# ---------- 本进程加载（sentence-transformers）----------

# 同一进程内的本地模型推理串行执行：PyTorch 的 MPS 后端不适合多线程并发调用
_lock = threading.Lock()


def pick_device(preferred: str = "auto") -> str:
    if preferred != "auto":
        return preferred
    import torch

    if torch.cuda.is_available():
        return "cuda"  # Windows / Linux + NVIDIA 显卡
    if torch.backends.mps.is_available():
        return "mps"  # Apple Silicon：走 Metal GPU
    return "cpu"


class LocalEmbedder:
    def __init__(self, model_name: str, device: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = 1024  # bge-m3 默认 8192，切块远没那么长，限一下省内存
        if device != "cpu":
            self.model.half()
        self.dim: int = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        with _lock:
            vecs = self.model.encode(texts, batch_size=16, normalize_embeddings=True, convert_to_numpy=True)
        return vecs.tolist()


class LocalReranker:
    def __init__(self, model_name: str, device: str) -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, device=device, max_length=1024)
        if device != "cpu":
            self.model.model.half()

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        with _lock:
            scores = self.model.predict([(query, t) for t in texts], batch_size=16, show_progress_bar=False)
        return [float(s) for s in scores]
