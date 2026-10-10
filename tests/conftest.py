"""测试统一用不需要模型、不需要外部服务的配置：不用向量和重排模型，Qdrant 用嵌入模式。"""
import pytest

from localbrain import config, models


@pytest.fixture
def env(tmp_path, monkeypatch):
    """每个测试一套独立的文档目录、SQLite 和 Qdrant 目录。子进程（解析进程池）会继承这些环境变量。"""
    docs = tmp_path / "docs"
    docs.mkdir()
    values = {
        "DOCS_DIR": str(docs),
        "DB_PATH": str(tmp_path / "test.db"),
        "QDRANT_URL": str(tmp_path / "qdrant"),
        "COLLECTION": "test",
        "EMBED_BACKEND": "none",
        "RERANK_BACKEND": "none",
        "PDF_PARSER": "simple",
        "INGEST_WORKERS": "1",
        "WEB_PASSWORD": "pw",
    }
    for k, v in values.items():
        monkeypatch.setenv(k, v)
    _clear_caches()
    yield docs
    _clear_caches()


def _clear_caches():
    config.get_settings.cache_clear()
    models.get_embedder.cache_clear()
    models.get_reranker.cache_clear()
