"""对外接口：OpenAI 兼容的 /v1/chat/completions，Open WebUI 等客户端可以直接接入。"""
from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from .agent import build_agent, stream_answer
from .config import get_settings
from .retrieval import Retriever
from .store import Registry, VectorStore

log = logging.getLogger(__name__)

MODEL_ID = "localbrain"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    if not settings.llm_api_key:
        log.warning("没有配置 LLM_API_KEY，提问会失败。请在 .env 里填写。")
    registry = Registry(settings.db_path)
    store = VectorStore(settings.qdrant_url, settings.collection)
    retriever = Retriever(settings, store)  # 启动时就加载向量和重排模型，第一次提问不用等
    app.state.agent = build_agent(settings, retriever, registry, store)
    yield


app = FastAPI(title="localbrain", lifespan=lifespan)


def check_api_key(authorization: str = Header(default="")) -> None:
    if authorization != f"Bearer {get_settings().localbrain_api_key}":
        raise HTTPException(status_code=401, detail="invalid api key")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/v1/models", dependencies=[Depends(check_api_key)])
async def list_models() -> dict:
    return {
        "object": "list",
        "data": [{"id": MODEL_ID, "object": "model", "created": 0, "owned_by": "localbrain"}],
    }


@app.post("/v1/chat/completions", dependencies=[Depends(check_api_key)])
async def chat_completions(request: Request):
    body = await request.json()
    messages = to_langchain_messages(body.get("messages", []))
    agent = request.app.state.agent
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    if body.get("stream"):
        return StreamingResponse(
            _sse(_answer(agent, messages), completion_id, created), media_type="text/event-stream"
        )

    content = "".join([text async for text in _answer(agent, messages)])
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": MODEL_ID,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
        ],
    }


def to_langchain_messages(messages: list[dict]) -> list[BaseMessage]:
    converted: list[BaseMessage] = []
    for m in messages:
        content = _text_content(m.get("content"))
        match m.get("role"):
            case "system":
                converted.append(SystemMessage(content))
            case "user":
                converted.append(HumanMessage(content))
            case "assistant":
                converted.append(AIMessage(content))
    return converted


def _text_content(content) -> str:
    """content 可能是字符串，也可能是多模态的分段列表；这里只取文字部分。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""


async def _answer(agent, messages: list[BaseMessage]) -> AsyncIterator[str]:
    try:
        async for text in stream_answer(agent, messages):
            yield text
    except Exception as e:
        log.exception("回答失败")
        yield f"\n\n出错了：{type(e).__name__}: {e}"


async def _sse(texts: AsyncIterator[str], completion_id: str, created: int) -> AsyncIterator[str]:
    def event(delta: dict, finish_reason: str | None = None) -> str:
        payload = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": MODEL_ID,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    yield event({"role": "assistant", "content": ""})
    async for text in texts:
        yield event({"content": text})
    yield event({}, "stop")
    yield "data: [DONE]\n\n"
