"""Agent：用 LangGraph 搭的工具调用循环，由模型自己决定查几次、查什么。"""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from datetime import date

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from .config import Settings
from .retrieval import Retriever, format_hit
from .store import Registry, VectorStore

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是 localbrain，用户的个人知识库助手，基于用户自己的文档回答问题。今天是 {today}。
你的底层大模型是 {model}。被问到你是什么模型时，如实回答这一点，不要自称其他模型。

工作方式：
1. 涉及具体事实、数据、文档内容的问题，先用 search_kb 检索，不要凭记忆回答。
2. 一次检索不够就换关键词、拆成子问题多查几次；片段上下文不完整时用 read_document 读前后内容。
3. 想知道库里有哪些文档时用 list_documents。
4. 闲聊或常识问题可以直接回答，不必检索。

回答要求：
- 先给结论再展开。
- 引用知识库内容时在句末标注出处：有页码写 [文件名 p.页码]；没有页码写 [文件名 § 章节]，章节取检索结果里"章节："后面最后一级的标题；都没有就写 [文件名]。文件名照抄检索结果里"文件："后面的内容，不要缩写。出处会自动变成可以点开原文的链接，你不用自己写链接。
- 知识库里查不到就明确说"知识库中没有找到相关内容"。可以补充你的一般知识，但要说明这部分不是来自知识库。"""


def build_agent(settings: Settings, retriever: Retriever, registry: Registry, store: VectorStore):
    @tool
    def search_kb(query: str, doc_id: str | None = None) -> str:
        """在用户的个人知识库中检索相关片段（关键词 + 语义混合检索，并经过重排）。

        query：检索语句，用具体的关键词或完整问题。
        doc_id：可选，只在某一篇文档内检索（doc_id 来自检索结果或 list_documents）。
        """
        hits = retriever.search(query, doc_id=doc_id)
        if not hits:
            return "没有检索到相关内容。"
        return "\n\n---\n\n".join(format_hit(h) for h in hits)

    @tool
    def read_document(doc_id: str, start: int = 0, count: int = 5) -> str:
        """按顺序读取某篇文档的连续片段，用来查看检索结果的上下文，或通读文档开头。

        doc_id：文档 ID。
        start：起始片段序号，即检索结果里的"第 N 段"；想看上文就往前减 1~2。
        count：读取的片段数，最多 10。
        """
        rec = registry.get(doc_id)
        if rec is None:
            return f"没有 doc_id={doc_id} 的文档。"
        start, count = max(0, start), max(1, min(count, 10))
        hits = store.get_chunks(doc_id, start, count)
        if not hits:
            return f"{rec.path} 共 {rec.num_chunks} 段，第 {start} 段之后没有内容了。"
        body = "\n\n---\n\n".join(format_hit(h) for h in hits)
        return f"文件：{rec.path}（共 {rec.num_chunks} 段）\n\n{body}"

    @tool
    def list_documents(keyword: str = "") -> str:
        """列出知识库里的文档（路径、doc_id、片段数）。keyword 按文件路径过滤，留空列出全部（最多 100 篇）。"""
        docs = registry.search(keyword)
        if not docs:
            return "知识库里没有匹配的文档。"
        return "\n".join(f"- {d.path}（doc_id={d.doc_id}，{d.num_chunks} 段）" for d in docs)

    tools = [search_kb, read_document, list_documents]
    llm = ChatOpenAI(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key or "missing",
        temperature=settings.llm_temperature,
        timeout=120,
        max_retries=2,
    ).bind_tools(tools)

    async def call_model(state: MessagesState) -> dict:
        system = SystemMessage(SYSTEM_PROMPT.format(today=date.today().isoformat(), model=settings.llm_model))
        response = await llm.ainvoke([system, *state["messages"]])
        # 记下接口实际返回的模型名，用来核对到底是哪个模型在回答（模型的自我介绍不可靠）
        log.info("大模型应答：%s", response.response_metadata.get("model_name", "未知"))
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)  # 有工具调用 → tools，否则结束
    graph.add_edge("tools", "agent")
    return graph.compile()


async def stream_answer(
    agent, messages: list[BaseMessage], steps: list[BaseMessage] | None = None
) -> AsyncIterator[str]:
    """把 Agent 的执行过程转成给人看的文本流：工具调用显示成引用行，模型回答逐字输出。

    传了 steps 的话，中间的工具调用和工具结果会依次追加进去（不含最后的回答），供下一轮复用。
    """
    async for mode, data in agent.astream(
        {"messages": messages}, {"recursion_limit": 25}, stream_mode=["messages", "updates"]
    ):
        if mode == "messages":
            chunk, meta = data
            if (
                meta.get("langgraph_node") == "agent"
                and isinstance(chunk, AIMessageChunk)
                and isinstance(chunk.content, str)
                and chunk.content
            ):
                yield chunk.content
        elif mode == "updates":
            for node, update in data.items():
                if not update:
                    continue
                if steps is not None:
                    steps.extend(
                        m for m in update.get("messages", []) if not (isinstance(m, AIMessage) and not m.tool_calls)
                    )
                if node != "agent":
                    continue
                for msg in update.get("messages", []):
                    for call in getattr(msg, "tool_calls", None) or []:
                        yield f"\n> {_describe_call(call)}\n\n"


def _describe_call(call: dict) -> str:
    args = call.get("args") or {}
    match call["name"]:
        case "search_kb":
            scope = f"（限定文档 {args['doc_id']}）" if args.get("doc_id") else ""
            return f"检索：{args.get('query', '')}{scope}"
        case "read_document":
            return f"阅读文档 {args.get('doc_id')}，从第 {args.get('start', 0)} 段开始"
        case "list_documents":
            return f"查看文档列表：{args['keyword']}" if args.get("keyword") else "查看文档列表"
        case name:
            return f"调用工具 {name}"
