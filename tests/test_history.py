from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from localbrain.history import ToolHistory


def test_restore_previous_turn_steps_despite_formatting_changes():
    h = ToolHistory()
    q1 = [HumanMessage("怎么查看分区 UUID？")]
    steps = [AIMessage("", tool_calls=[{"name": "search_kb", "args": {"query": "UUID"}, "id": "c1"}]),
             ToolMessage("blkid 可以列出 UUID", tool_call_id="c1")]
    h.remember(q1, "> 检索：UUID\n\n用 blkid。", steps)
    # Open WebUI 发回来的历史，空白和换行可能不一样
    nxt = [*q1, AIMessage(">  检索：UUID\n用 blkid。 "), HumanMessage("那第二个呢？")]
    restored, n = h.restore(nxt)
    assert n == 1
    assert [type(m).__name__ for m in restored] == ["HumanMessage", "AIMessage", "ToolMessage", "AIMessage", "HumanMessage"]


def test_unknown_history_is_left_alone():
    h = ToolHistory()
    msgs = [HumanMessage("a"), AIMessage("b"), HumanMessage("c")]
    assert h.restore(msgs) == (msgs, 0)
