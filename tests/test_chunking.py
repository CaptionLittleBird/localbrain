import time

from localbrain.chunking import _heading, chunk_sections
from localbrain.parsing import Section


def test_heading_levels_and_trailing_hashes():
    assert _heading("# 标题") == (1, "标题")
    assert _heading("## 2.1 inode ##") == (2, "2.1 inode")
    assert _heading("#标题") is None
    assert _heading("####### 七级") is None
    assert _heading("    # 缩进的代码注释") is None
    assert _heading("# " + "长" * 300) is None


def test_heading_regex_is_linear_time():
    # 旧的正则在这种行上会回溯到卡死
    start = time.time()
    _heading("# x" + " " * 50000 + "y")
    _heading("# x" + " " * 100 + "y")
    assert time.time() - start < 0.5
    assert _heading("# x" + " " * 100 + "y") == (1, "x" + " " * 100 + "y")


def test_heading_path_nesting():
    md = "# 第一章\n\n正文A\n\n## 1.1 小节\n\n正文B\n\n# 第二章\n\n正文C"
    got = [(c.heading, c.text) for c in chunk_sections([Section(md)])]
    assert got == [("第一章", "正文A"), ("第一章 > 1.1 小节", "正文B"), ("第二章", "正文C")]


def test_hash_lines_inside_code_fence_are_not_headings():
    md = "# 第一章\n\n```\n# yum install\n\n# ls -l\n```\n\n## 1.1 小节\n\n正文"
    chunks = chunk_sections([Section(md)])
    assert [c.heading for c in chunks] == ["第一章", "第一章 > 1.1 小节"]
    assert "# yum install" in chunks[0].text


def test_long_table_split_repeats_header():
    rows = "\n".join(f"| {i} | {'x' * 40} |" for i in range(60))
    md = "| 序号 | 内容 |\n|---|---|\n" + rows
    chunks = chunk_sections([Section(md)], chunk_size=500)
    assert len(chunks) > 1
    assert all(c.text.startswith("| 序号 | 内容 |\n|---|---|") for c in chunks)


def test_page_numbers_follow_sections():
    chunks = chunk_sections([Section("第一页内容", page=1), Section("第二页内容", page=2)], chunk_size=6)
    assert [(c.page_start, c.page_end) for c in chunks] == [(1, 1), (2, 2)]
