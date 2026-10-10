import random

import pytest

from localbrain.citations import CitationLinker, strip_links
from localbrain.store import DocRecord

BASE = "http://kb.test"


def rec(path, status="ok"):
    return DocRecord("x", path, path.split("/")[-1], 1, 1.0, "", 5, status, "", 0.0)


DOCS = [
    rec("书籍/【高清电子版】鸟哥的Linux私房菜-基础篇-第四版.pdf"),
    rec("工作/周报.docx"),
    rec("书/测试书.epub"),
    rec("a/笔记.md"),
    rec("b/笔记.md"),
    rec("坏的.pdf", "error"),
]


def link(text, chunks=None):
    lk = CitationLinker(DOCS, BASE)
    pieces = chunks or [text]
    return "".join(lk.feed(p) for p in pieces) + lk.flush()


def test_page_citation_with_shortened_name():
    out = link("见 [鸟哥的Linux私房菜-基础篇-第四版.pdf p.334-336]。")
    assert "/kb/open?path=" in out and "&page=334" in out
    assert out.endswith("。")


def test_section_citation():
    out = link("[测试书.epub § 2.1 inode]")
    assert "section=2.1%20inode" in out


def test_multiple_citations_in_one_bracket():
    out = link("[周报.docx；测试书.epub § 第一章]")
    assert out.count("/kb/open?") == 2


@pytest.mark.parametrize("text", [
    "命令行 [root@study ~]# ls",          # 不是出处
    "已有链接 [周报](http://x)",           # 本来就是链接
    "同名文件 [笔记.md] 不知道是哪个",       # 有歧义
    "失败的文档 [坏的.pdf]",               # 没入库成功
    "数组 a[0]",
])
def test_left_unchanged(text):
    assert link(text) == text


@pytest.mark.parametrize("text", [
    "用 tune2fs [鸟哥的Linux私房菜-基础篇-第四版.pdf p.12]，以及 [周报.docx] 和 [root@x ~]#",
    "未闭合 [abc\n下一行 [工作/周报.docx] 结尾",
])
def test_streaming_splits_give_same_result(text):
    expected = link(text)
    rnd = random.Random(0)
    for _ in range(50):
        cuts = sorted(rnd.sample(range(1, len(text)), k=min(len(text) - 1, rnd.randint(0, 10))))
        pieces = [text[i:j] for i, j in zip([0, *cuts], [*cuts, len(text)])]
        assert link(text, pieces) == expected


def test_strip_links_restores_plain_citation():
    original = "见 [周报.docx] 和 [测试书.epub § 2.1 inode]"
    assert strip_links(link(original)) == original
