from localbrain.retrieval import _dedupe
from localbrain.store import Hit


def hit(path, text):
    return Hit("d", path, 0, "", None, None, text)


def test_near_duplicates_removed_but_overlaps_kept():
    a = "硬链接共享同一个 inode，删除其中一个不影响另一个。软链接则是一个独立的文件，指向目标的路径。"
    hits = [hit("书.pdf", a), hit("书.epub", a.replace("，", ", ")), hit("书.pdf", "完全不同的内容，讲 inode 的结构。"), hit("书.epub", a[:30])]
    assert [h.path for h in _dedupe(hits)] == ["书.pdf", "书.pdf", "书.epub"]
