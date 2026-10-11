import pytest

from localbrain.config import SetupError, get_settings
from localbrain.ingest import Ingestor


def status(ing):
    return {r.path: (r.status, r.num_chunks) for r in ing.registry.all()}


def test_ingest_update_delete_and_duplicates(env):
    (env / "a.md").write_text("# 标题\n\n第一段内容。", encoding="utf-8")
    (env / "sub").mkdir()
    (env / "sub" / "b.txt").write_text("另一篇。", encoding="utf-8")
    (env / ".hidden.md").write_text("隐藏文件不入库", encoding="utf-8")
    (env / "empty.txt").write_text("", encoding="utf-8")
    ing = Ingestor(get_settings())
    ing.sync()
    assert status(ing) == {"a.md": ("ok", 1), "sub/b.txt": ("ok", 1), "empty.txt": ("error", 0)}

    # 完全相同的副本不重复入库；原件删了副本顶上
    (env / "a-copy.md").write_bytes((env / "a.md").read_bytes())
    ing.sync()
    assert status(ing)["a-copy.md"][0] == "duplicate"
    (env / "a.md").unlink()
    ing.sync()
    assert status(ing)["a-copy.md"] == ("ok", 1) and "a.md" not in status(ing)
    assert ing.store.count() == 2


def test_empty_or_missing_docs_dir_does_not_wipe_index(env):
    (env / "a.md").write_text("内容", encoding="utf-8")
    ing = Ingestor(get_settings())
    ing.sync()
    (env / "a.md").rename(env.parent / "a.md")  # 目录变空
    ing.sync()
    assert "a.md" in status(ing)
    env.rmdir()  # 目录不见了（外接硬盘没挂载）
    with pytest.raises(SetupError):
        ing.sync()
    assert "a.md" in status(ing)


def test_reingest_when_vector_store_was_wiped(env):
    (env / "a.md").write_text("内容", encoding="utf-8")
    ing = Ingestor(get_settings())
    ing.sync()
    ing.store.drop()
    ing.store.ensure_collection(None)
    ing.sync()
    assert ing.store.count() == 1


def test_files_still_being_written_are_deferred(env):
    (env / "a.md").write_text("刚写入", encoding="utf-8")
    ing = Ingestor(get_settings())
    assert ing.sync(settle=60) is True
    assert status(ing) == {}


def test_moving_and_renaming_does_not_reingest(env, monkeypatch):
    (env / "a.md").write_text("# 标题\n\n正文内容。", encoding="utf-8")
    (env / "书").mkdir()
    (env / "书" / "b.md").write_text("另一篇。", encoding="utf-8")
    ing = Ingestor(get_settings())
    ing.sync()
    before = ing.store.count()

    def no_reingest(jobs):
        raise AssertionError(f"不该重新入库：{[j.rel for j in jobs]}")

    monkeypatch.setattr(ing, "_ingest_all", no_reingest)
    (env / "归档").mkdir()
    (env / "a.md").rename(env / "归档" / "a-改名.md")   # 移动 + 改名
    (env / "书").rename(env / "图书")                    # 整个文件夹改名
    ing.sync()
    assert status(ing) == {"归档/a-改名.md": ("ok", 1), "图书/b.md": ("ok", 1)}
    assert ing.store.count() == before
    paths = {h.path for h in ing.store.doc_chunks(ing.registry.by_path("归档/a-改名.md").doc_id)}
    assert paths == {"归档/a-改名.md"}


def test_moved_and_modified_file_is_reingested(env):
    (env / "a.md").write_text("旧内容", encoding="utf-8")
    ing = Ingestor(get_settings())
    ing.sync()
    (env / "a.md").unlink()
    (env / "b.md").write_text("新内容不一样", encoding="utf-8")
    ing.sync()
    assert status(ing) == {"b.md": ("ok", 1)}
