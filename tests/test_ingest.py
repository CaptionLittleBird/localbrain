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
