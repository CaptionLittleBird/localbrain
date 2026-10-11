from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient

from localbrain.config import get_settings
from localbrain.ingest import Ingestor


@pytest.fixture
def client(env):
    (env / "书").mkdir()
    (env / "书" / "笔记.md").write_text("# 第一章\n\n## 1.1 小节\n\n正文内容。", encoding="utf-8")
    ing = Ingestor(get_settings())
    ing.sync()
    ing.store.client.close()  # 嵌入模式的 Qdrant 同一目录只能有一个客户端，下面服务启动时要再打开
    from localbrain.server import app

    with TestClient(app) as c:
        yield c


def login(c, password="pw"):
    return c.post("/kb/api/login", json={"password": password})


def test_login_required(client):
    assert client.get("/kb/api/docs").status_code == 401
    assert login(client, "wrong").status_code == 401
    assert login(client).status_code == 200
    assert client.get("/kb/api/docs").json()["docs"][0]["path"] == "书/笔记.md"


@pytest.mark.parametrize("bad", ["../x.md", "a/../../x.md", ".hidden.md", "x/.git/a.md", "~$lock.docx", "a.exe"])
def test_upload_rejects_bad_paths(client, bad):
    login(client)
    assert client.put(f"/kb/api/files?path={quote(bad)}", content=b"x").status_code == 400


def test_upload_conflict_and_overwrite(client, env):
    login(client)
    url = f"/kb/api/files?path={quote('新/文档.md')}"
    assert client.put(url, content="内容".encode()).status_code == 200
    assert client.put(url, content=b"x").status_code == 409
    assert client.put(url + "&overwrite=true", content="新内容".encode()).status_code == 200
    assert (env / "新" / "文档.md").read_text(encoding="utf-8") == "新内容"
    assert not list(env.rglob("*.uploading"))


def test_citation_link_redirects_through_login_then_to_section(client):
    url = f"/kb/open?path={quote('书/笔记.md')}&section={quote('1.1 小节')}"
    r = client.get(url, follow_redirects=False)
    assert r.status_code == 307 and r.headers["location"].startswith("/kb?next=")
    login(client)
    r = client.get(url, follow_redirects=False)
    assert r.headers["location"].startswith("/kb/view?")
    page = client.get(r.headers["location"]).text
    assert "正文内容" in page and "1.1 小节" in page


def test_html_files_are_downloaded_not_rendered(client, env):
    (env / "page.html").write_text("<script>alert(1)</script>", encoding="utf-8")
    login(client)
    r = client.get("/kb/api/raw?path=page.html")
    assert r.headers["content-disposition"].startswith("attachment")


def test_folder_create_move_rename_delete(client, env):
    login(client)
    assert client.post("/kb/api/folders", json={"path": "归档/2026"}).status_code == 200
    assert client.post("/kb/api/folders", json={"path": "归档/2026"}).status_code == 409
    assert "归档/2026" in client.get("/kb/api/docs").json()["folders"]  # 空文件夹也列出来

    # 移动文件到文件夹；文件夹不能移到自己里面；目标重名时整批都不动
    assert client.post("/kb/api/move", json={"paths": ["书/笔记.md"], "to": "归档/2026"}).status_code == 200
    assert (env / "归档" / "2026" / "笔记.md").exists()
    assert client.post("/kb/api/move", json={"paths": ["归档"], "to": "归档/2026"}).status_code == 400
    (env / "书" / "笔记.md").write_text("同名", encoding="utf-8")
    (env / "书" / "另一篇.md").write_text("x", encoding="utf-8")
    r = client.post("/kb/api/move", json={"paths": ["书/另一篇.md", "书/笔记.md"], "to": "归档/2026"})
    assert r.status_code == 409 and (env / "书" / "另一篇.md").exists()

    # 重命名：不能改成不支持的扩展名、不能带 /
    assert client.post("/kb/api/rename", json={"path": "归档/2026/笔记.md", "name": "笔记.exe"}).status_code == 400
    assert client.post("/kb/api/rename", json={"path": "归档/2026/笔记.md", "name": "a/b.md"}).status_code == 400
    assert client.post("/kb/api/rename", json={"path": "归档", "name": "旧资料"}).status_code == 200
    assert (env / "旧资料" / "2026" / "笔记.md").exists()

    # 删除非空文件夹要确认
    r = client.delete(f"/kb/api/folders?path={quote('旧资料')}")
    assert r.status_code == 409 and "1 个文件" in r.json()["detail"]
    assert client.delete(f"/kb/api/folders?path={quote('旧资料')}&recursive=true").json() == {"deleted_files": 1}
    assert not (env / "旧资料").exists()


def test_moved_file_keeps_status_before_watch_runs(client, env):
    login(client)
    client.post("/kb/api/move", json={"paths": ["书/笔记.md"], "to": ""})
    docs = client.get("/kb/api/docs").json()["docs"]
    assert [(d["path"], d["status"]) for d in docs] == [("笔记.md", "ok")]
