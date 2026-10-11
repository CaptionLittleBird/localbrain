"""文档管理页面 /kb：按文件夹管理文档（上传、新建文件夹、移动、重命名、删除），查看入库状态，在线看原文。

页面只管文档目录里的文件，入库仍由 watch 服务完成：上传的文件落到 DOCS_DIR 后，watch 自动解析入库；
删除文件后 watch 自动从向量库移除。
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import html
import os
import re
import shutil
import uuid
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from pydantic import BaseModel
from starlette.requests import ClientDisconnect

from .config import get_settings
from .ingest import is_ignored_name, scan_docs
from .parsing import SUPPORTED_SUFFIXES
from .store import Registry

router = APIRouter(prefix="/kb")

_COOKIE = "localbrain_session"
_MAX_UPLOAD = 2 << 30  # 单个文件最大 2GB
# 浏览器直接打开会执行脚本的格式，改为下载，免得上传的网页在本站点下运行
_ACTIVE_SUFFIXES = {".html", ".htm", ".xhtml", ".mhtml", ".mht", ".svg"}
_PAGE = Path(__file__).with_name("kb.html")


def _session_token(password: str) -> str:
    # 改了密码，之前登录的会话自动失效
    return hmac.new(password.encode(), b"localbrain-web-session", hashlib.sha256).hexdigest()


def _logged_in(request: Request) -> bool:
    password = get_settings().web_password
    return bool(password) and hmac.compare_digest(request.cookies.get(_COOKIE, ""), _session_token(password))


def require_login(request: Request) -> None:
    if not get_settings().web_password:
        raise HTTPException(503, "没有设置 WEB_PASSWORD：在 .env 里设置后重启 serve 服务")
    if not _logged_in(request):
        raise HTTPException(401, "请先登录")


def _registry(request: Request) -> Registry:
    return request.app.state.registry


def _docs_dir() -> Path:
    return get_settings().docs_dir


def _resolve(rel: str, *, upload: bool = False) -> Path:
    """把页面传来的相对路径转成文档目录里的绝对路径，拒绝 ..、隐藏文件和目录外的路径。"""
    parts = [p.strip() for p in rel.replace("\\", "/").split("/") if p.strip() not in ("", ".")]
    if not parts or any(p == ".." or p.startswith(".") for p in parts) or is_ignored_name(parts[-1]):
        raise HTTPException(400, f"路径不合法：{rel}")
    docs_dir = _docs_dir()
    path = docs_dir.joinpath(*parts)
    if not path.resolve().is_relative_to(docs_dir.resolve()):
        raise HTTPException(400, f"路径不合法：{rel}")
    if upload and path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise HTTPException(400, f"不支持 {path.suffix or '无扩展名'} 格式")
    return path


@router.get("", response_class=HTMLResponse, include_in_schema=False)
async def page() -> HTMLResponse:
    return HTMLResponse(_PAGE.read_text(encoding="utf-8"))


class Login(BaseModel):
    password: str


@router.post("/api/login")
async def login(body: Login, response: Response) -> dict:
    password = get_settings().web_password
    if not password:
        raise HTTPException(503, "没有设置 WEB_PASSWORD：在 .env 里设置后重启 serve 服务")
    if not hmac.compare_digest(body.password.encode(), password.encode()):
        await asyncio.sleep(1)  # 拖慢暴力猜密码
        raise HTTPException(401, "密码不对")
    # lax：从 Open WebUI 点出处链接跳过来时也带上登录状态；改数据的接口都不是 GET，跨站请求仍然带不上
    response.set_cookie(
        _COOKIE, _session_token(password), max_age=30 * 86400, httponly=True, samesite="lax"
    )
    return {"ok": True}


@router.post("/api/logout")
async def logout(response: Response) -> dict:
    response.delete_cookie(_COOKIE)
    return {"ok": True}


@router.get("/api/docs", dependencies=[Depends(require_login)])
def list_docs(request: Request) -> dict:
    """磁盘上的文件和入库记录合在一起看，得出每个文件当前的状态。"""
    docs_dir = _docs_dir()
    records = {r.path: r for r in _registry(request).all()}
    found = []
    for path in scan_docs(docs_dir):
        try:
            found.append((path.relative_to(docs_dir).as_posix(), path.stat()))
        except FileNotFoundError:
            continue
    on_disk = {rel for rel, _ in found}
    # 文件已经不在原处的记录。刚移动、改名的文件 watch 还没处理时，按大小和修改时间对上原来的记录，
    # 照样显示原来的状态，不会闪一下"等待入库"
    gone = {(r.size, r.mtime): r for p, r in records.items() if p not in on_disk}
    docs = []
    for rel, stat in found:
        rec = records.get(rel) or gone.pop((stat.st_size, stat.st_mtime), None)
        if rec is None or (rec.status != "indexing" and (rec.size, rec.mtime) != (stat.st_size, stat.st_mtime)):
            status = "pending"  # 新文件或改过的文件，等 watch 处理
        else:
            status = rec.status
        docs.append({
            "path": rel,
            "size": stat.st_size,
            "status": status,
            "chunks": rec.num_chunks if rec and status == "ok" else 0,
            "error": rec.error if rec and status in ("error", "duplicate") else "",
            "updated_at": rec.updated_at if rec else stat.st_mtime,
        })
    for rec in gone.values():  # 文件已删、记录还在：等 watch 从向量库移除
        docs.append({
            "path": rec.path, "size": rec.size, "status": "removing",
            "chunks": 0, "error": "", "updated_at": rec.updated_at,
        })
    return {"docs": docs, "folders": _folders(docs_dir), "suffixes": sorted(SUPPORTED_SUFFIXES)}


def _folders(docs_dir: Path) -> list[str]:
    """所有子文件夹，包括空的。"""
    out = []
    for p in docs_dir.rglob("*"):
        rel = p.relative_to(docs_dir)
        if p.is_dir() and not any(part.startswith(".") for part in rel.parts):
            out.append(rel.as_posix())
    return sorted(out)


def _folder(rel: str) -> Path:
    """文件夹路径，空字符串表示文档目录本身。"""
    return _docs_dir() if not rel.strip("/ ") else _resolve(rel)


def _rel(path: Path) -> str:
    return path.relative_to(_docs_dir()).as_posix()


class NewFolder(BaseModel):
    path: str


@router.post("/api/folders", dependencies=[Depends(require_login)])
def create_folder(body: NewFolder) -> dict:
    target = _resolve(body.path)
    if target.exists():
        raise HTTPException(409, "已经有同名的文件或文件夹")
    try:
        target.mkdir(parents=True)
    except OSError as e:
        raise HTTPException(400, f"创建失败：{e}") from e
    return {"path": _rel(target)}


@router.delete("/api/folders", dependencies=[Depends(require_login)])
def delete_folder(path: str, recursive: bool = False) -> dict:
    """删除文件夹。里面有文件时要 recursive=true 才删，文件会一起删掉并移出知识库。"""
    target = _resolve(path)
    if not target.is_dir():
        raise HTTPException(404, "文件夹不存在")
    files = [p for p in target.rglob("*") if p.is_file()]
    if files and not recursive:
        raise HTTPException(409, f"文件夹里还有 {len(files)} 个文件")
    shutil.rmtree(target)
    return {"deleted_files": len(files)}


class Move(BaseModel):
    paths: list[str]
    to: str  # 目标文件夹，空字符串是根目录


@router.post("/api/move", dependencies=[Depends(require_login)])
def move(body: Move) -> dict:
    """把文件或文件夹移到另一个文件夹。内容没变，watch 会认出是移动，不会重新入库。

    先全部检查一遍再动手，有一个不行就都不动。
    """
    dest_dir = _folder(body.to)
    if not dest_dir.is_dir():
        raise HTTPException(404, "目标文件夹不存在")
    plan = []
    for rel in dict.fromkeys(body.paths):
        src = _resolve(rel)
        if not src.exists():
            raise HTTPException(404, f"不存在：{rel}")
        dest = dest_dir / src.name
        if src.is_dir() and dest_dir.resolve().is_relative_to(src.resolve()):
            raise HTTPException(400, f"不能把文件夹移到它自己里面：{rel}")
        if dest == src:
            continue
        if dest.exists():
            raise HTTPException(409, f"目标文件夹里已经有「{src.name}」")
        plan.append((src, dest))
    for src, dest in plan:
        os.rename(src, dest)
    return {"moved": [_rel(d) for _, d in plan]}


class Rename(BaseModel):
    path: str
    name: str


@router.post("/api/rename", dependencies=[Depends(require_login)])
def rename(body: Rename) -> dict:
    src = _resolve(body.path)
    if not src.exists():
        raise HTTPException(404, "不存在")
    name = body.name.strip()
    if not name or "/" in name or "\\" in name or name in (".", "..") or is_ignored_name(name):
        raise HTTPException(400, "名字不合法（不能含 /，不能以 . 或 ~$ 开头）")
    dest = src.with_name(name)
    if src.is_file() and dest.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise HTTPException(400, f"改成 {dest.suffix or '无扩展名'} 就不会入库了，保留原来的扩展名吧")
    if dest.exists():
        raise HTTPException(409, "已经有同名的文件或文件夹")
    os.rename(src, dest)
    return {"path": _rel(dest)}


@router.put("/api/files", dependencies=[Depends(require_login)])
async def upload(request: Request, path: str, overwrite: bool = False) -> dict:
    """请求体就是文件内容，边收边写盘；先写到隐藏的临时文件，收完再改名，watch 不会读到半个文件。"""
    target = _resolve(path, upload=True)
    if target.exists() and not overwrite:
        raise HTTPException(409, "同名文件已存在")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise HTTPException(400, f"无法创建目录：{e}") from e
    tmp = target.with_name(f".{uuid.uuid4().hex}.uploading")
    size = 0
    try:
        with tmp.open("wb") as f:
            async for chunk in request.stream():
                size += len(chunk)
                if size > _MAX_UPLOAD:
                    raise HTTPException(413, "文件超过 2GB")
                f.write(chunk)
        os.replace(tmp, target)
    except ClientDisconnect:
        raise HTTPException(400, "上传中断") from None
    except OSError as e:
        raise HTTPException(400, f"保存失败：{e}") from e
    finally:
        tmp.unlink(missing_ok=True)
    return {"path": target.relative_to(_docs_dir()).as_posix(), "size": size}


@router.delete("/api/files", dependencies=[Depends(require_login)])
def delete(path: str) -> dict:
    target = _resolve(path)
    if not target.is_file():
        raise HTTPException(404, "文件不存在")
    target.unlink()
    return {"ok": True}


@router.post("/api/retry", dependencies=[Depends(require_login)])
def retry(path: str) -> dict:
    """更新文件的修改时间，watch 会当作文件变了，重新入库（失败过的文件也会重试）。"""
    target = _resolve(path)
    if not target.is_file():
        raise HTTPException(404, "文件不存在")
    os.utime(target)
    return {"ok": True}


def _to_login(request: Request) -> RedirectResponse:
    return RedirectResponse(f"/kb?next={quote(request.url.path + '?' + request.url.query, safe='')}")


@router.get("/open", include_in_schema=False)
def open_source(request: Request, path: str, page: int | None = None, section: str = "") -> RedirectResponse:
    """回答里出处链接指向这里：PDF 打开原文跳到对应页，其他格式打开阅读页定位到章节。没登录先去登录页，登录后再回来。"""
    if not _logged_in(request):
        return _to_login(request)
    target = _resolve(path)
    if not target.is_file():
        raise HTTPException(404, f"文件不存在（可能已删除或改名）：{path}")
    if target.suffix.lower() == ".pdf" and (page or not section):
        return RedirectResponse(f"/kb/api/raw?path={quote(path)}" + (f"#page={page}" if page else ""))
    url = f"/kb/view?path={quote(path)}"
    if page:
        url += f"&page={page}"
    if section:
        url += f"&section={quote(section)}"
    return RedirectResponse(url)


_NOT_WORD = re.compile(r"[\W_]+")
_MAX_SHOWN = 80


def _norm(text: str) -> str:
    return _NOT_WORD.sub("", text).lower()


def _pick_section(headings: list[str], section: str) -> str | None:
    """模型引用的章节可能是完整路径，也可能只有最后一级，按从严到宽的顺序找。"""
    key = _norm(section)
    if not key:
        return None
    for match in (
        lambda h: _norm(h) == key,
        lambda h: _norm(h.split(" > ")[-1]) == key,
        lambda h: _norm(h).endswith(key),
        lambda h: key in _norm(h),
    ):
        found = [h for h in headings if match(h)]
        if found:
            return found[0]
    return None


@router.get("/view", response_class=HTMLResponse, include_in_schema=False)
def view(request: Request, path: str, page: int | None = None, section: str = ""):
    """阅读页：按章节显示入库后的原文，左边是章节目录。给 EPUB、Word 这类没有页码的格式用。"""
    if not _logged_in(request):
        return _to_login(request)
    rec = _registry(request).by_path(path)
    if rec is None or rec.status != "ok":
        raise HTTPException(404, f"知识库里没有这篇文档：{path}")
    chunks = request.app.state.store.doc_chunks(rec.doc_id)
    headings = list(dict.fromkeys(c.heading for c in chunks))  # 按出现顺序去重
    note = ""
    if page:
        shown = [c for c in chunks if c.page_start is not None and c.page_start <= page <= (c.page_end or c.page_start)]
        current = shown[0].heading if shown else None
        title = f"第 {page} 页"
    else:
        current = _pick_section(headings, section) if section else None
        if section and current is None:
            note = f"没找到章节「{section}」，先显示文档开头，可以从目录里找。"
        current = current if current is not None else (headings[0] if headings else "")
        shown = [c for c in chunks if c.heading == current]
        title = current or "正文"
    if not shown:
        note = note or "没找到对应的内容，先显示文档开头。"
        shown = chunks[:5]
    if len(shown) > _MAX_SHOWN:  # 没有标题的长文档整篇都算一节，只显示前面一部分
        note = (note + " " if note else "") + f"这一节共 {len(shown)} 段，只显示前 {_MAX_SHOWN} 段，完整内容请打开原文件。"
        shown = shown[:_MAX_SHOWN]
    idx = headings.index(current) if current in headings else -1
    return HTMLResponse(_render_view(rec.path, title, note, shown, headings, idx))


def _toc(headings: list[str]) -> list[tuple[str, str]]:
    """目录条目 (标题路径, 点击后打开的章节)。只有小节没有正文的上级标题（比如"第二章"直接接"2.1"）
    也列出来，点它打开下面第一个有内容的小节。"""
    entries: dict[str, str] = {}
    for h in headings:
        parts = h.split(" > ")
        for depth in range(1, len(parts) + 1):
            entries.setdefault(" > ".join(parts[:depth]), h)
    return list(entries.items())


def _render_view(path: str, title: str, note: str, shown, headings: list[str], idx: int) -> str:
    e = html.escape
    link = lambda h: f"/kb/view?path={quote(path)}&section={quote(h)}"  # noqa: E731
    current = headings[idx] if idx >= 0 else None
    toc = "".join(
        f'<li class="{"cur" if label == current else ""}" style="padding-left:{label.count(" > ") * 12}px">'
        f'<a href="{e(link(target))}">{e(label.split(" > ")[-1] or "（开头）")}</a></li>'
        for label, target in _toc(headings)
    )
    nav = []
    if idx > 0:
        nav.append(f'<a href="{e(link(headings[idx - 1]))}">← 上一节</a>')
    if 0 <= idx < len(headings) - 1:
        nav.append(f'<a href="{e(link(headings[idx + 1]))}">下一节 →</a>')
    body = "".join(f'<div class="chunk">{e(c.text)}</div>' for c in shown)
    return _VIEW_PAGE.format(
        name=e(path.rsplit("/", 1)[-1]), path=e(path), raw=e(f"/kb/api/raw?path={quote(path)}"),
        title=e(title), note=f'<p class="note">{e(note)}</p>' if note else "", body=body, toc=toc,
        nav=" ".join(nav),
    )


_VIEW_PAGE = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{name}</title>
<style>
  :root {{ --bg:#f6f7f9; --card:#fff; --text:#1d2330; --muted:#6b7385; --line:#e3e6ec; --accent:#2f6fed; --hl:#e8effe; --warn:#a86400; }}
  @media (prefers-color-scheme: dark) {{ :root {{ --bg:#14171c; --card:#1c2027; --text:#e6e8ec; --muted:#9098a8; --line:#2c323c; --accent:#6b9bff; --hl:#1f2b44; --warn:#e7a94a; }} }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--text); font:15px/1.75 -apple-system, BlinkMacSystemFont, "PingFang SC", "Microsoft YaHei", sans-serif; }}
  .wrap {{ display:grid; grid-template-columns: 280px 1fr; gap:20px; max-width:1200px; margin:0 auto; padding:20px 16px; }}
  aside {{ position:sticky; top:16px; align-self:start; max-height:calc(100vh - 32px); overflow:auto; background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px; font-size:13px; }}
  aside ul {{ list-style:none; margin:0; padding:0; }}
  aside li {{ margin:2px 0; }}
  aside li.cur a {{ color:var(--accent); font-weight:600; }}
  a {{ color:var(--text); text-decoration:none; }} a:hover {{ color:var(--accent); }}
  main {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:20px 24px; min-width:0; }}
  .meta {{ color:var(--muted); font-size:13px; display:flex; justify-content:space-between; gap:12px; flex-wrap:wrap; }}
  .meta a {{ color:var(--accent); }}
  h1 {{ font-size:20px; margin:8px 0 16px; }}
  .chunk {{ white-space:pre-wrap; word-break:break-word; margin-bottom:14px; }}
  .note {{ color:var(--warn); font-size:13px; }}
  .nav {{ display:flex; justify-content:space-between; margin-top:20px; padding-top:12px; border-top:1px solid var(--line); }}
  .nav a {{ color:var(--accent); }}
  @media (max-width: 760px) {{ .wrap {{ grid-template-columns:1fr; }} aside {{ position:static; max-height:240px; }} }}
</style></head>
<body><div class="wrap">
<aside><ul>{toc}</ul></aside>
<main>
  <div class="meta"><span>{path}</span><a href="{raw}" target="_blank" rel="noopener">打开原文件</a></div>
  <h1>{title}</h1>
  {note}
  {body}
  <div class="nav">{nav}</div>
</main>
</div>
<script>document.querySelector("aside li.cur")?.scrollIntoView({{block: "center"}});</script>
</body></html>"""


@router.get("/api/raw", dependencies=[Depends(require_login)])
def raw(path: str) -> FileResponse:
    target = _resolve(path)
    if not target.is_file():
        raise HTTPException(404, "文件不存在")
    active = target.suffix.lower() in _ACTIVE_SUFFIXES
    return FileResponse(
        target,
        filename=target.name,
        content_disposition_type="attachment" if active else "inline",
        headers={"X-Content-Type-Options": "nosniff"},
    )
