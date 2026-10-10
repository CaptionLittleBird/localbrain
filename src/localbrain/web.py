"""文档管理页面 /kb：上传、查看入库状态、删除、重试、在线看原文。

页面只管文档目录里的文件，入库仍由 watch 服务完成：上传的文件落到 DOCS_DIR 后，watch 自动解析入库；
删除文件后 watch 自动从向量库移除。
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import os
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse
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


def require_login(request: Request) -> None:
    password = get_settings().web_password
    if not password:
        raise HTTPException(503, "没有设置 WEB_PASSWORD：在 .env 里设置后重启 serve 服务")
    if not hmac.compare_digest(request.cookies.get(_COOKIE, ""), _session_token(password)):
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
    response.set_cookie(
        _COOKIE, _session_token(password), max_age=30 * 86400, httponly=True, samesite="strict"
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
    docs, folders = [], set()
    for path in scan_docs(docs_dir):
        rel = path.relative_to(docs_dir).as_posix()
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        folder = path.parent.relative_to(docs_dir).as_posix()
        if folder != ".":
            folders.add(folder)
        rec = records.pop(rel, None)
        if rec is None or (rec.status != "indexing" and (rec.size, rec.mtime) != (stat.st_size, stat.st_mtime)):
            status = "pending"  # 新文件或改过的文件，等 watch 处理
        else:
            status = rec.status
        docs.append({
            "path": rel,
            "size": stat.st_size,
            "status": status,
            "chunks": rec.num_chunks if rec and status == "ok" else 0,
            "error": rec.error if rec and status == "error" else "",
            "updated_at": rec.updated_at if rec else stat.st_mtime,
        })
    for rec in records.values():  # 文件已删、记录还在：等 watch 从向量库移除
        docs.append({
            "path": rec.path, "size": rec.size, "status": "removing",
            "chunks": 0, "error": "", "updated_at": rec.updated_at,
        })
    return {"docs": docs, "folders": sorted(folders), "suffixes": sorted(SUPPORTED_SUFFIXES)}


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
