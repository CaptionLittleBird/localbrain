"""文档解析：TXT / MD 直接读取，其余格式交给 Docling 转成 Markdown。

- .doc / .rtf 在 macOS 上先用系统自带的 textutil 转成 HTML，不需要 LibreOffice
- .xls / .ppt（以及非 macOS 上的 .doc / .rtf）要装 LibreOffice，由 Docling 调用它转成新格式
- 图片、扫描版 PDF 用 OCR 识别文字，仅 macOS 可用
"""
from __future__ import annotations

import io
import logging
import os
import re
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .config import get_settings, resolve_path

log = logging.getLogger(__name__)

TEXT_SUFFIXES = {".txt", ".md", ".markdown"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
# 按页标注页码的格式：PDF 是页码，演示文稿是第几张幻灯片
PAGED_SUFFIXES = {".pdf", ".pptx", ".ppsx", ".pptm", ".ppt", ".odp", ".key"}
# 老格式：macOS 上用 textutil 转换，其他系统要 LibreOffice
TEXTUTIL_SUFFIXES = {".doc", ".rtf"}
LIBREOFFICE_SUFFIXES = {".xls", ".ppt"}
DOCLING_SUFFIXES = {
    ".pdf",
    ".docx", ".docm", ".pptx", ".ppsx", ".pptm", ".xlsx", ".xlsm", ".csv",  # Office
    ".odt", ".ods", ".odp",  # OpenDocument
    ".pages", ".key", ".numbers",  # iWork
    ".epub", ".html", ".htm", ".xhtml", ".mhtml", ".mht",
    ".eml", ".msg", ".adoc", ".asciidoc", ".tex",
    *TEXTUTIL_SUFFIXES, *LIBREOFFICE_SUFFIXES, *IMAGE_SUFFIXES,
}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | DOCLING_SUFFIXES

# 自动 OCR 的判断标准：一页少于这么多字符算没有文字层；没有文字层的页超过这个比例才 OCR
_TEXTLESS_CHARS = 10
_TEXTLESS_RATIO = 0.2


@dataclass
class Section:
    """解析结果的一段：分页的格式按页给出（用于标注页码），其他格式整篇一段。"""

    text: str
    page: int | None = None


def parse_file(path: Path) -> list[Section]:
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return [Section(_read_text(path))]
    if suffix == ".pdf" and get_settings().pdf_parser == "simple":
        return _parse_pdf_simple(path)
    if suffix == ".epub":
        _check_epub_drm(path)

    from docling.datamodel.base_models import ConversionStatus, DocumentStream

    source: Path | DocumentStream = path
    if suffix in TEXTUTIL_SUFFIXES and sys.platform == "darwin":
        source = DocumentStream(name=f"{path.stem}.html", stream=io.BytesIO(_textutil_html(path)))
    elif suffix in TEXTUTIL_SUFFIXES | LIBREOFFICE_SUFFIXES and not _libreoffice():
        raise RuntimeError(
            f"解析 {suffix} 需要安装 LibreOffice（Mac 上 brew install --cask libreoffice），"
            "或者把文件另存为 .docx / .xlsx / .pptx"
        )

    result = _converter(_use_ocr(path, suffix)).convert(source)
    if result.status != ConversionStatus.SUCCESS:
        log.warning("%s 只解析成功了一部分：%s", path.name, "; ".join(e.error_message for e in result.errors))
    if suffix == ".pdf":
        return _split_pages(result.document, _heading_filter(path))
    if suffix in PAGED_SUFFIXES:
        return _export_by_page(result.document)
    return [Section(_clean(result.document.export_to_markdown()))]


_PAGE_BREAK = "<!-- localbrain:page-break -->"


def _split_pages(doc, fix=None) -> list[Section]:
    """整本导出一次，再按分页标记拆回各页。

    不用 export_to_markdown(page_no=p) 逐页导出：每次调用都要遍历全文，几百页的书要几十分钟。
    Docling 只在有内容的页之间插入分页标记，所以拆出来的段和有内容的页按顺序一一对应。
    """
    markdown = doc.export_to_markdown(page_break_placeholder=_PAGE_BREAK)
    parts = (fix(markdown) if fix else markdown).split(_PAGE_BREAK)
    pages = sorted({item.prov[0].page_no for item, _ in doc.iterate_items() if getattr(item, "prov", None)})
    if len(parts) != len(pages):
        log.warning("分页标记和页码对不上（%d 段 / %d 页），这份文档不标注页码", len(parts), len(pages))
        return [Section(_clean("\n\n".join(parts)))]
    return [Section(_clean(text), page=page) for page, text in zip(pages, parts)]


def _export_by_page(doc) -> list[Section]:
    """演示文稿逐页导出。Docling 给演示文稿插的分页标记位置不对（会错开一页），不能用 _split_pages 的办法；
    幻灯片页数不多，逐页导出也不慢。"""
    pages = sorted({item.prov[0].page_no for item, _ in doc.iterate_items() if getattr(item, "prov", None)})
    sections = [Section(_clean(doc.export_to_markdown(page_no=p)), page=p) for p in pages]
    return [s for s in sections if s.text]


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "gb18030"):  # Windows 上保存的中文 txt 常是 GBK
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _parse_pdf_simple(path: Path) -> list[Section]:
    """只提取文字层，不识别标题和表格结构，也不做 OCR；好处是不需要任何模型。"""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(path)
    try:
        return [
            Section(page.get_textpage().get_text_range().replace("\r\n", "\n").strip(), page=i)
            for i, page in enumerate(pdf, start=1)
        ]
    finally:
        pdf.close()


def _use_ocr(path: Path, suffix: str) -> bool:
    mode = get_settings().ocr
    if sys.platform != "darwin" or mode == "never":
        return False
    if suffix in IMAGE_SUFFIXES or mode == "always":
        return True
    if suffix != ".pdf":
        return False
    textless, total = _count_textless_pages(path)
    if total and textless / total > _TEXTLESS_RATIO:
        log.info("%s：%d/%d 页没有文字层，按扫描件处理，开启 OCR", path.name, textless, total)
        return True
    return False


def _count_textless_pages(path: Path) -> tuple[int, int]:
    """数一下没有文字层的页。只读文字层，上千页的书也就一两秒。"""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(path)
    try:
        textless = 0
        for page in pdf:
            textpage = page.get_textpage()
            textless += textpage.count_chars() < _TEXTLESS_CHARS
            textpage.close()
            page.close()
        return textless, len(pdf)
    finally:
        pdf.close()


def _textutil_html(path: Path) -> bytes:
    """用 macOS 自带的 textutil 把 .doc / .rtf 转成 HTML，表格能保留下来。"""
    try:
        proc = subprocess.run(
            ["textutil", "-convert", "html", "-stdout", str(path)], capture_output=True, timeout=300
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError("textutil 转换超时") from e
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(f"textutil 转换失败：{proc.stderr.decode(errors='replace').strip()[:300]}")
    return proc.stdout


def _libreoffice() -> str | None:
    """和 Docling 查找 LibreOffice 的顺序一致。"""
    mac_app = "/Applications/LibreOffice.app/Contents/MacOS/soffice"
    return (
        os.environ.get("DOCLING_LIBREOFFICE_CMD")
        or shutil.which("libreoffice")
        or shutil.which("soffice")
        or (mac_app if os.path.isfile(mac_app) else None)
    )


_FONT_SUFFIXES = (".ttf", ".otf", ".woff", ".woff2")


def _check_epub_drm(path: Path) -> None:
    """有 DRM 的 EPUB 正文是密文，解析出来是空的或乱码，提前给出明确的原因。

    encryption.xml 里只加密字体是正常的（出版社的字体混淆），加密了正文才算 DRM。
    """
    import defusedxml.ElementTree as ET

    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        if "META-INF/rights.xml" in names:  # Adobe ADEPT
            raise ValueError("EPUB 有 DRM 加密（Adobe），读不到正文，需要先去掉 DRM")
        if "META-INF/encryption.xml" not in names:
            return
        root = ET.fromstring(z.read("META-INF/encryption.xml"))
    encrypted = [el.get("URI", "") for el in root.iter() if el.tag.endswith("CipherReference")]
    if any(not uri.lower().endswith(_FONT_SUFFIXES) for uri in encrypted):
        raise ValueError("EPUB 有 DRM 加密，读不到正文，需要先去掉 DRM")


# 明确的章节编号：第三章、3.2、3.2.1、Chapter 3、附录 A
_CHAPTER = re.compile(r"(第[一二三四五六七八九十百零〇\d]+[章节篇部卷回]|\d+(\.\d+)+|(chapter|part|appendix)\s|附录)", re.I)
# 宽松的编号：再加上 1.、一、(一) 这类，也可能是正文里的步骤，只在没有书签的文档里用
_NUMBERED = re.compile(
    r"(第[一二三四五六七八九十百零〇\d]+[章节篇部卷回]|\d+(\.\d+)*[.、．\s]|\d+(\.\d+)+|"
    r"[一二三四五六七八九十]+[、．.]|[(（][一二三四五六七八九十\d]+[)）]|"
    r"(chapter|part|section|appendix)\s|附录|前言|序言|序$|目录|参考文献|索引)",
    re.IGNORECASE,
)
_MD_HEADING = re.compile(r"^(#{1,6})[ \t]+(\S.*)$", re.MULTILINE)
_NOT_WORD = re.compile(r"[\W_]+")


def _heading_filter(path: Path):
    """PDF 的标题是版面模型"看"出来的，书里的提示框、强调句常被误认成标题，再被当成章节挂在后面内容的标题路径上。

    文档有明确的结构时（有书签，或者多数标题带章节编号），只保留和书签对得上、或者带编号的标题，
    其余的降为普通段落。有书签时"1."这种单级编号多半是正文里的操作步骤，不算。没有明确结构的文档（论文、报告）不动。
    """
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(path)
    try:
        bookmarks = {_norm(b.get_title()) for b in pdf.get_toc()} - {""}
    finally:
        pdf.close()

    def trusted(title: str) -> bool:
        if (_CHAPTER if bookmarks else _NUMBERED).match(title):
            return True
        key = _norm(title)
        return bool(key) and any(key == b or (len(key) >= 4 and (b.startswith(key) or key.startswith(b))) for b in bookmarks)

    def fix(markdown: str) -> str:
        titles = [m.group(2) for m in _MD_HEADING.finditer(markdown)]
        numbered = sum(1 for t in titles if _NUMBERED.match(t))
        if len(bookmarks) < 5 and numbered * 2 < len(titles):
            return markdown
        return _MD_HEADING.sub(lambda m: m.group(0) if trusted(m.group(2)) else _as_text(m.group(2)), markdown)

    return fix


def _as_text(title: str) -> str:
    # 书里的命令行常以 # 开头（root 提示符、注释），去掉标题标记后要转义，不然还是一行标题
    return "\\" + title if title.startswith("#") else title


def _norm(text: str) -> str:
    return _NOT_WORD.sub("", text).lower()


_IMAGE_PLACEHOLDER = re.compile(r"^\s*<!-- image -->\s*$", re.MULTILINE)
_PRIVATE_USE = re.compile("[\ue000-\uf8ff]+[ \t]*")  # 图标字体里的符号（项目符号等），提取出来是乱码


def _clean(markdown: str) -> str:
    return _PRIVATE_USE.sub("", _IMAGE_PLACEHOLDER.sub("", markdown)).strip()


@lru_cache
def _converter(ocr: bool):
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, ImageFormatOption, PdfFormatOption

    s = get_settings()
    opts = PdfPipelineOptions()
    if s.docling_artifacts_path:
        opts.artifacts_path = resolve_path(s.docling_artifacts_path)
    opts.do_table_structure = True
    # 版面模型只认得"这是标题"，分不出级别；按 PDF 书签和章节编号推断标题层级，切块时能带上"第几章 > 第几节"
    opts.heading_hierarchy_options.enabled = True
    # OCR 用系统自带的 Vision 框架，支持中文，不用另外下载 OCR 模型
    opts.do_ocr = ocr
    if ocr:
        from docling.datamodel.pipeline_options import OcrMacOptions

        opts.ocr_options = OcrMacOptions(lang=["zh-Hans", "en-US"])
    # 加速设备默认是 AUTO：Mac 上自动用 MPS（GPU），没有 N 卡的 Windows 用 CPU
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=opts),
            InputFormat.IMAGE: ImageFormatOption(pipeline_options=opts),
        },
    )
