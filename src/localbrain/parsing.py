"""文档解析：PDF / DOCX 交给 Docling 转成 Markdown，TXT / MD 直接读取。"""
from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .config import get_settings, resolve_path

log = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".txt", ".md"}


@dataclass
class Section:
    """解析结果的一段：PDF 按页给出（用于标注页码），其他格式整篇一段。"""

    text: str
    page: int | None = None


def parse_file(path: Path) -> list[Section]:
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md"):
        return [Section(_read_text(path))]
    if suffix == ".pdf" and get_settings().pdf_parser == "simple":
        return _parse_pdf_simple(path)
    # DOCX 由 Docling 直接读取文档结构，不需要模型；PDF 要跑版面分析和表格识别模型
    doc = _converter().convert(path).document
    if suffix == ".pdf":
        return _split_pages(doc)
    return [Section(_clean(doc.export_to_markdown()))]


_PAGE_BREAK = "<!-- localbrain:page-break -->"


def _split_pages(doc) -> list[Section]:
    """整本导出一次，再按分页标记拆回各页。

    不用 export_to_markdown(page_no=p) 逐页导出：每次调用都要遍历全文，几百页的书要几十分钟。
    Docling 只在有内容的页之间插入分页标记，所以拆出来的段和有内容的页按顺序一一对应。
    """
    parts = doc.export_to_markdown(page_break_placeholder=_PAGE_BREAK).split(_PAGE_BREAK)
    pages = sorted({item.prov[0].page_no for item, _ in doc.iterate_items() if getattr(item, "prov", None)})
    if len(parts) != len(pages):
        log.warning("分页标记和页码对不上（%d 段 / %d 页），这份 PDF 不标注页码", len(parts), len(pages))
        return [Section(_clean("\n\n".join(parts)))]
    return [Section(_clean(text), page=page) for page, text in zip(pages, parts)]


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


_IMAGE_PLACEHOLDER = re.compile(r"^\s*<!-- image -->\s*$", re.MULTILINE)


def _clean(markdown: str) -> str:
    return _IMAGE_PLACEHOLDER.sub("", markdown).strip()


@lru_cache
def _converter():
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    s = get_settings()
    opts = PdfPipelineOptions()
    if s.docling_artifacts_path:
        opts.artifacts_path = resolve_path(s.docling_artifacts_path)
    opts.do_table_structure = True
    # OCR 只在 macOS 上开：用系统自带的 Vision 框架，支持中文，不用另外下载 OCR 模型
    opts.do_ocr = s.ocr and sys.platform == "darwin"
    if opts.do_ocr:
        from docling.datamodel.pipeline_options import OcrMacOptions

        opts.ocr_options = OcrMacOptions(lang=["zh-Hans", "en-US"])
    # 加速设备默认是 AUTO：Mac 上自动用 MPS（GPU），没有 N 卡的 Windows 用 CPU
    return DocumentConverter(
        allowed_formats=[InputFormat.PDF, InputFormat.DOCX],
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)},
    )
