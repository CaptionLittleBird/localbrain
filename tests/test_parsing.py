import zipfile

import pytest

from localbrain.parsing import _CHAPTER, _NUMBERED, _as_text, _check_epub_drm, _clean, parse_file


def make_epub(path, drm=False):
    chap = lambda t, body: f'<?xml version="1.0" encoding="utf-8"?><html xmlns="http://www.w3.org/1999/xhtml"><head><title>{t}</title></head><body>{body}</body></html>'  # noqa: E731
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        z.writestr("META-INF/container.xml", '<?xml version="1.0"?><container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles></container>')
        z.writestr("OEBPS/content.opf", '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="2.0" unique-identifier="id"><metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>测试书</dc:title><dc:identifier id="id">x</dc:identifier></metadata><manifest><item id="c2" href="c2.xhtml" media-type="application/xhtml+xml"/><item id="c1" href="c1.xhtml" media-type="application/xhtml+xml"/></manifest><spine><itemref idref="c1"/><itemref idref="c2"/></spine></package>')
        z.writestr("OEBPS/c1.xhtml", chap("一", "<h1>第一章 进程</h1><p>进程是正在执行的程序。</p><h2>1.1 ps</h2><p>ps aux 列出进程。</p>"))
        z.writestr("OEBPS/c2.xhtml", chap("二", "<h1>第二章 文件系统</h1><p>inode 记录元数据。</p>"))
        if drm:
            z.writestr("META-INF/encryption.xml", '<encryption xmlns="urn:oasis:names:tc:opendocument:xmlns:container" xmlns:enc="http://www.w3.org/2001/04/xmlenc#"><enc:EncryptedData><enc:CipherData><enc:CipherReference URI="OEBPS/c1.xhtml"/></enc:CipherData></enc:EncryptedData></encryption>')


def test_epub_follows_spine_order_and_headings(tmp_path):
    path = tmp_path / "书.epub"
    make_epub(path)
    text = parse_file(path)[0].text
    assert text.index("第一章") < text.index("第二章")  # 按 spine 顺序，不是 manifest 顺序
    assert "## 1.1 ps" in text


def test_epub_with_drm_gives_clear_error(tmp_path):
    path = tmp_path / "drm.epub"
    make_epub(path, drm=True)
    with pytest.raises(ValueError, match="DRM"):
        _check_epub_drm(path)


def test_text_files_with_gbk_encoding(tmp_path):
    path = tmp_path / "a.txt"
    path.write_bytes("中文内容".encode("gbk"))
    assert parse_file(path)[0].text == "中文内容"


def test_chapter_numbering_patterns():
    for title in ["第四章、首次登入", "4.1 首次登入", "4.1.2 GNOME", "Chapter 3 Intro", "附录 A"]:
        assert _CHAPTER.match(title), title
    for title in ["1. 暂时关闭 quota", "一、概述"]:
        assert not _CHAPTER.match(title) and _NUMBERED.match(title), title
    for title in ["这个指令请特别小心", "2015年的数据"]:
        assert not _NUMBERED.match(title), title


def test_demoted_shell_prompt_is_escaped():
    # 书里的 "# 命令" 被认成标题，降级时不能变回一行标题
    assert _as_text("# 输入两次密码") == "\\# 输入两次密码"
    assert _as_text("普通句子") == "普通句子"


def test_clean_removes_icon_glyphs_and_image_placeholders():
    assert _clean(" groupdel\n\n<!-- image -->\n\n正文") == "groupdel\n\n正文"
