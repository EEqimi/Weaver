# tests/test_cleaner.py
"""Gutenberg 清洗：头尾/TOC/图注去除、Unicode/换行归一化、段落保留。"""
import unicodedata

from knowledge.corpus.cleaner import clean
from knowledge.corpus.discover import discover
from knowledge.corpus.metadata import CORPUS

GUTENBERG_SAMPLE = """The Project Gutenberg eBook of Foo\r\n\r\nThis eBook is for the use of anyone anywhere in the United States and\r\nmost other parts of the world at no cost and with almost no restrictions.\r\n\r\n*** START OF THE PROJECT GUTENBERG EBOOK FOO ***\r\n\r\nFoo\r\n\r\nby Some Author\r\n\r\nCONTENTS\r\n\r\nCHAPTER I. A Beginning\r\nCHAPTER II. The Middle\r\n\r\nCHAPTER I. A Beginning\r\n\r\nIt was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness.\r\n\r\nCHAPTER II. The Middle\r\n\r\nThis is the second chapter, and it has plenty of prose to be detected as real text by the cleaner.\r\n\r\n*** END OF THE PROJECT GUTENBERG EBOOK FOO ***\r\n\r\nwww.gutenberg.org license text.\r\n"""


def test_clean_removes_header_footer_and_toc():
    c = clean(GUTENBERG_SAMPLE, "foo")
    assert c.header_removed and c.footer_removed and c.front_matter_removed
    assert "Project Gutenberg" not in c.text
    assert "www.gutenberg.org" not in c.text
    assert "CONTENTS" not in c.text
    assert "by Some Author" not in c.text          # 标题/作者页被去除
    assert c.text.startswith("CHAPTER I. A Beginning")  # 正文首章，而非 TOC


def test_clean_normalizes_newlines_and_unicode():
    decomposed = "café"  # e + 组合重音
    raw = ("*** START OF THE PROJECT GUTENBERG EBOOK X ***\n\nCHAPTER I\n\n"
           + decomposed + " is " + "fiancé" + " and " + "naïve" + ".\n\n"
           + "*** END OF THE PROJECT GUTENBERG EBOOK X ***")
    c = clean(raw, "x")
    assert "café" in c.text                       # NFC 合成
    assert "\r" not in c.text


def test_clean_preserves_paragraphs_and_punctuation():
    c = clean(GUTENBERG_SAMPLE, "foo")
    # 段落边界：两个章节各一段，正文段落用空行分隔
    assert "\n\n" in c.text
    # 标点保留
    assert "It was the best of times," in c.text
    assert "foolishness." in c.text


_OPENINGS = {
    "pride_and_prejudice": "It is a truth universally acknowledged",
    "emma": "Emma Woodhouse, handsome, clever",
    "persuasion": "Sir Walter Elliot, of Kellynch Hall",
    "great_expectations": "My father",
    "david_copperfield": "Whether I shall turn out to be the hero",
    "tale_of_two_cities": "It was the best of times",
}


def test_real_corpus_cleaning(real_corpus_root):
    found = discover(real_corpus_root)
    for m in CORPUS:
        raw = found[m.work_id].read_text(encoding="utf-8")
        c = clean(raw, m.work_id)
        assert c.header_removed and c.footer_removed
        low = c.text.lower()
        for residue in ("project gutenberg", "www.gutenberg.org",
                        "*** start of", "*** end of", "[illustration"):
            assert residue not in low, f"{m.work_id} 残留: {residue!r}"
        assert _OPENINGS[m.work_id] in c.text
        assert c.char_count > 100_000, m.work_id
