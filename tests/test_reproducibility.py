# tests/test_reproducibility.py
"""SHA256 复现性：清洗与分块对同一输入必产出相同结果。"""
import hashlib

from knowledge.corpus.cleaner import clean
from knowledge.corpus.chunker import chunk_text
from knowledge.corpus.discover import discover
from knowledge.corpus.metadata import CORPUS


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_clean_reproducible(real_corpus_root):
    found = discover(real_corpus_root)
    m = next(m for m in CORPUS if m.work_id == "persuasion")
    raw = found[m.work_id].read_text(encoding="utf-8")
    c1 = clean(raw, m.work_id)
    c2 = clean(raw, m.work_id)
    assert _sha256(c1.text) == _sha256(c2.text)
    assert c1.text == c2.text


def test_chunk_reproducible(real_corpus_root):
    found = discover(real_corpus_root)
    m = next(m for m in CORPUS if m.work_id == "persuasion")
    raw = found[m.work_id].read_text(encoding="utf-8")
    text = clean(raw, m.work_id).text
    a = chunk_text(text, m.work_id, 2000)
    b = chunk_text(text, m.work_id, 2000)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert [_sha256(c.text) for c in a] == [_sha256(c.text) for c in b]
