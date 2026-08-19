# tests/test_chunker.py
"""确定性分块：不跨章、段落优先、尺寸接近目标、可复现。"""
from knowledge.corpus.chunker import chunk_text, _split_long_paragraph


def _para(tok: str, n: int) -> str:
    return " ".join([tok] * n)


def _two_chapter_text():
    # 每章 5 段，每段约 40 个词（~200 字符）
    ch1 = "\n\n".join(_para(f"alpha{i}", 40) for i in range(5))
    ch2 = "\n\n".join(_para(f"beta{i}", 40) for i in range(5))
    return f"CHAPTER I\n\n{ch1}\n\nCHAPTER II\n\n{ch2}"


def test_no_cross_chapter_boundaries():
    chunks = chunk_text(_two_chapter_text(), "t", 1000)
    for c in chunks:
        # 每个 chunk 只能来自单一章节：正文里要么只有 alpha 词，要么只有 beta 词
        has_alpha = "alpha" in c.text
        has_beta = "beta" in c.text
        assert not (has_alpha and has_beta), c.chunk_id
        assert c.chapter in (1, 2)


def test_chapter_assignment():
    chunks = chunk_text(_two_chapter_text(), "t", 1000)
    assert {c.chapter for c in chunks} == {1, 2}
    assert chunks[0].chapter == 1
    assert chunks[-1].chapter == 2


def test_paragraph_priority_and_size():
    text = "\n\n".join(_para(f"word{i}", 40) for i in range(10))  # 无章节
    chunks = chunk_text(text, "t", 1000)
    # 每段 ~200 字符，1000 目标 → 每块约 4-5 段，尺寸不会远超目标
    for c in chunks[:-1]:
        assert c.char_count <= 1000 * 1.5 + 50
        assert c.char_count >= 1000 * 0.5  # 段落优先，非精确切分
    assert chunks[-1].char_count > 0


def test_deterministic():
    text = _two_chapter_text()
    a = chunk_text(text, "t", 2000)
    b = chunk_text(text, "t", 2000)
    assert [c.text for c in a] == [c.text for c in b]
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]


def test_word_and_char_count_consistent():
    chunks = chunk_text(_two_chapter_text(), "t", 1000)
    for c in chunks:
        assert c.char_count == len(c.text)
        assert c.word_count == len(c.text.split())


# ---- 超长段落与句切分（Phase 1–2.1，item 2/3） ----

def _sent(k: int) -> str:
    """约 36 字符的完整句子（以句点结尾，可被句切分识别）。"""
    return f"sentence{k} goes here and says a thing."


def _long_paragraph(n_sent: int) -> str:
    """由 n_sent 个完整句子拼成的单个超长段落。"""
    return " ".join(_sent(i) for i in range(n_sent))


def test_sentence_split_long_paragraph():
    # 单个超长段落（> 1.5×target）应被拆为多块，且切分点落在句末（各块以句点结尾）
    para = _long_paragraph(50)          # ~1800 字符 > 1500
    chunks = chunk_text(para, "t", 1000)
    assert len(chunks) >= 2
    for c in chunks:
        assert c.char_count <= 1000 * 1.5 + 100   # 句切分不超硬上限（容差句长）
        assert c.text.rstrip().endswith(".")


def test_long_paragraph_remainder_merges_forward():
    # 超长段落句切分的最后一片（< 0.5×target）应并入后续段落，而非孤立成小块
    long_para = _long_paragraph(60)      # ~2329 字符 → 余片 ~389
    pieces = _split_long_paragraph(long_para, 1000)
    assert len(pieces[-1]) < 1000 * 0.5  # 测试前提：余片确实过小
    tail = [_para("tail", 20)] * 5       # 5 × ~99 字符，足以吸收余片
    text = "\n\n".join([long_para] + tail)
    chunks = chunk_text(text, "t", 1000)
    assert chunks, "应有输出"
    for c in chunks:
        assert c.char_count >= 1000 * 0.5, c.chunk_id


def test_chapter_end_long_paragraph_tail():
    # 超长段落位于章节末尾时，句切分余片可作为章节尾小尾巴（预期，允许 < 0.5×target）
    long_para = _long_paragraph(60)      # ~2329 字符 → 余片 ~389（< 0.5×target）
    text = f"CHAPTER I\n\n{long_para}\n\nCHAPTER II\n\n" + _para("body", 40)
    chunks = chunk_text(text, "t", 1000)
    ch1 = [c for c in chunks if c.chapter == 1]
    assert len(ch1) >= 2                  # 超长段落确实被句切分
    # 章节尾小尾巴允许 < 0.5×target；但非尾块不应过小
    assert ch1[-1].char_count < 1000 * 0.5
    for c in ch1[:-1]:
        assert c.char_count >= 1000 * 0.5, c.chunk_id


def test_small_paragraph_not_isolated():
    # 小段落（< 0.5×target）后接大段落时，应合并而非孤立成微型块
    small = _para("small", 40)           # ~240 字符
    big = _para("big", 200)              # ~800 字符
    text = "\n\n".join([small, big])     # 合计 ~1040 > target
    chunks = chunk_text(text, "t", 1000)
    assert len(chunks) == 1
    assert chunks[0].char_count >= 1000 * 0.5
