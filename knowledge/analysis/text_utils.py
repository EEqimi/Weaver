# knowledge/analysis/text_utils.py
"""确定性文本切分的共享工具（供 Layer A 统计与 Layer D 文体学复用）。

约定：
    - token 指英文单词（小写，含撇号缩略/所有格，如 "don't"、"austen's"）；
    - sentence 以 [.!?;] 结尾为界；
    - paragraph 以空行（\\n\\n）为界；
    - 所有函数确定、可复现，不依赖外部 NLP。

注意：这些是 V0.1 的轻量启发式，对 19 世纪英文小说足够；对现代/非英文
文本需替换为更稳健的分词器（通过 analyzer 可替换，见 spec §17.2）。
"""
from __future__ import annotations

import re
from collections import Counter

# 单词：小写字母串，可含内部撇号（缩略/所有格）
_WORD_RE = re.compile(r"[a-z]+(?:['’][a-z]+)?")

# 句末切分：在 [.!?;] 之后的空白处断开
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+")


def tokens(text: str) -> list[str]:
    """返回小写单词 token 列表（含缩略/所有格）。"""
    return _WORD_RE.findall(text.lower())


def sentences(text: str) -> list[str]:
    """返回句子字符串列表（不含末尾句点，空句已剔除）。"""
    return [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]


def paragraphs(text: str) -> list[str]:
    """返回段落字符串列表（空段落已剔除）。"""
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def token_frequencies(text: str) -> Counter:
    return Counter(tokens(text))
