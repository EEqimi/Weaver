# knowledge/analysis/evidence.py
"""共享的证据校验工具（Phase 3–4.1，task item 5）。

LLM analyzer 声称 evidence 是"逐字摘自 passage"，因此必须校验，不能静默接受
可能被模型编造的引文。校验规则：

    1. Unicode 一致归一化（NFC）；
    2. 引号/撇号/破折号等排版变体映射到 ASCII 等价形式；
    3. 折叠平凡空白（安全：只折叠空白，不折叠标点/单词）；
    4. 仅当（归一化后的）引文是（归一化后的）passage 的**子串**时判定为逐字吻合。

对无法验证的引文：不静默丢弃，而是显式标记为 unverified，由调用方决定
reject 或记录（本工具只负责判定与分组）。
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Any, Iterable

# 排版变体 → ASCII 等价（引号/撇号/破折号；与 cleaner 的 NFC 归一化互补）
_PUNCT_TRANS = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "―": "-",
})


def normalize_quote(text: str) -> str:
    """归一化一段文本用于逐字比对：NFC + 排版变体映射 + 空白折叠。"""
    if text is None:
        return ""
    s = unicodedata.normalize("NFC", str(text))
    s = s.translate(_PUNCT_TRANS)
    # 折叠所有空白为单空格，并去掉首尾空白（平凡空白归一化）
    return " ".join(s.split())


def quote_in_passage(quote: str, passage: str) -> bool:
    """判定 quote 是否逐字出现在 passage 中（归一化后子串匹配）。"""
    q = normalize_quote(quote)
    if not q:
        return False
    p = normalize_quote(passage)
    return q in p


def _extract_quote_text(item: Any) -> str:
    """从证据条目中提取可比对的文本：str 直接用，dict 取 evidence/quote/text。"""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("evidence", "quote", "text"):
            v = item.get(key)
            if isinstance(v, str):
                return v
    return ""


@dataclass
class EvidenceCheck:
    """证据校验结果：verified 与 unverified 分组（保留原始条目，便于追溯）。"""
    verified: list[Any] = field(default_factory=list)
    unverified: list[Any] = field(default_factory=list)

    @property
    def n_verified(self) -> int:
        return len(self.verified)

    @property
    def n_unverified(self) -> int:
        return len(self.unverified)

    def all_verified(self) -> bool:
        return self.n_unverified == 0

    def to_dict(self) -> dict:
        return {
            "n_verified": self.n_verified,
            "n_unverified": self.n_unverified,
            "verified": [e for e in self.verified],
            "unverified": [e for e in self.unverified],
        }


def verify_evidence_quotes(quotes: Iterable[Any], passage: str) -> EvidenceCheck:
    """把证据条目分为 verified / unverified 两组（不静默丢弃）。"""
    check = EvidenceCheck()
    for item in quotes or []:
        text = _extract_quote_text(item)
        if quote_in_passage(text, passage):
            check.verified.append(item)
        else:
            check.unverified.append(item)
    return check
