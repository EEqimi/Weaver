# knowledge/evaluation/effect.py
"""Phase 8.2 改写有效性分析（Revision Effect 门）——确定性、零 LLM、零 token。

核心函数 `normalize_for_revision_comparison` 做 canonical 归一化：只归一化 Unicode
排版标点（弯引号 / 弯连字符 / 省略号）与空白（换行 / NBSP / 多空格），**绝不**触碰
词形或词序（不会把 "don't" 合并成 "do not"，不会把 "happy" 换成 "glad"，不会重排
词序）。因此归一化后仍不相等 ⇔ 确实存在词级差异。

`RevisionEffectAnalyzer.analyze(original, revised)` 输出 `RevisionEffectResult`：
    - IDENTICAL        字节级相等
    - FORMATTING_ONLY  仅空白 / 换行差异（normalize_whitespace 后相等）
    - PUNCTUATION_ONLY 仅 Unicode 排版标点归一化差异（canonical 后相等）
    - SUBSTANTIVE      存在词级 token 差异

铁律（spec §十四 / §十六）：只有 `substantive_edit == True`，after-measurement
（重测 / 文学评价 / 策略匹配）才有资格参与比较；否则记 `no_effect`，绝不把 LLM
测量噪声（对等价文本的评分漂移）解释为真实改善。
"""
from __future__ import annotations

import difflib
import re

from ..generation.schema import output_hash
from ..schema.versions import REVISION_EFFECT_ANALYZER_VERSION
from .schema import (
    EFFECT_FORMATTING_ONLY, EFFECT_IDENTICAL, EFFECT_PUNCTUATION_ONLY,
    EFFECT_SUBSTANTIVE, REVISION_EFFECT_SCHEMA_VERSION, RevisionEffectResult,
)

ANALYZER_ID = "RevisionEffectAnalyzer"
ANALYZER_VERSION = REVISION_EFFECT_ANALYZER_VERSION

# Unicode 排版标点 → ASCII（只影响字形，不改词形/词序）。
_PUNCT_TRANSLATE = str.maketrans({
    "‘": "'", "’": "'", "‛": "'",          # 弯单引号 / 高位单引号
    "“": '"', "”": '"', "‟": '"',          # 弯双引号 / 高位双引号
    "‐": "-", "‑": "-", "‒": "-", "–": "-",  # 连字符族
    "—": "-", "―": "-",                          # em dash / 水平横线
    "…": "...",                                        # 水平省略号
})

# 常见 Unicode 空格（NBSP / 窄不换行空格 / 全角空格等）→ 普通空格。
_WS_CHARS = (
    " "   # no-break space
    " "   # ogham space mark
    "        "  # en/em/hair/… quad spaces
    "   "  # punctuation/thin/hair spaces
    " "   # narrow no-break space
    " "   # medium mathematical space
    "　"   # ideographic space
)
_WS_TRANSLATE = str.maketrans({ord(c): " " for c in _WS_CHARS})

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def normalize_whitespace(text: str) -> str:
    """仅归一化空白：换行（CRLF/LF/CR）→ 折叠、Unicode 空格→普通空格、多空格→单空格、
    去首尾空白。**不**做任何标点/词形归一化（用于区分 FORMATTING_ONLY）。"""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = t.translate(_WS_TRANSLATE)
    return re.sub(r"\s+", " ", t).strip()


def normalize_for_revision_comparison(text: str) -> str:
    """Canonical 归一化（spec §六）：排版标点 + 空白。绝不改词形/词序。"""
    return normalize_whitespace(text).translate(_PUNCT_TRANSLATE)


def _word_tokens(text: str) -> list[str]:
    return [t for t in text.split() if t]


def _sentence_tokens(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def _changed_token_count(before: list[str], after: list[str]) -> int:
    sm = difflib.SequenceMatcher(a=before, b=after, autojunk=False)
    changed = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        changed += max(i2 - i1, j2 - j1)
    return changed


class RevisionEffectAnalyzer:
    """确定性改写有效性分析（零 LLM）。输入 original / revised 文本，输出有效性分类
    与词/句改动计数。绝不烧 token；只依赖纯文本 diff。"""

    def analyze(self, original_text: str, revised_text: str) -> RevisionEffectResult:
        canonical_original = normalize_for_revision_comparison(original_text)
        canonical_revised = normalize_for_revision_comparison(revised_text)

        original_words = _word_tokens(canonical_original)
        revised_words = _word_tokens(canonical_revised)

        byte_identical = original_text == revised_text
        ws_identical = normalize_whitespace(original_text) == normalize_whitespace(revised_text)
        normalized_identical = canonical_original == canonical_revised

        if byte_identical:
            effect_status = EFFECT_IDENTICAL
        elif ws_identical:
            effect_status = EFFECT_FORMATTING_ONLY
        elif normalized_identical:
            effect_status = EFFECT_PUNCTUATION_ONLY
        else:
            effect_status = EFFECT_SUBSTANTIVE

        original_word_count = len(original_words)
        revised_word_count = len(revised_words)
        word_change_count = _changed_token_count(original_words, revised_words)
        word_change_ratio = (word_change_count / max(original_word_count, 1)
                             if original_word_count else 0.0)
        sentence_change_count = _changed_token_count(
            _sentence_tokens(canonical_original), _sentence_tokens(canonical_revised))

        substantive_edit = effect_status == EFFECT_SUBSTANTIVE
        reason = self._reason(effect_status, byte_identical, word_change_count)

        return RevisionEffectResult(
            schema_version=REVISION_EFFECT_SCHEMA_VERSION,
            analyzer_version=ANALYZER_VERSION,
            effect_status=effect_status,
            substantive_edit=substantive_edit,
            byte_identical=byte_identical,
            normalized_identical=normalized_identical,
            original_word_count=original_word_count,
            revised_word_count=revised_word_count,
            word_change_count=word_change_count,
            word_change_ratio=word_change_ratio,
            sentence_change_count=sentence_change_count,
            original_text_hash=output_hash(original_text),
            revised_text_hash=output_hash(revised_text),
            canonical_original_hash=output_hash(canonical_original),
            canonical_revised_hash=output_hash(canonical_revised),
            reason=reason,
        )

    @staticmethod
    def _reason(effect_status: str, byte_identical: bool,
                word_change_count: int) -> str:
        if effect_status == EFFECT_IDENTICAL:
            return "revised_text is byte-identical to the original"
        if effect_status == EFFECT_FORMATTING_ONLY:
            return "only whitespace / line-ending differences (no lexical change)"
        if effect_status == EFFECT_PUNCTUATION_ONLY:
            return ("only typographic punctuation normalization (no lexical "
                    "change)")
        return f"lexical word-token differences present ({word_change_count})"
