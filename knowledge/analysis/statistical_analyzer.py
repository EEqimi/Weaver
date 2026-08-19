# knowledge/analysis/statistical_analyzer.py
"""Layer A 确定性统计特征分析器（不调用 LLM、不依赖 NLP）。

对 FeatureRegistry 中 analyzer == "StatisticalAnalyzer" 的可解释特征做确定性
测量。每个结果保留 raw_value / sample_count / variance / evidence；
normalized_value 留空，由聚合阶段按特征声明的 normalization 方式计算。

只实现"可靠可测"的特征；需要 NLP/LLM 的特征仍路由到 NlpAnalyzer /
LlmFeatureAnalyzer（见 feature_registry 的 analyzer 字段）。每个特征的计算
函数按 feature_id 注册到 _COMPUTERS，新增特征只需加一个计算函数（MUST-3）。
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable

from ..schema.feature_registry import FeatureDefinition
from ..schema.style_schema import FeatureValue
from ..schema.versions import FEATURE_SCHEMA_VERSION, STATISTICAL_ANALYZER_VERSION
from .text_utils import paragraphs, sentences, token_frequencies, tokens

ANALYZER_ID = "StatisticalAnalyzer"
ANALYZER_VERSION = STATISTICAL_ANALYZER_VERSION

# 英语逻辑连接词（discourse/cohesion 的 connective_density）
_CONNECTIVES = frozenset({
    "and", "but", "or", "nor", "so", "for", "yet", "because", "since", "while",
    "whereas", "though", "although", "if", "unless", "until", "when", "where",
    "whenever", "however", "therefore", "thus", "hence", "moreover", "furthermore",
    "nevertheless", "consequently", "accordingly", "then", "still", "instead",
    "otherwise", "meanwhile", "besides", "otherwise", "whereupon", "whilst",
})

# 双引号字符（对话/引用分界；单引号因与所有格/缩写混淆，不计入）
_OPEN_QUOTES = frozenset('"“')
_CLOSE_QUOTES = frozenset('"”')
_QUOTE_CHARS = frozenset('"“”')

# 短/长句阈值（词数，V0.1 约定，可后续实验调整）
_SHORT_SENTENCE_TOKENS = 10
_LONG_SENTENCE_TOKENS = 30
# MSTTR 窗口大小（词数）
_MSTTR_WINDOW = 100
# word_repetition_rate 的 top 词数
_TOP_TYPES = 50


def _word_length(token: str) -> int:
    return len(token.replace("'", "").replace("’", ""))


@dataclass
class _TextStats:
    tokens: list[str]
    token_count: int
    type_count: int
    freq: Counter
    sentence_lengths: list[int]
    paragraph_lengths: list[int]
    n_sentences: int
    n_paragraphs: int
    char_count: int
    comma: int
    period: int
    semicolon: int
    dash: int
    exclam: int
    question: int
    quote: int
    dialogue_chars: int
    n_quoted_spans: int


def _stats(text: str) -> _TextStats:
    toks = tokens(text)
    freq = Counter(toks)
    sent_lens = [len(tokens(s)) for s in sentences(text)]
    para_lens = [len(tokens(p)) for p in paragraphs(text)]
    dash = len(re.findall(r"—|--", text))
    dialogue_chars = 0
    n_spans = 0
    inside = False
    for ch in text:
        if ch in _OPEN_QUOTES and not inside:
            inside = True
            n_spans += 1
        elif ch in _CLOSE_QUOTES and inside:
            inside = False
        elif inside:
            dialogue_chars += 1
    return _TextStats(
        tokens=toks, token_count=len(toks), type_count=len(freq), freq=freq,
        sentence_lengths=sent_lens, paragraph_lengths=para_lens,
        n_sentences=len(sent_lens), n_paragraphs=len(para_lens),
        char_count=len(text),
        comma=text.count(","), period=text.count("."),
        semicolon=text.count(";"), dash=dash,
        exclam=text.count("!"), question=text.count("?"),
        quote=sum(text.count(c) for c in _QUOTE_CHARS),
        dialogue_chars=dialogue_chars, n_quoted_spans=n_spans,
    )


def _per_1000_tokens(count: int, token_count: int) -> float:
    return (count / token_count * 1000.0) if token_count else 0.0


class StatisticalAnalyzer:
    """对一段文本计算所有已注册的确定性统计特征。"""

    def __init__(self) -> None:
        self._computers: dict[str, Callable[[FeatureDefinition, _TextStats], FeatureValue]] = {
            "lexical_diversity": self._lexical_diversity,
            "type_token_ratio": self._type_token_ratio,
            "hapax_ratio": self._hapax_ratio,
            "word_repetition_rate": self._word_repetition_rate,
            "mean_word_length": self._mean_word_length,
            "word_length_variance": self._word_length_variance,
            "mean_sentence_length": self._mean_sentence_length,
            "sentence_length_variance": self._sentence_length_variance,
            "sentence_length_cv": self._sentence_length_cv,
            "short_sentence_ratio": self._short_sentence_ratio,
            "long_sentence_ratio": self._long_sentence_ratio,
            "mean_paragraph_length": self._mean_paragraph_length,
            "paragraph_length_variance": self._paragraph_length_variance,
            "comma_density": self._comma_density,
            "period_density": self._period_density,
            "semicolon_density": self._semicolon_density,
            "dash_density": self._dash_density,
            "exclamation_frequency": self._exclamation_frequency,
            "question_frequency": self._question_frequency,
            "quotation_density": self._quotation_density,
            "dialogue_ratio": self._dialogue_ratio,
            "connective_density": self._connective_density,
        }

    # ---- 公共入口 ----
    def analyze(self, text: str, feature: FeatureDefinition) -> FeatureValue | None:
        """计算单个特征；文本无词则返回 None（由调用方跳过）。"""
        if feature.id not in self._computers:
            raise ValueError(f"StatisticalAnalyzer 未实现特征 {feature.id}")
        st = _stats(text)
        if st.token_count == 0:
            return None
        return self._computers[feature.id](feature, st)

    def analyze_many(self, text: str, features: list[FeatureDefinition]) -> list[FeatureValue]:
        """计算一组特征，跳过无法计算的（空文本 / 未实现）。"""
        out: list[FeatureValue] = []
        for f in features:
            fv = self.analyze(text, f)
            if fv is not None:
                out.append(fv)
        return out

    # ---- 结果构造 ----
    @staticmethod
    def _fv(feature: FeatureDefinition, value: float, raw_value: float,
            sample_count: int, variance: float | None = None,
            evidence: list[str] | None = None) -> FeatureValue:
        return FeatureValue(
            feature_id=feature.id,
            value=value,
            raw_value=raw_value,
            normalized_value=None,
            value_type=feature.value_type.value,
            measurement_type=feature.measurement_type.value,
            confidence=None,
            evidence=evidence or [],
            sample_count=sample_count,
            variance=variance,
            analyzer_id=ANALYZER_ID,
            analyzer_version=ANALYZER_VERSION,
            schema_version=FEATURE_SCHEMA_VERSION,
            provenance={"analyzer": ANALYZER_ID},
        )

    # ---- 词汇 ----
    def _type_token_ratio(self, f, st: _TextStats) -> FeatureValue:
        v = st.type_count / st.token_count
        return self._fv(f, v, v, st.token_count)

    def _lexical_diversity(self, f, st: _TextStats) -> FeatureValue:
        # MSTTR：非重叠 100 词窗口的 TTR 均值（比裸 TTR 更稳健、长度无关）
        wins = [st.tokens[i:i + _MSTTR_WINDOW]
                for i in range(0, st.token_count, _MSTTR_WINDOW)]
        tt_rs = [len(set(w)) / len(w) for w in wins if w]
        mean = sum(tt_rs) / len(tt_rs)
        var = (sum((x - mean) ** 2 for x in tt_rs) / len(tt_rs)) if len(tt_rs) > 1 else 0.0
        return self._fv(f, mean, mean, len(tt_rs), variance=var)

    def _hapax_ratio(self, f, st: _TextStats) -> FeatureValue:
        v = sum(1 for c in st.freq.values() if c == 1) / st.token_count
        return self._fv(f, v, v, st.token_count)

    def _word_repetition_rate(self, f, st: _TextStats) -> FeatureValue:
        # 前 50 高频词型占总词频的份额（重复/高频集中度）
        top = sum(c for _, c in st.freq.most_common(_TOP_TYPES))
        v = top / st.token_count
        return self._fv(f, v, v, st.token_count)

    def _mean_word_length(self, f, st: _TextStats) -> FeatureValue:
        lengths = [_word_length(t) for t in st.tokens]
        mean = sum(lengths) / len(lengths)
        var = sum((x - mean) ** 2 for x in lengths) / len(lengths)
        return self._fv(f, mean, mean, len(lengths), variance=var)

    def _word_length_variance(self, f, st: _TextStats) -> FeatureValue:
        lengths = [_word_length(t) for t in st.tokens]
        mean = sum(lengths) / len(lengths)
        var = sum((x - mean) ** 2 for x in lengths) / len(lengths)
        return self._fv(f, var, var, len(lengths))

    # ---- 句法 ----
    def _mean_sentence_length(self, f, st: _TextStats) -> FeatureValue:
        return self._mean_of(f, st.sentence_lengths)

    def _sentence_length_variance(self, f, st: _TextStats) -> FeatureValue:
        return self._variance_of(f, st.sentence_lengths)

    def _sentence_length_cv(self, f, st: _TextStats) -> FeatureValue:
        lens = st.sentence_lengths
        mean = sum(lens) / len(lens)
        std = math.sqrt(sum((x - mean) ** 2 for x in lens) / len(lens))
        v = std / mean if mean else 0.0
        return self._fv(f, v, v, len(lens))

    def _short_sentence_ratio(self, f, st: _TextStats) -> FeatureValue:
        v = sum(1 for x in st.sentence_lengths if x < _SHORT_SENTENCE_TOKENS) / st.n_sentences
        return self._fv(f, v, v, st.n_sentences)

    def _long_sentence_ratio(self, f, st: _TextStats) -> FeatureValue:
        v = sum(1 for x in st.sentence_lengths if x > _LONG_SENTENCE_TOKENS) / st.n_sentences
        return self._fv(f, v, v, st.n_sentences)

    # ---- 节奏 / 段落 ----
    def _mean_paragraph_length(self, f, st: _TextStats) -> FeatureValue:
        return self._mean_of(f, st.paragraph_lengths)

    def _paragraph_length_variance(self, f, st: _TextStats) -> FeatureValue:
        return self._variance_of(f, st.paragraph_lengths)

    # ---- 标点（每千词） ----
    def _comma_density(self, f, st: _TextStats) -> FeatureValue:
        return self._rate(f, st.comma, st)

    def _period_density(self, f, st: _TextStats) -> FeatureValue:
        return self._rate(f, st.period, st)

    def _semicolon_density(self, f, st: _TextStats) -> FeatureValue:
        return self._rate(f, st.semicolon, st)

    def _dash_density(self, f, st: _TextStats) -> FeatureValue:
        return self._rate(f, st.dash, st)

    def _exclamation_frequency(self, f, st: _TextStats) -> FeatureValue:
        return self._rate(f, st.exclam, st)

    def _question_frequency(self, f, st: _TextStats) -> FeatureValue:
        return self._rate(f, st.question, st)

    def _quotation_density(self, f, st: _TextStats) -> FeatureValue:
        return self._rate(f, st.quote, st)

    # ---- 对话 ----
    def _dialogue_ratio(self, f, st: _TextStats) -> FeatureValue:
        v = st.dialogue_chars / st.char_count if st.char_count else 0.0
        return self._fv(f, v, v, st.n_quoted_spans)

    # ---- 篇章 ----
    def _connective_density(self, f, st: _TextStats) -> FeatureValue:
        n = sum(1 for t in st.tokens if t in _CONNECTIVES)
        return self._rate(f, n, st)

    # ---- 通用 ----
    def _rate(self, f, count: int, st: _TextStats) -> FeatureValue:
        v = _per_1000_tokens(count, st.token_count)
        return self._fv(f, v, v, st.token_count)

    def _mean_of(self, f, xs: list[int]) -> FeatureValue:
        mean = sum(xs) / len(xs)
        var = sum((x - mean) ** 2 for x in xs) / len(xs)
        return self._fv(f, mean, mean, len(xs), variance=var)

    def _variance_of(self, f, xs: list[int]) -> FeatureValue:
        mean = sum(xs) / len(xs)
        var = sum((x - mean) ** 2 for x in xs) / len(xs)
        return self._fv(f, var, var, len(xs))
