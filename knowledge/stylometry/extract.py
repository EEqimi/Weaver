# knowledge/stylometry/extract.py
"""Layer D 文体学特征提取（非 LLM）。

用 scikit-learn 的 CountVectorizer 构建字符 n-gram / 词 unigram 词汇表，加
固定功能词表，输出"按家族归一化的相对频率"特征矩阵（稠密 numpy）。

POS 特征在 V0.1 刻意留空（NLTK 未安装，spec §7.1 "POS 可选/stub"）。
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from ..analysis.text_utils import tokens as _tokens
from ..schema.versions import STYLOMETRY_VERSION

ANALYZER_ID = "StylometricExtractor"
ANALYZER_VERSION = STYLOMETRY_VERSION

_WORD_PATTERN = r"[a-z]+(?:['’][a-z]+)?"

# 英语功能词（封闭类词表）：跨 chunk 固定可比
FUNCTION_WORDS: list[str] = [
    "the", "a", "an", "and", "or", "but", "nor", "for", "yet", "so",
    "of", "in", "on", "at", "by", "to", "from", "with", "into", "upon",
    "over", "under", "between", "among", "through", "during", "before", "after",
    "above", "below", "behind", "beside", "beyond", "within", "without",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
    "us", "them", "my", "your", "his", "its", "our", "their", "mine", "yours",
    "hers", "ours", "theirs", "myself", "yourself", "himself", "herself",
    "itself", "ourselves", "yourselves", "themselves",
    "this", "that", "these", "those", "such", "which", "who", "whom", "whose",
    "what", "whatever", "whoever", "whichever",
    "be", "am", "is", "are", "was", "were", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing",
    "shall", "should", "will", "would", "may", "might", "can", "could",
    "must", "ought",
    "not", "no", "never", "nothing", "none",
    "as", "than", "because", "since", "while", "though", "although",
    "if", "unless", "until", "when", "where", "whether",
    "there", "here", "then", "now", "thus", "therefore", "hence",
    "very", "too", "more", "most", "less", "least", "much", "many",
    "some", "any", "all", "both", "each", "every", "either", "neither",
    "other", "another", "own", "same", "only", "even", "just", "still",
]


class StylometricVectorizer:
    """把文本集转为 (n_texts × n_features) 的相对频率矩阵。

    词汇表在 fit 时确定；transform 按家族归一化：
        - 功能词 / 词 unigram → 除以该文本词数；
        - 字符 n-gram          → 除以该文本字符数。
    """

    def __init__(self, char_n: int = 3, char_top_k: int = 400,
                 word_top_k: int = 400, function_words: list[str] | None = None):
        self.char_n = char_n
        self.char_top_k = char_top_k
        self.word_top_k = word_top_k
        self.function_words = list(function_words) if function_words is not None else list(FUNCTION_WORDS)
        self.feature_names_: list[str] = []
        self._fw_vec: CountVectorizer | None = None
        self._char_vec: CountVectorizer | None = None
        self._word_vec: CountVectorizer | None = None

    def fit(self, texts: list[str]) -> "StylometricVectorizer":
        self._fw_vec = CountVectorizer(vocabulary=self.function_words,
                                       token_pattern=_WORD_PATTERN, lowercase=True)
        self._fw_vec.fit(texts)
        self._char_vec = CountVectorizer(analyzer="char",
                                         ngram_range=(self.char_n, self.char_n),
                                         max_features=self.char_top_k, lowercase=True)
        self._char_vec.fit(texts)
        self._word_vec = CountVectorizer(analyzer="word", ngram_range=(1, 1),
                                         max_features=self.word_top_k,
                                         token_pattern=_WORD_PATTERN, lowercase=True)
        self._word_vec.fit(texts)
        self.feature_names_ = (
            [f"fw:{w}" for w in self._fw_vec.get_feature_names_out()]
            + [f"char:{g}" for g in self._char_vec.get_feature_names_out()]
            + [f"word:{w}" for w in self._word_vec.get_feature_names_out()]
        )
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        if self._fw_vec is None or self._char_vec is None or self._word_vec is None:
            raise RuntimeError("请先调用 fit()")
        n_tokens = np.array([len(_tokens(t)) for t in texts], dtype=float)
        n_chars = np.array([len(t) for t in texts], dtype=float)
        fw = self._fw_vec.transform(texts).toarray() / np.maximum(n_tokens[:, None], 1.0)
        ch = self._char_vec.transform(texts).toarray() / np.maximum(n_chars[:, None], 1.0)
        wd = self._word_vec.transform(texts).toarray() / np.maximum(n_tokens[:, None], 1.0)
        return np.hstack([fw, ch, wd])

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        return self.fit(texts).transform(texts)
