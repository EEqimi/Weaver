# knowledge/schema/feature_registry.py
"""候选特征注册表（spec §4/§17.1）。

Feature 通过 Registry 注册，而非散落 hard-code（MUST-3）。
每个 FeatureDefinition 至少声明：id / category / measurement_type / value_type /
control_role / normalization / analyzer / schema_version。

四种 measurement_type（spec §4）：
    statistical  程序直接统计
    nlp          NLP + 统计
    hybrid       NLP/程序 + LLM evidence
    judgment     LLM evidence-based judgment
四种 control_role（spec §9 映射）：
    candidate_core / descriptive / diagnostic / experimental
    （core 保留给未来"跨作品稳定 + 生成可控"验证通过后的正式核心特征；
     当前 V0.1 的所有"核心候选"一律标记 candidate_core，不视为已验证。）
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .versions import FEATURE_SCHEMA_VERSION


class MeasurementType(str, Enum):
    STATISTICAL = "statistical"
    NLP = "nlp"
    HYBRID = "hybrid"
    JUDGMENT = "judgment"


class ValueType(str, Enum):
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    CATEGORICAL = "categorical"
    DISTRIBUTION = "distribution"


class ControlRole(str, Enum):
    CORE = "core"                      # 正式核心（验证通过后），当前预留
    CANDIDATE_CORE = "candidate_core"  # 候选核心：V0.1 的核心特征，未验证
    DESCRIPTIVE = "descriptive"
    DIAGNOSTIC = "diagnostic"
    EXPERIMENTAL = "experimental"


@dataclass(frozen=True)
class FeatureDefinition:
    id: str
    category: str
    measurement_type: MeasurementType
    value_type: ValueType
    control_role: ControlRole
    normalization: str          # none / zscore / minmax / corpus_percentile
    analyzer: str               # analyzer 名称（Registry 解耦，不硬编码分支）
    schema_version: str = FEATURE_SCHEMA_VERSION
    description: str = ""
    # 测量协议标签（LLM 派生特征必填；详见 schema/rubrics.py）：
    #   "frequency" —— LLM 识别实例，程序计数/折算
    #   "ordinal"   —— 锚定序数/程度量表（每档有显式定义）
    measurement_protocol: str = ""
    # 测量协议版本（与 analyzer/schema 版本分离，保证量表变更可追溯）
    protocol_version: str = FEATURE_SCHEMA_VERSION


class FeatureRegistry:
    """候选特征注册表。"""

    def __init__(self) -> None:
        self._features: dict[str, FeatureDefinition] = {}

    def register(self, feature: FeatureDefinition) -> None:
        if feature.id in self._features:
            raise ValueError(f"重复的 feature id: {feature.id}")
        self._features[feature.id] = feature

    def get(self, feature_id: str) -> FeatureDefinition:
        return self._features[feature_id]

    def has(self, feature_id: str) -> bool:
        return feature_id in self._features

    def all(self) -> list[FeatureDefinition]:
        return list(self._features.values())

    def by_category(self, category: str) -> list[FeatureDefinition]:
        return [f for f in self._features.values() if f.category == category]

    def by_control_role(self, role: ControlRole) -> list[FeatureDefinition]:
        return [f for f in self._features.values() if f.control_role == role]

    def by_measurement_type(self, mt: MeasurementType) -> list[FeatureDefinition]:
        return [f for f in self._features.values() if f.measurement_type == mt]

    def __len__(self) -> int:
        return len(self._features)

    def __iter__(self):
        return iter(self._features.values())


# ---- 便捷构造 ----
def _f(fid, category, mt, vt, role, norm, analyzer, desc="", protocol=""):
    return FeatureDefinition(
        id=fid, category=category, measurement_type=MeasurementType(mt),
        value_type=ValueType(vt), control_role=ControlRole(role),
        normalization=norm, analyzer=analyzer, description=desc,
        measurement_protocol=protocol,
    )


def build_default_registry() -> FeatureRegistry:
    """V0.1 候选特征池：确定性可算的 + 待 LLM/NLP 接口的。

    analyzer 字段指向抽象 analyzer 名称，不硬编码实现；V0.1 仅实现 statistical
    子集，nlp/hybrid/judgment 的 analyzer 作为接口占位（spec §3）。
    """
    reg = FeatureRegistry()
    S, N, H, J = ("statistical", "nlp", "hybrid", "judgment")
    CONT, DISC, DIST = ("continuous", "discrete", "distribution")

    defs = [
        # —— 1. Lexical & Register ——
        _f("lexical_diversity", "lexical_register", S, CONT, "candidate_core",
           "corpus_percentile", "StatisticalAnalyzer", "类型-词符比/词汇多样性"),
        _f("type_token_ratio", "lexical_register", S, CONT, "descriptive",
           "none", "StatisticalAnalyzer"),
        _f("hapax_ratio", "lexical_register", S, CONT, "descriptive",
           "none", "StatisticalAnalyzer", "单现词占比"),
        _f("word_repetition_rate", "lexical_register", S, CONT, "descriptive",
           "none", "StatisticalAnalyzer", "高频词重复倾向"),
        _f("mean_word_length", "lexical_register", S, CONT, "descriptive",
           "none", "StatisticalAnalyzer", "平均词长（字母数）"),
        _f("word_length_variance", "lexical_register", S, CONT, "descriptive",
           "none", "StatisticalAnalyzer", "词长方差"),
        _f("rare_word_ratio", "lexical_register", N, CONT, "experimental",
           "zscore", "NlpAnalyzer"),

        # —— 2. Syntax ——
        _f("mean_sentence_length", "syntax", S, CONT, "candidate_core",
           "corpus_percentile", "StatisticalAnalyzer"),
        _f("sentence_length_variance", "syntax", S, CONT, "descriptive",
           "none", "StatisticalAnalyzer"),
        _f("sentence_length_cv", "syntax", S, CONT, "descriptive",
           "none", "StatisticalAnalyzer", "句长变异系数"),
        _f("short_sentence_ratio", "syntax", S, CONT, "descriptive",
           "none", "StatisticalAnalyzer"),
        _f("long_sentence_ratio", "syntax", S, CONT, "descriptive",
           "none", "StatisticalAnalyzer"),
        _f("pos_noun_ratio", "syntax", N, CONT, "experimental",
           "zscore", "NlpAnalyzer"),
        _f("pos_verb_ratio", "syntax", N, CONT, "experimental",
           "zscore", "NlpAnalyzer"),
        _f("pos_adj_ratio", "syntax", N, CONT, "experimental",
           "zscore", "NlpAnalyzer"),
        _f("pos_pronoun_ratio", "syntax", N, CONT, "experimental",
           "zscore", "NlpAnalyzer"),

        # —— 3. Rhythm & Punctuation ——
        _f("mean_paragraph_length", "rhythm_punctuation", S, CONT, "candidate_core",
           "corpus_percentile", "StatisticalAnalyzer"),
        _f("paragraph_length_variance", "rhythm_punctuation", S, CONT,
           "descriptive", "none", "StatisticalAnalyzer"),
        _f("comma_density", "rhythm_punctuation", S, CONT, "descriptive",
           "none", "StatisticalAnalyzer"),
        _f("period_density", "rhythm_punctuation", S, CONT, "descriptive",
           "none", "StatisticalAnalyzer"),
        _f("semicolon_density", "rhythm_punctuation", S, CONT, "descriptive",
           "none", "StatisticalAnalyzer"),
        _f("dash_density", "rhythm_punctuation", S, CONT, "descriptive",
           "none", "StatisticalAnalyzer"),
        _f("exclamation_frequency", "rhythm_punctuation", S, CONT,
           "descriptive", "none", "StatisticalAnalyzer"),
        _f("question_frequency", "rhythm_punctuation", S, CONT,
           "descriptive", "none", "StatisticalAnalyzer"),

        # —— 4. Rhetoric & Imagery ——
        # frequency-like：LLM 识别隐喻/明喻实例，程序计数（见 rubrics.py）
        _f("metaphor_frequency", "rhetoric_imagery", H, CONT, "experimental",
           "zscore", "LlmFeatureAnalyzer", "隐喻频率（LLM evidence）",
           protocol="frequency"),
        _f("simile_frequency", "rhetoric_imagery", H, CONT, "experimental",
           "zscore", "LlmFeatureAnalyzer", "明喻频率（LLM evidence）",
           protocol="frequency"),

        # —— 5. Voice & Pragmatics ——
        _f("irony_frequency", "voice_pragmatics", J, CONT, "experimental",
           "zscore", "LlmFeatureAnalyzer", "反讽频率（LLM evidence）",
           protocol="frequency"),
        # intensity/degree-like：锚定序数 0–4（见 rubrics.py 的 ordinal 协议）
        _f("irony_intensity", "voice_pragmatics", J, DISC, "experimental",
           "none", "LlmFeatureAnalyzer", "反讽强度（0–4 序数）",
           protocol="ordinal"),
        _f("narrator_evaluative_intervention", "voice_pragmatics", J, DISC,
           "experimental", "none", "LlmFeatureAnalyzer",
           "叙述者评价性介入程度（0–4 序数）", protocol="ordinal"),

        # —— 6. Character Representation ——
        _f("dialogue_ratio", "character_representation", S, CONT, "candidate_core",
           "corpus_percentile", "StatisticalAnalyzer", "对话占比"),
        _f("quotation_density", "character_representation", S, CONT, "descriptive",
           "none", "StatisticalAnalyzer", "引号密度（双引号/千词）"),
        _f("psychological_representation", "character_representation", H, DISC,
           "experimental", "none", "LlmFeatureAnalyzer",
           "心理呈现程度（0–4 序数）", protocol="ordinal"),

        # —— 7. Emotion & Semantic Texture ——
        _f("emotional_restraint", "emotion_semantics", J, DISC, "experimental",
           "none", "LlmFeatureAnalyzer", "情感克制（0–4 序数）",
           protocol="ordinal"),
        _f("emotional_intensity", "emotion_semantics", H, DISC, "experimental",
           "none", "LlmFeatureAnalyzer", "情绪强度（0–4 序数）",
           protocol="ordinal"),

        # —— 8. Discourse & Cohesion ——
        _f("connective_density", "discourse_cohesion", S, CONT, "descriptive",
           "none", "StatisticalAnalyzer", "连接词密度"),
        _f("pos_bigram_distribution", "discourse_cohesion", N, DIST,
           "experimental", "none", "NlpAnalyzer"),

        # —— 9. Stylometric（诊断用，spec §7.1，控制角色为 diagnostic）——
        _f("mfw_frequency", "stylometric", S, DIST, "diagnostic",
           "none", "StylometricExtractor", "最常用词频分布"),
        _f("function_word_frequency", "stylometric", S, DIST, "diagnostic",
           "none", "StylometricExtractor", "功能词频分布"),
        _f("char_trigram_frequency", "stylometric", S, DIST, "diagnostic",
           "none", "StylometricExtractor", "字符 3-gram 频分布"),
    ]
    for d in defs:
        reg.register(d)
    return reg
