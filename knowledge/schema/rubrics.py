# knowledge/schema/rubrics.py
"""Feature measurement rubrics（Phase 3–4.1 calibration readiness）。

每个 LLM 派生的特征都必须有显式的"测量协议/量表"，绝不只要求模型"返回一个
float"。这里定义两种协议族（task item 1）：

    frequency
        LLM 只负责识别证据实例（span/event）；程序对**已验证**的实例做确定性
        计数，并归一化为每 1000 词（token）的实例率。最终 value = raw_count /
        exposure_tokens × 1000，由证据推导，而非一个不透明的浮点（task item 1）。

    ordinal
        有锚定、有界的序数/程度量表，每一档都有显式定义（如 0=absent … 4=dominant）。
        LLM 返回评估状态（observed / insufficient_evidence / not_observable）+ 档位
        + 证据；当状态非 observed 时档位必须为 null，程序绝不把"无法评估"折算成 0
        （task item 2）。

每个 rubric 带 protocol_version，使测量结果可追溯（与 analyzer/schema 版本
并列，见 spec §17.6 的版本分离原则）。

不同文学属性不共享同一量表：frequency 与 ordinal 的 value 语义完全不同，
且 ordinal 特征的档位定义逐特征定制。
"""
from __future__ import annotations

from dataclasses import dataclass

from .versions import SCHEMA_VERSION

FREQUENCY_PROTOCOL = "frequency"
ORDINAL_PROTOCOL = "ordinal"

# frequency 特征的 value 单位：程序归一化为每 1000 词（token）的实例率。
# LLM 只负责识别实例，绝不参与归一化（task item 1）。
FREQUENCY_UNIT = "instances per 1000 tokens"
FREQUENCY_DENOMINATOR = 1000

# ordinal 特征的评估状态（task item 2）：显式区分"观察到缺失"与"无法评估"。
ASSESSMENT_OBSERVED = "observed"
ASSESSMENT_INSUFFICIENT = "insufficient_evidence"
ASSESSMENT_NOT_OBSERVABLE = "not_observable"
ASSESSMENT_STATUSES = (ASSESSMENT_OBSERVED, ASSESSMENT_INSUFFICIENT,
                       ASSESSMENT_NOT_OBSERVABLE)

# 通用锚定序数档（task item 1.B 的示例量表）
DEFAULT_ORDINAL_LEVELS: tuple["ScaleLevel", ...] = ()


@dataclass(frozen=True)
class ScaleLevel:
    """序数量表的一档：档值 + 标签 + 显式定义。"""
    value: int
    label: str
    definition: str


@dataclass(frozen=True)
class MeasurementRubric:
    """单个特征的测量协议。

    protocol       ∈ {frequency, ordinal}
    levels         ordinal 特征的档位定义（frequency 特征为空）
    unit           frequency 特征的计数单位（如 "instances" / "rate per 1000 tokens"）
    min_evidence   高置信正向判定所需的最少已验证证据数
    instruction    注入 LLM prompt 的测量指令（含量表定义）
    """
    feature_id: str
    protocol: str
    protocol_version: str
    levels: tuple[ScaleLevel, ...] = ()
    unit: str = "instances"
    min_evidence: int = 1
    instruction: str = ""

    def is_frequency(self) -> bool:
        return self.protocol == FREQUENCY_PROTOCOL

    def is_ordinal(self) -> bool:
        return self.protocol == ORDINAL_PROTOCOL


def _levels(entries: list[tuple[int, str, str]]) -> tuple[ScaleLevel, ...]:
    return tuple(ScaleLevel(v, label, defn) for v, label, defn in entries)


# 0=absent 1=weak 2=moderate 3=strong 4=dominant（task item 1.B）
_ABSENT_WEAK_MODERATE_STRONG_DOMINANT = _levels([
    (0, "absent", "该属性在段落中完全未出现"),
    (1, "weak", "该属性微弱、偶发，或仅有一处隐约迹象"),
    (2, "moderate", "该属性清晰存在，但未主导段落"),
    (3, "strong", "该属性显著、反复出现，是段落的重要特征"),
    (4, "dominant", "该属性主导段落，是段落最突出的特征"),
])

# 强度/程度类可复用 "absent … dominant"，但语义需按特征定制描述文本
# 关键契约（task item 2）：显式区分"观察到缺失"（level 0）与"无法评估"
# （assessment_status != observed → level 必须为 null，程序不折算成 0）。
_INTENSITY_INSTRUCTION = (
    "Measure the DEGREE of {desc} in the passage on an anchored ordinal scale. "
    "Return an `assessment_status` in {{observed, insufficient_evidence, "
    "not_observable}} plus an integer `level`:\n"
    "  observed: you can assess the degree from the passage. Then `level` is an "
    "integer in {{0,1,2,3,4}} with these definitions:\n"
    "    0 = absent: {desc_absent}\n"
    "    1 = weak: a faint or occasional trace of {desc}\n"
    "    2 = moderate: {desc} is clearly present but not dominant\n"
    "    3 = strong: {desc} is pronounced and recurrent\n"
    "    4 = dominant: {desc} dominates the passage\n"
    "  insufficient_evidence: the passage does not give enough text to assess it. "
    "Then `level` must be null and `evidence` an empty array.\n"
    "  not_observable: the property is not observable in this kind of passage. "
    "Then `level` must be null and `evidence` an empty array.\n"
    "Do NOT emit a non-numeric label for this scale, and do NOT return level 0 "
    "to mean 'cannot assess' — level 0 means observed absence only."
)

_FREQUENCY_INSTRUCTION = (
    "Identify every DISTINCT instance of {desc} in the passage. Return a list "
    "`instances`; each element is an object with:\n"
    '  "evidence": a SHORT VERBATIM quote from the passage containing the instance,\n'
    '  "label": a brief label of the instance.\n'
    "Do NOT return a number — the program will count your validated instances "
    "and normalize to a rate per 1000 tokens. "
    "If there are no instances, return an empty list."
)


def _rubric(feature_id: str, protocol: str, instruction: str,
            levels: tuple[ScaleLevel, ...] = (), unit: str = "instances",
            min_evidence: int = 1, protocol_version: str = SCHEMA_VERSION) -> MeasurementRubric:
    return MeasurementRubric(
        feature_id=feature_id, protocol=protocol,
        protocol_version=protocol_version, levels=levels, unit=unit,
        min_evidence=min_evidence, instruction=instruction,
    )


class RubricRegistry:
    """feature_id → MeasurementRubric 的注册表（与 FeatureRegistry 解耦）。"""

    def __init__(self) -> None:
        self._rubrics: dict[str, MeasurementRubric] = {}

    def register(self, rubric: MeasurementRubric) -> None:
        if rubric.feature_id in self._rubrics:
            raise ValueError(f"重复的 rubric feature id: {rubric.feature_id}")
        self._rubrics[rubric.feature_id] = rubric

    def get(self, feature_id: str) -> MeasurementRubric:
        return self._rubrics[feature_id]

    def has(self, feature_id: str) -> bool:
        return feature_id in self._rubrics

    def all(self) -> list[MeasurementRubric]:
        return list(self._rubrics.values())

    def __len__(self) -> int:
        return len(self._rubrics)


def build_default_rubrics() -> RubricRegistry:
    """V0.1 LLM 特征的默认测量协议（frequency 与 ordinal 两类）。"""
    reg = RubricRegistry()

    # —— frequency-like（程序对已验证实例计数，并归一化为每 1000 词的率）——
    reg.register(_rubric("metaphor_frequency", FREQUENCY_PROTOCOL,
                         _FREQUENCY_INSTRUCTION.format(desc="metaphor"),
                         unit=FREQUENCY_UNIT, min_evidence=1))
    reg.register(_rubric("simile_frequency", FREQUENCY_PROTOCOL,
                         _FREQUENCY_INSTRUCTION.format(desc="simile"),
                         unit=FREQUENCY_UNIT, min_evidence=1))
    reg.register(_rubric("irony_frequency", FREQUENCY_PROTOCOL,
                         _FREQUENCY_INSTRUCTION.format(desc="ironic statement or event"),
                         unit=FREQUENCY_UNIT, min_evidence=1))

    # —— intensity / degree-like（锚定序数 0–4）——
    reg.register(_rubric("irony_intensity", ORDINAL_PROTOCOL,
                         _INTENSITY_INSTRUCTION.format(
                             desc="irony", desc_absent="no irony is present"),
                         levels=_ABSENT_WEAK_MODERATE_STRONG_DOMINANT, unit="ordinal",
                         min_evidence=1))
    reg.register(_rubric("narrator_evaluative_intervention", ORDINAL_PROTOCOL,
                         _INTENSITY_INSTRUCTION.format(
                             desc="explicit narrator evaluation or commentary",
                             desc_absent="the narrator makes no evaluative intervention"),
                         levels=_ABSENT_WEAK_MODERATE_STRONG_DOMINANT, unit="ordinal",
                         min_evidence=1))
    reg.register(_rubric("psychological_representation", ORDINAL_PROTOCOL,
                         _INTENSITY_INSTRUCTION.format(
                             desc="psychological representation of a character's inner state",
                             desc_absent="no psychological representation is present"),
                         levels=_ABSENT_WEAK_MODERATE_STRONG_DOMINANT, unit="ordinal",
                         min_evidence=1))
    reg.register(_rubric("emotional_restraint", ORDINAL_PROTOCOL,
                         _INTENSITY_INSTRUCTION.format(
                             desc="emotional restraint (understatement of affect)",
                             desc_absent="no emotional restraint is present"),
                         levels=_ABSENT_WEAK_MODERATE_STRONG_DOMINANT, unit="ordinal",
                         min_evidence=1))
    reg.register(_rubric("emotional_intensity", ORDINAL_PROTOCOL,
                         _INTENSITY_INSTRUCTION.format(
                             desc="emotional intensity of the passage",
                             desc_absent="the passage is emotionally neutral"),
                         levels=_ABSENT_WEAK_MODERATE_STRONG_DOMINANT, unit="ordinal",
                         min_evidence=1))

    return reg
