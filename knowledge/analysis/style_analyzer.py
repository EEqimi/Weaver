# knowledge/analysis/style_analyzer.py
"""Layer A 的 LLM 判断/混合特征分析器（Phase 3 §3 + Phase 3–4.1 标定就绪）。

把"文本 → analyzer → 结构化判断 → confidence → evidence → FeatureValue"这条
链路做通，但**只用于受控采样**，绝不对全语料跑。

标定就绪（task item 1/5）：
    - 每个 LLM 特征都从 RubricRegistry 取显式测量协议，绝不要求模型"返回一个
      float"；
    - frequency 特征：LLM 识别证据实例，程序对**已验证**实例计数；
    - ordinal 特征：锚定序数量表（每档有显式定义），程序校验档位；
    - 所有 evidence 经共享校验（analysis/evidence.py）逐字比对 passage；
    - 高置信正向判定必须满足最小已验证证据数，否则显式报错，绝不静默接受
      可能被编造的引文。

安全约定（沿用）：
    - 结果必须附带逐字 evidence 引用，禁止无依据的文学断言；
    - 只存 reasoning_summary，不存隐藏思维链；
    - 默认盲测（不注入作者身份），避免确认偏差（spec §11）；
    - 无 provider 时返回 AnalysisUnavailable，绝不伪造；
    - 结果按稳定缓存键缓存（text hash + 版本 + model/provider）。
"""
from __future__ import annotations

from ..providers.llm_provider import LLMProvider, cache_key
from ..schema.feature_registry import FeatureDefinition
from ..schema.rubrics import (
    ASSESSMENT_OBSERVED, ASSESSMENT_STATUSES, FREQUENCY_DENOMINATOR, FREQUENCY_UNIT,
    MeasurementRubric, RubricRegistry, build_default_rubrics,
)
from ..schema.style_schema import FeatureValue
from ..schema.versions import FEATURE_SCHEMA_VERSION, LLM_ANALYZER_VERSION
from .base import AnalysisUnavailable, LLMResponseError, parse_json_response
from .evidence import verify_evidence_quotes
from .text_utils import tokens

ANALYZER_ID = "LlmFeatureAnalyzer"
ANALYZER_VERSION = LLM_ANALYZER_VERSION

# 高置信阈值：conf 达到该值即视为"高置信"，须满足最小已验证证据数
_CONFIDENT_THRESHOLD = 0.6

_SYSTEM_PROMPT_HEAD = (
    "You are a careful literary stylometric analyst. You will be shown a short "
    "text passage and asked to measure ONE specific style feature.\n"
    "Rules:\n"
    "- Base every judgment on explicit textual evidence from the passage.\n"
    "- Do not invent claims the text does not support.\n"
    "- Do not assume or mention the author's identity.\n"
    "- Evidence quotes must be SHORT and VERBATIM from the passage.\n"
    "- Return ONLY a JSON object, no prose, no markdown fences.\n\n"
    "MEASUREMENT PROTOCOL (protocol_version={protocol_version}):\n"
    "{instruction}\n\n"
    "The JSON object must contain exactly these keys:\n"
    "{schema_description}\n"
    '  "confidence": a number between 0 and 1 for how well-evidenced the judgment is,\n'
    '  "reasoning_summary": a concise 1-3 sentence justification (no hidden reasoning).\n'
)


def _ordinal_schema_description(rubric: MeasurementRubric) -> str:
    levels = ", ".join(f"{l.value}={l.label}" for l in rubric.levels)
    statuses = ", ".join(ASSESSMENT_STATUSES)
    return (
        f'  "assessment_status": one of {{{statuses}}} (observed / insufficient_evidence '
        f'/ not_observable),\n'
        f'  "level": when assessment_status is "observed", an integer in {{{levels}}} '
        f'(the anchored scale value); otherwise null,\n'
        f'  "evidence": an array of 1-5 SHORT VERBATIM quotes from the passage that '
        f'support the assigned level,\n'
    )


def _frequency_schema_description(rubric: MeasurementRubric) -> str:
    return (
        '  "instances": an array of objects, each with "evidence" (SHORT VERBATIM '
        'quote) and "label" (a short label) — empty array if none,\n'
    )


class LLMFeatureAnalyzer:
    def __init__(self, provider: LLMProvider, blind: bool = True,
                 rubrics: RubricRegistry | None = None):
        self._provider = provider
        self.blind = blind
        self._rubrics = rubrics if rubrics is not None else build_default_rubrics()

    def analyze(self, text: str, feature: FeatureDefinition, chunk_id: str = "",
                author: str | None = None) -> FeatureValue | AnalysisUnavailable:
        if not self._provider.is_configured():
            return AnalysisUnavailable(feature.id, ANALYZER_ID, ANALYZER_VERSION,
                                       "未配置 LLM provider")

        rubric = self._rubric_for(feature)
        user = (
            f"Feature to measure: {feature.id} ({feature.category}).\n"
            f"Description: {feature.description or feature.id}.\n\n"
            "PASSAGE:\n"
            f'"""{text}"""'
        )
        system = _SYSTEM_PROMPT_HEAD.format(
            protocol_version=rubric.protocol_version,
            instruction=rubric.instruction,
            schema_description=(_ordinal_schema_description(rubric)
                                if rubric.is_ordinal()
                                else _frequency_schema_description(rubric)),
        )
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        key = cache_key(
            text=text, analyzer_id=ANALYZER_ID, analyzer_version=ANALYZER_VERSION,
            schema_version=FEATURE_SCHEMA_VERSION, model=self._provider.model,
            provider_id=self._provider.provider_id,
            prompt_name=(f"llm_feature:{feature.id}:blind={self.blind}:"
                         f"protocol={rubric.protocol}:v={rubric.protocol_version}"),
        )
        raw = self._provider.complete(messages, cache_hint=key)
        data = parse_json_response(raw)
        return self._to_feature_value(feature, rubric, chunk_id, text, data)

    def _rubric_for(self, feature: FeatureDefinition) -> MeasurementRubric:
        if not self._rubrics.has(feature.id):
            raise LLMResponseError(
                f"特征 {feature.id} 缺少测量协议（rubric）。所有 LLM 派生特征"
                f"必须在 RubricRegistry 中声明 protocol，禁止裸 float 输出。")
        rubric = self._rubrics.get(feature.id)
        if not rubric.is_frequency() and not rubric.is_ordinal():
            raise LLMResponseError(
                f"特征 {feature.id} 的 rubric protocol 非法: {rubric.protocol!r}")
        return rubric

    @staticmethod
    def _confidence(data: dict) -> float | None:
        confidence = data.get("confidence")
        if confidence is None:
            return None
        if isinstance(confidence, bool):
            raise LLMResponseError(f"confidence 必须是数值: {confidence!r}")
        confidence = float(confidence)
        if not 0.0 <= confidence <= 1.0:
            raise LLMResponseError(f"confidence 越界: {confidence}")
        return confidence

    @staticmethod
    def _reasoning(data: dict) -> str:
        rs = data.get("reasoning_summary", "")
        return rs if isinstance(rs, str) else ""

    def _to_feature_value(self, feature: FeatureDefinition, rubric: MeasurementRubric,
                          chunk_id: str, text: str, data: dict) -> FeatureValue:
        confidence = self._confidence(data)
        reasoning = self._reasoning(data)

        if rubric.is_frequency():
            return self._to_frequency_value(feature, rubric, chunk_id, text, data,
                                            confidence, reasoning)
        return self._to_ordinal_value(feature, rubric, chunk_id, text, data,
                                      confidence, reasoning)

    def _to_frequency_value(self, feature, rubric, chunk_id, text, data,
                            confidence, reasoning) -> FeatureValue:
        instances = data.get("instances")
        if not isinstance(instances, list):
            raise LLMResponseError(
                f"frequency 特征 {feature.id} 的响应缺少 instances 列表")
        check = verify_evidence_quotes(instances, text)
        # "正向判定" = 模型声称存在实例（len(instances) > 0），而非已验证计数：
        # 声称有实例却全部无法逐字验证 → 高置信正向判定缺证据，拒绝。
        self._enforce_evidence_count(feature, rubric, confidence, len(instances) > 0,
                                     check.n_verified)
        evidence = [self._quote_text(e) for e in check.verified]

        # 真实频率归一化（task item 1）：value = 已验证实例数 / 词数 × 1000。
        # 分母用项目统一 tokenizer（text_utils.tokens），绝不使用 LLM 输出归一化。
        raw_count = check.n_verified
        exposure = len(tokens(text))
        value = (raw_count * float(FREQUENCY_DENOMINATOR) / exposure) if exposure > 0 else 0.0

        return FeatureValue(
            feature_id=feature.id,
            value=value,
            raw_value=float(raw_count),
            normalized_value=None,
            value_type=feature.value_type.value,
            measurement_type=feature.measurement_type.value,
            confidence=confidence,
            evidence=evidence,
            sample_count=1,
            variance=None,
            analyzer_id=ANALYZER_ID,
            analyzer_version=ANALYZER_VERSION,
            schema_version=FEATURE_SCHEMA_VERSION,
            provenance={
                "chunk_id": chunk_id,
                "reasoning_summary": reasoning,
                "raw_count": raw_count,
                "exposure_tokens": exposure,
                "unit": FREQUENCY_UNIT,
                "n_instances_identified": len(instances),
                "n_instances_verified": check.n_verified,
                "n_instances_unverified": check.n_unverified,
                "instance_labels": [self._instance_label(e) for e in check.verified],
                "unverified_evidence": [self._quote_text(e) for e in check.unverified],
            },
        )

    def _to_ordinal_value(self, feature, rubric, chunk_id, text, data,
                          confidence, reasoning) -> FeatureValue:
        # 评估状态（task item 2）：显式区分"观察到"与"无法评估"，绝不把后者折算成 0。
        status = data.get("assessment_status", ASSESSMENT_OBSERVED)
        if status not in ASSESSMENT_STATUSES:
            raise LLMResponseError(
                f"ordinal 特征 {feature.id} 的 assessment_status={status!r} 非法")

        if status != ASSESSMENT_OBSERVED:
            # 无法评估：value / raw_value 置 None，保留状态，绝不折算成 0。
            level = data.get("level")
            if level is not None:
                raise LLMResponseError(
                    f"ordinal 特征 {feature.id} 状态 {status} 时 level 必须为 null，"
                    f"实际 {level!r}")
            return FeatureValue(
                feature_id=feature.id,
                value=None,
                raw_value=None,
                normalized_value=None,
                value_type=feature.value_type.value,
                measurement_type=feature.measurement_type.value,
                confidence=confidence,
                evidence=[],
                sample_count=1,
                variance=None,
                analyzer_id=ANALYZER_ID,
                analyzer_version=ANALYZER_VERSION,
                schema_version=FEATURE_SCHEMA_VERSION,
                provenance={
                    "chunk_id": chunk_id,
                    "reasoning_summary": reasoning,
                    "assessment_status": status,
                },
            )

        if "level" not in data:
            raise LLMResponseError(
                f"ordinal 特征 {feature.id} 的响应缺少 level")
        level = data["level"]
        if isinstance(level, bool) or not isinstance(level, (int, float)):
            raise LLMResponseError(f"ordinal 特征 {feature.id} 的 level 必须是整数: {level!r}")
        level_int = int(level)
        if float(level) != float(level_int):
            raise LLMResponseError(f"ordinal 特征 {feature.id} 的 level 必须是整数: {level!r}")
        allowed = {l.value for l in rubric.levels}
        if level_int not in allowed:
            raise LLMResponseError(
                f"ordinal 特征 {feature.id} 的 level={level_int} 不在量表 {sorted(allowed)} 中")

        raw_evidence = data.get("evidence", [])
        if not isinstance(raw_evidence, list) or not all(isinstance(e, str) for e in raw_evidence):
            raise LLMResponseError("evidence 必须是字符串列表")
        check = verify_evidence_quotes(raw_evidence, text)
        self._enforce_evidence_count(feature, rubric, confidence, level_int > 0,
                                     check.n_verified)

        level_label = next((l.label for l in rubric.levels if l.value == level_int), "")
        return FeatureValue(
            feature_id=feature.id,
            value=float(level_int),
            raw_value=float(level_int),
            normalized_value=None,
            value_type=feature.value_type.value,
            measurement_type=feature.measurement_type.value,
            confidence=confidence,
            evidence=[self._quote_text(e) for e in check.verified],
            sample_count=1,
            variance=None,
            analyzer_id=ANALYZER_ID,
            analyzer_version=ANALYZER_VERSION,
            schema_version=FEATURE_SCHEMA_VERSION,
            provenance={
                "chunk_id": chunk_id,
                "reasoning_summary": reasoning,
                "assessment_status": status,
                "level_label": level_label,
                "unverified_evidence": [self._quote_text(e) for e in check.unverified],
            },
        )

    @staticmethod
    def _enforce_evidence_count(feature, rubric, confidence, positive, n_verified):
        """高置信正向判定必须附带足够多的已验证证据，否则显式报错。"""
        if (confidence is not None and confidence >= _CONFIDENT_THRESHOLD
                and positive and n_verified < rubric.min_evidence):
            raise LLMResponseError(
                f"特征 {feature.id} 的高置信正向判定（conf={confidence:.2f}）缺少"
                f"已验证证据：需要 >= {rubric.min_evidence} 条，实际 {n_verified} 条。"
                f"拒绝静默接受无法验证的引文。")

    @staticmethod
    def _quote_text(item) -> str:
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            for k in ("evidence", "quote", "text"):
                if isinstance(item.get(k), str):
                    return item[k]
        return ""

    @staticmethod
    def _instance_label(item) -> str:
        if isinstance(item, dict):
            lbl = item.get("label")
            if isinstance(lbl, str):
                return lbl
        return ""
