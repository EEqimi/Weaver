# knowledge/analysis/style_analyzer.py
"""Layer A 的 LLM 判断/混合特征分析器（Phase 3 §3）。

把"文本 → analyzer → 结构化判断 → confidence → evidence → FeatureValue"这条
链路做通，但**只用于受控采样**，绝不对全语料跑。

安全约定：
    - 结果必须附带逐字 evidence 引用，禁止无依据的文学断言；
    - 只存 reasoning_summary，不存隐藏思维链；
    - 默认盲测（不注入作者身份），避免确认偏差（spec §11）；
    - 无 provider 时返回 AnalysisUnavailable，绝不伪造；
    - 结果按稳定缓存键缓存（text hash + 版本 + model/provider）。
"""
from __future__ import annotations

from ..providers.llm_provider import LLMProvider, cache_key
from ..schema.feature_registry import FeatureDefinition
from ..schema.style_schema import FeatureValue
from ..schema.versions import FEATURE_SCHEMA_VERSION, LLM_ANALYZER_VERSION
from .base import AnalysisUnavailable, LLMResponseError, parse_json_response

ANALYZER_ID = "LlmFeatureAnalyzer"
ANALYZER_VERSION = LLM_ANALYZER_VERSION

_SYSTEM_PROMPT = (
    "You are a careful literary stylometric analyst. You will be shown a short "
    "text passage and asked to measure ONE specific style feature.\n"
    "Rules:\n"
    "- Base every judgment on explicit textual evidence from the passage.\n"
    "- Do not invent claims the text does not support.\n"
    "- Do not assume or mention the author's identity.\n"
    "- Return ONLY a JSON object, no prose, no markdown fences.\n"
    "The JSON object must have exactly these keys:\n"
    '  "value": the measured value (a number for continuous features; '
    'a short string for categorical/discrete features),\n'
    '  "confidence": a number between 0 and 1 for how well-evidenced the judgment is,\n'
    '  "evidence": an array of 1-5 SHORT VERBATIM quotes from the passage that '
    'support the judgment,\n'
    '  "reasoning_summary": a concise 1-3 sentence justification (no hidden reasoning).\n'
)


def _value_type_hint(value_type: str) -> str:
    if value_type == "continuous":
        return "a single number (float)"
    if value_type == "categorical":
        return "a single category label (string)"
    if value_type == "distribution":
        return "a JSON object mapping categories to proportions (numbers summing to 1)"
    return "a single value"


class LLMFeatureAnalyzer:
    def __init__(self, provider: LLMProvider, blind: bool = True):
        self._provider = provider
        self.blind = blind

    def analyze(self, text: str, feature: FeatureDefinition, chunk_id: str = "",
                author: str | None = None) -> FeatureValue | AnalysisUnavailable:
        if not self._provider.is_configured():
            return AnalysisUnavailable(feature.id, ANALYZER_ID, ANALYZER_VERSION,
                                       "未配置 LLM provider")

        user = (
            f"Feature to measure: {feature.id} ({feature.category}).\n"
            f"Description: {feature.description or feature.id}.\n"
            f"Expected value type: {_value_type_hint(feature.value_type.value)}.\n\n"
            "PASSAGE:\n"
            f'"""{text}"""'
        )
        messages = [{"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user}]
        key = cache_key(
            text=text, analyzer_id=ANALYZER_ID, analyzer_version=ANALYZER_VERSION,
            schema_version=FEATURE_SCHEMA_VERSION, model=self._provider.model,
            provider_id=self._provider.provider_id,
            prompt_name=f"llm_feature:{feature.id}:blind={self.blind}",
        )
        raw = self._provider.complete(messages, cache_hint=key)
        data = parse_json_response(raw)
        return self._to_feature_value(feature, chunk_id, data)

    @staticmethod
    def _to_feature_value(feature: FeatureDefinition, chunk_id: str,
                          data: dict) -> FeatureValue:
        if "value" not in data:
            raise LLMResponseError(f"LLM 响应缺少 value: {data}")
        value = data["value"]
        if feature.value_type.value == "continuous":
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise LLMResponseError(f"continuous 特征 value 必须是数值: {value!r}")
            value = float(value)
        confidence = data.get("confidence")
        if confidence is not None:
            confidence = float(confidence)
            if not 0.0 <= confidence <= 1.0:
                raise LLMResponseError(f"confidence 越界: {confidence}")
        evidence = data.get("evidence", [])
        if not isinstance(evidence, list) or not all(isinstance(e, str) for e in evidence):
            raise LLMResponseError("evidence 必须是字符串列表")

        return FeatureValue(
            feature_id=feature.id,
            value=value,
            raw_value=value,
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
            provenance={"chunk_id": chunk_id, "reasoning_summary": data.get("reasoning_summary", "")},
        )
