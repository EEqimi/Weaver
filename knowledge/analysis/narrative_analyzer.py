# knowledge/analysis/narrative_analyzer.py
"""Layer B 叙事分析器（Phase 3 §4）。

对单个 chunk 输出结构化 NarrativeObservation：视角/聚焦/距离/信息/时间/详略，
区分"观察证据"与"解释"，并明确不得从单 chunk 推断作者级结论。

同样受控采样、盲测默认、无 provider 时返回 AnalysisUnavailable。
"""
from __future__ import annotations

from ..providers.llm_provider import LLMProvider, cache_key
from ..schema.narrative_schema import (
    DETAIL_DIMENSIONS, FOCALIZATION_VALUES, INFORMATION_ACCESS_VALUES,
    PACE_DIMENSIONS, POV_VALUES, PRESENCE_VALUES, STABILITY_VALUES,
    TEMPORAL_ORDER_VALUES, DISTANCE_VALUES, UNKNOWN_VALUES,
    NarrativeObservation, validate_narrative,
)
from ..schema.versions import NARRATIVE_ANALYZER_VERSION, NARRATIVE_SCHEMA_VERSION
from .base import AnalysisUnavailable, parse_json_response
from .evidence import verify_evidence_quotes

ANALYZER_ID = "NarrativeAnalyzer"
ANALYZER_VERSION = NARRATIVE_ANALYZER_VERSION

# 高置信阈值：conf 达到该值即视为"高置信实质判断"，须至少一条已验证证据（task item 3）
_HIGH_CONFIDENCE = 0.9
# 实质判断字段：任一字段给出非 unknown 的实际取值，即视为做出了实质叙事判断
_SUBSTANTIVE_FIELDS = ("pov", "focalization", "narrative_distance",
                       "narrator_presence", "information_access", "temporal_order")


def _is_substantive(obs: NarrativeObservation) -> bool:
    """实质判断：至少一个实质字段给出了非 unknown 的实际取值。"""
    return any(getattr(obs, f) not in UNKNOWN_VALUES for f in _SUBSTANTIVE_FIELDS)


def _build_system_prompt() -> str:
    return (
        "You are a narratologist analyzing a single text passage. Produce a "
        "structured observation of its narration. Base everything on the passage; "
        "do NOT infer author-level or work-level conclusions from this one chunk.\n"
        "Distinguish OBSERVED evidence (verbatim quotes) from INTERPRETATION.\n"
        "Do not assume the author's identity.\n"
        "Return ONLY a JSON object with these keys and allowed values:\n"
        f'  "pov": one of {list(POV_VALUES)},\n'
        f'  "focalization": one of {list(FOCALIZATION_VALUES)},\n'
        '  "focalizer": who/what is the focalizer (short string or null),\n'
        f'  "perspective_stability": one of {list(STABILITY_VALUES)},\n'
        f'  "narrative_distance": one of {list(DISTANCE_VALUES)},\n'
        f'  "narrator_presence": one of {list(PRESENCE_VALUES)},\n'
        f'  "narrator_evaluative_intervention": one of {list(PRESENCE_VALUES)},\n'
        f'  "information_access": one of {list(INFORMATION_ACCESS_VALUES)},\n'
        '  "information_withholding": short string or null,\n'
        '  "revelation_timing": short string or null,\n'
        f'  "temporal_order": one of {list(TEMPORAL_ORDER_VALUES)},\n'
        f'  "temporal_pace": object mapping {list(PACE_DIMENSIONS)} to proportions summing to 1,\n'
        f'  "scene_detail": object mapping {list(DETAIL_DIMENSIONS)} to proportions summing to 1,\n'
        '  "observed_evidence": array of 1-5 VERBATIM quotes from the passage,\n'
        '  "interpretation": concise 1-3 sentence interpretation,\n'
        '  "confidence": number in [0,1].\n'
    )


class NarrativeAnalyzer:
    def __init__(self, provider: LLMProvider, blind: bool = True):
        self._provider = provider
        self.blind = blind

    def analyze(self, text: str, chunk_id: str = "",
                author: str | None = None) -> NarrativeObservation | AnalysisUnavailable:
        if not self._provider.is_configured():
            return AnalysisUnavailable("narrative", ANALYZER_ID, ANALYZER_VERSION,
                                       "未配置 LLM provider")
        messages = [
            {"role": "system", "content": _build_system_prompt()},
            {"role": "user", "content": f'PASSAGE:\n"""{text}"""'},
        ]
        key = cache_key(
            text=text, analyzer_id=ANALYZER_ID, analyzer_version=ANALYZER_VERSION,
            schema_version=NARRATIVE_SCHEMA_VERSION, model=self._provider.model,
            provider_id=self._provider.provider_id,
            prompt_name=f"narrative:blind={self.blind}",
        )
        raw = self._provider.complete(messages, cache_hint=key)
        data = parse_json_response(raw)
        data = dict(data)
        data["chunk_id"] = chunk_id
        obs = validate_narrative(data)
        # 证据校验（task item 5）：逐字比对 passage，未验证引文显式标记，不静默丢弃
        check = verify_evidence_quotes(obs.observed_evidence, text)
        obs.observed_evidence = [e for e in check.verified if isinstance(e, str)]
        obs.unverified_evidence = [e for e in check.unverified if isinstance(e, str)]
        # 证据充分性（task item 3）：高置信实质判断却无已验证证据 → 确定性降级，
        # 绝不静默保留 confidence 0.9+ 且零证据的结果。
        if (obs.confidence >= _HIGH_CONFIDENCE and not obs.observed_evidence
                and _is_substantive(obs)):
            obs.confidence = 0.0
            obs.evidence_issues.append(
                "high_confidence_substantive_without_verified_evidence")
        return obs
