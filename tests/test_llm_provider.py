# tests/test_llm_provider.py
"""LLM 分析器与 provider 抽象测试（spec §12）：无 provider 显式不可用、
schema 校验、malformed 响应报错、缓存键可复现。"""
import json

import pytest

from knowledge.analysis.base import AnalysisUnavailable, LLMResponseError, parse_json_response
from knowledge.analysis.narrative_analyzer import NarrativeAnalyzer
from knowledge.analysis.style_analyzer import LLMFeatureAnalyzer
from knowledge.providers.llm_provider import (
    DummyLLMProvider, UnconfiguredLLMProvider, cache_key,
)
from knowledge.schema.feature_registry import build_default_registry

PASSAGE = "He walked alone down the lane, and did not once look back."


def _judgment_feature():
    return build_default_registry().get("irony_frequency")


# ---- 无 provider：显式不可用，绝不伪造 ----
def test_llm_feature_analyzer_unavailable_without_provider():
    a = LLMFeatureAnalyzer(UnconfiguredLLMProvider())
    result = a.analyze(PASSAGE, _judgment_feature())
    assert isinstance(result, AnalysisUnavailable)
    assert result.status == "unavailable"
    assert result.analyzer_id == "LlmFeatureAnalyzer"


def test_narrative_analyzer_unavailable_without_provider():
    a = NarrativeAnalyzer(UnconfiguredLLMProvider())
    result = a.analyze(PASSAGE)
    assert isinstance(result, AnalysisUnavailable)


# ---- LLM schema 校验 ----
def test_llm_feature_analyzer_valid_response():
    resp = json.dumps({"value": 0.7, "confidence": 0.9,
                       "evidence": ["did not once look back"],
                       "reasoning_summary": "克制表达"})
    a = LLMFeatureAnalyzer(DummyLLMProvider(response=resp))
    fv = a.analyze(PASSAGE, _judgment_feature(), chunk_id="c1")
    assert fv.feature_id == "irony_frequency"
    assert fv.value == 0.7
    assert fv.evidence == ["did not once look back"]
    assert fv.provenance["chunk_id"] == "c1"


def test_llm_feature_analyzer_missing_value():
    resp = json.dumps({"confidence": 0.9, "evidence": []})
    a = LLMFeatureAnalyzer(DummyLLMProvider(response=resp))
    with pytest.raises(LLMResponseError):
        a.analyze(PASSAGE, _judgment_feature())


def test_llm_feature_analyzer_non_numeric_continuous():
    resp = json.dumps({"value": "high", "confidence": 0.9, "evidence": []})
    a = LLMFeatureAnalyzer(DummyLLMProvider(response=resp))
    with pytest.raises(LLMResponseError):
        a.analyze(PASSAGE, _judgment_feature())


def test_llm_feature_analyzer_confidence_out_of_range():
    resp = json.dumps({"value": 0.5, "confidence": 1.7, "evidence": []})
    a = LLMFeatureAnalyzer(DummyLLMProvider(response=resp))
    with pytest.raises(LLMResponseError):
        a.analyze(PASSAGE, _judgment_feature())


def test_narrative_analyzer_rejects_illegal_enum():
    resp = json.dumps({"pov": "fourth", "focalization": "internal",
                       "perspective_stability": "stable", "narrative_distance": "medium",
                       "narrator_presence": "low", "narrator_evaluative_intervention": "low",
                       "information_access": "limited", "temporal_order": "chronological",
                       "observed_evidence": ["He walked alone"],
                       "confidence": 0.8})
    a = NarrativeAnalyzer(DummyLLMProvider(response=resp))
    with pytest.raises(ValueError):
        a.analyze(PASSAGE)


# ---- malformed 响应 ----
def test_parse_json_response_tolerates_fences():
    assert parse_json_response('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_response('prefix {"a": 1} suffix') == {"a": 1}


def test_parse_json_response_raises_on_garbage():
    with pytest.raises(LLMResponseError):
        parse_json_response("not json at all")


def test_llm_analyzer_raises_on_malformed():
    a = LLMFeatureAnalyzer(DummyLLMProvider(response="not json"))
    with pytest.raises(LLMResponseError):
        a.analyze(PASSAGE, _judgment_feature())


# ---- 缓存键可复现 ----
def test_cache_key_reproducible():
    k1 = cache_key(text=PASSAGE, analyzer_id="A", analyzer_version="0.1.0",
                   schema_version="0.1.0", model="m", provider_id="p",
                   prompt_name="n")
    k2 = cache_key(text=PASSAGE, analyzer_id="A", analyzer_version="0.1.0",
                   schema_version="0.1.0", model="m", provider_id="p",
                   prompt_name="n")
    assert k1 == k2


def test_cache_key_changes_with_text():
    k1 = cache_key(text=PASSAGE, analyzer_id="A", analyzer_version="0.1.0",
                   schema_version="0.1.0", model="m", provider_id="p",
                   prompt_name="n")
    k2 = cache_key(text="different text", analyzer_id="A", analyzer_version="0.1.0",
                   schema_version="0.1.0", model="m", provider_id="p",
                   prompt_name="n")
    assert k1 != k2


def test_cache_key_changes_with_version():
    k1 = cache_key(text=PASSAGE, analyzer_id="A", analyzer_version="0.1.0",
                   schema_version="0.1.0", model="m", provider_id="p",
                   prompt_name="n")
    k2 = cache_key(text=PASSAGE, analyzer_id="A", analyzer_version="0.2.0",
                   schema_version="0.1.0", model="m", provider_id="p",
                   prompt_name="n")
    assert k1 != k2
