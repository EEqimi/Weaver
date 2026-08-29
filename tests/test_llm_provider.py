# tests/test_llm_provider.py
"""LLM 分析器与 provider 抽象测试（spec §12）：无 provider 显式不可用、
schema 校验、malformed 响应报错、缓存键可复现。"""
import json
import os
from unittest import mock
from urllib.error import HTTPError

import pytest

from knowledge.analysis.base import (
    AnalysisUnavailable, LLMNotConfiguredError, LLMResponseError, parse_json_response,
)
from knowledge.analysis.narrative_analyzer import NarrativeAnalyzer
from knowledge.analysis.style_analyzer import LLMFeatureAnalyzer
from knowledge.providers.llm_provider import (
    CacheBackedLLMProvider, DashScopeProvider, DeepSeekProvider, DummyLLMProvider,
    LLMCache, LLMTransportError, OpenAICompatibleProvider,
    UnconfiguredLLMProvider, cache_key,
)
from knowledge.schema.feature_registry import build_default_registry

PASSAGE = "He walked alone down the lane, and did not once look back."


def _judgment_feature():
    # frequency 协议特征：LLM 识别实例，程序对已验证实例计数
    return build_default_registry().get("irony_frequency")


def _ordinal_feature():
    # ordinal 协议特征：锚定序数 0–4，程序校验档位
    return build_default_registry().get("irony_intensity")


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
def test_llm_feature_analyzer_valid_frequency_response():
    # frequency 协议：LLM 返回 instances 列表，程序对已验证实例计数并归一化为
    # 每 1000 词的率（task item 1）。PASSAGE 共 12 词。
    resp = json.dumps({"instances": [{"evidence": "did not once look back",
                                      "label": "negation"}],
                       "confidence": 0.9, "reasoning_summary": "克制表达"})
    a = LLMFeatureAnalyzer(DummyLLMProvider(response=resp))
    fv = a.analyze(PASSAGE, _judgment_feature(), chunk_id="c1")
    assert fv.feature_id == "irony_frequency"
    assert fv.raw_value == 1.0                  # raw_count = 已验证实例数
    assert fv.value == pytest.approx(1000 / 12)  # value = 1 / 12 × 1000
    assert fv.evidence == ["did not once look back"]
    assert fv.provenance["chunk_id"] == "c1"
    assert fv.provenance["raw_count"] == 1
    assert fv.provenance["exposure_tokens"] == 12
    assert fv.provenance["unit"] == "instances per 1000 tokens"
    assert fv.provenance["n_instances_verified"] == 1


def test_llm_feature_analyzer_valid_ordinal_response():
    # ordinal 协议：LLM 返回锚定档位 level + 证据，程序校验档位（task item 1）
    resp = json.dumps({"level": 2, "confidence": 0.7,
                       "evidence": ["did not once look back"],
                       "reasoning_summary": "中等强度"})
    a = LLMFeatureAnalyzer(DummyLLMProvider(response=resp))
    fv = a.analyze(PASSAGE, _ordinal_feature(), chunk_id="c1")
    assert fv.feature_id == "irony_intensity"
    assert fv.value == 2.0
    assert fv.provenance["level_label"] == "moderate"


def test_llm_feature_analyzer_ordinal_level_out_of_range():
    resp = json.dumps({"level": 9, "confidence": 0.5, "evidence": []})
    a = LLMFeatureAnalyzer(DummyLLMProvider(response=resp))
    with pytest.raises(LLMResponseError):
        a.analyze(PASSAGE, _ordinal_feature())


def test_llm_feature_analyzer_rejects_unverified_confident_positive():
    # 高置信正向判定但证据无法逐字对应 passage → 报错，绝不静默接受编造引文
    resp = json.dumps({"instances": [{"evidence": "completely fabricated quote",
                                      "label": "x"}],
                       "confidence": 0.9, "reasoning_summary": "..."})
    a = LLMFeatureAnalyzer(DummyLLMProvider(response=resp))
    with pytest.raises(LLMResponseError):
        a.analyze(PASSAGE, _judgment_feature())


def test_llm_feature_analyzer_missing_instances():
    resp = json.dumps({"confidence": 0.9, "evidence": []})
    a = LLMFeatureAnalyzer(DummyLLMProvider(response=resp))
    with pytest.raises(LLMResponseError):
        a.analyze(PASSAGE, _judgment_feature())


def test_llm_feature_analyzer_non_numeric_ordinal_level():
    resp = json.dumps({"level": "high", "confidence": 0.9, "evidence": []})
    a = LLMFeatureAnalyzer(DummyLLMProvider(response=resp))
    with pytest.raises(LLMResponseError):
        a.analyze(PASSAGE, _ordinal_feature())


def test_llm_feature_analyzer_confidence_out_of_range():
    resp = json.dumps({"instances": [], "confidence": 1.7})
    a = LLMFeatureAnalyzer(DummyLLMProvider(response=resp))
    with pytest.raises(LLMResponseError):
        a.analyze(PASSAGE, _judgment_feature())


def test_llm_feature_analyzer_frequency_normalizes_to_rate():
    # task item 1：多条实例 → value = raw_count / tokens × 1000
    resp = json.dumps({"instances": [
        {"evidence": "walked alone", "label": "a"},
        {"evidence": "did not once look back", "label": "b"},
    ], "confidence": 0.8, "reasoning_summary": "..."})
    a = LLMFeatureAnalyzer(DummyLLMProvider(response=resp))
    fv = a.analyze(PASSAGE, _judgment_feature())
    assert fv.raw_value == 2.0
    assert fv.value == pytest.approx(2 * 1000 / 12)
    assert fv.provenance["raw_count"] == 2
    assert fv.provenance["exposure_tokens"] == 12


# ---- ordinal 评估状态（task item 2）----
def test_llm_feature_analyzer_ordinal_not_observable():
    resp = json.dumps({"assessment_status": "not_observable", "level": None,
                       "confidence": 0.5, "reasoning_summary": "..."})
    a = LLMFeatureAnalyzer(DummyLLMProvider(response=resp))
    fv = a.analyze(PASSAGE, _ordinal_feature(), chunk_id="c1")
    assert fv.value is None                 # 绝不折算成 0
    assert fv.raw_value is None
    assert fv.provenance["assessment_status"] == "not_observable"


def test_llm_feature_analyzer_ordinal_insufficient_evidence():
    resp = json.dumps({"assessment_status": "insufficient_evidence", "level": None,
                       "confidence": 0.4, "reasoning_summary": "..."})
    a = LLMFeatureAnalyzer(DummyLLMProvider(response=resp))
    fv = a.analyze(PASSAGE, _ordinal_feature())
    assert fv.value is None
    assert fv.provenance["assessment_status"] == "insufficient_evidence"


def test_llm_feature_analyzer_ordinal_observed_requires_null_level_when_unobservable():
    # 状态非 observed 时 level 必须为 null，否则报错
    resp = json.dumps({"assessment_status": "not_observable", "level": 0,
                       "confidence": 0.5})
    a = LLMFeatureAnalyzer(DummyLLMProvider(response=resp))
    with pytest.raises(LLMResponseError):
        a.analyze(PASSAGE, _ordinal_feature())


def test_llm_feature_analyzer_ordinal_invalid_status():
    resp = json.dumps({"assessment_status": "maybe", "level": None, "confidence": 0.5})
    a = LLMFeatureAnalyzer(DummyLLMProvider(response=resp))
    with pytest.raises(LLMResponseError):
        a.analyze(PASSAGE, _ordinal_feature())


# ---- narrative 证据充分性（task item 3）----
def test_narrative_downgrades_high_confidence_without_verified_evidence():
    # 高置信 + 实质判断（pov=third）+ 零已验证证据 → 确定性降级
    resp = json.dumps({"pov": "third", "focalization": "internal",
                       "perspective_stability": "stable", "narrative_distance": "medium",
                       "narrator_presence": "low", "narrator_evaluative_intervention": "low",
                       "information_access": "limited", "temporal_order": "chronological",
                       "observed_evidence": ["totally fabricated quote"],
                       "confidence": 0.95})
    a = NarrativeAnalyzer(DummyLLMProvider(response=resp))
    obs = a.analyze(PASSAGE)
    assert obs.confidence == 0.0
    assert "high_confidence_substantive_without_verified_evidence" in obs.evidence_issues


def test_narrative_keeps_verified_evidence_confidence():
    # 有已验证证据时，高置信不降级
    resp = json.dumps({"pov": "third", "focalization": "internal",
                       "perspective_stability": "stable", "narrative_distance": "medium",
                       "narrator_presence": "low", "narrator_evaluative_intervention": "low",
                       "information_access": "limited", "temporal_order": "chronological",
                       "observed_evidence": ["did not once look back"],
                       "confidence": 0.95})
    a = NarrativeAnalyzer(DummyLLMProvider(response=resp))
    obs = a.analyze(PASSAGE)
    assert obs.confidence == 0.95
    assert obs.evidence_issues == []


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


# ---- OpenAI 兼容 provider（真实 HTTP 后端，标准库实现，无第三方依赖）----
def _http_response(content: str, usage: dict | None = None) -> mock.MagicMock:
    data = {"choices": [{"message": {"content": content}}]}
    if usage is not None:
        data["usage"] = usage
    resp = mock.MagicMock()
    resp.read.return_value = json.dumps(data).encode("utf-8")
    cm = mock.MagicMock()
    cm.__enter__.return_value = resp
    return cm


def test_openai_provider_unconfigured_without_key():
    p = OpenAICompatibleProvider(api_key="")
    assert p.is_configured() is False


def test_openai_provider_configured_with_key():
    p = OpenAICompatibleProvider(api_key="test-key")
    assert p.is_configured() is True


def test_openai_provider_complete_unconfigured_raises():
    p = OpenAICompatibleProvider(api_key="")
    with pytest.raises(LLMNotConfiguredError):
        p.complete([{"role": "user", "content": "hi"}])


def test_openai_provider_success_accumulates_usage():
    usage = {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}
    p = OpenAICompatibleProvider(api_key="test-key")
    with mock.patch("knowledge.providers.llm_provider.urllib.request.urlopen",
                    return_value=_http_response("ok", usage)):
        out = p.complete([{"role": "user", "content": "hi"}])
    assert out == "ok"
    assert p.n_calls == 1
    assert p.n_success == 1
    assert p.n_retries == 0
    assert p.usage == usage


def test_openai_provider_retries_transient_then_raises():
    # 429 视为瞬态：按确定性退避重试，耗尽后抛 LLMTransportError（区别于 schema 失败）
    p = OpenAICompatibleProvider(api_key="test-key", max_retries=2)
    err = HTTPError("http://x", 429, "Too Many Requests", {}, None)
    with mock.patch("knowledge.providers.llm_provider.urllib.request.urlopen",
                    side_effect=err), \
         mock.patch("knowledge.providers.llm_provider.time.sleep") as sleep:
        with pytest.raises(LLMTransportError):
            p.complete([{"role": "user", "content": "hi"}])
    assert p.n_calls == 1
    assert p.n_success == 0
    assert p.n_retries == 2
    assert sleep.call_count == 2


def test_openai_provider_does_not_retry_permanent_4xx():
    p = OpenAICompatibleProvider(api_key="test-key", max_retries=2)
    err = HTTPError("http://x", 400, "Bad Request", {}, None)
    with mock.patch("knowledge.providers.llm_provider.urllib.request.urlopen",
                    side_effect=err):
        with pytest.raises(LLMTransportError):
            p.complete([{"role": "user", "content": "hi"}])
    assert p.n_retries == 0


def test_openai_provider_accumulate_usage_ignores_non_numeric():
    p = OpenAICompatibleProvider(api_key="test-key")
    p._accumulate_usage({"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7})
    p._accumulate_usage({"prompt_tokens": "x", "completion_tokens": True})
    assert p.usage == {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}


def test_openai_provider_captures_http_error_body():
    # 400 响应体应被读入错误详情（定位内容审核/参数越界等真实原因）
    import io
    err = HTTPError("http://x", 400, "Bad Request", {},
                    io.BytesIO(b'{"error": {"message": "content filtered"}}'))
    detail = OpenAICompatibleProvider._error_detail(err)
    assert "400" in detail and "content filtered" in detail


def test_openai_provider_error_detail_non_http():
    detail = OpenAICompatibleProvider._error_detail(KeyError("boom"))
    assert "KeyError" in detail


# ---- 真实 bug 回归（Request C）：畸形生成响应绝不抛裸 AttributeError ----
def _raw_http_response(raw_body: str) -> mock.MagicMock:
    """构造一个返回原始（可能是畸形）响应体的 urlopen 上下文管理器。"""
    resp = mock.MagicMock()
    resp.read.return_value = raw_body.encode("utf-8")
    cm = mock.MagicMock()
    cm.__enter__.return_value = resp
    return cm


@pytest.mark.parametrize("raw_body", [
    '[1, 2, 3]',                                   # 顶层非 object
    '"just a string"',                             # 顶层非 object
    '{"choices": "not-a-list", "usage": {}}',      # choices 非 array
    '{"choices": [null], "usage": {}}',            # choices[0] 非 object
    '{"choices": [{"message": {"content": [1, 2]},'  # content 非字符串
    ' "finish_reason": "stop"}], "usage": {}}',
])
def test_openai_provider_malformed_response_raises_transport_error(raw_body):
    """旧代码对畸形响应会抛裸 AttributeError/TypeError（不在 except 元组里，一路逃逸
    到 UI 显示 "生成失败：AttributeError"）。修复后改为明确的 LLMTransportError。"""
    p = OpenAICompatibleProvider(api_key="test-key")
    with mock.patch("knowledge.providers.llm_provider.urllib.request.urlopen",
                    return_value=_raw_http_response(raw_body)):
        with pytest.raises(LLMTransportError):
            p.complete_with_metadata([{"role": "user", "content": "hi"}])


def test_openai_provider_null_message_yields_empty_content_not_attributeerror():
    """message 为 null 时旧代码 `None.get("content")` 抛 AttributeError；修复后返回
    空内容，由下游 GeneratedPassage 以 GenerationError 拒绝（绝不裸抛 AttributeError）。"""
    p = OpenAICompatibleProvider(api_key="test-key")
    raw = '{"choices": [{"message": null, "finish_reason": "stop"}], "usage": {}}'
    with mock.patch("knowledge.providers.llm_provider.urllib.request.urlopen",
                    return_value=_raw_http_response(raw)):
        out = p.complete_with_metadata([{"role": "user", "content": "hi"}])
    assert out["content"] == ""
    assert out["finish_reason"] == "stop"


# ---- 具体 provider 预设：DeepSeek（新默认）与 DashScope（保留）----
def test_deepseek_provider_defaults():
    p = DeepSeekProvider(api_key="sk-test")
    assert p.provider_id == "deepseek"
    assert p.model == "deepseek-chat"
    assert p._base_url == "https://api.deepseek.com"
    assert p.is_configured() is True


def test_deepseek_provider_unconfigured_without_key():
    with mock.patch.dict(os.environ, {"DEEPSEEK_API_KEY": ""}):
        p = DeepSeekProvider()
        assert p.is_configured() is False


def test_deepseek_provider_reads_env_key():
    with mock.patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-env"}):
        p = DeepSeekProvider()
        assert p.is_configured() is True


def test_dashscope_provider_preserves_legacy_defaults():
    p = DashScopeProvider(api_key="sk-test")
    assert p.provider_id == "dashscope"
    assert p.model == "qwen-plus"
    assert p._base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"


# ---- 缓存命中/未命中计量 ----
def test_cache_backed_provider_counts_hits_and_misses(tmp_path):
    inner = DummyLLMProvider(response="hello", provider_id="dummy", model="m")
    p = CacheBackedLLMProvider(inner, LLMCache(tmp_path / "cache"))
    assert p.complete([{"role": "user", "content": "x"}], cache_hint="k1") == "hello"
    assert (p.cache_hits, p.cache_misses) == (0, 1)
    assert p.complete([{"role": "user", "content": "x"}], cache_hint="k1") == "hello"
    assert (p.cache_hits, p.cache_misses) == (1, 1)
    # 无 cache_hint 的调用不改变命中/未命中计数
    p.complete([{"role": "user", "content": "x"}])
    assert (p.cache_hits, p.cache_misses) == (1, 1)
