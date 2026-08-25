# tests/test_strategy_consolidation.py
"""Phase 4.5 作者级策略合并测试（spec Phase 4.5 §三/§四/§五/§七/§十四）。

全部确定性：不调用真实 LLM，用 DummyLLMProvider 或直接驱动纯函数校验。
"""
import json

import pytest

from knowledge.analysis.base import LLMResponseError
from knowledge.providers.llm_provider import DummyLLMProvider
from knowledge.schema.strategy_schema import (
    ConsolidationGroup, RawStrategy, StrategyEvidence, StrategyStatus,
    canonical_strategy_id,
)
from knowledge.strategies.consolidation import (
    ConsolidationError, StrategyConsolidator,
)


def _raw(sid, author="austen", name=None, desc="d", evidence=None,
         triggers=None, operations=None, effects=None):
    return RawStrategy(
        strategy_id=sid, author_id=author, name=name or f"strategy {sid}",
        description=desc, triggers=triggers or ["t"], operations=operations or ["o"],
        intended_effects=effects or ["e"], evidence=evidence or [],
    )


def _group(name, source_ids, desc="d"):
    return ConsolidationGroup(canonical_name=name, canonical_description=desc,
                              source_strategy_ids=source_ids)


# ---- §十四.1：author scope 隔离 ----
def test_author_scope_isolation_rejects_mixed_authors():
    raws = [_raw("a", author="austen"), _raw("b", author="dickens")]
    with pytest.raises(ConsolidationError):
        StrategyConsolidator.validate_author_scope(raws, "austen")


# ---- §十四.2：missing author ----
def test_missing_author_rejected():
    with pytest.raises(ConsolidationError):
        StrategyConsolidator.validate_author_scope([_raw("a", author="")], "austen")
    with pytest.raises(ConsolidationError):
        StrategyConsolidator.validate_author_scope([_raw("a", author="austen")], "")


# ---- §十四.3：complete source coverage ----
def test_complete_source_coverage_ok():
    ids = [f"r{i}" for i in range(10)]
    groups = [_group("g1", ["r0", "r1", "r2"]),
              _group("g2", ["r3", "r4"]),
              _group("g3", ["r5"]),
              _group("g4", ["r6", "r7", "r8", "r9"])]
    StrategyConsolidator.validate_mapping(ids, groups)  # 不应抛异常


# ---- §十四.4：duplicate source assignment ----
def test_duplicate_source_assignment_rejected():
    groups = [_group("g1", ["r1", "r2"]), _group("g2", ["r1"])]
    with pytest.raises(ConsolidationError):
        StrategyConsolidator.validate_mapping(["r1", "r2"], groups)


# ---- §十四.5：hallucinated source ID ----
def test_hallucinated_source_id_rejected():
    groups = [_group("g", ["r1", "ghost"])]
    with pytest.raises(ConsolidationError):
        StrategyConsolidator.validate_mapping(["r1"], groups)


# ---- §十四.6：missing source ID ----
def test_missing_source_id_rejected():
    groups = [_group("g", ["r1"])]
    with pytest.raises(ConsolidationError):
        StrategyConsolidator.validate_mapping(["r1", "r2"], groups)


# ---- §十四.7：canonical provenance（raw → chunk → work → evidence）----
def test_canonical_provenance_traceable():
    ev = StrategyEvidence("c1", "w1", "austen", strategy_id="r1", quote="q")
    raw = _raw("r1", name="Free indirect discourse", evidence=[ev])
    canonicals = StrategyConsolidator.build_canonicals(
        {"r1": raw}, [_group("Free indirect discourse", ["r1"])], "austen")
    assert len(canonicals) == 1
    cs = canonicals[0]
    assert cs.canonical_strategy_id == "austen::free_indirect_discourse"
    assert cs.source_strategy_ids == ["r1"]
    assert cs.supporting_chunk_ids == ["c1"]
    assert cs.supporting_work_ids == ["w1"]
    assert cs.number_of_raw_observations == 1
    assert cs.number_of_distinct_chunks == 1
    assert cs.number_of_distinct_works == 1
    assert cs.support_status == StrategyStatus.DISCOVERED.value
    # 追溯到证据：chunk/work/quote 完整
    assert cs.evidence[0].chunk_id == "c1"
    assert cs.evidence[0].work_id == "w1"
    assert cs.evidence[0].quote == "q"


# ---- §十四.8：跨作者同名 canonical 不冲突 ----
def test_same_canonical_name_across_authors_distinct_ids():
    assert canonical_strategy_id("austen", "dramatic irony") == "austen::dramatic_irony"
    assert canonical_strategy_id("dickens", "dramatic irony") == "dickens::dramatic_irony"
    # 完整构建也产生不同 id
    a = StrategyConsolidator.build_canonicals(
        {"r1": _raw("r1", author="austen", name="dramatic irony")},
        [_group("dramatic irony", ["r1"])], "austen")[0]
    d = StrategyConsolidator.build_canonicals(
        {"r2": _raw("r2", author="dickens", name="dramatic irony")},
        [_group("dramatic irony", ["r2"])], "dickens")[0]
    assert a.canonical_strategy_id != d.canonical_strategy_id
    assert a.canonical_strategy_id == "austen::dramatic_irony"
    assert d.canonical_strategy_id == "dickens::dramatic_irony"


# ---- 补充：canonical id 稳定（不依赖 description hash）----
def test_canonical_id_stable_regardless_of_description():
    assert canonical_strategy_id("austen", "social dialogue") == canonical_strategy_id(
        "austen", "social dialogue")
    # 描述措辞不同不影响 identity（slug 只来自 name）
    assert canonical_strategy_id("austen", "Social  Dialogue!!") == "austen::social_dialogue"


# ---- 补充：精确去重折叠（内容完全一致才折叠）----
def test_exact_duplicate_fold_merges_source_ids():
    a = _raw("a", name="Same Name", desc="same", triggers=["t"], operations=["o"], effects=["e"])
    b = _raw("b", name="Same Name", desc="same", triggers=["t"], operations=["o"], effects=["e"])
    prepared = StrategyConsolidator().prepare([a, b])
    assert len(prepared) == 1
    assert set(prepared[0].source_strategy_ids) == {"a", "b"}


# ---- 补充：不因名称近似做语义合并 ----
def test_no_semantic_merge_by_name_similarity():
    a = _raw("a", name="Physical gesture as psychological revelation", desc="d1")
    b = _raw("b", name="Physical gesture as psychological revealation", desc="d2")  # 近似但非精确
    prepared = StrategyConsolidator().prepare([a, b])
    assert len(prepared) == 2  # 语义合并归 LLM，确定性预处理绝不越权


# ---- 补充：dummy 端到端 consolidate ----
def test_consolidate_end_to_end_with_dummy_provider():
    resp = json.dumps({"groups": [{
        "canonical_name": "Merged mechanism",
        "canonical_description": "combined",
        "source_strategy_ids": ["a", "b"],
        "trigger_summary": "t", "operation_summary": "o", "effect_summary": "e",
        "reasoning_summary": "same mechanism", "confidence": 0.9,
    }]})
    c = StrategyConsolidator(DummyLLMProvider(response=resp))
    canonicals = c.consolidate([_raw("a"), _raw("b")], "austen")
    assert len(canonicals) == 1
    assert set(canonicals[0].source_strategy_ids) == {"a", "b"}
    assert canonicals[0].author_id == "austen"
    assert canonicals[0].canonical_strategy_id == "austen::merged_mechanism"
    assert canonicals[0].confidence == 0.9


# ---- 补充：consolidate 拒绝跨作者输入（在调用 LLM 前）----
def test_consolidate_rejects_cross_author_before_llm():
    c = StrategyConsolidator(DummyLLMProvider(response=json.dumps({"groups": []})))
    with pytest.raises(ConsolidationError):
        c.consolidate([_raw("a", author="austen"), _raw("b", author="dickens")], "austen")


# ---- Phase 4.5.1 §1：prompt 含紧凑支持上下文 + 已验证引文 ----
def test_prompt_includes_compact_support_and_verified_quotes():
    ev = StrategyEvidence(
        "c1", "w1", "austen", strategy_id="a",
        quote="first verified", quotes=["first verified quote", "second verified quote"],
        unverified_quotes=["unverified should not appear"], confidence=0.7)
    raw = RawStrategy(
        strategy_id="a", author_id="austen", name="Irony", description="d",
        triggers=["t"], operations=["o"], intended_effects=["e"],
        confidence=0.8, evidence=[ev])
    _, user = StrategyConsolidator().build_prompt([raw], "austen")
    assert "support: chunks=1 works=1 status=discovered confidence=0.8" in user
    assert 'evidence: "first verified quote" | "second verified quote"' in user
    assert "unverified should not appear" not in user


def test_prompt_caps_verified_quotes_at_two():
    ev = StrategyEvidence(
        "c1", "w1", "austen", strategy_id="a",
        quotes=["q1", "q2", "q3"], confidence=0.7)
    raw = RawStrategy(
        strategy_id="a", author_id="austen", name="Irony", description="d",
        triggers=["t"], operations=["o"], intended_effects=["e"], evidence=[ev])
    _, user = StrategyConsolidator().build_prompt([raw], "austen")
    assert '"q1"' in user and '"q2"' in user
    assert '"q3"' not in user


# ---- Phase 4.5.1 §2：严格输出校验（非法必需字段显式报错，绝不静默）----
def test_consolidation_group_rejects_empty_canonical_name():
    with pytest.raises(ValueError):
        ConsolidationGroup.from_dict({"canonical_name": "", "canonical_description": "d",
                                      "source_strategy_ids": ["a"], "confidence": 0.5})


def test_consolidation_group_rejects_empty_canonical_description():
    with pytest.raises(ValueError):
        ConsolidationGroup.from_dict({"canonical_name": "n", "canonical_description": "  ",
                                      "source_strategy_ids": ["a"], "confidence": 0.5})


def test_consolidation_group_rejects_invalid_confidence_type():
    with pytest.raises(ValueError):
        ConsolidationGroup.from_dict({"canonical_name": "n", "canonical_description": "d",
                                      "source_strategy_ids": ["a"], "confidence": "0.8"})


def test_consolidation_group_rejects_confidence_out_of_range():
    with pytest.raises(ValueError):
        ConsolidationGroup.from_dict({"canonical_name": "n", "canonical_description": "d",
                                      "source_strategy_ids": ["a"], "confidence": -0.1})
    with pytest.raises(ValueError):
        ConsolidationGroup.from_dict({"canonical_name": "n", "canonical_description": "d",
                                      "source_strategy_ids": ["a"], "confidence": 1.2})


def test_consolidate_wraps_invalid_fields_as_llm_response_error():
    resp = json.dumps({"groups": [{
        "canonical_name": "Merged", "canonical_description": "d",
        "source_strategy_ids": ["a", "b"], "confidence": "high",
    }]})
    c = StrategyConsolidator(DummyLLMProvider(response=resp))
    with pytest.raises(LLMResponseError):
        c.consolidate([_raw("a"), _raw("b")], "austen")


# ---- Phase 4.5.1（追加）：consolidation 输出需放宽 max_tokens，防止 JSON 截断 ----
def test_consolidate_passes_max_tokens_to_provider():
    class SpyProvider:
        provider_id = "spy"
        model = "spy-model"

        def __init__(self):
            self.kwargs = None

        def is_configured(self):
            return True

        def complete(self, messages, **kwargs):
            self.kwargs = kwargs
            return json.dumps({"groups": [{
                "canonical_name": "M", "canonical_description": "d",
                "source_strategy_ids": ["a"], "confidence": 0.5,
            }]})

    spy = SpyProvider()
    StrategyConsolidator(spy, max_tokens=6000).consolidate([_raw("a")], "austen")
    assert spy.kwargs.get("max_tokens") == 6000


# ---- Phase 4.5.1（追加）：首轮遗漏 → 覆盖修复（merge 进已有组 / 新建组）----
class _ScriptedProvider:
    """按顺序返回预设响应，验证 repair 二次调用路径。"""
    provider_id = "scripted"
    model = "m"

    def __init__(self, responses):
        self._responses = list(responses)
        self._i = 0

    def is_configured(self):
        return True

    def complete(self, messages, **kwargs):
        r = self._responses[self._i]
        self._i += 1
        return r


def test_consolidate_repairs_missing_source_id_into_existing_group():
    first = json.dumps({"groups": [{
        "canonical_name": "Merged", "canonical_description": "d",
        "source_strategy_ids": ["a"], "confidence": 0.8}]})
    repair = json.dumps({"groups": [{
        "canonical_name": "Merged", "canonical_description": "d",
        "source_strategy_ids": ["b"], "confidence": 0.7}]})
    c = StrategyConsolidator(_ScriptedProvider([first, repair]))
    canonicals = c.consolidate([_raw("a"), _raw("b")], "austen")
    assert len(canonicals) == 1
    assert set(canonicals[0].source_strategy_ids) == {"a", "b"}


def test_consolidate_repair_can_create_new_group():
    first = json.dumps({"groups": [{
        "canonical_name": "A", "canonical_description": "d",
        "source_strategy_ids": ["a"], "confidence": 0.8}]})
    repair = json.dumps({"groups": [{
        "canonical_name": "B", "canonical_description": "d",
        "source_strategy_ids": ["b"], "confidence": 0.7}]})
    c = StrategyConsolidator(_ScriptedProvider([first, repair]))
    canonicals = c.consolidate([_raw("a"), _raw("b")], "austen")
    assert len(canonicals) == 2
    assert {cs.canonical_strategy_id for cs in canonicals} == {"austen::a", "austen::b"}
