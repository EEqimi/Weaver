# tests/test_strategy_miner.py
"""Layer C 策略挖掘器测试（spec §12）：匹配、发现、未知策略忽略。"""
import json

from knowledge.analysis.base import AnalysisUnavailable
from knowledge.analysis.strategy_miner import StrategyMiner
from knowledge.providers.llm_provider import DummyLLMProvider, UnconfiguredLLMProvider
from knowledge.schema.strategy_schema import StrategyStatus
from knowledge.strategies.registry import seed_default_registry

PASSAGE = "He was perfectly astonished at the sight."


def test_match_returns_known_strategy_evidence():
    resp = json.dumps({"matches": [
        {"strategy_id": "free_indirect_discourse",
         "evidence": ["He was perfectly astonished"], "confidence": 0.8}]})
    miner = StrategyMiner(DummyLLMProvider(response=resp), seed_default_registry())
    out = miner.match(PASSAGE, chunk_id="c1", work_id="w1", author_id="austen")
    assert isinstance(out, list)
    assert len(out) == 1
    sid, ev = out[0]
    assert sid == "free_indirect_discourse"
    assert ev.chunk_id == "c1" and ev.work_id == "w1" and ev.author_id == "austen"


def test_match_ignores_unknown_strategy():
    # 未知策略被忽略；已知策略使用可验证证据（否则高置信正向判定会因无证据被拒）
    resp = json.dumps({"matches": [
        {"strategy_id": "made_up_strategy",
         "evidence": ["He was perfectly astonished"], "confidence": 0.9},
        {"strategy_id": "dramatic_irony",
         "evidence": ["He was perfectly astonished"], "confidence": 0.9}]})
    miner = StrategyMiner(DummyLLMProvider(response=resp), seed_default_registry())
    out = miner.match(PASSAGE)
    assert [sid for sid, _ in out] == ["dramatic_irony"]


def test_match_rejects_zero_verified_evidence_regardless_of_confidence():
    # task item 6：零验证证据的正向匹配绝不构成生命周期证据，无论置信度高低
    resp = json.dumps({"matches": [
        {"strategy_id": "dramatic_irony",
         "evidence": ["fabricated quote"], "confidence": 0.9},
        {"strategy_id": "free_indirect_discourse",
         "evidence": ["another fabricated quote"], "confidence": 0.2}]})
    miner = StrategyMiner(DummyLLMProvider(response=resp), seed_default_registry())
    out = miner.match(PASSAGE)
    assert out == []


def test_discover_rejects_zero_verified_evidence():
    # task item 6：零验证证据的候选发现也不得产出
    resp = json.dumps({"strategies": [
        {"name": "Fabricated Strategy", "description": "d",
         "triggers": ["t"], "operations": ["o"], "intended_effects": ["e"],
         "evidence": ["not in the passage"], "confidence": 0.3}]})
    miner = StrategyMiner(DummyLLMProvider(response=resp), seed_default_registry())
    out = miner.discover(PASSAGE)
    assert out == []


def test_match_preserves_all_verified_quotes_and_confidence():
    # task item 7：保留 match confidence 与全部有效引文（不只第一条）
    resp = json.dumps({"matches": [
        {"strategy_id": "free_indirect_discourse",
         "evidence": ["He was perfectly astonished", "at the sight"],
         "confidence": 0.85}]})
    miner = StrategyMiner(DummyLLMProvider(response=resp), seed_default_registry())
    out = miner.match(PASSAGE, chunk_id="c1", work_id="w1", author_id="austen")
    sid, ev = out[0]
    assert ev.quotes == ["He was perfectly astonished", "at the sight"]
    assert ev.quote == "He was perfectly astonished"
    assert ev.confidence == 0.85
    assert ev.analyzer_id == "StrategyMiner"
    assert ev.analyzer_version and ev.schema_version


def test_discover_returns_discovered_strategy():
    resp = json.dumps({"strategies": [
        {"name": "Understated Revelation", "description": "d",
         "triggers": ["t"], "operations": ["o"], "intended_effects": ["e"],
         "evidence": ["He was perfectly astonished"], "confidence": 0.7}]})
    miner = StrategyMiner(DummyLLMProvider(response=resp), seed_default_registry())
    out = miner.discover(PASSAGE, chunk_id="c1", work_id="w1", author_id="austen")
    assert isinstance(out, list) and len(out) == 1
    s = out[0]
    assert s.status == StrategyStatus.DISCOVERED.value
    assert s.source_work == "w1" and s.source_author == "austen"


def test_miner_unavailable_without_provider():
    miner = StrategyMiner(UnconfiguredLLMProvider(), seed_default_registry())
    assert isinstance(miner.match(PASSAGE), AnalysisUnavailable)
    assert isinstance(miner.discover(PASSAGE), AnalysisUnavailable)


# ---- 拒绝收集器（供冒烟/标定报表统计被拒输出）----
def test_match_records_zero_evidence_rejection():
    resp = json.dumps({"matches": [
        {"strategy_id": "dramatic_irony",
         "evidence": ["fabricated quote"], "confidence": 0.9}]})
    rejections: list[dict] = []
    miner = StrategyMiner(DummyLLMProvider(response=resp), seed_default_registry(),
                          rejections=rejections)
    out = miner.match(PASSAGE)
    assert out == []
    assert rejections == [{"stage": "match", "reason": "zero_verified_evidence",
                           "strategy_id": "dramatic_irony", "confidence": 0.9}]


def test_match_records_unknown_strategy_rejection():
    resp = json.dumps({"matches": [
        {"strategy_id": "made_up_strategy", "evidence": ["x"], "confidence": 0.5}]})
    rejections: list[dict] = []
    miner = StrategyMiner(DummyLLMProvider(response=resp), seed_default_registry(),
                          rejections=rejections)
    assert miner.match(PASSAGE) == []
    assert rejections == [{"stage": "match", "reason": "unknown_strategy",
                           "strategy_id": "made_up_strategy"}]


def test_discover_records_zero_evidence_rejection():
    resp = json.dumps({"strategies": [
        {"name": "Fabricated Strategy", "description": "d",
         "triggers": ["t"], "operations": ["o"], "intended_effects": ["e"],
         "evidence": ["not in the passage"], "confidence": 0.6}]})
    rejections: list[dict] = []
    miner = StrategyMiner(DummyLLMProvider(response=resp), seed_default_registry(),
                          rejections=rejections)
    assert miner.discover(PASSAGE) == []
    assert len(rejections) == 1
    assert rejections[0]["stage"] == "discover"
    assert rejections[0]["reason"] == "zero_verified_evidence"
    assert rejections[0]["name"] == "Fabricated Strategy"


def test_no_rejection_collector_is_noop():
    # 默认 rejections=None：行为与无收集器一致，绝不抛异常
    resp = json.dumps({"matches": [
        {"strategy_id": "dramatic_irony",
         "evidence": ["fabricated"], "confidence": 0.9}]})
    miner = StrategyMiner(DummyLLMProvider(response=resp), seed_default_registry())
    assert miner.match(PASSAGE) == []
