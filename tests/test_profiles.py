# tests/test_profiles.py
"""Phase 4 类型感知聚合 + 策略生命周期测试（spec §12）。"""
import math

import pytest

from knowledge.profiles.aggregation import (
    Aggregator, ChunkProfile, aggregate_categorical, aggregate_continuous,
    aggregate_distribution, aggregate_feature_values, aggregate_narrative,
)
from knowledge.schema.narrative_schema import NarrativeObservation
from knowledge.schema.style_schema import FeatureValue
from knowledge.schema.strategy_schema import StrategyEvidence, StrategyStatus
from knowledge.strategies.registry import StrategyRegistry, seed_default_registry


def _fv(fid, value, value_type="continuous"):
    return FeatureValue(feature_id=fid, value=value, raw_value=value,
                        value_type=value_type, measurement_type="statistical")


# ---- 连续聚合 ----
def test_aggregate_continuous_mean_variance():
    s = aggregate_continuous([1.0, 2.0, 3.0])
    assert s["mean"] == 2.0
    assert math.isclose(s["variance"], 2 / 3)
    assert s["min"] == 1.0 and s["max"] == 3.0
    assert s["n"] == 3


# ---- 类别聚合 ----
def test_aggregate_categorical_distribution():
    s = aggregate_categorical(["a", "a", "b"])
    assert s["counts"] == {"a": 2, "b": 1}
    assert s["proportions"]["a"] == pytest.approx(2 / 3)
    assert s["mode"] == "a"


# ---- 分布聚合 ----
def test_aggregate_distribution_mean():
    s = aggregate_distribution([{"x": 0.5, "y": 0.5}, {"x": 0.9, "y": 0.1}])
    assert s["mean_distribution"]["x"] == pytest.approx(0.7)
    assert s["mean_distribution"]["y"] == pytest.approx(0.3)


# ---- 按 value_type 分派 ----
def test_aggregate_feature_values_continuous_dispatch():
    s = aggregate_feature_values([_fv("a", 1.0), _fv("a", 3.0)])
    assert s["mean"] == 2.0 and s["n"] == 2


def test_aggregate_feature_values_categorical_dispatch():
    s = aggregate_feature_values([_fv("a", "x", "categorical"),
                                  _fv("a", "y", "categorical")])
    assert set(s["proportions"]) == {"x", "y"}


def test_aggregate_feature_values_distribution_dispatch():
    s = aggregate_feature_values([_fv("a", {"p": 1.0}, "distribution"),
                                  _fv("a", {"p": 0.5}, "distribution")])
    assert s["mean_distribution"]["p"] == pytest.approx(0.75)


def test_aggregate_feature_values_empty():
    assert aggregate_feature_values([]) == {
        "n": 0, "n_expected": 0, "n_total": 0, "n_valid": 0, "n_missing": 0,
        "n_unobservable": 0, "n_insufficient": 0,
    }


def test_aggregate_feature_values_preserves_missing_confidence_evidence():
    # task item 4：聚合必须保留 n_total/n_valid/n_missing、独立 confidence 汇总、
    # evidence 引用、analyzer/schema 溯源；绝不把 confidence 平均进 value。
    fvs = [
        _fv("a", 1.0),
        FeatureValue(feature_id="a", value=None, raw_value=None,
                     value_type="continuous", measurement_type="statistical",
                     confidence=0.8, evidence=["e1"], analyzer_id="X",
                     analyzer_version="1.0", provenance={"chunk_id": "c2"}),
        _fv("a", 3.0),
    ]
    s = aggregate_feature_values(fvs)
    assert s["n_total"] == 3
    assert s["n_valid"] == 2
    assert s["n_missing"] == 1
    # value 汇总基于有效值，不受 confidence 污染
    assert s["mean"] == 2.0
    # confidence 独立汇总
    assert s["confidence"]["mean"] == 0.8
    assert s["evidence_refs"] == [["e1"]]
    assert s["analyzer_ids"] == ["X"]
    assert s["analyzer_versions"] == ["1.0"]
    assert s["provenance"] == [{"chunk_id": "c2"}]


def test_aggregate_feature_values_expected_sample_accounting():
    # task item 7：n_expected 表示预期接受分析的 chunk 数；缺失/不可观察/不充分
    # 分别计数，且不可观察值绝不拉低数值均值。
    fvs = [
        _fv("a", 1.0, "discrete"),
        FeatureValue(feature_id="a", value=None, raw_value=None,
                     value_type="discrete", measurement_type="judgment",
                     provenance={"assessment_status": "not_observable"}),
        FeatureValue(feature_id="a", value=None, raw_value=None,
                     value_type="discrete", measurement_type="judgment",
                     provenance={"assessment_status": "insufficient_evidence"}),
        _fv("a", 3.0, "discrete"),
    ]
    s = aggregate_feature_values(fvs, n_expected=5)  # 5 预期，1 个 chunk 无 FeatureValue
    assert s["n_expected"] == 5
    assert s["n_total"] == 5
    assert s["n_valid"] == 2
    assert s["n_unobservable"] == 1
    assert s["n_insufficient"] == 1
    assert s["n_missing"] == 1           # 5 - 2 - 1 - 1
    assert s["mean"] == 2.0              # 不可观察/不充分不拉低均值


# ---- narrative 聚合 ----
def _obs(pov="third", focalization="internal"):
    return NarrativeObservation(chunk_id="c", pov=pov, focalization=focalization)


def test_aggregate_narrative_categorical():
    s = aggregate_narrative([_obs("third", "internal"), _obs("third", "zero")])
    assert s["pov"]["proportions"] == {"third": 1.0}
    assert s["focalization"]["proportions"]["internal"] == pytest.approx(0.5)
    assert s["n"] == 2


def test_aggregate_narrative_preserves_evidence_and_confidence():
    # task item 4：叙事聚合保留逐 chunk 证据引用、未验证引文与 confidence 汇总
    o1 = NarrativeObservation(chunk_id="c1", observed_evidence=["q1"], confidence=0.9)
    o2 = NarrativeObservation(chunk_id="c2", unverified_evidence=["u1"], confidence=0.5)
    s = aggregate_narrative([o1, o2])
    assert s["observed_evidence"] == {"c1": ["q1"]}
    assert s["unverified_evidence"] == {"c2": ["u1"]}
    assert s["confidence"]["mean"] == pytest.approx(0.7)
    assert s["n_total"] == 2
    assert s["chunk_provenance"] == ["c1", "c2"]


# ---- Work/Author 画像 ----
def test_aggregate_work_profile():
    p1 = ChunkProfile("c1", "w", "a", feature_values={"f": _fv("f", 1.0)})
    p2 = ChunkProfile("c2", "w", "a", feature_values={"f": _fv("f", 3.0)})
    wp = Aggregator().aggregate_work([p1, p2])
    assert wp.work_id == "w" and wp.author_id == "a" and wp.chunk_count == 2
    assert wp.features["f"]["mean"] == 2.0


def test_aggregate_author_profile_multi_work():
    p1 = ChunkProfile("c1", "w1", "a", feature_values={"f": _fv("f", 1.0)})
    p2 = ChunkProfile("c2", "w2", "a", feature_values={"f": _fv("f", 3.0)})
    ap = Aggregator().aggregate_author([p1, p2])
    assert ap.author_id == "a" and ap.work_ids == ["w1", "w2"]


# ---- 策略生命周期 ----
def test_strategy_lifecycle_monotonic_promotion():
    # 使用预置注册表里的 candidate 策略验证跨作品晋升
    reg = seed_default_registry()
    sid = "free_indirect_discourse"
    assert reg.get(sid).status == StrategyStatus.CANDIDATE.value
    # 两条同作者、不同 work 的证据 → validated
    reg.record_evidence(sid, StrategyEvidence("c1", "w1", "austen"))
    reg.record_evidence(sid, StrategyEvidence("c2", "w2", "austen"))
    assert reg.get(sid).status == StrategyStatus.VALIDATED.value


def test_strategy_lifecycle_never_regresses():
    reg = seed_default_registry()
    sid = "dramatic_irony"
    reg.record_evidence(sid, StrategyEvidence("c1", "w1", "austen"))
    reg.record_evidence(sid, StrategyEvidence("c2", "w2", "austen"))
    assert reg.get(sid).status == StrategyStatus.VALIDATED.value
    # 再追加一条单 chunk 证据，不应降级
    reg.record_evidence(sid, StrategyEvidence("c3", "w2", "austen"))
    assert reg.get(sid).status == StrategyStatus.VALIDATED.value


def test_strategy_single_chunk_stays_discovered():
    reg = StrategyRegistry()
    from knowledge.schema.strategy_schema import CreativeStrategy
    reg.register(CreativeStrategy(strategy_id="s", name="n", description="d",
                                  status=StrategyStatus.DISCOVERED.value))
    reg.record_evidence("s", StrategyEvidence("c1", "w1", "austen"))
    assert reg.get("s").status == StrategyStatus.DISCOVERED.value


def test_strategy_validation_requires_consistent_non_empty_author():
    # task item 5：一条 Austen 证据 + 一条空 author 证据跨两作品 → 不得 validated
    reg = seed_default_registry()
    sid = "free_indirect_discourse"
    reg.record_evidence(sid, StrategyEvidence("c1", "w1", "austen"))
    reg.record_evidence(sid, StrategyEvidence("c2", "w2", ""))  # 空 author
    assert reg.get(sid).status == StrategyStatus.CANDIDATE.value
    # 空 work_id 同样不得参与作者级验证
    reg.record_evidence(sid, StrategyEvidence("c3", "", "austen"))
    assert reg.get(sid).status == StrategyStatus.CANDIDATE.value


def test_strategy_validation_requires_two_distinct_works():
    # task item 5：>= 2 个不同 work 才可 validated；同 work 多 chunk 仅 candidate
    reg = seed_default_registry()
    sid = "free_indirect_discourse"
    reg.record_evidence(sid, StrategyEvidence("c1", "w1", "austen"))
    reg.record_evidence(sid, StrategyEvidence("c2", "w1", "austen"))
    assert reg.get(sid).status == StrategyStatus.CANDIDATE.value


def test_seed_registry_has_four_candidates():
    reg = seed_default_registry()
    assert len(reg) == 4
    assert all(s.status == StrategyStatus.CANDIDATE.value for s in reg.all())


def test_strategy_evidence_carries_strategy_id():
    # 回归：StrategyEvidence 必须携带 strategy_id，否则 work 画像按策略计数
    # 会 AttributeError（Phase 4.4 首次填充 strategy_evidence 时暴露）。
    ev = StrategyEvidence("c1", "w1", "austen", strategy_id="sid_a")
    assert ev.strategy_id == "sid_a"
    # 缺省为空，兼容旧的位置参数构造
    assert StrategyEvidence("c1", "w1", "austen").strategy_id == ""


def test_work_profile_counts_strategy_evidence_by_strategy_id():
    p1 = ChunkProfile("c1", "w1", "austen", strategy_evidence=[
        StrategyEvidence("c1", "w1", "austen", strategy_id="sid_a"),
        StrategyEvidence("c1", "w1", "austen", strategy_id="sid_b"),
    ])
    p2 = ChunkProfile("c2", "w1", "austen", strategy_evidence=[
        StrategyEvidence("c2", "w1", "austen", strategy_id="sid_a"),
    ])
    wp = Aggregator().aggregate_work([p1, p2])
    assert wp.strategies == {"sid_a": 2, "sid_b": 1}
