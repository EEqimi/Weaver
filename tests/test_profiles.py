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
    assert aggregate_feature_values([]) == {"n": 0}


# ---- narrative 聚合 ----
def _obs(pov="third", focalization="internal"):
    return NarrativeObservation(chunk_id="c", pov=pov, focalization=focalization)


def test_aggregate_narrative_categorical():
    s = aggregate_narrative([_obs("third", "internal"), _obs("third", "zero")])
    assert s["pov"]["proportions"] == {"third": 1.0}
    assert s["focalization"]["proportions"]["internal"] == pytest.approx(0.5)
    assert s["n"] == 2


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


def test_seed_registry_has_four_candidates():
    reg = seed_default_registry()
    assert len(reg) == 4
    assert all(s.status == StrategyStatus.CANDIDATE.value for s in reg.all())
