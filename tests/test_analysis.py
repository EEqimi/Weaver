# tests/test_analysis.py
"""Layer A 确定性分析器与 FeatureValue 序列化 / 版本记录测试（spec §12）。"""
from knowledge.analysis.statistical_analyzer import (
    ANALYZER_ID, ANALYZER_VERSION, StatisticalAnalyzer,
)
from knowledge.schema.feature_registry import build_default_registry
from knowledge.schema.style_schema import FeatureValue
from knowledge.schema.versions import FEATURE_SCHEMA_VERSION

SAMPLE = (
    "It is a truth universally acknowledged, that a single man in possession of "
    "a good fortune, must be in want of a wife. However little known the feelings "
    "or views of such a man may be on his first entering a neighbourhood, this "
    "truth is so well fixed in the minds of the surrounding families, that he is "
    "considered as the rightful property of some one or other of their daughters."
)


def _stat_features():
    reg = build_default_registry()
    return [f for f in reg.all() if f.analyzer == "StatisticalAnalyzer"]


def test_statistical_analyzer_deterministic():
    a = StatisticalAnalyzer()
    fv1 = a.analyze_many(SAMPLE, _stat_features())
    fv2 = a.analyze_many(SAMPLE, _stat_features())
    assert len(fv1) == len(fv2) == 22
    d1 = {f.feature_id: f.value for f in fv1}
    d2 = {f.feature_id: f.value for f in fv2}
    assert d1 == d2


def test_type_token_ratio_value():
    a = StatisticalAnalyzer()
    f = build_default_registry().get("type_token_ratio")
    fv = a.analyze(SAMPLE, f)
    assert fv is not None
    assert 0.0 < fv.value < 1.0
    assert fv.sample_count > 0


def test_feature_value_serialization_roundtrip():
    fv = FeatureValue(
        feature_id="x", value=1.5, raw_value=1.5, value_type="continuous",
        measurement_type="statistical", confidence=0.9, evidence=["q"],
        sample_count=3, variance=0.2, analyzer_id="StatisticalAnalyzer",
        analyzer_version=ANALYZER_VERSION,
    )
    d = fv.to_dict()
    assert d["feature_id"] == "x"
    assert d["value"] == 1.5
    assert d["analyzer_version"] == ANALYZER_VERSION
    # 关键字段必须可序列化且无缺省丢字段
    for key in ("feature_id", "value", "raw_value", "value_type",
                "measurement_type", "analyzer_id", "analyzer_version",
                "schema_version"):
        assert key in d


def test_analyzer_version_recorded():
    a = StatisticalAnalyzer()
    f = build_default_registry().get("mean_sentence_length")
    fv = a.analyze(SAMPLE, f)
    assert fv is not None
    assert fv.analyzer_id == ANALYZER_ID
    assert fv.analyzer_version == ANALYZER_VERSION
    assert fv.schema_version == FEATURE_SCHEMA_VERSION
