# tests/test_narrative_schema.py
"""narrative 比例校验与证据充分性 schema 测试（Phase 3–4.2，task item 4）。"""
import pytest

from knowledge.schema.narrative_schema import (
    DETAIL_DIMENSIONS, PACE_DIMENSIONS, validate_narrative,
)


def _base():
    return {
        "pov": "third", "focalization": "internal",
        "perspective_stability": "stable", "narrative_distance": "medium",
        "narrator_presence": "low", "narrator_evaluative_intervention": "low",
        "information_access": "limited", "temporal_order": "chronological",
        "observed_evidence": [], "confidence": 0.8,
    }


def test_proportion_value_greater_than_one_rejected():
    d = _base()
    d["temporal_pace"] = {"scene": 1.5, "summary": 0.0, "ellipsis": 0.0}
    with pytest.raises(ValueError):
        validate_narrative(d)


def test_proportion_negative_value_rejected():
    d = _base()
    d["scene_detail"] = {"psychology": -0.2, "action": 0.8, "dialogue": 0.4}
    with pytest.raises(ValueError):
        validate_narrative(d)


def test_proportion_all_zero_distribution_flagged():
    d = _base()
    d["temporal_pace"] = {"scene": 0.0, "summary": 0.0, "ellipsis": 0.0}
    obs = validate_narrative(d)
    assert any("全零" in issue or "insufficient" in issue for issue in obs.proportion_issues)


def test_proportion_sum_far_from_one_flagged():
    d = _base()
    d["temporal_pace"] = {"scene": 0.3, "summary": 0.3, "ellipsis": 0.1}
    obs = validate_narrative(d)
    assert any("偏离 1" in issue for issue in obs.proportion_issues)
    # 不静默重归一化：保留原值
    assert obs.temporal_pace == {"scene": 0.3, "summary": 0.3, "ellipsis": 0.1}


def test_proportion_valid_approximate_distribution():
    d = _base()
    d["temporal_pace"] = {"scene": 0.4, "summary": 0.35, "ellipsis": 0.25}
    obs = validate_narrative(d)
    assert obs.proportion_issues == []
    assert obs.temporal_pace["scene"] == pytest.approx(0.4)


def test_proportion_unknown_key_reported_not_silently_dropped():
    d = _base()
    d["temporal_pace"] = {"scene": 0.5, "summary": 0.5, "ellipsis": 0.0, "mystery": 0.3}
    obs = validate_narrative(d)
    assert any("未知维度" in issue for issue in obs.proportion_issues)
    # 未知键被显式忽略，合法键保留
    assert "mystery" not in obs.temporal_pace
    assert "scene" in obs.temporal_pace
