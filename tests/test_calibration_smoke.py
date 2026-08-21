# tests/test_calibration_smoke.py
"""冒烟标定模块的确定性单元测试：_bump 安全累加、_feature_report 拍平。

冒烟报表要"看到一切意外"，因此未知错误类型绝不能因缺键而 KeyError（回归锁定）。
"""
from knowledge.calibration.smoke import _bump, _feature_report
from knowledge.schema.rubrics import ASSESSMENT_OBSERVED


def test_bump_creates_key_for_unknown_error_type():
    bucket = {"schema_json": 0, "transport": 0}
    _bump(bucket, "KeyError")
    _bump(bucket, "KeyError")
    assert bucket["KeyError"] == 2
    assert bucket["schema_json"] == 0
    assert bucket["transport"] == 0


def test_bump_increments_existing_key():
    bucket = {"transport": 1}
    _bump(bucket, "transport")
    assert bucket["transport"] == 2


def test_feature_report_flattens_feature_value():
    fv = {
        "value": 12.5,
        "raw_value": 1.0,
        "confidence": 0.8,
        "evidence": ["quote a"],
        "provenance": {"assessment_status": "observed",
                       "unverified_evidence": ["quote b"]},
    }
    r = _feature_report("irony_intensity", {"status": "ok", "result": fv})
    assert r["feature"] == "irony_intensity"
    assert r["value"] == 12.5
    assert r["raw_value"] == 1.0
    assert r["confidence"] == 0.8
    assert r["assessment_status"] == "observed"
    assert r["verified_evidence"] == ["quote a"]
    assert r["unverified_evidence"] == ["quote b"]


def test_feature_report_defaults_assessment_status():
    # frequency 特征 provenance 无 assessment_status → 默认 observed
    fv = {"value": 0.0, "raw_value": 0.0, "confidence": None,
          "evidence": [], "provenance": {}}
    r = _feature_report("metaphor_frequency", {"status": "ok", "result": fv})
    assert r["assessment_status"] == ASSESSMENT_OBSERVED
    assert r["verified_evidence"] == []
    assert r["unverified_evidence"] == []
