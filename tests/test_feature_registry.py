# tests/test_feature_registry.py
"""特征注册表：注册/查重/分类覆盖。"""
import pytest

from knowledge.schema.feature_registry import (
    FeatureDefinition, FeatureRegistry, MeasurementType, ControlRole,
    build_default_registry,
)


def test_no_duplicate_ids():
    reg = build_default_registry()
    ids = [f.id for f in reg.all()]
    assert len(ids) == len(set(ids))


def test_categories_cover_eight_families():
    reg = build_default_registry()
    cats = {f.category for f in reg.all()}
    assert cats >= {
        "lexical_register", "syntax", "rhythm_punctuation", "rhetoric_imagery",
        "voice_pragmatics", "character_representation", "emotion_semantics",
        "discourse_cohesion",
    }


def test_measurement_and_control_types_covered():
    reg = build_default_registry()
    assert {f.measurement_type for f in reg.all()} == set(MeasurementType)
    roles = {f.control_role for f in reg.all()}
    assert ControlRole.CANDIDATE_CORE in roles
    # core 保留给"验证通过"后的正式核心，当前 V0.1 不分配（候选一律 candidate_core）
    assert ControlRole.CORE not in roles


def test_duplicate_registration_raises():
    reg = FeatureRegistry()
    f = FeatureDefinition("x", "syntax", MeasurementType.STATISTICAL,
                          "continuous", ControlRole.CORE, "none", "A")
    reg.register(f)
    with pytest.raises(ValueError):
        reg.register(f)


def test_lookup_and_filters():
    reg = build_default_registry()
    assert reg.has("mean_sentence_length")
    assert reg.get("dialogue_ratio").control_role == ControlRole.CANDIDATE_CORE
    core = reg.by_control_role(ControlRole.CANDIDATE_CORE)
    assert all(f.control_role == ControlRole.CANDIDATE_CORE for f in core)
    stats = reg.by_measurement_type(MeasurementType.STATISTICAL)
    assert all(f.measurement_type == MeasurementType.STATISTICAL for f in stats)
    assert len(reg) == len(reg.all())


def test_llm_features_registered_as_interfaces():
    reg = build_default_registry()
    irony = reg.get("irony_frequency")
    assert irony.measurement_type == MeasurementType.JUDGMENT
    assert irony.analyzer == "LlmFeatureAnalyzer"


def test_candidate_core_marked_provisional():
    # 核心候选不得被视为已验证（item 5）：不得出现正式 core 角色
    reg = build_default_registry()
    for f in reg.all():
        assert f.control_role != ControlRole.CORE, f.id
