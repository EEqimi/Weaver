# tests/test_style_profile.py
"""Phase 5 作者风格画像合成测试（全部确定性，无 LLM）。

覆盖：control-role 映射、diagnostic 不进入 generation controls、direct/conditional/
reference-only 正确分桶、canonical 数量不丢失、support_status 保持、不确定性不伪造 0、
full-corpus vs sampled scope 区分、held-out 隔离、strategy 优先级确定性、字节级复现。
"""
import json
from pathlib import Path

import numpy as np
import pytest

from knowledge.calibration.synthesize import _author_targets_from_matrix
from knowledge.profiles.style_profile import (
    AuthorStyleProfile, AuthorStyleProfileSynthesizer,
    ProfileSchemaError, ProfileSynthesisError,
    _profile_role, rank_canonical_strategies,
)


# --------------------------------------------------------------------------- #
# 构造辅助
# --------------------------------------------------------------------------- #
def _feature_summary(value_type="continuous", measurement_type="statistical", **over):
    base = {
        "n": 10, "mean": 0.5, "variance": 0.01, "std": 0.1,
        "n_expected": 10, "n_total": 10, "n_valid": 10,
        "n_missing": 0, "n_unobservable": 0, "n_insufficient": 0,
        "value_type": value_type, "measurement_type": measurement_type,
        "confidence": {"n": 10, "mean": 0.8},
    }
    base.update(over)
    return base


_TRAIN_WORKS = ["emma", "pride_and_prejudice"]


def _canonical(cid, name="N", status="discovered", works=1, chunks=1, conf=0.8, raw=1):
    return {
        "canonical_strategy_id": cid,
        "canonical_name": name,
        "canonical_description": "d",
        "trigger_summary": "t",
        "operation_summary": "o",
        "effect_summary": "e",
        "support_status": status,
        "confidence": conf,
        "source_strategy_ids": [f"{cid}_raw"],
        "supporting_work_ids": _TRAIN_WORKS[:works],
        "supporting_chunk_ids": [f"c{i}" for i in range(chunks)],
        "number_of_distinct_works": works,
        "number_of_distinct_chunks": chunks,
        "number_of_raw_observations": raw,
    }


def _synthesize(**kw):
    defaults = dict(
        author_id="austen",
        train_work_ids=["emma", "pride_and_prejudice"],
        held_out_work_ids=["persuasion"],
        profile_work_ids=["emma", "pride_and_prejudice"],
        full_corpus_features={},
        sampled_features={},
        sampled_narrative={},
        canonical_strategies=[],
        stylometry_author_target={},
        stylometry_validation_metadata={"n_features": 954},
    )
    defaults.update(kw)
    return AuthorStyleProfileSynthesizer().synthesize(**defaults)


# --------------------------------------------------------------------------- #
# control-role 映射
# --------------------------------------------------------------------------- #
def test_control_role_mapping():
    assert _profile_role("candidate_core") == "direct_control"
    assert _profile_role("descriptive") == "direct_control"
    assert _profile_role("core") == "direct_control"
    assert _profile_role("diagnostic") == "diagnostic"
    assert _profile_role("experimental") == "reference_only"
    assert _profile_role("totally_unknown") == "reference_only"  # 未知不猜测


def test_direct_control_enters_generation_controls():
    profile = _synthesize(
        full_corpus_features={"mean_sentence_length": _feature_summary()})
    g = profile.generation_controls["mean_sentence_length"]
    assert g.control_role == "direct_control"
    assert g.source_scope == "full_train_corpus"


def test_reference_only_sampled_feature_keeps_scope():
    profile = _synthesize(
        sampled_features={"simile_frequency": _feature_summary(
            measurement_type="hybrid")})
    g = profile.generation_controls["simile_frequency"]
    assert g.control_role == "reference_only"
    assert g.source_scope == "calibration_sample"
    assert g.registry_control_role == "experimental"


def test_diagnostic_feature_never_enters_generation_controls():
    # char_trigram_frequency 的 registry 角色为 diagnostic，即便被误传入 features，
    # 也绝不进入 generation_controls（stylometric 指纹只用于诊断）。
    profile = _synthesize(
        full_corpus_features={"char_trigram_frequency": _feature_summary()})
    assert "char_trigram_frequency" not in profile.generation_controls
    assert "diagnostic" not in {g.control_role for g in profile.generation_controls.values()}


def test_conditional_strategy_enters_strategy_controls():
    profile = _synthesize(canonical_strategies=[_canonical("austen::x", status="validated")])
    assert len(profile.strategy_controls) == 1
    assert profile.strategy_controls[0].control_role == "conditional_control"


# --------------------------------------------------------------------------- #
# 数量保持 / support_status 保持（不因 priority/filtering 丢策略）
# --------------------------------------------------------------------------- #
def test_canonical_count_preserved():
    canonicals = [_canonical(f"austen::s{i}", status="discovered") for i in range(26)]
    profile = _synthesize(canonical_strategies=canonicals)
    assert len(profile.strategy_controls) == 26


def test_support_status_preserved():
    canonicals = [
        _canonical("austen::v", status="validated"),
        _canonical("austen::c", status="candidate"),
        _canonical("austen::d", status="discovered"),
    ]
    profile = _synthesize(canonical_strategies=canonicals)
    got = {s.canonical_strategy_id: s.support_status for s in profile.strategy_controls}
    assert got == {"austen::v": "validated", "austen::c": "candidate", "austen::d": "discovered"}


# --------------------------------------------------------------------------- #
# 不确定性：not_observable / insufficient / missing 绝不伪造为 0
# --------------------------------------------------------------------------- #
def test_uncertainty_counts_not_zeroed():
    summary = _feature_summary(value_type="discrete", measurement_type="judgment",
                               n_valid=0, n_unobservable=5, n_insufficient=3,
                               n_missing=2, n_expected=10, n_total=10, mean=None,
                               variance=None, std=None)
    profile = _synthesize(sampled_features={"irony_intensity": summary})
    g = profile.generation_controls["irony_intensity"]
    assert g.summary["n_unobservable"] == 5
    assert g.summary["n_insufficient"] == 3
    assert g.summary["n_missing"] == 2
    assert g.summary["mean"] is None  # 不可观察绝不合成均值


def test_narrative_not_observable_preserved_as_category():
    # 全为 not_observable 的类别字段必须保留其类别计数，绝不转为 0/伪造。
    narrative = {
        "n": 10, "n_total": 10, "n_valid": 0, "n_missing": 0,
        "pov": {"n": 10, "counts": {"not_observable": 10},
                "proportions": {"not_observable": 1.0}, "mode": "not_observable"},
    }
    profile = _synthesize(sampled_narrative=narrative)
    nc = profile.narrative_controls["pov"]
    assert nc.summary["counts"]["not_observable"] == 10
    assert nc.control_role == "direct_control"
    assert nc.source_scope == "calibration_sample"


# --------------------------------------------------------------------------- #
# held-out 隔离
# --------------------------------------------------------------------------- #
def test_held_out_isolation_clean():
    profile = _synthesize(
        canonical_strategies=[_canonical("austen::x", works=2)])
    assert profile.author_scope["held_out_isolation"]["clean"] is True


def test_strategy_held_out_contamination_raises():
    # 某 canonical 引用了 held-out 作品 persuasion → fail-closed：拒绝合成，而非仅标记。
    canon = _canonical("austen::bad", works=1)
    canon["supporting_work_ids"] = ["persuasion"]
    with pytest.raises(ProfileSynthesisError) as exc:
        _synthesize(canonical_strategies=[canon])
    assert "held-out" in str(exc.value)


def test_profile_work_ids_contamination_raises():
    with pytest.raises(ProfileSynthesisError):
        _synthesize(profile_work_ids=["emma", "persuasion"])


# --------------------------------------------------------------------------- #
# strategy 优先级：确定性、可复现、status 层级正确
# --------------------------------------------------------------------------- #
def test_strategy_priority_deterministic_ordering():
    canonicals = [
        _canonical("austen::d1", status="discovered", works=1, chunks=1, conf=0.9),
        _canonical("austen::v1", status="validated", works=2, chunks=5, conf=0.7),
        _canonical("austen::c1", status="candidate", works=1, chunks=3, conf=0.8),
        _canonical("austen::v2", status="validated", works=2, chunks=10, conf=0.9),
    ]
    ranked = rank_canonical_strategies(canonicals)
    order = [c["canonical_strategy_id"] for c in ranked]
    # validated 恒高于 candidate 恒高于 discovered；同层内 chunk 多者优先
    assert order[0] == "austen::v2"   # validated, 10 chunks
    assert order[1] == "austen::v1"   # validated, 5 chunks
    assert order[2] == "austen::c1"   # candidate
    assert order[3] == "austen::d1"   # discovered
    # priority 1..N 不重不漏
    assert [c["control_priority"] for c in ranked] == [1, 2, 3, 4]


def test_strategy_priority_stable_across_runs():
    canonicals = [_canonical(f"austen::s{i}", status="candidate",
                             works=1, chunks=i + 1, conf=0.5) for i in range(5)]
    a = rank_canonical_strategies(canonicals)
    b = rank_canonical_strategies(canonicals)
    assert [c["control_priority"] for c in a] == [c["control_priority"] for c in b]


# --------------------------------------------------------------------------- #
# 复现性：同输入 → 同结构 + 同 hash
# --------------------------------------------------------------------------- #
def test_synthesis_deterministic():
    args = dict(
        full_corpus_features={"mean_sentence_length": _feature_summary(),
                              "dialogue_ratio": _feature_summary()},
        sampled_features={"simile_frequency": _feature_summary(measurement_type="hybrid")},
        sampled_narrative={"n": 10, "n_total": 10, "n_valid": 10, "n_missing": 0,
                           "pov": {"n": 10, "counts": {"third": 8, "unknown": 2},
                                   "proportions": {"third": 0.8, "unknown": 0.2},
                                   "mode": "third"}},
        canonical_strategies=[_canonical("austen::x", status="validated", works=2, chunks=5)],
        stylometry_author_target={"author_id": "austen", "n_samples": 833,
                                  "source_work_ids": ["emma", "pride_and_prejudice"],
                                  "centroid_norm": 0.123456},
        stylometry_validation_metadata={"n_features": 954},
    )
    p1 = _synthesize(**args)
    p2 = _synthesize(**args)
    assert p1.to_dict() == p2.to_dict()
    assert p1.reproducibility_hash == p2.reproducibility_hash
    assert p1.reproducibility_hash  # 非空


# --------------------------------------------------------------------------- #
# Phase 5.1：diagnostics 拆分 author_target / validation_metadata
# --------------------------------------------------------------------------- #
def test_diagnostics_split_author_target_vs_validation():
    profile = _synthesize(
        stylometry_author_target={"author_id": "austen", "n_samples": 833},
        stylometry_validation_metadata={"n_features": 954, "heldout_accuracy": 0.745},
    )
    stylo = profile.diagnostics["stylometry"]
    assert stylo["control_role"] == "diagnostic"
    assert stylo["author_target"]["author_id"] == "austen"
    assert stylo["author_target"]["n_samples"] == 833
    assert stylo["validation_metadata"]["n_features"] == 954
    assert stylo["validation_metadata"]["heldout_accuracy"] == 0.745
    # 作者专属字段绝不混入全局验证元数据
    assert "n_samples" not in stylo["validation_metadata"]


def test_author_target_distinct_compact_values():
    # 不同 author_target 输入必须原样落到各自的 diagnostics，而非共享同一份。
    a = _synthesize(author_id="austen",
                    stylometry_author_target={"author_id": "austen", "centroid_norm": 0.11})
    d = _synthesize(author_id="dickens",
                    stylometry_author_target={"author_id": "dickens", "centroid_norm": 0.22})
    assert a.diagnostics["stylometry"]["author_target"]["centroid_norm"] == 0.11
    assert d.diagnostics["stylometry"]["author_target"]["centroid_norm"] == 0.22
    assert a.diagnostics["stylometry"]["author_target"] != d.diagnostics["stylometry"]["author_target"]


# --------------------------------------------------------------------------- #
# Phase 5.1：concrete provenance（无 {author_id} 占位符）
# --------------------------------------------------------------------------- #
def test_concrete_provenance_paths_no_placeholder():
    profile = _synthesize(
        author_id="austen",
        canonical_strategies=[_canonical("austen::x", status="validated")])
    for s in profile.strategy_controls:
        assert "{author_id}" not in s.source_artifact
        assert s.source_artifact == "data/analysis/consolidation/austen_canonical_strategies.json"
    for src in profile.provenance["sources"]:
        assert "{author_id}" not in src


# --------------------------------------------------------------------------- #
# Phase 5.1：作者专属文体学目标（纯函数，TRAIN-only，作者互异）
# --------------------------------------------------------------------------- #
def test_author_targets_distinct_and_train_only():
    X = np.array([
        [1.0, 0.0, 0.0],   # austen emma
        [0.9, 0.1, 0.0],   # austen p&p
        [0.0, 2.0, 0.0],   # dickens ge
        [0.0, 1.8, 0.2],   # dickens dc
    ])
    authors = ["austen", "austen", "dickens", "dickens"]
    works = ["emma", "pride_and_prejudice", "great_expectations", "david_copperfield"]
    full, compact = _author_targets_from_matrix(X, authors, works, "0.1.0")

    assert set(compact) == {"austen", "dickens"}
    # 作者质心互不相同（不共享同一文体学目标）
    assert compact["austen"]["centroid_norm"] != compact["dickens"]["centroid_norm"]
    assert full["austen"]["centroid"] != full["dickens"]["centroid"]
    assert compact["austen"]["source_work_ids"] == ["emma", "pride_and_prejudice"]
    assert compact["dickens"]["source_work_ids"] == ["david_copperfield", "great_expectations"]
    # TRAIN-only：绝不出现 held-out 作品
    for c in compact.values():
        assert "persuasion" not in c["source_work_ids"]
        assert "tale_of_two_cities" not in c["source_work_ids"]
        assert c["fit_scope"] == "train_only"
    # full 落盘载体含质心/离散度向量（此处 feature_dim=3）
    assert full["austen"]["feature_dim"] == 3
    assert len(full["austen"]["centroid"]) == 3
    assert len(full["austen"]["dispersion"]) == 3


def test_real_artifacts_austen_dickens_distinct_targets():
    # 集成校验：真实 Layer D TRAIN 矩阵 → Austen/Dickens 目标互异且仅 TRAIN。
    from knowledge.calibration.synthesize import _compute_stylometric_author_targets
    base = Path("data")
    if not (base / "analysis" / "stylometry" / "matrix.npz").exists():
        pytest.skip("data/ artifacts 不存在（gitignored）")
    full, compact = _compute_stylometric_author_targets(base)
    assert set(compact) == {"austen", "dickens"}
    assert compact["austen"]["centroid_norm"] != compact["dickens"]["centroid_norm"]
    assert compact["austen"]["source_work_ids"] == ["emma", "pride_and_prejudice"]
    assert compact["dickens"]["source_work_ids"] == ["david_copperfield", "great_expectations"]
    for c in compact.values():
        assert "persuasion" not in c["source_work_ids"]
        assert "tale_of_two_cities" not in c["source_work_ids"]


# --------------------------------------------------------------------------- #
# Phase 5.1：反序列化 / 校验
# --------------------------------------------------------------------------- #
def _sample_profile_dict():
    return _synthesize(
        author_id="austen",
        full_corpus_features={"mean_sentence_length": _feature_summary()},
        sampled_features={"simile_frequency": _feature_summary(measurement_type="hybrid")},
        sampled_narrative={"n": 10, "n_total": 10, "n_valid": 10, "n_missing": 0,
                           "pov": {"n": 10, "counts": {"third": 8, "unknown": 2},
                                   "proportions": {"third": 0.8, "unknown": 0.2},
                                   "mode": "third"}},
        canonical_strategies=[_canonical("austen::x", status="validated", works=2, chunks=5)],
        stylometry_author_target={"author_id": "austen", "n_samples": 833,
                                  "source_work_ids": ["emma", "pride_and_prejudice"],
                                  "centroid_norm": 0.123456},
        stylometry_validation_metadata={"n_features": 954},
    ).to_dict()


def test_serialization_round_trip_exact():
    d = _sample_profile_dict()
    d2 = json.loads(json.dumps(d))  # 模拟 JSON 落盘再读回
    p = AuthorStyleProfile.from_dict(d2)
    assert p.to_dict() == d


def test_reproducibility_hash_verified_after_reload():
    d = _sample_profile_dict()
    p = AuthorStyleProfile.from_dict(json.loads(json.dumps(d)))
    assert p.verify_reproducibility_hash() is True


def test_malformed_profile_wrong_schema_version_rejected():
    d = _sample_profile_dict()
    d["schema_version"] = "9.9.9"
    with pytest.raises(ProfileSchemaError):
        AuthorStyleProfile.from_dict(d)


def test_malformed_profile_missing_field_rejected():
    d = _sample_profile_dict()
    del d["provenance"]
    with pytest.raises(ProfileSchemaError):
        AuthorStyleProfile.from_dict(d)
