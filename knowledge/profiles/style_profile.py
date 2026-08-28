# knowledge/profiles/style_profile.py
"""Phase 5：作者风格画像合成（Author Style Profile Synthesis）。

把 Phase 1–4.5 已有的测量结果**确定性合成**为统一的 `AuthorStyleProfile`，供未来的
Style Planner 使用。本模块**绝不**：重新分析文本、调用任何 LLM、生成文章、做风格混合、
修改既有 consolidation 产物、把 held-out 作品混入画像。

核心原则（spec Phase 5）：
    - 不是所有"能区分作者"的指标都该直接控制生成：stylometric fingerprint（char
      3-gram / function word / PCA / Delta）进入 diagnostics，**绝不**进入 generation
      controls。
    - control_role 复用既有 `FeatureRegistry.control_role`，经确定性映射得到画像角色，
      不新建第二套互相冲突的 role system。
    - 不确定性是一等数据：连续量带 variance/distribution，n_expected / n_valid /
      n_missing / n_unobservable / n_insufficient 全保留；not_observable /
      insufficient / missing **绝不**伪造为 0。
    - sampled LLM 结果绝不表述为"全语料确定真值"：source_scope 区分
      `full_train_corpus` 与 `calibration_sample`。
    - 确定性可复现：无随机数、无 LLM、无时间戳内容字段；以 reproducibility_hash 校验。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from ..config import data_root as default_data_root
from ..schema.feature_registry import build_default_registry
from ..schema.versions import AUTHOR_STYLE_PROFILE_SCHEMA_VERSION
from ..schema.narrative_schema import (
    DETAIL_DIMENSIONS, PACE_DIMENSIONS,
)

# --------------------------------------------------------------------------- #
# 画像角色（placement/bucket 概念，派生自已有的 registry control_role，见映射表）
# --------------------------------------------------------------------------- #
class ProfileControlRole(str, Enum):
    DIRECT_CONTROL = "direct_control"          # 直接转成写作控制指令（进入 generation controls）
    CONDITIONAL_CONTROL = "conditional_control"  # canonical strategy（trigger→operation→effect）
    DIAGNOSTIC = "diagnostic"                  # 生成后相似度/诊断用（绝不控制生成）
    REFERENCE_ONLY = "reference_only"          # 保留但证据/可控性不足，不主动控制


# 数据来源范围：全文训练语料 vs 40-chunk 标定样本
SCOPE_FULL_CORPUS = "full_train_corpus"
SCOPE_CALIBRATION_SAMPLE = "calibration_sample"

# 既有 registry ControlRole → Phase 5 ProfileControlRole 的确定性映射。
# 依据：registry 里 `diagnostic` 就是 stylometric 指纹；`experimental` 是仅 40-chunk
# 样本的 LLM 派生特征（STATUS 明确"candidate_core 仍不得晋升"），故保持 reference_only；
# `candidate_core` / `descriptive` 是全语料统计且人类可解释、可直接左右写作，故 direct_control。
_REGISTRY_ROLE_TO_PROFILE_ROLE: dict[str, str] = {
    "core": ProfileControlRole.DIRECT_CONTROL.value,
    "candidate_core": ProfileControlRole.DIRECT_CONTROL.value,
    "descriptive": ProfileControlRole.DIRECT_CONTROL.value,
    "diagnostic": ProfileControlRole.DIAGNOSTIC.value,
    "experimental": ProfileControlRole.REFERENCE_ONLY.value,
}

# strategy 优先级：support_status 的层级（validated > candidate > discovered）
_STATUS_TIER = {"validated": 3, "candidate": 2, "discovered": 1}

# narrative 里作为"控制维度"的枚举字段与分布字段（与 aggregation 对齐）
_NARRATIVE_ENUM_FIELDS = (
    "pov", "focalization", "perspective_stability", "narrative_distance",
    "narrator_presence", "narrator_evaluative_intervention",
    "information_access", "temporal_order",
)
_NARRATIVE_DIST_FIELDS = ("temporal_pace", "scene_detail")


def _registry_role(feature_id: str) -> str:
    """从默认 FeatureRegistry 读取某特征的既有 control_role（字符串值）。"""
    reg = build_default_registry()
    if reg.has(feature_id):
        return reg.get(feature_id).control_role.value
    return "experimental"  # 未知特征保守归入 reference_only（不猜测）


def _profile_role(registry_role: str) -> str:
    """确定性映射 registry control_role → 画像角色（未知 → reference_only，不猜测）。"""
    return _REGISTRY_ROLE_TO_PROFILE_ROLE.get(registry_role, ProfileControlRole.REFERENCE_ONLY.value)


# --------------------------------------------------------------------------- #
# 结构
# --------------------------------------------------------------------------- #
@dataclass
class GenerationControl:
    """一个语言特征 generation control（direct_control 或 reference_only）。"""
    feature_id: str
    control_role: str                     # ProfileControlRole 值
    source_scope: str                     # full_train_corpus | calibration_sample
    measurement_type: str
    value_type: str
    registry_control_role: str            # 既有 registry 角色（溯源，不丢）
    summary: dict[str, Any]               # 完整聚合摘要（含 uncertainty counts）
    source_artifact: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "control_role": self.control_role,
            "source_scope": self.source_scope,
            "measurement_type": self.measurement_type,
            "value_type": self.value_type,
            "registry_control_role": self.registry_control_role,
            "summary": self.summary,
            "source_artifact": self.source_artifact,
        }


@dataclass
class NarrativeControl:
    """一个叙事控制维度（Layer B，sampled evidence）。"""
    field: str
    control_role: str                     # direct_control
    source_scope: str                     # calibration_sample
    value_type: str                       # categorical | distribution
    summary: dict[str, Any]
    source_artifact: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "control_role": self.control_role,
            "source_scope": self.source_scope,
            "value_type": self.value_type,
            "summary": self.summary,
            "source_artifact": self.source_artifact,
        }


@dataclass
class StrategyControl:
    """一个 canonical strategy control（conditional_control）。"""
    canonical_strategy_id: str
    canonical_name: str
    canonical_description: str
    trigger_summary: str
    operation_summary: str
    effect_summary: str
    control_role: str                     # conditional_control
    source_scope: str                     # calibration_sample
    support_status: str
    confidence: float | None
    source_strategy_ids: list[str]
    supporting_work_ids: list[str]
    supporting_chunk_ids: list[str]
    control_priority: int                 # 1 = 最高（确定性排名）
    priority_components: dict[str, Any]
    source_artifact: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_strategy_id": self.canonical_strategy_id,
            "canonical_name": self.canonical_name,
            "canonical_description": self.canonical_description,
            "trigger_summary": self.trigger_summary,
            "operation_summary": self.operation_summary,
            "effect_summary": self.effect_summary,
            "control_role": self.control_role,
            "source_scope": self.source_scope,
            "support_status": self.support_status,
            "confidence": self.confidence,
            "source_strategy_ids": self.source_strategy_ids,
            "supporting_work_ids": self.supporting_work_ids,
            "supporting_chunk_ids": self.supporting_chunk_ids,
            "control_priority": self.control_priority,
            "priority_components": self.priority_components,
            "source_artifact": self.source_artifact,
        }


@dataclass
class AuthorStyleProfile:
    """统一的作者风格画像（机器可读、可解释、可追溯、供 Style Planner 使用）。"""
    author_id: str
    schema_version: str
    author_scope: dict[str, Any]
    generation_controls: dict[str, GenerationControl] = field(default_factory=dict)
    narrative_controls: dict[str, NarrativeControl] = field(default_factory=dict)
    strategy_controls: list[StrategyControl] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    uncertainty: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    reproducibility_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "author_id": self.author_id,
            "schema_version": self.schema_version,
            "author_scope": self.author_scope,
            "generation_controls": {k: v.to_dict() for k, v in self.generation_controls.items()},
            "narrative_controls": {k: v.to_dict() for k, v in self.narrative_controls.items()},
            "strategy_controls": [s.to_dict() for s in self.strategy_controls],
            "diagnostics": self.diagnostics,
            "uncertainty": self.uncertainty,
            "provenance": self.provenance,
            "reproducibility_hash": self.reproducibility_hash,
        }


# --------------------------------------------------------------------------- #
# 合成器
# --------------------------------------------------------------------------- #
def _canonical_json(data: dict[str, Any]) -> str:
    """稳定序列化（结构级可复现：sort_keys + 固定分隔符）。"""
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _reproducibility_hash(data: dict[str, Any]) -> str:
    """对画像内容（不含 reproducibility_hash 自身）求 sha256。"""
    return hashlib.sha256(_canonical_json(data).encode("utf-8")).hexdigest()


def _priority_sort_key(cs: dict[str, Any]) -> tuple:
    """strategy 优先级主键（降序）：status tier → 跨作品数 → chunk 数 → confidence → raw 数。

    透明、可复现、可测试：validated 恒高于 candidate 恒高于 discovered；同层内按证据
    广度（跨作品 > 跨 chunk）与 confidence 排序，绝不把 status 简单映射成固定权重了事。
    """
    conf = cs.get("confidence")
    return (
        _STATUS_TIER.get(cs.get("support_status"), 0),
        int(cs.get("number_of_distinct_works") or 0),
        int(cs.get("number_of_distinct_chunks") or 0),
        float(conf) if isinstance(conf, (int, float)) and not isinstance(conf, bool) else 0.0,
        int(cs.get("number_of_raw_observations") or 0),
    )


def rank_canonical_strategies(canonicals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """给 canonical strategies 打上确定性 control_priority（1..N，不重不漏）。

    返回按优先级降序排列的新列表（不改动输入）。排序稳定：先按 canonical_strategy_id
    升序定底，再按 _priority_sort_key 降序，使同分项次序确定。
    """
    base = sorted(canonicals, key=lambda c: str(c["canonical_strategy_id"]))
    ordered = sorted(base, key=_priority_sort_key, reverse=True)
    out: list[dict[str, Any]] = []
    for rank, cs in enumerate(ordered, start=1):
        cs = dict(cs)
        cs["control_priority"] = rank
        cs["priority_components"] = {
            "support_status": cs.get("support_status"),
            "tier": _STATUS_TIER.get(cs.get("support_status"), 0),
            "n_distinct_works": int(cs.get("number_of_distinct_works") or 0),
            "n_distinct_chunks": int(cs.get("number_of_distinct_chunks") or 0),
            "confidence": cs.get("confidence"),
            "n_raw_observations": int(cs.get("number_of_raw_observations") or 0),
        }
        out.append(cs)
    return out


class AuthorStyleProfileSynthesizer:
    """确定性合成器：已有 AuthorProfile + canonical strategies + stylometry → AuthorStyleProfile。

    纯函数：不读文件、不调 LLM、不用随机数/时间。I/O 由 calibration runner 负责。
    """

    def __init__(self) -> None:
        pass

    def synthesize(
        self,
        author_id: str,
        *,
        train_work_ids: list[str],
        held_out_work_ids: list[str],
        profile_work_ids: list[str],
        full_corpus_features: dict[str, dict[str, Any]],
        sampled_features: dict[str, dict[str, Any]],
        sampled_narrative: dict[str, Any],
        canonical_strategies: list[dict[str, Any]],
        stylometry_diagnostics: dict[str, Any],
    ) -> AuthorStyleProfile:
        profile = AuthorStyleProfile(
            author_id=author_id,
            schema_version=AUTHOR_STYLE_PROFILE_SCHEMA_VERSION,
            author_scope=self._build_author_scope(
                author_id, train_work_ids, held_out_work_ids, profile_work_ids,
                canonical_strategies),
            generation_controls=self._build_generation_controls(
                full_corpus_features, sampled_features),
            narrative_controls=self._build_narrative_controls(sampled_narrative),
            strategy_controls=self._build_strategy_controls(canonical_strategies),
            diagnostics=self._build_diagnostics(stylometry_diagnostics),
            uncertainty=self._build_uncertainty(
                full_corpus_features, sampled_features, sampled_narrative,
                canonical_strategies),
            provenance=self._build_provenance(),
        )
        # reproducibility hash 覆盖除 hash 自身外的全部内容
        body = profile.to_dict()
        body.pop("reproducibility_hash", None)
        profile.reproducibility_hash = _reproducibility_hash(body)
        return profile

    # ------------------------------------------------------------------ #
    # 各部分构建
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_author_scope(
        author_id: str, train_work_ids: list[str], held_out_work_ids: list[str],
        profile_work_ids: list[str],
        canonical_strategies: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """作者范围 + held-out 隔离校验（绝不偷偷吸入 held-out 作品）。"""
        train = set(train_work_ids)
        held = sorted(set(held_out_work_ids))

        # 画像声称的 work 范围必须 ⊆ train（held-out 排除）
        profile_contaminated = sorted(set(profile_work_ids) - train)
        strategy_works: set[str] = set()
        for cs in canonical_strategies:
            strategy_works.update(cs.get("supporting_work_ids", []))
        strategy_contaminated = sorted(strategy_works - train)

        return {
            "author_id": author_id,
            "profile_work_ids": sorted(set(profile_work_ids)),
            "train_work_ids": sorted(train),
            "held_out_work_ids": held,
            "strategy_supporting_work_ids": sorted(strategy_works),
            "held_out_isolation": {
                "checked": True,
                "profile_held_out_contamination": profile_contaminated,
                "strategy_held_out_contamination": strategy_contaminated,
                "clean": not profile_contaminated and not strategy_contaminated,
            },
        }

    @staticmethod
    def _build_generation_controls(
        full_corpus_features: dict[str, dict[str, Any]],
        sampled_features: dict[str, dict[str, Any]],
    ) -> dict[str, GenerationControl]:
        """语言特征控制：全语料统计 + sampled LLM 特征，各按 registry 角色定画像角色。"""
        controls: dict[str, GenerationControl] = {}
        for feature_id, summ in full_corpus_features.items():
            role = _profile_role(_registry_role(feature_id))
            if role == ProfileControlRole.DIAGNOSTIC.value:
                continue  # diagnostic（stylometric 指纹）绝不进入 generation controls
            controls[feature_id] = GenerationControl(
                feature_id=feature_id, control_role=role,
                source_scope=SCOPE_FULL_CORPUS,
                measurement_type=summ.get("measurement_type", "statistical"),
                value_type=summ.get("value_type", "continuous"),
                registry_control_role=_registry_role(feature_id),
                summary=summ, source_artifact="data/analysis/profiles/author_profiles.json",
            )
        for feature_id, summ in sampled_features.items():
            role = _profile_role(_registry_role(feature_id))
            if role == ProfileControlRole.DIAGNOSTIC.value:
                continue
            controls[feature_id] = GenerationControl(
                feature_id=feature_id, control_role=role,
                source_scope=SCOPE_CALIBRATION_SAMPLE,
                measurement_type=summ.get("measurement_type", "judgment"),
                value_type=summ.get("value_type", "continuous"),
                registry_control_role=_registry_role(feature_id),
                summary=summ, source_artifact="data/analysis/calibration/profiles/author_profiles.json",
            )
        return controls

    @staticmethod
    def _build_narrative_controls(narrative: dict[str, Any]) -> dict[str, NarrativeControl]:
        """Layer B 叙事控制维度（sampled；not_observable/insufficient/unknown 原样保留）。"""
        controls: dict[str, NarrativeControl] = {}
        for f in _NARRATIVE_ENUM_FIELDS:
            if f not in narrative:
                continue
            controls[f] = NarrativeControl(
                field=f, control_role=ProfileControlRole.DIRECT_CONTROL.value,
                source_scope=SCOPE_CALIBRATION_SAMPLE, value_type="categorical",
                summary=narrative[f],
                source_artifact="data/analysis/calibration/profiles/author_profiles.json",
            )
        for f in _NARRATIVE_DIST_FIELDS:
            if f not in narrative:
                continue
            controls[f] = NarrativeControl(
                field=f, control_role=ProfileControlRole.DIRECT_CONTROL.value,
                source_scope=SCOPE_CALIBRATION_SAMPLE, value_type="distribution",
                summary=narrative[f],
                source_artifact="data/analysis/calibration/profiles/author_profiles.json",
            )
        return controls

    @staticmethod
    def _build_strategy_controls(canonicals: list[dict[str, Any]]) -> list[StrategyControl]:
        """canonical strategies → conditional_control，附确定性 priority。"""
        ranked = rank_canonical_strategies(canonicals)
        out: list[StrategyControl] = []
        for cs in ranked:
            out.append(StrategyControl(
                canonical_strategy_id=cs["canonical_strategy_id"],
                canonical_name=cs.get("canonical_name", ""),
                canonical_description=cs.get("canonical_description", ""),
                trigger_summary=cs.get("trigger_summary", ""),
                operation_summary=cs.get("operation_summary", ""),
                effect_summary=cs.get("effect_summary", ""),
                control_role=ProfileControlRole.CONDITIONAL_CONTROL.value,
                source_scope=SCOPE_CALIBRATION_SAMPLE,
                support_status=cs.get("support_status", ""),
                confidence=cs.get("confidence"),
                source_strategy_ids=list(cs.get("source_strategy_ids", [])),
                supporting_work_ids=list(cs.get("supporting_work_ids", [])),
                supporting_chunk_ids=list(cs.get("supporting_chunk_ids", [])),
                control_priority=cs["control_priority"],
                priority_components=cs["priority_components"],
                source_artifact="data/analysis/consolidation/{author_id}_canonical_strategies.json",
            ))
        return out

    @staticmethod
    def _build_diagnostics(stylometry_diagnostics: dict[str, Any]) -> dict[str, Any]:
        """stylometric diagnostics：只做生成后相似度诊断，绝不进入 generation controls。"""
        return {
            "stylometry": {
                "control_role": ProfileControlRole.DIAGNOSTIC.value,
                "note": ("char 3-gram / function-word fingerprint / PCA / Delta 等微观统计指纹，"
                         "仅用于生成后相似度与作者判别诊断，不作为写作控制指令。"),
                **stylometry_diagnostics,
            },
        }

    @staticmethod
    def _build_uncertainty(
        full_corpus_features: dict[str, dict[str, Any]],
        sampled_features: dict[str, dict[str, Any]],
        sampled_narrative: dict[str, Any],
        canonical_strategies: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """一等不确定性汇总：绝不只给最终值，绝不把 missing/insufficient 合成 0。"""
        def _counts(features: dict[str, dict[str, Any]]) -> dict[str, int]:
            n_expected = n_valid = n_missing = n_unobservable = n_insufficient = 0
            for summ in features.values():
                n_expected += int(summ.get("n_expected") or 0)
                n_valid += int(summ.get("n_valid") or 0)
                n_missing += int(summ.get("n_missing") or 0)
                n_unobservable += int(summ.get("n_unobservable") or 0)
                n_insufficient += int(summ.get("n_insufficient") or 0)
            return {
                "n_expected": n_expected, "n_valid": n_valid, "n_missing": n_missing,
                "n_unobservable": n_unobservable, "n_insufficient": n_insufficient,
            }

        strategy_support = {"validated": 0, "candidate": 0, "discovered": 0}
        strategy_confidence: list[float] = []
        for cs in canonical_strategies:
            st = cs.get("support_status")
            if st in strategy_support:
                strategy_support[st] += 1
            conf = cs.get("confidence")
            if isinstance(conf, (int, float)) and not isinstance(conf, bool):
                strategy_confidence.append(float(conf))

        return {
            "language_full_corpus": _counts(full_corpus_features),
            "language_sampled": _counts(sampled_features),
            "narrative_sample": {
                "n": int(sampled_narrative.get("n") or 0),
                "n_valid": int(sampled_narrative.get("n_valid") or 0),
                "n_missing": int(sampled_narrative.get("n_missing") or 0),
            },
            "strategy_support": strategy_support,
            "strategy_confidence": {
                "n": len(strategy_confidence),
                "mean": round(sum(strategy_confidence) / len(strategy_confidence), 6)
                if strategy_confidence else None,
            },
        }

    @staticmethod
    def _build_provenance() -> dict[str, Any]:
        return {
            "sources": [
                "data/analysis/profiles/author_profiles.json",          # 全语料 Layer A 统计
                "data/analysis/calibration/profiles/author_profiles.json",  # sampled Layer A/B/C
                "data/analysis/consolidation/{author_id}_canonical_strategies.json",  # 作者级 canonical
                "data/analysis/stylometry/{baseline,index}.json",       # Layer D 诊断
            ],
            "authoritative_strategy_source": "author-scoped canonical strategies + support_status",
            "note": ("global strategy_registry.status 为跨作者单调生命周期，不作为作者级 canonical "
                     "支持结论；作者策略以 canonical strategy 的 support_status 为准。"),
        }
