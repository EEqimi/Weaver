# knowledge/schema/narrative_schema.py
"""Narrative Profile 的结构化 schema（Layer B，spec §5）。

叙事特征多为离散/序数/结构化取值，**不**用连续平均表示。例如 POV 不会
被编码成 "0.73 第三人称"，而是保留类别与分布。

关键约束（Phase 3 §4）：
    - 区分"观察到的证据"（observed_evidence，逐字引用）与"解释"（interpretation）；
    - 单 chunk 不得推断 work/author 级结论（字段 scope 明确为 chunk）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .versions import NARRATIVE_SCHEMA_VERSION


# 允许的取值（供 analyzer 输出校验与聚合使用）
POV_VALUES = ("first", "second", "third", "mixed")
FOCALIZATION_VALUES = ("zero", "internal", "external", "mixed")
DISTANCE_VALUES = ("close", "medium", "distant", "mixed")
PRESENCE_VALUES = ("none", "low", "medium", "high")
STABILITY_VALUES = ("stable", "mostly_stable", "shifting", "mixed")
INFORMATION_ACCESS_VALUES = ("omniscient", "limited", "restricted", "dramatic", "mixed")
TEMPORAL_ORDER_VALUES = ("chronological", "analepsis", "prolepsis", "insertion", "mixed")
DETAIL_DIMENSIONS = ("psychology", "action", "dialogue", "objects",
                     "environment", "social_relations")
PACE_DIMENSIONS = ("scene", "summary", "ellipsis")


@dataclass
class NarrativeObservation:
    """单个 chunk 的叙事观察（LLM 产出，经 schema 校验后落盘）。"""

    # 身份
    chunk_id: str
    schema_version: str = NARRATIVE_SCHEMA_VERSION

    # 视角与聚焦
    pov: str = "third"                       # POV_VALUES
    focalization: str = "external"           # FOCALIZATION_VALUES
    focalizer: str | None = None             # 聚焦者（人名/叙述者，自由文本）
    perspective_stability: str = "stable"    # STABILITY_VALUES

    # 距离与叙述者
    narrative_distance: str = "medium"       # DISTANCE_VALUES
    narrator_presence: str = "low"           # PRESENCE_VALUES
    narrator_evaluative_intervention: str = "low"  # PRESENCE_VALUES

    # 信息
    information_access: str = "limited"      # INFORMATION_ACCESS_VALUES
    information_withholding: str | None = None
    revelation_timing: str | None = None

    # 时间
    temporal_order: str = "chronological"    # TEMPORAL_ORDER_VALUES
    temporal_pace: dict[str, float] = field(default_factory=dict)   # PACE_DIMENSIONS → 占比
    scene_detail: dict[str, float] = field(default_factory=dict)    # DETAIL_DIMENSIONS → 占比

    # 证据与解释（二者分离）
    observed_evidence: list[str] = field(default_factory=list)      # 逐字引用
    interpretation: str = ""                                        # 解释（非隐藏思维链）

    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "schema_version": self.schema_version,
            "pov": self.pov,
            "focalization": self.focalization,
            "focalizer": self.focalizer,
            "perspective_stability": self.perspective_stability,
            "narrative_distance": self.narrative_distance,
            "narrator_presence": self.narrator_presence,
            "narrator_evaluative_intervention": self.narrator_evaluative_intervention,
            "information_access": self.information_access,
            "information_withholding": self.information_withholding,
            "revelation_timing": self.revelation_timing,
            "temporal_order": self.temporal_order,
            "temporal_pace": self.temporal_pace,
            "scene_detail": self.scene_detail,
            "observed_evidence": self.observed_evidence,
            "interpretation": self.interpretation,
            "confidence": self.confidence,
        }


def validate_narrative(data: dict[str, Any]) -> NarrativeObservation:
    """校验 LLM 产出的叙事 JSON，非法取值抛 ValueError。"""
    def _check(name: str, value: Any, allowed: tuple) -> None:
        if value not in allowed:
            raise ValueError(f"narrative.{name}={value!r} 不在 {allowed} 中")

    _check("pov", data.get("pov"), POV_VALUES)
    _check("focalization", data.get("focalization"), FOCALIZATION_VALUES)
    _check("perspective_stability", data.get("perspective_stability"), STABILITY_VALUES)
    _check("narrative_distance", data.get("narrative_distance"), DISTANCE_VALUES)
    _check("narrator_presence", data.get("narrator_presence"), PRESENCE_VALUES)
    _check("narrator_evaluative_intervention",
           data.get("narrator_evaluative_intervention"), PRESENCE_VALUES)
    _check("information_access", data.get("information_access"), INFORMATION_ACCESS_VALUES)
    _check("temporal_order", data.get("temporal_order"), TEMPORAL_ORDER_VALUES)

    observed = data.get("observed_evidence", [])
    if not isinstance(observed, list) or not all(isinstance(e, str) for e in observed):
        raise ValueError("narrative.observed_evidence 必须是字符串列表")
    confidence = data.get("confidence", 0.0)
    if not (isinstance(confidence, (int, float)) and 0.0 <= confidence <= 1.0):
        raise ValueError("narrative.confidence 必须是 [0,1] 数值")

    return NarrativeObservation(
        chunk_id=data.get("chunk_id", ""),
        pov=data["pov"],
        focalization=data["focalization"],
        focalizer=data.get("focalizer"),
        perspective_stability=data["perspective_stability"],
        narrative_distance=data["narrative_distance"],
        narrator_presence=data["narrator_presence"],
        narrator_evaluative_intervention=data["narrator_evaluative_intervention"],
        information_access=data["information_access"],
        information_withholding=data.get("information_withholding"),
        revelation_timing=data.get("revelation_timing"),
        temporal_order=data["temporal_order"],
        temporal_pace={k: v for k, v in data.get("temporal_pace", {}).items()
                       if k in PACE_DIMENSIONS},
        scene_detail={k: v for k, v in data.get("scene_detail", {}).items()
                      if k in DETAIL_DIMENSIONS},
        observed_evidence=observed,
        interpretation=data.get("interpretation", ""),
        confidence=float(confidence),
    )
