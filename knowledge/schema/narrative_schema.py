# knowledge/schema/narrative_schema.py
"""Narrative Profile 的结构化 schema（Layer B，spec §5）。

叙事特征多为离散/序数/结构化取值，**不**用连续平均表示。例如 POV 不会
被编码成 "0.73 第三人称"，而是保留类别与分布。

关键约束（Phase 3 §4）：
    - 区分"观察到的证据"（observed_evidence，逐字引用）与"解释"（interpretation）；
    - 单 chunk 不得推断 work/author 级结论（字段 scope 明确为 chunk）。

标定就绪（Phase 3–4.1，task item 6）：
    - 每个类别字段允许显式"不可判定"表示：unknown / insufficient_evidence /
      not_observable，绝不因 dataclass 默认值把缺失字段伪装成"已观察事实"；
    - temporal_pace / scene_detail 校验数值、范围、键与近似归一化；
    - 未通过证据校验的引文显式标记，不静默丢弃。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .versions import NARRATIVE_SCHEMA_VERSION

# 显式"不可判定/不可观察"表示（task item 6）：替代强迫性类别判断
UNKNOWN = "unknown"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"
NOT_OBSERVABLE = "not_observable"
UNKNOWN_VALUES = (UNKNOWN, INSUFFICIENT_EVIDENCE, NOT_OBSERVABLE)


def _with_unknown(*values: str) -> tuple[str, ...]:
    """类别取值 + 三种"不可判定"表示。"""
    return tuple(values) + UNKNOWN_VALUES


# 允许的取值（供 analyzer 输出校验与聚合使用；含显式"不可判定"表示）
POV_VALUES = _with_unknown("first", "second", "third", "mixed")
FOCALIZATION_VALUES = _with_unknown("zero", "internal", "external", "mixed")
DISTANCE_VALUES = _with_unknown("close", "medium", "distant", "mixed")
PRESENCE_VALUES = _with_unknown("none", "low", "medium", "high")
STABILITY_VALUES = _with_unknown("stable", "mostly_stable", "shifting", "mixed")
INFORMATION_ACCESS_VALUES = _with_unknown("omniscient", "limited", "restricted", "dramatic", "mixed")
TEMPORAL_ORDER_VALUES = _with_unknown("chronological", "analepsis", "prolepsis", "insertion", "mixed")
DETAIL_DIMENSIONS = ("psychology", "action", "dialogue", "objects",
                     "environment", "social_relations")
PACE_DIMENSIONS = ("scene", "summary", "ellipsis")

# 类别字段 → 允许取值（供 validate_narrative 使用）
_CATEGORICAL_FIELDS: dict[str, tuple[str, ...]] = {
    "pov": POV_VALUES,
    "focalization": FOCALIZATION_VALUES,
    "perspective_stability": STABILITY_VALUES,
    "narrative_distance": DISTANCE_VALUES,
    "narrator_presence": PRESENCE_VALUES,
    "narrator_evaluative_intervention": PRESENCE_VALUES,
    "information_access": INFORMATION_ACCESS_VALUES,
    "temporal_order": TEMPORAL_ORDER_VALUES,
}

# 比例字段的近似归一化容差（绝对值）
_PROPORTION_TOLERANCE = 0.05


@dataclass
class NarrativeObservation:
    """单个 chunk 的叙事观察（LLM 产出，经 schema 校验后落盘）。"""

    # 身份
    chunk_id: str
    schema_version: str = NARRATIVE_SCHEMA_VERSION

    # 视角与聚焦（默认 unknown：缺失字段不伪装成"已观察事实"）
    pov: str = UNKNOWN                       # POV_VALUES
    focalization: str = UNKNOWN              # FOCALIZATION_VALUES
    focalizer: str | None = None             # 聚焦者（人名/叙述者，自由文本）
    perspective_stability: str = UNKNOWN     # STABILITY_VALUES

    # 距离与叙述者
    narrative_distance: str = UNKNOWN        # DISTANCE_VALUES
    narrator_presence: str = UNKNOWN         # PRESENCE_VALUES
    narrator_evaluative_intervention: str = UNKNOWN  # PRESENCE_VALUES

    # 信息
    information_access: str = UNKNOWN        # INFORMATION_ACCESS_VALUES
    information_withholding: str | None = None
    revelation_timing: str | None = None

    # 时间
    temporal_order: str = UNKNOWN            # TEMPORAL_ORDER_VALUES
    temporal_pace: dict[str, float] = field(default_factory=dict)   # PACE_DIMENSIONS → 占比
    scene_detail: dict[str, float] = field(default_factory=dict)    # DETAIL_DIMENSIONS → 占比

    # 证据与解释（二者分离）
    observed_evidence: list[str] = field(default_factory=list)      # 逐字引用（已验证）
    unverified_evidence: list[str] = field(default_factory=list)    # 无法验证的引文（显式标记）
    interpretation: str = ""                                        # 解释（非隐藏思维链）

    # 比例字段的校验问题（键/数值/归一化），供 QC 与追溯
    proportion_issues: list[str] = field(default_factory=list)

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
            "unverified_evidence": self.unverified_evidence,
            "interpretation": self.interpretation,
            "proportion_issues": self.proportion_issues,
            "confidence": self.confidence,
        }


def _validate_proportions(name: str, data: Any,
                          dims: tuple[str, ...]) -> tuple[dict[str, float], list[str]]:
    """校验比例字段：数值、范围、键、近似归一化；返回 (干净 dict, 问题列表)。

    - 未知键：显式记录（不静默丢弃）；
    - 非数值 / 负值：抛 ValueError（malformed）；
    - 比例之和偏离 1（超过容差）：记录问题，保留原值（不伪造归一化）。
    """
    if data is None:
        return {}, []
    if not isinstance(data, dict):
        raise ValueError(f"narrative.{name} 必须是对象")
    clean: dict[str, float] = {}
    issues: list[str] = []
    for k, v in data.items():
        if k not in dims:
            issues.append(f"{name} 含未知维度 {k!r}（忽略）")
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"narrative.{name}.{k} 必须是数值: {v!r}")
        if v < 0:
            raise ValueError(f"narrative.{name}.{k} 必须 >= 0: {v!r}")
        clean[k] = float(v)
    if clean:
        s = sum(clean.values())
        if s > 0 and not math.isclose(s, 1.0, abs_tol=_PROPORTION_TOLERANCE):
            issues.append(f"{name} 比例之和 {s:.3f} 偏离 1（容差 {_PROPORTION_TOLERANCE}）")
    return clean, issues


def validate_narrative(data: dict[str, Any]) -> NarrativeObservation:
    """校验 LLM 产出的叙事 JSON；非法取值抛 ValueError，缺失类别字段显式置 unknown。"""
    def _check(name: str, value: Any, allowed: tuple) -> str:
        if value not in allowed:
            raise ValueError(f"narrative.{name}={value!r} 不在 {allowed} 中")
        return value

    # 类别字段：缺失 → unknown（不伪造），非法 → 报错
    resolved: dict[str, str] = {}
    for field_name, allowed in _CATEGORICAL_FIELDS.items():
        value = data.get(field_name)
        if value is None or value == "":
            value = UNKNOWN
        resolved[field_name] = _check(field_name, value, allowed)

    observed = data.get("observed_evidence", [])
    if not isinstance(observed, list) or not all(isinstance(e, str) for e in observed):
        raise ValueError("narrative.observed_evidence 必须是字符串列表")
    confidence = data.get("confidence", 0.0)
    if not (isinstance(confidence, (int, float)) and 0.0 <= confidence <= 1.0):
        raise ValueError("narrative.confidence 必须是 [0,1] 数值")

    pace, pace_issues = _validate_proportions("temporal_pace",
                                              data.get("temporal_pace"), PACE_DIMENSIONS)
    detail, detail_issues = _validate_proportions("scene_detail",
                                                  data.get("scene_detail"), DETAIL_DIMENSIONS)

    return NarrativeObservation(
        chunk_id=data.get("chunk_id", ""),
        pov=resolved["pov"],
        focalization=resolved["focalization"],
        focalizer=data.get("focalizer"),
        perspective_stability=resolved["perspective_stability"],
        narrative_distance=resolved["narrative_distance"],
        narrator_presence=resolved["narrator_presence"],
        narrator_evaluative_intervention=resolved["narrator_evaluative_intervention"],
        information_access=resolved["information_access"],
        information_withholding=data.get("information_withholding"),
        revelation_timing=data.get("revelation_timing"),
        temporal_order=resolved["temporal_order"],
        temporal_pace=pace,
        scene_detail=detail,
        observed_evidence=observed,
        interpretation=data.get("interpretation", ""),
        proportion_issues=pace_issues + detail_issues,
        confidence=float(confidence),
    )
