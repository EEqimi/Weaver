# knowledge/planning/schema.py
"""Phase 6 纯数据 schema：WritingRequest / StylePlan / PlannerPolicy + 计划项。

只定义结构，不含 I/O、LLM、随机。所有 to_dict / from_dict 均稳定往返（sort 无关，
键顺序固定）；from_dict 校验必填字段与 schema_version。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..schema.versions import (
    STYLE_PLAN_SCHEMA_VERSION, STYLE_PLANNER_VERSION, WRITING_REQUEST_SCHEMA_VERSION,
)


class PlanningError(Exception):
    """planning 被拒绝（profile 完整性校验失败 / held-out 隔离不干净等）。"""


class PromptBudgetError(Exception):
    """提示词预算无法容纳**强制**（不可降级）内容时抛出。

    语义（Phase 6.1 §3）：预算降级只丢弃可降级内容（策略 / secondary / weak 语言控制 /
    可选解释措辞）；若连强制内容（ROLE / CONTENT / 剩余强控制 / NARRATIVE / IMPORTANT）
    都放不下，绝不硬截断用户内容，而是显式失败。"""


class ActivationLevel(str, Enum):
    """激活级别（有限枚举，绝不使用伪精确连续权重）。"""
    STRONG = "strong"          # 强控制（本次直接写入指令）
    MEDIUM = "medium"          # 中控制
    WEAK = "weak"              # 弱/辅助控制
    REFERENCE = "reference"    # 保留但不激活（供未来版本使用）
    SUPPRESSED = "suppressed"  # 明确禁用，必须记录原因（证据不足 / 预算 / 覆盖）


_ACTIVATION_RANK = {
    ActivationLevel.STRONG.value: 0,
    ActivationLevel.MEDIUM.value: 1,
    ActivationLevel.WEAK.value: 2,
    ActivationLevel.REFERENCE.value: 3,
    ActivationLevel.SUPPRESSED.value: 4,
}


def _require(d: dict[str, Any], *keys: str) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise ValueError(f"缺少必填字段: {missing}")


def _canonical_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha16(*parts: Any) -> str:
    blob = _canonical_json(list(parts))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


@dataclass
class WritingRequest:
    """用户本次写作需求（不是作者风格数据）。planner 绝不改写用户内容意图。"""
    content: str
    desired_length: str = "short_scene"
    target_words: int | None = None
    language: str = "english"
    pov: str | None = None            # 用户显式视角偏好（可覆盖作者倾向，产生 warning）
    constraints: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("content 必须为非空字符串")
        if not isinstance(self.language, str) or not self.language.strip():
            raise ValueError("language 必须为非空字符串")

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "desired_length": self.desired_length,
            "target_words": self.target_words,
            "language": self.language,
            "pov": self.pov,
            "constraints": list(self.constraints),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "WritingRequest":
        _require(d, "content", "desired_length", "target_words", "language",
                 "pov", "constraints")
        if not isinstance(d["content"], str) or not d["content"].strip():
            raise ValueError("content 必须为非空字符串")
        if not isinstance(d["language"], str) or not d["language"].strip():
            raise ValueError("language 必须为非空字符串")
        return cls(
            content=d["content"], desired_length=d["desired_length"],
            target_words=d["target_words"], language=d["language"],
            pov=d["pov"], constraints=list(d["constraints"]),
        )


@dataclass
class PlannedControl:
    """一个被计划的语言控制（language control）。"""
    feature_id: str
    registry_control_role: str        # core / candidate_core / descriptive / experimental
    activation: str                   # ActivationLevel 值
    bucket: str                       # primary / secondary / reference / suppressed
    source_scope: str                 # full_train_corpus / calibration_sample
    support: dict[str, Any]           # n_valid/n_expected/variance/confidence 摘要
    reason: str                       # 为什么被选中（可解释性）
    guidance: str                     # 自然语言：本次如何使用
    source: str                       # provenance（来源产物路径）

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "registry_control_role": self.registry_control_role,
            "activation": self.activation,
            "bucket": self.bucket,
            "source_scope": self.source_scope,
            "support": self.support,
            "reason": self.reason,
            "guidance": self.guidance,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PlannedControl":
        _require(d, "feature_id", "registry_control_role", "activation", "bucket",
                 "source_scope", "support", "reason", "guidance", "source")
        return cls(**d)


@dataclass
class PlannedNarrativeControl:
    """一个被计划的叙事控制（Layer B，sampled evidence）。"""
    field: str
    activation: str                   # ActivationLevel 值
    value_type: str                   # categorical / distribution
    summary: dict[str, Any]           # mode/counts 或 mean_distribution
    reason: str
    guidance: str
    overridden: bool = False          # True：用户约束覆盖作者倾向（保留 warning）

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field, "activation": self.activation,
            "value_type": self.value_type, "summary": self.summary,
            "reason": self.reason, "guidance": self.guidance,
            "overridden": self.overridden,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PlannedNarrativeControl":
        _require(d, "field", "activation", "value_type", "summary", "reason",
                 "guidance", "overridden")
        return cls(**d)


@dataclass
class PlannedStrategy:
    """一个被计划的 canonical strategy（conditional control）。"""
    canonical_strategy_id: str
    canonical_name: str
    support_status: str               # validated / candidate / discovered
    confidence: float | None
    control_priority: int
    n_supporting_works: int
    n_supporting_chunks: int
    activation: str                   # "active"（进入 prompt）或 "reference"（保留）
    trigger: str
    operation: str
    effect: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_strategy_id": self.canonical_strategy_id,
            "canonical_name": self.canonical_name,
            "support_status": self.support_status,
            "confidence": self.confidence,
            "control_priority": self.control_priority,
            "n_supporting_works": self.n_supporting_works,
            "n_supporting_chunks": self.n_supporting_chunks,
            "activation": self.activation,
            "trigger": self.trigger,
            "operation": self.operation,
            "effect": self.effect,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PlannedStrategy":
        _require(d, "canonical_strategy_id", "canonical_name", "support_status",
                 "confidence", "control_priority", "n_supporting_works",
                 "n_supporting_chunks", "activation", "trigger", "operation",
                 "effect", "reason")
        return cls(**d)


@dataclass
class PlannerPolicy:
    """可配置的 planner 策略（控制预算 + candidate_core 门槛）。绝不散落 hard-code。"""
    max_primary_controls: int = 6
    max_secondary_controls: int = 4
    max_narrative_controls: int = 4
    max_strategies: int = 6
    max_prompt_chars: int = 6000
    candidate_core_min_completeness: float = 0.5
    allow_discovered_strategies_as_active: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_primary_controls": self.max_primary_controls,
            "max_secondary_controls": self.max_secondary_controls,
            "max_narrative_controls": self.max_narrative_controls,
            "max_strategies": self.max_strategies,
            "max_prompt_chars": self.max_prompt_chars,
            "candidate_core_min_completeness": self.candidate_core_min_completeness,
            "allow_discovered_strategies_as_active": self.allow_discovered_strategies_as_active,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PlannerPolicy":
        _require(d, "max_primary_controls", "max_secondary_controls",
                 "max_narrative_controls", "max_strategies", "max_prompt_chars",
                 "candidate_core_min_completeness",
                 "allow_discovered_strategies_as_active")
        return cls(**d)


@dataclass
class StylePlan:
    """planner 输出：本次写作激活哪些风格控制（非画像、非 prompt）。"""
    style_plan_id: str
    schema_version: str
    author_id: str
    source_profile_hash: str
    writing_request: dict[str, Any]
    language_controls: list[PlannedControl] = field(default_factory=list)
    narrative_controls: list[PlannedNarrativeControl] = field(default_factory=list)
    strategy_controls: list[PlannedStrategy] = field(default_factory=list)
    reference_controls: list[PlannedControl] = field(default_factory=list)
    reference_strategy_controls: list[PlannedStrategy] = field(default_factory=list)
    suppressed_controls: list[PlannedControl] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    planner_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "style_plan_id": self.style_plan_id,
            "schema_version": self.schema_version,
            "author_id": self.author_id,
            "source_profile_hash": self.source_profile_hash,
            "writing_request": self.writing_request,
            "language_controls": [c.to_dict() for c in self.language_controls],
            "narrative_controls": [c.to_dict() for c in self.narrative_controls],
            "strategy_controls": [c.to_dict() for c in self.strategy_controls],
            "reference_controls": [c.to_dict() for c in self.reference_controls],
            "reference_strategy_controls": [c.to_dict() for c in self.reference_strategy_controls],
            "suppressed_controls": [c.to_dict() for c in self.suppressed_controls],
            "warnings": list(self.warnings),
            "planner_metadata": self.planner_metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StylePlan":
        _require(d, "style_plan_id", "schema_version", "author_id",
                 "source_profile_hash", "writing_request", "language_controls",
                 "narrative_controls", "strategy_controls", "reference_controls",
                 "reference_strategy_controls", "suppressed_controls", "warnings",
                 "planner_metadata")
        if d["schema_version"] != STYLE_PLAN_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version 不匹配: 期望 {STYLE_PLAN_SCHEMA_VERSION}, 得到 {d['schema_version']!r}")
        return cls(
            style_plan_id=d["style_plan_id"], schema_version=d["schema_version"],
            author_id=d["author_id"], source_profile_hash=d["source_profile_hash"],
            writing_request=d["writing_request"],
            language_controls=[PlannedControl.from_dict(c) for c in d["language_controls"]],
            narrative_controls=[PlannedNarrativeControl.from_dict(c) for c in d["narrative_controls"]],
            strategy_controls=[PlannedStrategy.from_dict(c) for c in d["strategy_controls"]],
            reference_controls=[PlannedControl.from_dict(c) for c in d["reference_controls"]],
            reference_strategy_controls=[PlannedStrategy.from_dict(c) for c in d["reference_strategy_controls"]],
            suppressed_controls=[PlannedControl.from_dict(c) for c in d["suppressed_controls"]],
            warnings=list(d["warnings"]),
            planner_metadata=d["planner_metadata"],
        )


def make_style_plan_id(author_id: str, source_profile_hash: str,
                       request: WritingRequest, policy: PlannerPolicy) -> str:
    """确定性 style_plan_id：同一 (author, profile, request, policy) 恒得同一 id。"""
    return _sha16(author_id, source_profile_hash, request.to_dict(), policy.to_dict())


def make_intensity_plan_id(base_plan_id: str, intensity: str) -> str:
    """强度覆写后的确定性 plan id（§19.5 可控性实验）。

    基础 plan（(author, profile, request, policy) 派生）经 apply_intensity 覆写激活
    级别后，其控制已变，须获得**互异**的确定性 id，否则同 id 异 prompt、身份混乱。
    派生自 base_plan_id + intensity，绝不依赖时间。
    """
    return _sha16(base_plan_id, intensity)
