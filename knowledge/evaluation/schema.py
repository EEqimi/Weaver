# knowledge/evaluation/schema.py
"""Phase 8 纯数据 schema：ActualStyleProfile / 偏差 / 文学评价 / 改写计划与结果。

只定义结构，不含 I/O、LLM、随机。所有 to_dict / from_dict 均稳定往返（键顺序固定，
sort 无关）；from_dict 校验必填字段与 schema_version（镜像 planning/schema.py）。

铁律（spec §15 / §19.5 / §21）：
    - 文学评价是**独立 LLM** 判定（6 维 1–10 + 证据引文），与目标画像的测量分离；
    - RevisionItem.instruction 只含**可解释**的自然语言指令（来自字面 guidance 或
      有限词汇），绝不含作者名、原始数值、或微观 stylometric 指纹（"char 3-gram"）；
    - stylometric 指纹只出现在诊断字段（layer_d_stylometric），绝不生成改写指令。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..schema.versions import (
    CONTENT_INTEGRITY_VERSION, EVALUATION_SCHEMA_VERSION,
    FEEDBACK_DECISION_SCHEMA_VERSION, LITERARY_EVALUATION_SCHEMA_VERSION,
    LITERARY_EVALUATOR_VERSION, REVISION_REWRITER_VERSION,
)

# 6 个文学评价维度（独立 LLM 文学评价，README_AGENTS "评价迭代器"）。
LITERARY_DIMENSIONS: tuple[tuple[str, str], ...] = (
    ("plot_logic", "Plot Logic"),
    ("characterization", "Characterization"),
    ("language_texture", "Language Texture"),
    ("theme_expression", "Theme Expression"),
    ("pacing", "Pacing"),
    ("emotional_resonance", "Emotional Resonance"),
)

# 各维度默认权重（总分 = Σ score × weight / Σ weight）。可配置。
DEFAULT_DIMENSION_WEIGHTS: dict[str, float] = {
    "plot_logic": 0.20,
    "characterization": 0.20,
    "language_texture": 0.20,
    "theme_expression": 0.15,
    "pacing": 0.10,
    "emotional_resonance": 0.15,
}

# 改写优先级（spec §15.1）：P0 故事情节/语义连贯 → P1 叙事 → P2 策略 → P3 语言 →
# P4 stylometric。P0 永不因低优先级风格编辑而被破坏。
REVISION_PRIORITIES: tuple[str, ...] = ("P0", "P1", "P2", "P3", "P4")

_PRIORITY_RANK: dict[str, int] = {p: i for i, p in enumerate(REVISION_PRIORITIES)}

# 改写项类别 → 优先级（spec §15.1 的确定性映射）。
CATEGORY_TO_PRIORITY: dict[str, str] = {
    "story_coherence": "P0",
    "narrative": "P1",
    "strategy": "P2",
    "language": "P3",
    "stylometric": "P4",
}

# 文学维度评估状态（Phase 8.1 evidence contract）：至少 1 条逐字验证证据 → observed；
# 证据全部验证失败 → insufficient_evidence（该维不进加权总分）。
ASSESSMENT_OBSERVED = "observed"
ASSESSMENT_INSUFFICIENT = "insufficient_evidence"
ASSESSMENT_STATUSES: tuple[str, ...] = (ASSESSMENT_OBSERVED, ASSESSMENT_INSUFFICIENT)

# 反馈决策结果（Phase 8.1 语义）：no_action 独立于 roll_back（改写计划为空 = 未执行任何
# 改写，不是"回滚"）。
FEEDBACK_ACCEPT = "accept"
FEEDBACK_CONTINUE = "continue"
FEEDBACK_ROLL_BACK = "roll_back"
FEEDBACK_NO_ACTION = "no_action"
FEEDBACK_OUTCOMES: tuple[str, ...] = (
    FEEDBACK_ACCEPT, FEEDBACK_CONTINUE, FEEDBACK_ROLL_BACK, FEEDBACK_NO_ACTION,
)

# 内容完整性违规严重度：critical 使 passed=False；warning 仅记录不阻断。
INTEGRITY_CRITICAL = "critical"
INTEGRITY_WARNING = "warning"


class EvalError(Exception):
    """evaluation 失败（provider 未配置 / 传输失败 / schema 校验失败 / 泄露等）。"""


def _require(d: dict[str, Any], *keys: str) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise EvalError(f"缺少必填字段: {missing}")


def _guard_version(d: dict[str, Any],
                   expected: str = EVALUATION_SCHEMA_VERSION) -> None:
    if d.get("schema_version") != expected:
        raise EvalError(
            f"schema_version 不匹配: 期望 {expected}, "
            f"得到 {d.get('schema_version')!r}")


def priority_rank(priority: str) -> int:
    """改写优先级 → 数值（P0=0 … P4=4），用于确定性排序。"""
    return _PRIORITY_RANK.get(priority, len(_PRIORITY_RANK))


# --------------------------------------------------------------------------- #
# 偏差（目标 vs 实际）
# --------------------------------------------------------------------------- #
@dataclass
class FeatureDeviation:
    """一个激活语言控制的目标带 vs 实测带（经验 band：low/medium/high）。"""
    feature_id: str
    target_band: str | None           # low / medium / high / None（无阈值）
    actual_band: str | None
    target_value: float | None        # 画像 mean（仅记录，绝不进入改写指令）
    actual_value: float | None
    status: str                       # on_target / above / below / not_measurable
    measurable: bool
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "target_band": self.target_band,
            "actual_band": self.actual_band,
            "target_value": self.target_value,
            "actual_value": self.actual_value,
            "status": self.status,
            "measurable": self.measurable,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FeatureDeviation":
        _require(d, "feature_id", "target_band", "actual_band", "target_value",
                 "actual_value", "status", "measurable", "reason")
        return cls(**d)


@dataclass
class NarrativeDeviation:
    """一个激活叙事控制的目标取值 vs 实测取值（类别字段）。"""
    field: str
    target_value: str | None          # 画像 mode（如 "third"）
    actual_value: str | None          # 实测取值（如 "third" / "unknown"）
    status: str                       # on_target / off_target / not_measurable
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field, "target_value": self.target_value,
            "actual_value": self.actual_value, "status": self.status,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "NarrativeDeviation":
        _require(d, "field", "target_value", "actual_value", "status", "reason")
        return cls(**d)


@dataclass
class StrategyCoverage:
    """一个激活 canonical 策略是否在生成正文中被 match 到。"""
    strategy_id: str
    active: bool
    matched: bool
    evidence_quotes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id, "active": self.active,
            "matched": self.matched,
            "evidence_quotes": list(self.evidence_quotes),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StrategyCoverage":
        _require(d, "strategy_id", "active", "matched", "evidence_quotes")
        return cls(strategy_id=d["strategy_id"], active=d["active"],
                   matched=d["matched"], evidence_quotes=list(d["evidence_quotes"]))


@dataclass
class ComparisonResult:
    """compare_target_actual 的纯函数输出：三类偏差 + 汇总。"""
    author_id: str
    passage_id: str
    language_deviations: list[FeatureDeviation] = field(default_factory=list)
    narrative_deviations: list[NarrativeDeviation] = field(default_factory=list)
    strategy_coverage: list[StrategyCoverage] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "author_id": self.author_id, "passage_id": self.passage_id,
            "language_deviations": [d.to_dict() for d in self.language_deviations],
            "narrative_deviations": [d.to_dict() for d in self.narrative_deviations],
            "strategy_coverage": [s.to_dict() for s in self.strategy_coverage],
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ComparisonResult":
        _require(d, "author_id", "passage_id", "language_deviations",
                 "narrative_deviations", "strategy_coverage", "summary")
        return cls(
            author_id=d["author_id"], passage_id=d["passage_id"],
            language_deviations=[FeatureDeviation.from_dict(x)
                                 for x in d["language_deviations"]],
            narrative_deviations=[NarrativeDeviation.from_dict(x)
                                  for x in d["narrative_deviations"]],
            strategy_coverage=[StrategyCoverage.from_dict(x)
                               for x in d["strategy_coverage"]],
            summary=d["summary"],
        )


# --------------------------------------------------------------------------- #
# 实测画像（对生成正文的再测量结果）
# --------------------------------------------------------------------------- #
@dataclass
class ActualStyleProfile:
    """生成正文的"实际风格画像"（spec §15 再测量输出）。

    各层测量结果以既有 schema 的 to_dict 形式内嵌（Layer A 的 FeatureValue dict、
    Layer B 的 NarrativeObservation dict、Layer C 的 StrategyEvidence dict），
    避免在此重复实现它们的 from_dict；本结构只做身份 + 元数据 + 校验。
    """
    schema_version: str
    author_id: str
    passage_id: str                    # 被测正文的 generation_id
    passage_hash: str                  # 被测正文 sha256
    style_plan_id: str
    layer_a_statistical: dict[str, dict[str, Any]] = field(default_factory=dict)
    layer_a_judgment: dict[str, dict[str, Any]] = field(default_factory=dict)
    layer_b_narrative: dict[str, Any] | None = None
    layer_c_strategies: list[dict[str, Any]] = field(default_factory=list)
    layer_d_stylometric: dict[str, Any] | None = None
    # 各层中 AnalysisUnavailable 的 kind 列表（绝不静默丢弃不可用测量）
    unavailable: dict[str, list[str]] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "author_id": self.author_id,
            "passage_id": self.passage_id,
            "passage_hash": self.passage_hash,
            "style_plan_id": self.style_plan_id,
            "layer_a_statistical": self.layer_a_statistical,
            "layer_a_judgment": self.layer_a_judgment,
            "layer_b_narrative": self.layer_b_narrative,
            "layer_c_strategies": self.layer_c_strategies,
            "layer_d_stylometric": self.layer_d_stylometric,
            "unavailable": self.unavailable,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ActualStyleProfile":
        _require(d, "schema_version", "author_id", "passage_id", "passage_hash",
                 "style_plan_id", "layer_a_statistical", "layer_a_judgment",
                 "layer_b_narrative", "layer_c_strategies", "layer_d_stylometric",
                 "unavailable", "provenance")
        _guard_version(d)
        return cls(
            schema_version=d["schema_version"], author_id=d["author_id"],
            passage_id=d["passage_id"], passage_hash=d["passage_hash"],
            style_plan_id=d["style_plan_id"],
            layer_a_statistical=d["layer_a_statistical"],
            layer_a_judgment=d["layer_a_judgment"],
            layer_b_narrative=d["layer_b_narrative"],
            layer_c_strategies=d["layer_c_strategies"],
            layer_d_stylometric=d["layer_d_stylometric"],
            unavailable=d["unavailable"], provenance=d["provenance"],
        )


# --------------------------------------------------------------------------- #
# 独立 LLM 文学评价（6 维）
# --------------------------------------------------------------------------- #
@dataclass
class DimensionScore:
    """单维文学评价：1–10 分 + 简述 + 至少一个优点 + 至少一个缺点 + 逐字证据引文。

    Phase 8.1 evidence contract：`assessment_status` 区分 `observed`（≥1 条逐字验证
    证据）与 `insufficient_evidence`（证据全部验证失败，该维不进加权总分）。
    """
    dimension: str
    label: str
    score: float                       # 1–10
    summary: str
    strength: str
    weakness: str
    evidence: list[str] = field(default_factory=list)   # 逐字引文（校验后）
    assessment_status: str = ASSESSMENT_OBSERVED        # observed / insufficient_evidence
    verified_evidence_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension, "label": self.label, "score": self.score,
            "summary": self.summary, "strength": self.strength,
            "weakness": self.weakness, "evidence": list(self.evidence),
            "assessment_status": self.assessment_status,
            "verified_evidence_count": self.verified_evidence_count,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DimensionScore":
        _require(d, "dimension", "label", "score", "summary", "strength",
                 "weakness", "evidence")
        return cls(
            dimension=d["dimension"], label=d["label"], score=d["score"],
            summary=d["summary"], strength=d["strength"], weakness=d["weakness"],
            evidence=list(d["evidence"]),
            assessment_status=d.get("assessment_status", ASSESSMENT_OBSERVED),
            verified_evidence_count=d.get("verified_evidence_count",
                                          len(list(d["evidence"]))),
        )


@dataclass
class LiteraryEvaluation:
    """独立 LLM 文学评价结果（与目标画像测量完全分离）。"""
    schema_version: str
    author_id: str
    passage_id: str
    dimensions: dict[str, DimensionScore]
    weights: dict[str, float]
    total_score: float
    summary: str
    blind: bool
    evaluator_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "author_id": self.author_id,
            "passage_id": self.passage_id,
            "dimensions": {k: v.to_dict() for k, v in self.dimensions.items()},
            "weights": self.weights,
            "total_score": self.total_score,
            "summary": self.summary,
            "blind": self.blind,
            "evaluator_version": self.evaluator_version,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LiteraryEvaluation":
        _require(d, "schema_version", "author_id", "passage_id", "dimensions",
                 "weights", "total_score", "summary", "blind", "evaluator_version")
        _guard_version(d, LITERARY_EVALUATION_SCHEMA_VERSION)
        return cls(
            schema_version=d["schema_version"], author_id=d["author_id"],
            passage_id=d["passage_id"],
            dimensions={k: DimensionScore.from_dict(v)
                        for k, v in d["dimensions"].items()},
            weights=d["weights"], total_score=d["total_score"], summary=d["summary"],
            blind=d["blind"], evaluator_version=d["evaluator_version"],
        )


# --------------------------------------------------------------------------- #
# 改写计划与结果
# --------------------------------------------------------------------------- #
@dataclass
class RevisionItem:
    """一条改写项：优先级（P0–P4）+ 可解释指令（绝不含作者名 / 原始数值 / 微观指纹）。"""
    priority: str                      # P0..P4
    category: str                      # story_coherence / narrative / strategy / language / stylometric
    target: str                        # feature_id / field / strategy_id / dimension
    instruction: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "priority": self.priority, "category": self.category,
            "target": self.target, "instruction": self.instruction,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RevisionItem":
        _require(d, "priority", "category", "target", "instruction", "reason")
        return cls(**d)


@dataclass
class RevisionPlan:
    """优先化的改写计划（P0→P4 有序）。"""
    schema_version: str
    author_id: str
    passage_id: str
    style_plan_id: str
    revision_items: list[RevisionItem] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "author_id": self.author_id,
            "passage_id": self.passage_id,
            "style_plan_id": self.style_plan_id,
            "revision_items": [i.to_dict() for i in self.revision_items],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RevisionPlan":
        _require(d, "schema_version", "author_id", "passage_id", "style_plan_id",
                 "revision_items", "metadata")
        _guard_version(d)
        return cls(
            schema_version=d["schema_version"], author_id=d["author_id"],
            passage_id=d["passage_id"], style_plan_id=d["style_plan_id"],
            revision_items=[RevisionItem.from_dict(i) for i in d["revision_items"]],
            metadata=d["metadata"],
        )


@dataclass
class RevisionResult:
    """一次最小编辑改写的结果：新正文 + 变更说明（局部性映射）+ 溯源。"""
    schema_version: str
    author_id: str
    passage_id: str
    original_passage_hash: str
    revised_passage_hash: str
    revised_text: str
    change_descriptions: list[str] = field(default_factory=list)
    revision_items_applied: list[str] = field(default_factory=list)
    blind: bool = True
    rewriter_version: str = REVISION_REWRITER_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "author_id": self.author_id,
            "passage_id": self.passage_id,
            "original_passage_hash": self.original_passage_hash,
            "revised_passage_hash": self.revised_passage_hash,
            "revised_text": self.revised_text,
            "change_descriptions": list(self.change_descriptions),
            "revision_items_applied": list(self.revision_items_applied),
            "blind": self.blind,
            "rewriter_version": self.rewriter_version,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RevisionResult":
        _require(d, "schema_version", "author_id", "passage_id",
                 "original_passage_hash", "revised_passage_hash", "revised_text",
                 "change_descriptions", "revision_items_applied", "blind",
                 "rewriter_version")
        _guard_version(d)
        return cls(
            schema_version=d["schema_version"], author_id=d["author_id"],
            passage_id=d["passage_id"],
            original_passage_hash=d["original_passage_hash"],
            revised_passage_hash=d["revised_passage_hash"],
            revised_text=d["revised_text"],
            change_descriptions=list(d["change_descriptions"]),
            revision_items_applied=list(d["revision_items_applied"]),
            blind=d["blind"], rewriter_version=d["rewriter_version"],
        )


# --------------------------------------------------------------------------- #
# Phase 8.1：评价策略 + 内容完整性 + 反馈决策
# --------------------------------------------------------------------------- #
@dataclass
class EvaluationPolicy:
    """可配置决策策略（统一配置，不散落硬编码常数；spec §二 STEP 2）。

    `max_literary_drop`：文学总分允许的最大下降（超过 → roll_back）。默认 0.5 是合理
    经验值，绝非科学真值，且可配置、有文档、有测试。
    """
    max_literary_drop: float = 0.5
    weak_score_threshold: float = 5.0   # 文学维度低于此分产生一条改写项

    def to_dict(self) -> dict[str, Any]:
        return {"max_literary_drop": self.max_literary_drop,
                "weak_score_threshold": self.weak_score_threshold}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EvaluationPolicy":
        return cls(
            max_literary_drop=float(d.get("max_literary_drop", 0.5)),
            weak_score_threshold=float(d.get("weak_score_threshold", 5.0)),
        )


@dataclass
class ContentIntegrityViolation:
    """一条内容完整性违规（critical 阻断；warning 仅记录）。"""
    kind: str          # plot_facts / characters / relationships / constraints /
                       # new_event / removed_event
    severity: str      # critical / warning
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "severity": self.severity,
                "description": self.description}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ContentIntegrityViolation":
        _require(d, "kind", "severity", "description")
        return cls(kind=d["kind"], severity=d["severity"],
                   description=d["description"])


@dataclass
class ContentIntegrityResult:
    """内容完整性检查结果：改写是否破坏用户内容（plot/角色/关系/约束/事件增删）。"""
    schema_version: str
    checker_version: str
    passed: bool
    plot_facts_preserved: bool
    characters_preserved: bool
    relationships_preserved: bool
    constraints_preserved: bool
    new_major_events: bool              # True = 违规（新增主要事件）
    removed_major_events: bool          # True = 违规（删除主要事件）
    violations: list[ContentIntegrityViolation] = field(default_factory=list)
    reasoning_summary: str = ""         # 简短，绝不保存 hidden chain-of-thought
    deterministic: bool = False         # True = 确定性短路判定（未调 LLM）
    blind: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "checker_version": self.checker_version,
            "passed": self.passed,
            "plot_facts_preserved": self.plot_facts_preserved,
            "characters_preserved": self.characters_preserved,
            "relationships_preserved": self.relationships_preserved,
            "constraints_preserved": self.constraints_preserved,
            "new_major_events": self.new_major_events,
            "removed_major_events": self.removed_major_events,
            "violations": [v.to_dict() for v in self.violations],
            "reasoning_summary": self.reasoning_summary,
            "deterministic": self.deterministic,
            "blind": self.blind,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ContentIntegrityResult":
        _require(d, "schema_version", "checker_version", "passed",
                 "plot_facts_preserved", "characters_preserved",
                 "relationships_preserved", "constraints_preserved",
                 "new_major_events", "removed_major_events", "violations",
                 "reasoning_summary", "deterministic", "blind")
        _guard_version(d)
        return cls(
            schema_version=d["schema_version"],
            checker_version=d["checker_version"],
            passed=d["passed"],
            plot_facts_preserved=d["plot_facts_preserved"],
            characters_preserved=d["characters_preserved"],
            relationships_preserved=d["relationships_preserved"],
            constraints_preserved=d["constraints_preserved"],
            new_major_events=d["new_major_events"],
            removed_major_events=d["removed_major_events"],
            violations=[ContentIntegrityViolation.from_dict(v)
                        for v in d["violations"]],
            reasoning_summary=d["reasoning_summary"],
            deterministic=d["deterministic"],
            blind=d["blind"],
        )


@dataclass
class FeedbackDecision:
    """反馈决策（可审计）：Style Fidelity 与 Literary Quality 分别报告，绝不合并成
    单一加权总分。决策基于 gate/规则，而非一个神秘加权分。"""
    schema_version: str
    outcome: str                       # accept / continue / roll_back / no_action
    reason: str
    content_integrity_passed: bool | None      # None = 未运行（no_action / 改写失败）
    content_integrity: dict[str, Any] | None   # ContentIntegrityResult.to_dict() 或 None
    style_fidelity: dict[str, Any]             # {before_n, after_n, improved}
    literary_quality: dict[str, Any]           # {before, after, drop, tolerance,
                                               #  drop_exceeded, evaluated}
    iteration: int
    max_iterations: int
    author_id: str = ""
    passage_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "outcome": self.outcome,
            "reason": self.reason,
            "content_integrity_passed": self.content_integrity_passed,
            "content_integrity": self.content_integrity,
            "style_fidelity": self.style_fidelity,
            "literary_quality": self.literary_quality,
            "iteration": self.iteration,
            "max_iterations": self.max_iterations,
            "author_id": self.author_id,
            "passage_id": self.passage_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FeedbackDecision":
        _require(d, "schema_version", "outcome", "reason",
                 "content_integrity_passed", "content_integrity",
                 "style_fidelity", "literary_quality", "iteration",
                 "max_iterations")
        _guard_version(d, FEEDBACK_DECISION_SCHEMA_VERSION)
        return cls(
            schema_version=d["schema_version"], outcome=d["outcome"],
            reason=d["reason"],
            content_integrity_passed=d["content_integrity_passed"],
            content_integrity=d["content_integrity"],
            style_fidelity=d["style_fidelity"],
            literary_quality=d["literary_quality"],
            iteration=d["iteration"], max_iterations=d["max_iterations"],
            author_id=d.get("author_id", ""), passage_id=d.get("passage_id", ""),
        )
