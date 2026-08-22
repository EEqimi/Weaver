# knowledge/schema/strategy_schema.py
"""Creative Strategy 的结构化 schema（Layer C，spec §6）。

Creative Strategy 表达"在什么条件下 → 执行什么写作操作 → 产生什么文学效果"
（TRIGGER → OPERATION → EFFECT），而非"风格形容词列表"或"写复杂句"之类的
模糊观察。

生命周期（Phase 3 §5）：
    discovered（单 chunk 证据）
    → candidate（多 chunk 证据）
    → validated（跨作品证据，作者级稳定策略）
新策略绝不立即成为 Author Strategy。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .versions import (
    CANONICAL_STRATEGY_SCHEMA_VERSION, STRATEGY_MINER_VERSION, STRATEGY_SCHEMA_VERSION,
)


class StrategyStatus(str, Enum):
    DISCOVERED = "discovered"   # 单 chunk 提出
    CANDIDATE = "candidate"     # 同作品多 chunk 证据
    VALIDATED = "validated"     # 跨作品（作者级）证据


@dataclass
class StrategyEvidence:
    """一条策略证据：来自哪个 chunk/work 的哪段引用。

    task item 7：保留 match confidence 与全部**有效**证据引文（而非只保留第一条、
    丢弃置信度）；未验证引文显式标记；附 analyzer/schema 溯源。
    """
    chunk_id: str
    work_id: str
    author_id: str
    strategy_id: str = ""                             # 该证据支持哪个策略（聚合计数需要）
    quote: str = ""                                   # 首条有效引文（向后兼容）
    quotes: list[str] = field(default_factory=list)   # 全部有效引文
    unverified_quotes: list[str] = field(default_factory=list)  # 无法验证的引文
    confidence: float | None = None                   # match confidence（不丢弃）
    analyzer_id: str = "StrategyMiner"
    analyzer_version: str = STRATEGY_MINER_VERSION
    schema_version: str = STRATEGY_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "work_id": self.work_id,
            "author_id": self.author_id,
            "strategy_id": self.strategy_id,
            "quote": self.quote,
            "quotes": self.quotes,
            "unverified_quotes": self.unverified_quotes,
            "confidence": self.confidence,
            "analyzer_id": self.analyzer_id,
            "analyzer_version": self.analyzer_version,
            "schema_version": self.schema_version,
        }


@dataclass
class CreativeStrategy:
    strategy_id: str
    name: str
    description: str

    triggers: list[str] = field(default_factory=list)           # 触发条件
    operations: list[str] = field(default_factory=list)         # 写作操作
    intended_effects: list[str] = field(default_factory=list)   # 预期效果

    constraints: list[str] = field(default_factory=list)
    incompatibilities: list[str] = field(default_factory=list)
    compatible_strategies: list[str] = field(default_factory=list)

    strength: float | None = None
    confidence: float | None = None
    evidence: list[StrategyEvidence] = field(default_factory=list)

    source_author: str | None = None
    source_work: str | None = None

    status: str = StrategyStatus.DISCOVERED.value
    schema_version: str = STRATEGY_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "name": self.name,
            "description": self.description,
            "triggers": self.triggers,
            "operations": self.operations,
            "intended_effects": self.intended_effects,
            "constraints": self.constraints,
            "incompatibilities": self.incompatibilities,
            "compatible_strategies": self.compatible_strategies,
            "strength": self.strength,
            "confidence": self.confidence,
            "evidence": [e.to_dict() for e in self.evidence],
            "source_author": self.source_author,
            "source_work": self.source_work,
            "status": self.status,
            "schema_version": self.schema_version,
        }


# --------------------------------------------------------------------------- #
# Phase 4.5：作者级策略合并的两层结构
# --------------------------------------------------------------------------- #
# 层次一（raw）：CreativeStrategy + StrategyEvidence 记录"某个 chunk 上 LLM 实际
# 发现了什么"（原始观察，永久保留，绝不覆盖）。RawStrategy 是把某个 strategy_id
# 在**某位作者**名下收拢后的合并输入单元。
# 层次二（canonical）：CanonicalStrategy 记录"作者级 consolidation 后该作者稳定
# 使用的规范化创作策略"；通过 source_strategy_ids → evidence → chunk/work 完全可追溯。
def _safe_confidence(value: Any) -> float | None:
    """把 LLM 返回的 confidence 规范化为 [0,1] 浮点或 None（非数值/越界/布尔 → None）。"""
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    v = float(value)
    return v if 0.0 <= v <= 1.0 else None


def canonical_strategy_id(author_id: str, canonical_name: str) -> str:
    """作者范围内稳定的 canonical id：`author_id::slug(canonical_name)`。

    只从 canonical_name 派生 slug（小写、非字母数字转下划线），绝不依赖 description
    的自由文本 hash，因此描述措辞的微小变化不会改变 identity；两位作者可拥有同名
    canonical strategy 而 id 不冲突（`austen::dramatic_irony` vs `dickens::dramatic_irony`）。
    """
    slug = re.sub(r"[^a-z0-9]+", "_", canonical_name.lower()).strip("_")
    return f"{author_id}::{slug or 'strategy'}"


@dataclass
class RawStrategy:
    """作者级合并的输入单元：某位作者名下的一个 raw 策略观察。

    与 CreativeStrategy（discover 的原始输出）不同：RawStrategy 是**作者归一**后的
    合并输入，把某个 strategy_id 在该作者名下的证据收拢，供 consolidation 使用。
    绝不删除/覆盖原始 CreativeStrategy —— 这里只是引用其字段与作者范围内证据。
    """
    strategy_id: str
    author_id: str
    name: str
    description: str
    triggers: list[str] = field(default_factory=list)
    operations: list[str] = field(default_factory=list)
    intended_effects: list[str] = field(default_factory=list)
    status: str = StrategyStatus.DISCOVERED.value
    confidence: float | None = None
    evidence: list[StrategyEvidence] = field(default_factory=list)
    source_work: str | None = None
    source_strategy_ids: list[str] = field(default_factory=list)  # 精确去重折叠后合并的原始 id
    schema_version: str = STRATEGY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.source_strategy_ids:
            self.source_strategy_ids = [self.strategy_id]

    @classmethod
    def from_creative_strategy(cls, strategy: CreativeStrategy, author_id: str,
                               evidence: list[StrategyEvidence] | None = None) -> "RawStrategy":
        """从 CreativeStrategy 构造作者范围内的 raw 输入（evidence 缺省取策略自身证据）。"""
        return cls(
            strategy_id=strategy.strategy_id, author_id=author_id,
            name=strategy.name, description=strategy.description,
            triggers=list(strategy.triggers), operations=list(strategy.operations),
            intended_effects=list(strategy.intended_effects),
            status=strategy.status, confidence=strategy.confidence,
            evidence=list(evidence if evidence is not None else strategy.evidence),
            source_work=strategy.source_work,
            source_strategy_ids=[strategy.strategy_id],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "author_id": self.author_id,
            "name": self.name,
            "description": self.description,
            "triggers": self.triggers,
            "operations": self.operations,
            "intended_effects": self.intended_effects,
            "status": self.status,
            "confidence": self.confidence,
            "evidence": [e.to_dict() for e in self.evidence],
            "source_work": self.source_work,
            "source_strategy_ids": self.source_strategy_ids,
            "schema_version": self.schema_version,
        }


@dataclass
class ConsolidationGroup:
    """LLM consolidation 返回的单个 canonical 分组（结构化映射，绝不直接改数据）。

    每个分组声明把哪些 source_strategy_ids 合并为一个 canonical strategy；确定性
    校验（见 consolidator）保证覆盖完整、无重复赋值、无幻觉、无丢失。
    """
    canonical_name: str
    canonical_description: str
    source_strategy_ids: list[str] = field(default_factory=list)
    trigger_summary: str = ""
    operation_summary: str = ""
    effect_summary: str = ""
    reasoning_summary: str = ""
    confidence: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConsolidationGroup":
        src = data.get("source_strategy_ids") or []
        if isinstance(src, str):
            src = [src]
        return cls(
            canonical_name=str(data.get("canonical_name", "")).strip(),
            canonical_description=str(data.get("canonical_description", "")).strip(),
            source_strategy_ids=[str(x) for x in src if isinstance(x, str)],
            trigger_summary=str(data.get("trigger_summary", "")),
            operation_summary=str(data.get("operation_summary", "")),
            effect_summary=str(data.get("effect_summary", "")),
            reasoning_summary=str(data.get("reasoning_summary", "")),
            confidence=_safe_confidence(data.get("confidence")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_name": self.canonical_name,
            "canonical_description": self.canonical_description,
            "source_strategy_ids": self.source_strategy_ids,
            "trigger_summary": self.trigger_summary,
            "operation_summary": self.operation_summary,
            "effect_summary": self.effect_summary,
            "reasoning_summary": self.reasoning_summary,
            "confidence": self.confidence,
        }


@dataclass
class CanonicalStrategy:
    """作者级 consolidation 后的规范化创作策略（作者画像中的稳定表示）。

    canonical_strategy_id = "{author_id}::{slug(canonical_name)}"，作者范围内稳定。
    通过 source_strategy_ids → evidence → chunk/work 完全可追溯；不同作者可拥有同名
    canonical strategy 而 id 不冲突。绝不删除/覆盖原始 raw strategies。
    """
    canonical_strategy_id: str
    author_id: str
    canonical_name: str
    canonical_description: str
    trigger_summary: str
    operation_summary: str
    effect_summary: str
    source_strategy_ids: list[str]
    supporting_chunk_ids: list[str]
    supporting_work_ids: list[str]
    reasoning_summary: str
    confidence: float | None
    number_of_raw_observations: int
    number_of_distinct_chunks: int
    number_of_distinct_works: int
    support_status: str
    evidence: list[StrategyEvidence] = field(default_factory=list)
    schema_version: str = CANONICAL_STRATEGY_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_strategy_id": self.canonical_strategy_id,
            "author_id": self.author_id,
            "canonical_name": self.canonical_name,
            "canonical_description": self.canonical_description,
            "trigger_summary": self.trigger_summary,
            "operation_summary": self.operation_summary,
            "effect_summary": self.effect_summary,
            "source_strategy_ids": self.source_strategy_ids,
            "supporting_chunk_ids": self.supporting_chunk_ids,
            "supporting_work_ids": self.supporting_work_ids,
            "reasoning_summary": self.reasoning_summary,
            "confidence": self.confidence,
            "number_of_raw_observations": self.number_of_raw_observations,
            "number_of_distinct_chunks": self.number_of_distinct_chunks,
            "number_of_distinct_works": self.number_of_distinct_works,
            "support_status": self.support_status,
            "evidence": [e.to_dict() for e in self.evidence],
            "schema_version": self.schema_version,
        }
