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

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .versions import STRATEGY_MINER_VERSION, STRATEGY_SCHEMA_VERSION


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
