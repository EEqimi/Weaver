# knowledge/schema/style_schema.py
"""Feature 测量结果的数据结构（spec §4 概念结构）。

LLM/混合/判断型特征绝不只存裸 0-1 分数：必须附带 confidence、evidence 引用与
analyzer 版本（MUST）。raw_value 与 normalized_value 尽量同时保留。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .versions import FEATURE_SCHEMA_VERSION


@dataclass
class FeatureValue:
    feature_id: str
    value: Any                      # 汇总/展示值
    raw_value: Any                  # 原始测量值
    normalized_value: Any | None = None
    value_type: str = "continuous"
    measurement_type: str = "statistical"
    confidence: float | None = None
    evidence: list[Any] = field(default_factory=list)
    sample_count: int = 1
    variance: float | None = None
    analyzer_id: str = ""
    analyzer_version: str = ""       # 与 analyzer_id / schema_version 分离的 analyzer 版本
    schema_version: str = FEATURE_SCHEMA_VERSION
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "value": self.value,
            "raw_value": self.raw_value,
            "normalized_value": self.normalized_value,
            "value_type": self.value_type,
            "measurement_type": self.measurement_type,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "sample_count": self.sample_count,
            "variance": self.variance,
            "analyzer_id": self.analyzer_id,
            "analyzer_version": self.analyzer_version,
            "schema_version": self.schema_version,
            "provenance": self.provenance,
        }
