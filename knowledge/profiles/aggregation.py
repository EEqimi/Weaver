# knowledge/profiles/aggregation.py
"""Phase 4 类型感知聚合（spec §10）：把逐 chunk 的测量聚合成工作/作者画像。

原则（spec §10 / §11）：
    - 连续量绝不只存一个均值：同时保留 variance 与 min/max/quartiles 分布；
    - 类别量存类别分布（proportions），绝不硬编码成"0.73 第三人称"；
    - 每层都记录 sample_count，避免"单数据点看似确凿"的误导；
    - narrative 的枚举字段按类别分布聚合，比例字段（pace/detail）逐类取均值。
"""
from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from ..schema.narrative_schema import (
    DETAIL_DIMENSIONS, PACE_DIMENSIONS, NarrativeObservation,
)
from ..schema.style_schema import FeatureValue
from ..schema.strategy_schema import StrategyEvidence
from ..schema.versions import AGGREGATION_VERSION, NARRATIVE_SCHEMA_VERSION

# narrative 中按"类别分布"聚合的枚举字段
_NARRATIVE_ENUM_FIELDS = (
    "pov", "focalization", "perspective_stability", "narrative_distance",
    "narrator_presence", "narrator_evaluative_intervention",
    "information_access", "temporal_order",
)
# narrative 中按"逐类均值"聚合的比例字段
_NARRATIVE_DIST_FIELDS = {
    "temporal_pace": PACE_DIMENSIONS,
    "scene_detail": DETAIL_DIMENSIONS,
}


# --------------------------------------------------------------------------- #
# 基础聚合函数
# --------------------------------------------------------------------------- #
def _mean_var(xs: list[float]) -> tuple[float, float, float]:
    """返回 (mean, variance, std)；总体方差（/n），与 analyzer 一致。"""
    mean = sum(xs) / len(xs)
    var = sum((x - mean) ** 2 for x in xs) / len(xs)
    std = var ** 0.5
    return mean, var, std


def _quartiles(xs: list[float]) -> list[float]:
    """返回 [min, q1, median, q3, max]（线性插值近似）。"""
    s = sorted(xs)
    n = len(s)
    q = lambda p: s[min(n - 1, max(0, round(p * (n - 1))))]
    return [s[0], q(0.25), q(0.5), q(0.75), s[-1]]


def aggregate_continuous(values: list[float]) -> dict[str, Any]:
    """连续量：mean / variance / std / 分布，绝不只存均值。"""
    if not values:
        return {"n": 0, "mean": None}
    mean, var, std = _mean_var(values)
    return {
        "n": len(values),
        "mean": mean,
        "variance": var,
        "std": std,
        "min": min(values),
        "q1": _quartiles(values)[1],
        "median": statistics.median(values),
        "q3": _quartiles(values)[3],
        "max": max(values),
    }


def aggregate_categorical(values: list[str]) -> dict[str, Any]:
    """类别量：计数 + 占比（类别按名称稳定排序）。"""
    if not values:
        return {"n": 0, "counts": {}, "proportions": {}}
    counts = Counter(values)
    n = len(values)
    return {
        "n": n,
        "counts": {k: counts[k] for k in sorted(counts)},
        "proportions": {k: counts[k] / n for k in sorted(counts)},
        "mode": max(counts, key=counts.get),
    }


def aggregate_distribution(dists: list[dict[str, float]]) -> dict[str, Any]:
    """分布量：逐类均值（每 chunk 已是 category → proportion 的分布）。"""
    if not dists:
        return {"n": 0, "mean_distribution": {}}
    keys = set()
    for d in dists:
        keys.update(d.keys())
    mean_dist = {k: sum(d.get(k, 0.0) for d in dists) / len(dists) for k in sorted(keys)}
    return {"n": len(dists), "mean_distribution": mean_dist}


def _numeric_values(fvs: list[FeatureValue]) -> list[float]:
    out: list[float] = []
    for fv in fvs:
        v = fv.value
        if isinstance(v, bool) or v is None:
            continue
        if isinstance(v, (int, float)):
            out.append(float(v))
    return out


def aggregate_feature_values(fvs: list[FeatureValue]) -> dict[str, Any]:
    """按 value_type 分派聚合，并保留不确定性与证据溯源（task item 4）。

    保留内容：
        - value summary/distribution（连续量绝不只存均值）；
        - n_total / n_valid / n_missing（显式区分缺失，绝不静默丢弃）；
        - confidence 独立汇总（**绝不并入 value**）；
        - evidence refs、analyzer id/version、schema version、provenance（溯源）。
    空列表返回 {n: 0}。
    """
    if not fvs:
        return {"n": 0}
    value_type = fvs[0].value_type
    n_total = len(fvs)
    valid = [fv for fv in fvs if fv.value is not None]
    n_valid = len(valid)
    n_missing = n_total - n_valid

    if value_type == "categorical":
        summary = aggregate_categorical([str(fv.value) for fv in valid])
    elif value_type == "distribution":
        summary = aggregate_distribution([fv.value for fv in valid if isinstance(fv.value, dict)])
    else:
        # continuous / discrete 走数值聚合
        summary = aggregate_continuous(_numeric_values(valid))
    summary["n"] = n_valid  # 对齐到有效样本数（n_total 另存，不静默丢弃缺失）

    # confidence 独立汇总（绝不并入 value）
    confs = [float(fv.confidence) for fv in fvs
             if isinstance(fv.confidence, (int, float)) and not isinstance(fv.confidence, bool)]
    confidence = aggregate_continuous(confs) if confs else {"n": 0, "mean": None}

    return {
        **summary,
        "n_total": n_total,
        "n_valid": n_valid,
        "n_missing": n_missing,
        "value_type": value_type,
        "measurement_type": fvs[0].measurement_type,
        "confidence": confidence,
        "evidence_refs": [fv.evidence for fv in fvs if fv.evidence],
        "analyzer_ids": sorted({fv.analyzer_id for fv in fvs if fv.analyzer_id}),
        "analyzer_versions": sorted({fv.analyzer_version for fv in fvs if fv.analyzer_version}),
        "schema_versions": sorted({fv.schema_version for fv in fvs if fv.schema_version}),
        "provenance": [fv.provenance for fv in fvs if fv.provenance],
    }


# --------------------------------------------------------------------------- #
# 画像结构
# --------------------------------------------------------------------------- #
@dataclass
class ChunkProfile:
    chunk_id: str
    work_id: str
    author_id: str
    feature_values: dict[str, FeatureValue] = field(default_factory=dict)
    narrative: NarrativeObservation | None = None
    strategy_evidence: list[StrategyEvidence] = field(default_factory=list)
    schema_version: str = AGGREGATION_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "work_id": self.work_id,
            "author_id": self.author_id,
            "features": {k: v.to_dict() for k, v in self.feature_values.items()},
            "narrative": self.narrative.to_dict() if self.narrative else None,
            "strategy_evidence": [e.to_dict() for e in self.strategy_evidence],
            "schema_version": self.schema_version,
        }


@dataclass
class WorkProfile:
    work_id: str
    author_id: str
    chunk_count: int
    features: dict[str, dict[str, Any]] = field(default_factory=dict)
    narrative: dict[str, Any] = field(default_factory=dict)
    strategies: dict[str, int] = field(default_factory=dict)  # strategy_id -> 证据 chunk 数
    schema_version: str = AGGREGATION_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "work_id": self.work_id,
            "author_id": self.author_id,
            "chunk_count": self.chunk_count,
            "features": self.features,
            "narrative": self.narrative,
            "strategies": self.strategies,
            "schema_version": self.schema_version,
        }


@dataclass
class AuthorProfile:
    author_id: str
    work_ids: list[str]
    chunk_count: int
    features: dict[str, dict[str, Any]] = field(default_factory=dict)
    narrative: dict[str, Any] = field(default_factory=dict)
    strategies: dict[str, str] = field(default_factory=dict)  # strategy_id -> status
    schema_version: str = AGGREGATION_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "author_id": self.author_id,
            "work_ids": self.work_ids,
            "chunk_count": self.chunk_count,
            "features": self.features,
            "narrative": self.narrative,
            "strategies": self.strategies,
            "schema_version": self.schema_version,
        }


# --------------------------------------------------------------------------- #
# narrative 聚合
# --------------------------------------------------------------------------- #
def aggregate_narrative(obs: list[NarrativeObservation]) -> dict[str, Any]:
    """聚合叙事观察：字段级摘要 + 不确定性/证据溯源（task item 4/6）。

    - 枚举 → 类别分布（含显式 unknown/insufficient_evidence/not_observable，
      绝不把缺失字段伪装成已观察事实）；
    - 比例 → 逐类均值；
    - 额外保留逐 chunk 的证据引用、未验证引文、比例问题、confidence 汇总与溯源。
    """
    if not obs:
        return {"n": 0}
    n_total = len(obs)
    out: dict[str, Any] = {
        "n": n_total, "n_total": n_total, "n_valid": n_total, "n_missing": 0,
    }
    for f in _NARRATIVE_ENUM_FIELDS:
        out[f] = aggregate_categorical([getattr(o, f) for o in obs])
    for f, dims in _NARRATIVE_DIST_FIELDS.items():
        dists = [{k: v for k, v in getattr(o, f).items() if k in dims} for o in obs]
        out[f] = aggregate_distribution([d for d in dists if d])

    # 证据与溯源：保留逐 chunk 引用，绝不静默丢弃（task item 4）
    confs = [float(o.confidence) for o in obs
             if isinstance(o.confidence, (int, float)) and not isinstance(o.confidence, bool)]
    out["confidence"] = aggregate_continuous(confs) if confs else {"n": 0, "mean": None}
    out["observed_evidence"] = {o.chunk_id: list(o.observed_evidence) for o in obs
                                if o.observed_evidence}
    out["unverified_evidence"] = {o.chunk_id: list(o.unverified_evidence) for o in obs
                                  if o.unverified_evidence}
    out["proportion_issues"] = {o.chunk_id: list(o.proportion_issues) for o in obs
                                if o.proportion_issues}
    out["schema_versions"] = sorted({o.schema_version for o in obs if o.schema_version})
    out["chunk_provenance"] = [o.chunk_id for o in obs]
    return out


# --------------------------------------------------------------------------- #
# 聚合器
# --------------------------------------------------------------------------- #
class Aggregator:
    """把 chunk 画像聚合成 work / author 画像（类型感知）。"""

    def __init__(self) -> None:
        pass

    def aggregate_work(self, profiles: list[ChunkProfile]) -> WorkProfile:
        if not profiles:
            raise ValueError("空 chunk 画像无法聚合 work profile")
        work_id = profiles[0].work_id
        author_id = profiles[0].author_id
        return WorkProfile(
            work_id=work_id,
            author_id=author_id,
            chunk_count=len(profiles),
            features=self._aggregate_features(profiles),
            narrative=aggregate_narrative([p.narrative for p in profiles if p.narrative]),
            strategies=self._count_strategy_evidence(profiles),
        )

    def aggregate_author(self, profiles: list[ChunkProfile]) -> AuthorProfile:
        if not profiles:
            raise ValueError("空 chunk 画像无法聚合 author profile")
        author_id = profiles[0].author_id
        works = sorted({p.work_id for p in profiles})
        return AuthorProfile(
            author_id=author_id,
            work_ids=works,
            chunk_count=len(profiles),
            features=self._aggregate_features(profiles),
            narrative=aggregate_narrative([p.narrative for p in profiles if p.narrative]),
            strategies={},  # 状态由 registry 生命周期决定，聚合器不越权定级
        )

    @staticmethod
    def _aggregate_features(profiles: list[ChunkProfile]) -> dict[str, dict[str, Any]]:
        by_feature: dict[str, list[FeatureValue]] = {}
        for p in profiles:
            for fid, fv in p.feature_values.items():
                by_feature.setdefault(fid, []).append(fv)
        return {fid: aggregate_feature_values(fvs) for fid, fvs in by_feature.items()}

    @staticmethod
    def _count_strategy_evidence(profiles: list[ChunkProfile]) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for p in profiles:
            for e in p.strategy_evidence:
                counts[e.strategy_id] += 1
        return dict(sorted(counts.items()))
