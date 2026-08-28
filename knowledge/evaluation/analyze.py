# knowledge/evaluation/analyze.py
"""Phase 8 再测量：把生成正文重新送入测量管线 → ActualStyleProfile。

复用既有 analyzer（绝不另写第二套测量实现）：
    - Layer A 统计（22，确定性）：StatisticalAnalyzer.analyze_many
    - Layer A 判断/hybrid（8，LLM，盲测）：LLMFeatureAnalyzer
    - Layer B 叙事（LLM，盲测）：NarrativeAnalyzer
    - Layer C 策略（LLM，盲测）：StrategyMiner.match（用作者 canonical 策略构造注册表，
      使 match 到的 strategy_id 与 StylePlan.strategy_controls 的 canonical id 对齐）
    - Layer D stylometric（确定性）：StylometricVectorizer 在 TRAIN chunk 上**重拟合**，
      与持久化 `stylometry/index.json` 的 feature_names 严格比对（fail-closed），
      再把正文 transform 后对作者质心（`stylometric_author_targets.json`）算余弦距离。

铁律：
    - 全部盲测（不注入作者身份）；无 provider 的 LLM 步骤返回 AnalysisUnavailable，
      记录进 `unavailable`，绝不伪造测量值；
    - Layer D 只作诊断，产出余弦距离，绝不生成改写指令；
    - 密钥只读；provider 由调用方注入（测试用 DummyLLMProvider）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..analysis.narrative_analyzer import NarrativeAnalyzer
from ..analysis.statistical_analyzer import StatisticalAnalyzer
from ..analysis.strategy_miner import StrategyMiner
from ..analysis.style_analyzer import LLMFeatureAnalyzer
from ..analysis.base import AnalysisUnavailable
from ..config import data_layout, data_root as default_data_root
from ..generation.schema import output_hash
from ..profiles.style_profile import AuthorStyleProfile
from ..providers.llm_provider import LLMProvider
from ..schema.feature_registry import build_default_registry
from ..schema.strategy_schema import CreativeStrategy
from ..schema.versions import EVALUATION_SCHEMA_VERSION
from ..strategies.registry import StrategyRegistry
from ..stylometry.delta import cosine_distance
from ..stylometry.extract import StylometricVectorizer
from .schema import ActualStyleProfile, EvalError

_LAYER_A_STAT = "layer_a_statistical"
_LAYER_A_JUDGMENT = "layer_a_judgment"
_LAYER_B = "layer_b_narrative"
_LAYER_C = "layer_c_strategies"
_LAYER_D = "layer_d_stylometric"

_LLM_ANALYZER = "LlmFeatureAnalyzer"


def _author_strategy_registry(profile: AuthorStyleProfile) -> StrategyRegistry:
    """用作者 canonical 策略构造匹配注册表（strategy_id = canonical_strategy_id）。

    这样 StrategyMiner.match 命中项的 strategy_id 与 StylePlan.strategy_controls 的
    canonical id 直接对齐，compare 阶段可对 coverage 做精确比对。
    """
    reg = StrategyRegistry()
    for sc in profile.strategy_controls:
        reg.register(CreativeStrategy(
            strategy_id=sc.canonical_strategy_id,
            name=sc.canonical_name,
            description=sc.canonical_description,
            triggers=[sc.trigger_summary],
            operations=[sc.operation_summary],
            intended_effects=[sc.effect_summary],
            status=sc.support_status,
        ))
    return reg


def _layer_d_diagnostic(text: str, author_id: str,
                        base: Path) -> dict[str, Any]:
    """Layer D：在 TRAIN chunk 上重拟合向量器，算正文到作者质心的余弦距离。

    fail-closed：重拟合的 feature_names 必须与持久化 `stylometry/index.json` 完全
    一致，否则抛 EvalError（绝不产生可能错位对齐的距离值）。
    """
    stylo_dir = base / "analysis" / "stylometry"
    index_path = stylo_dir / "index.json"
    targets_path = base / "analysis" / "style_profiles" / "stylometric_author_targets.json"
    if not index_path.exists():
        raise EvalError(f"缺 stylometry index: {index_path}")
    if not targets_path.exists():
        raise EvalError(f"缺 stylometric author targets: {targets_path}")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    targets = json.loads(targets_path.read_text(encoding="utf-8"))
    author_target = (targets.get("authors") or {}).get(author_id)
    if not author_target or "centroid" not in author_target:
        raise EvalError(f"{author_id}: stylometric author target 缺 centroid")

    layout = data_layout(base)
    train_work_ids = sorted(set(index.get("train_work_ids", [])))
    texts: list[str] = []
    for work_id in train_work_ids:
        chunk_path = layout["chunks"] / f"{work_id}__2000.jsonl"
        if not chunk_path.exists():
            raise EvalError(f"缺 TRAIN chunk 文件（重拟合 Layer D 需要）: {chunk_path}")
        for line in chunk_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                texts.append(json.loads(line)["text"])

    vec = StylometricVectorizer().fit(texts)
    if list(vec.feature_names_) != list(index.get("feature_names", [])):
        raise EvalError(
            f"{author_id}: Layer D 重拟合 feature_names 与持久化 index 不一致，"
            f"拒绝产生错位对齐的距离（fail-closed）")

    centroid = np.asarray(author_target["centroid"], dtype=float)
    vector = np.asarray(vec.transform([text]), dtype=float)[0]
    dist = cosine_distance(vector, centroid)
    return {
        "cosine_distance": round(dist, 8),
        "n_features": int(len(vec.feature_names_)),
        "feature_names_match": True,
        "train_work_ids": train_work_ids,
        "n_train_chunks": len(texts),
        "note": ("stylometric 指纹仅作生成后相似度诊断，绝不生成改写指令"),
    }


def measure_actual_profile(
    text: str,
    *,
    author_id: str,
    passage_id: str,
    style_plan_id: str,
    profile: AuthorStyleProfile,
    provider: LLMProvider,
    data_root_: Path | None = None,
) -> ActualStyleProfile:
    """对生成正文跑五步再测量，返回 ActualStyleProfile。provider 由调用方注入。"""
    base = Path(data_root_) if data_root_ is not None else default_data_root()
    registry = build_default_registry()
    stat_features = [f for f in registry.all() if f.analyzer == "StatisticalAnalyzer"]
    llm_features = [f for f in registry.all() if f.analyzer == _LLM_ANALYZER]

    unavailable: dict[str, list[str]] = {}
    stat_out: dict[str, dict[str, Any]] = {}
    judgment_out: dict[str, dict[str, Any]] = {}
    narrative_out: dict[str, Any] | None = None
    strategies_out: list[dict[str, Any]] = []

    # ---- Layer A 统计（确定性） ----
    for fv in StatisticalAnalyzer().analyze_many(text, stat_features):
        stat_out[fv.feature_id] = fv.to_dict()

    # ---- Layer A 判断（LLM，盲测） ----
    feature_analyzer = LLMFeatureAnalyzer(provider, blind=True)
    for feat in llm_features:
        res = feature_analyzer.analyze(text, feat, chunk_id=passage_id)
        if isinstance(res, AnalysisUnavailable):
            unavailable.setdefault(_LAYER_A_JUDGMENT, []).append(feat.id)
            continue
        judgment_out[feat.id] = res.to_dict()

    # ---- Layer B 叙事（LLM，盲测） ----
    narrative_analyzer = NarrativeAnalyzer(provider, blind=True)
    res = narrative_analyzer.analyze(text, chunk_id=passage_id)
    if isinstance(res, AnalysisUnavailable):
        unavailable.setdefault(_LAYER_B, []).append("narrative")
    else:
        narrative_out = res.to_dict()

    # ---- Layer C 策略（LLM，盲测；作者 canonical 注册表） ----
    miner = StrategyMiner(provider, _author_strategy_registry(profile), blind=True)
    res = miner.match(text, chunk_id=passage_id, work_id="", author_id=author_id)
    if isinstance(res, AnalysisUnavailable):
        unavailable.setdefault(_LAYER_C, []).append("strategy")
    else:
        strategies_out = [
            {"strategy_id": sid, "evidence": ev.to_dict()} for sid, ev in res
        ]

    # ---- Layer D stylometric（确定性诊断） ----
    stylometric = _layer_d_diagnostic(text, author_id, base)

    return ActualStyleProfile(
        schema_version=EVALUATION_SCHEMA_VERSION,
        author_id=author_id,
        passage_id=passage_id,
        passage_hash=output_hash(text),
        style_plan_id=style_plan_id,
        layer_a_statistical=stat_out,
        layer_a_judgment=judgment_out,
        layer_b_narrative=narrative_out,
        layer_c_strategies=strategies_out,
        layer_d_stylometric=stylometric,
        unavailable=unavailable,
        provenance={
            "layer_a_statistical_analyzer": "StatisticalAnalyzer",
            "layer_a_judgment_analyzer": _LLM_ANALYZER,
            "layer_b_analyzer": "NarrativeAnalyzer",
            "layer_c_analyzer": "StrategyMiner",
            "layer_d_analyzer": "StylometricExtractor",
            "blind": True,
        },
    )
