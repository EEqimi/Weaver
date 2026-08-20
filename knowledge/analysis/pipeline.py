# knowledge/analysis/pipeline.py
"""Phase 3+4 编排：确定性分析 + Layer D 基线 + 采样清单（**不调用 LLM**）。

执行范围（spec §14 / §13 的"现在做"部分）：
    1. Layer A 确定性统计特征：对 TRAIN 作品全部 chunk 运行；
    2. Layer D 文体学：向量器仅在 TRAIN 上 fit，held-out 只做 transform 与
       留出验证（held-out 绝不进入训练，杜绝泄漏）；
    3. 聚合 ChunkProfile → WorkProfile → AuthorProfile（仅 TRAIN）；
    4. 生成分层标定采样清单（仅 TRAIN，8–12 chunk/作品）。

明确不做（spec §13）：
    - 不自动触发 LLM（Layer A 判断/hybrid、Layer B、Layer C 由采样清单另行标定）；
    - 不用 held-out 调 schema；不产出作者级结论。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import data_layout, data_root as default_data_root
from ..corpus.metadata import CORPUS, held_out_works, train_works
from ..profiles.aggregation import Aggregator, ChunkProfile
from ..sampling.calibration import build_calibration_manifest
from ..schema.feature_registry import FeatureDefinition, build_default_registry
from ..schema.versions import (
    AGGREGATION_VERSION, SAMPLING_VERSION, SCHEMA_VERSION, STYLOMETRY_VERSION,
)
from .statistical_analyzer import StatisticalAnalyzer
from ..stylometry.validation import evaluate_heldout, grouped_cross_validation_texts
from ..stylometry.extract import StylometricVectorizer

DEFAULT_TARGET_CHARS = 2000
ANALYSIS_DIRNAME = "analysis"


def analysis_layout(data_root_: Path | None = None) -> dict[str, Path]:
    base = (Path(data_root_) if data_root_ is not None else default_data_root())
    a = base / ANALYSIS_DIRNAME
    return {
        "root": a,
        "stylometry": a / "stylometry",
        "profiles": a / "profiles",
    }


# --------------------------------------------------------------------------- #
# 数据加载
# --------------------------------------------------------------------------- #
def load_work_chunks(work_id: str, layout: dict[str, Path],
                     target_chars: int = DEFAULT_TARGET_CHARS) -> list[dict]:
    path = layout["chunks"] / f"{work_id}__{target_chars}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"缺少 chunk 文件 {path}（先运行 corpus.pipeline.build_corpus）")
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


# --------------------------------------------------------------------------- #
# Layer A 确定性
# --------------------------------------------------------------------------- #
def run_layer_a(chunks: list[dict], work_id: str, author_id: str,
                features: list[FeatureDefinition],
                analyzer: StatisticalAnalyzer) -> list[ChunkProfile]:
    """对单个作品的全部 chunk 运行确定性统计特征，产出 ChunkProfile 列表。"""
    profiles: list[ChunkProfile] = []
    for c in chunks:
        fvs = {fv.feature_id: fv for fv in analyzer.analyze_many(c["text"], features)}
        profiles.append(ChunkProfile(
            chunk_id=c["chunk_id"], work_id=work_id, author_id=author_id,
            feature_values=fvs,
        ))
    return profiles


# --------------------------------------------------------------------------- #
# Layer D 文体学（fit 仅 TRAIN）
# --------------------------------------------------------------------------- #
def run_layer_d(train_chunks: dict[str, list[dict]],
                heldout_chunks: dict[str, list[dict]],
                layout: dict[str, Path]) -> dict:
    """提取文体学特征矩阵并跑基线（分组 CV + 留出验证）。"""
    def _records(chunks_by_work: dict[str, list[dict]], work_meta):
        texts, ids, works, authors = [], [], [], []
        for work_id in sorted(chunks_by_work):
            meta = work_meta[work_id]
            for c in chunks_by_work[work_id]:
                texts.append(c["text"])
                ids.append(c["chunk_id"])
                works.append(work_id)
                authors.append(meta.author_id)
        return texts, ids, works, authors

    meta_by_work = {m.work_id: m for m in CORPUS}
    train_meta = {w: meta_by_work[w] for w in train_chunks}
    held_meta = {w: meta_by_work[w] for w in heldout_chunks}

    tr_texts, tr_ids, tr_works, tr_authors = _records(train_chunks, train_meta)
    ho_texts, ho_ids, ho_works, ho_authors = _records(heldout_chunks, held_meta)

    vec = StylometricVectorizer()
    X_train = vec.fit_transform(tr_texts)
    X_heldout = vec.transform(ho_texts) if ho_texts else np.zeros((0, X_train.shape[1]))

    stylo_dir = layout["stylometry"]
    stylo_dir.mkdir(parents=True, exist_ok=True)
    np.savez(str(stylo_dir / "matrix.npz"),
             X_train=X_train, X_heldout=X_heldout)
    (stylo_dir / "index.json").write_text(json.dumps({
        "feature_names": vec.feature_names_,
        "train_chunk_ids": tr_ids, "train_work_ids": tr_works, "train_author_ids": tr_authors,
        "heldout_chunk_ids": ho_ids, "heldout_work_ids": ho_works,
        "heldout_author_ids": ho_authors,
        "stylometry_version": STYLOMETRY_VERSION,
    }, ensure_ascii=False), encoding="utf-8")

    baseline = {
        "n_train_chunks": len(tr_texts),
        "n_heldout_chunks": len(ho_texts),
        "n_features": int(X_train.shape[1]),
        # 泄漏安全：每折重拟合向量器（左出作品不参与词汇选择），见 validation.py
        "grouped_cv_accuracy": grouped_cross_validation_texts(
            tr_texts, tr_authors, tr_works, classifier="svm"),
        "grouped_cv_leak_free": True,
        "stylometric_family_overlap": vec.family_overlap(),
        "heldout_eval": {},
    }
    if ho_texts:
        # 拼接 TRAIN + HELD-OUT 做留出评估（训练仅用 TRAIN）
        acc, preds, truths = evaluate_heldout(
            np.vstack([X_train, X_heldout]), tr_authors + ho_authors,
            tr_works + ho_works, held_out_works=[w for w in heldout_chunks],
            classifier="svm")
        baseline["heldout_eval"] = {"accuracy": acc, "predicted": preds, "truth": truths}

    (stylo_dir / "baseline.json").write_text(
        json.dumps(baseline, ensure_ascii=False, indent=2), encoding="utf-8")
    return baseline


# --------------------------------------------------------------------------- #
# 编排
# --------------------------------------------------------------------------- #
def run_deterministic_pipeline(data_root_: Path | None = None,
                               target_chars: int = DEFAULT_TARGET_CHARS) -> dict:
    """确定性分析（Layer A + Layer D）+ 采样清单；不调用 LLM。"""
    layout = data_layout(data_root_)
    alayout = analysis_layout(data_root_)
    for p in alayout.values():
        p.mkdir(parents=True, exist_ok=True)

    registry = build_default_registry()
    analyzer = StatisticalAnalyzer()
    stat_features = [f for f in registry.all() if f.analyzer == "StatisticalAnalyzer"]

    train = train_works()
    held = held_out_works()

    # Layer A + 画像
    train_chunks: dict[str, list[dict]] = {}
    all_profiles: list[ChunkProfile] = []
    for m in train:
        chunks = load_work_chunks(m.work_id, layout, target_chars)
        train_chunks[m.work_id] = chunks
        all_profiles.extend(run_layer_a(chunks, m.work_id, m.author_id,
                                        stat_features, analyzer))

    # Layer D
    held_chunks: dict[str, list[dict]] = {
        m.work_id: load_work_chunks(m.work_id, layout, target_chars) for m in held
    }
    baseline = run_layer_d(train_chunks, held_chunks, alayout)

    # 聚合
    agg = Aggregator()
    by_author: dict[str, list[ChunkProfile]] = {}
    by_work: dict[str, list[ChunkProfile]] = {}
    for p in all_profiles:
        by_author.setdefault(p.author_id, []).append(p)
        by_work.setdefault(p.work_id, []).append(p)

    work_profiles = {wid: agg.aggregate_work(by_work[wid]) for wid in sorted(by_work)}
    author_profiles = {aid: agg.aggregate_author(by_author[aid]) for aid in sorted(by_author)}

    # 写画像
    prof_dir = alayout["profiles"]
    with (prof_dir / "chunk_profiles.jsonl").open("w", encoding="utf-8") as fh:
        for p in all_profiles:
            fh.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")
    (prof_dir / "work_profiles.json").write_text(json.dumps(
        {k: v.to_dict() for k, v in work_profiles.items()},
        ensure_ascii=False, indent=2), encoding="utf-8")
    (prof_dir / "author_profiles.json").write_text(json.dumps(
        {k: v.to_dict() for k, v in author_profiles.items()},
        ensure_ascii=False, indent=2), encoding="utf-8")

    # 采样清单
    sample = build_calibration_manifest(
        train_chunks,
        target_per_work=10,
        author_by_work={m.work_id: m.author_id for m in CORPUS},
    )
    (alayout["root"] / "calibration_sample.json").write_text(
        json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "schema_version": SCHEMA_VERSION,
        "aggregation_version": AGGREGATION_VERSION,
        "sampling_version": SAMPLING_VERSION,
        "layer_a": {
            "n_chunks": len(all_profiles),
            "n_features_per_chunk": len(stat_features),
        },
        "layer_d": baseline,
        "profiles": {
            "n_work_profiles": len(work_profiles),
            "n_author_profiles": len(author_profiles),
        },
        "calibration_sample": {
            "n_works": sample["totals"]["n_works"],
            "n_chunks": sample["totals"]["n_chunks"],
        },
        "outputs": {k: str(v) for k, v in alayout.items()},
    }
