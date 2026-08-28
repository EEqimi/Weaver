# knowledge/planning/bands.py
"""Phase 6.1 经验 band 阈值：用 TRAIN-only chunk 分布的分位数替代人工绝对阈值。

原则（spec Phase 6.1 §1/§2）：
    - 阈值只从 TRAIN chunk-level 分布派生；held-out 作品绝不参与；
    - 阈值持久化 + 版本化（`band_thresholds.json`），planning 读取同一份，跨运行一致；
    - 同一输入 → 同一阈值 → 同一 guidance；
    - guidance 只字面描述"测得什么"，绝不自造未测量的文学机制（如"插入语/对仗结构"）；
    - 缺乏经验 band 的特征 → `describe_feature` 返回 None → planner 标 reference（不编造标签）。

分位数政策（文档化）：`low < Q33`、`medium ∈ [Q33, Q67]`、`high > Q67`；
分位数用线性插值（numpy "linear" 方法，确定性）。
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..config import data_root as default_data_root
from ..schema.versions import BAND_SCHEMA_VERSION

LOW_Q = 1.0 / 3.0
HIGH_Q = 2.0 / 3.0
_ROUND = 8

BAND_THRESHOLDS_FILENAME = "band_thresholds.json"
CHUNK_PROFILES_RELPATH = "analysis/profiles/chunk_profiles.jsonl"


# --------------------------------------------------------------------------- #
# 分位数（线性插值，确定性）
# --------------------------------------------------------------------------- #
def _percentile(sorted_vals: list[float], p: float) -> float:
    """线性插值分位数（等价 numpy `linear` 方法）。输入必须已排序。"""
    n = len(sorted_vals)
    if n == 0:
        raise ValueError("空序列无法求分位数")
    if n == 1:
        return sorted_vals[0]
    rank = p * (n - 1)
    lo = int(rank)
    hi = min(lo + 1, n - 1)
    frac = rank - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


# --------------------------------------------------------------------------- #
# 经验阈值计算（纯函数）
# --------------------------------------------------------------------------- #
def compute_band_thresholds(
    chunks: list[dict[str, Any]],
    train_work_ids: list[str] | set[str] | None = None,
) -> dict[str, Any]:
    """从 TRAIN chunk 画像派生每个特征的经验 band 阈值。

    chunks：逐 chunk 画像 dict，含 `work_id` 与 `features`（feature_id → FeatureValue dict，
    其 `value` 为数值）。`train_work_ids` 提供时，非 TRAIN 的 chunk 被显式排除（held-out 隔离
    在签名层保证，绝不偷偷吸入）。

    返回可持久化的阈值 artifact dict（含 `features` 映射；n==0 的特征标 not_compilable）。
    """
    train = set(train_work_ids) if train_work_ids is not None else None
    values_by_feature: dict[str, list[float]] = defaultdict(list)
    used_work_ids: set[str] = set()
    n_chunks = 0

    for c in chunks:
        wid = c.get("work_id")
        if train is not None and wid not in train:
            continue
        used_work_ids.add(wid)
        n_chunks += 1
        for fid, fv in (c.get("features") or {}).items():
            if not isinstance(fv, dict):
                continue
            v = fv.get("value")
            if isinstance(v, bool) or v is None:
                continue
            if isinstance(v, (int, float)):
                values_by_feature[fid].append(float(v))

    features: dict[str, Any] = {}
    for fid in sorted(values_by_feature):
        vals = sorted(values_by_feature[fid])
        if not vals:
            features[fid] = {"n": 0, "not_compilable": True}
            continue
        features[fid] = {
            "q33": round(_percentile(vals, LOW_Q), _ROUND),
            "q67": round(_percentile(vals, HIGH_Q), _ROUND),
            "n": len(vals),
            "min": round(vals[0], _ROUND),
            "median": round(_percentile(vals, 0.5), _ROUND),
            "max": round(vals[-1], _ROUND),
        }

    return {
        "stage": "band_thresholds",
        "schema_version": BAND_SCHEMA_VERSION,
        "quantile_policy": {
            "low_below": round(LOW_Q, 6),
            "high_above": round(HIGH_Q, 6),
            "method": "linear_interpolation",
            "note": "low < Q33, medium ∈ [Q33, Q67], high > Q67",
        },
        "derived_from": f"data/{CHUNK_PROFILES_RELPATH}",
        "train_only": True,
        "n_train_chunks": n_chunks,
        "train_work_ids": sorted(used_work_ids),
        "features": features,
    }


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def load_band_thresholds(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def band_thresholds_from_chunk_file(chunk_path: Path,
                                    train_work_ids: list[str] | None = None) -> dict[str, Any]:
    chunks = [json.loads(line) for line in chunk_path.read_text(encoding="utf-8").splitlines()
              if line.strip()]
    return compute_band_thresholds(chunks, train_work_ids=train_work_ids)


def build_band_thresholds_artifact(data_root_: Path | None = None,
                                   train_work_ids: list[str] | None = None) -> dict[str, Any]:
    """读 TRAIN chunk 画像 → 计算经验阈值 → 落盘 `band_thresholds.json` → 返回阈值 dict。

    确定性：同一 TRAIN chunk 数据恒得同一阈值。`train_work_ids` 提供时，非 TRAIN 的
    chunk 被显式排除（held-out 隔离 fail-closed，绝不偷偷吸入）。
    """
    base = Path(data_root_) if data_root_ is not None else default_data_root()
    chunk_path = base / CHUNK_PROFILES_RELPATH
    thresholds = band_thresholds_from_chunk_file(chunk_path, train_work_ids=train_work_ids)
    out_dir = base / "analysis" / "planning"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / BAND_THRESHOLDS_FILENAME).write_text(
        json.dumps(thresholds, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8")
    return thresholds


# --------------------------------------------------------------------------- #
# 字面 guidance（只描述测得什么，绝不推断未测机制）
# --------------------------------------------------------------------------- #
# feature_id → {low, medium, high} 字面指令。避免任何未测量的文学机制（如"插入语/
# 对仗结构/从句"）与任何作者专属知识。
_LITERAL_GUIDANCE: dict[str, dict[str, str]] = {
    "dialogue_ratio": {
        "low": "Use dialogue relatively little.",
        "medium": "Use dialogue in moderate proportion.",
        "high": "Use dialogue relatively often."},
    "mean_sentence_length": {
        "low": "Favor relatively short sentences.",
        "medium": "Use a moderate sentence length.",
        "high": "Favor relatively long sentences."},
    "mean_paragraph_length": {
        "low": "Favor relatively short paragraphs.",
        "medium": "Use a moderate paragraph length.",
        "high": "Favor relatively long paragraphs."},
    "lexical_diversity": {
        "low": "Reuse words relatively often.",
        "medium": "Use moderate lexical variety.",
        "high": "Vary vocabulary relatively widely."},
    "comma_density": {
        "low": "Use commas relatively rarely.",
        "medium": "Use commas moderately.",
        "high": "Use commas relatively frequently."},
    "semicolon_density": {
        "low": "Use semicolons relatively rarely.",
        "medium": "Use semicolons moderately.",
        "high": "Use semicolons relatively frequently."},
    "dash_density": {
        "low": "Use dashes relatively rarely.",
        "medium": "Use dashes moderately.",
        "high": "Use dashes relatively frequently."},
    "quotation_density": {
        "low": "Use direct quotation relatively little.",
        "medium": "Use direct quotation moderately.",
        "high": "Use direct quotation relatively often."},
    "exclamation_frequency": {
        "low": "Use exclamation marks relatively rarely.",
        "medium": "Use exclamation marks moderately.",
        "high": "Use exclamation marks relatively frequently."},
    "question_frequency": {
        "low": "Use question sentences relatively rarely.",
        "medium": "Use question sentences moderately.",
        "high": "Use question sentences relatively frequently."},
    "period_density": {
        "low": "Use sentence-ending periods relatively rarely.",
        "medium": "Use sentence-ending periods moderately.",
        "high": "Use sentence-ending periods relatively frequently."},
    "long_sentence_ratio": {
        "low": "Use long sentences relatively rarely.",
        "medium": "Use long sentences moderately.",
        "high": "Use long sentences relatively frequently."},
    "short_sentence_ratio": {
        "low": "Use short sentences relatively rarely.",
        "medium": "Use short sentences moderately.",
        "high": "Use short sentences relatively frequently."},
    "sentence_length_cv": {
        "low": "Keep sentence length relatively uniform.",
        "medium": "Vary sentence length moderately.",
        "high": "Vary sentence length relatively widely."},
    "type_token_ratio": {
        "low": "Reuse words relatively often.",
        "medium": "Use moderate lexical variety.",
        "high": "Vary vocabulary relatively widely."},
    "hapax_ratio": {
        "low": "Use rare single-occurrence words relatively rarely.",
        "medium": "Use rare single-occurrence words moderately.",
        "high": "Use rare single-occurrence words relatively often."},
    "word_repetition_rate": {
        "low": "Repeat high-frequency words relatively little.",
        "medium": "Repeat high-frequency words moderately.",
        "high": "Repeat high-frequency words relatively often."},
    "mean_word_length": {
        "low": "Favor relatively short words.",
        "medium": "Use a moderate word length.",
        "high": "Favor relatively long words."},
    "connective_density": {
        "low": "Use connectives relatively rarely.",
        "medium": "Use connectives moderately.",
        "high": "Use connectives relatively frequently."},
    "word_length_variance": {
        "low": "Keep word length relatively uniform.",
        "medium": "Vary word length moderately.",
        "high": "Vary word length relatively widely."},
    "sentence_length_variance": {
        "low": "Keep sentence length relatively uniform.",
        "medium": "Vary sentence length moderately.",
        "high": "Vary sentence length relatively widely."},
    "paragraph_length_variance": {
        "low": "Keep paragraph length relatively uniform.",
        "medium": "Vary paragraph length moderately.",
        "high": "Vary paragraph length relatively widely."},
}


def band_label(feature_id: str, value: float,
               thresholds: dict[str, Any] | None) -> str | None:
    """给定经验阈值返回 low/medium/high；无阈值返回 None（not_compilable）。"""
    ft = (thresholds or {}).get("features", {}).get(feature_id)
    if not ft or "q33" not in ft or "q67" not in ft:
        return None
    if value < ft["q33"]:
        return "low"
    if value > ft["q67"]:
        return "high"
    return "medium"


def describe_feature(feature_id: str, summary: dict[str, Any],
                     thresholds: dict[str, Any] | None) -> str | None:
    """数值特征 → 字面 guidance（经验 band）；不可编译返回 None（绝不编造标签）。"""
    mean = summary.get("mean")
    if not isinstance(mean, (int, float)) or isinstance(mean, bool):
        return None
    band = band_label(feature_id, mean, thresholds)
    if band is None:
        return None
    triple = _LITERAL_GUIDANCE.get(feature_id)
    if triple is None:
        return None
    return triple[band]
