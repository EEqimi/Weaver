# knowledge/calibration/synthesize.py
"""Phase 5：执行作者风格画像合成（确定性，无 LLM，无随机数，无时间戳内容）。

只读 Phase 1–4.5 已有产物：
    - 全语料作者画像 `data/analysis/profiles/author_profiles.json`（Layer A 统计，full train）
    - sampled 作者画像 `data/analysis/calibration/profiles/author_profiles.json`（Layer A/B/C，40-chunk）
    - 作者级 canonical strategies `data/analysis/consolidation/{author}_canonical_strategies.json`
    - stylometry 诊断 `data/analysis/stylometry/{baseline,index}.json`
合成 `AuthorStyleProfile` 并落盘 `data/analysis/style_profiles/`。

绝不：调用 LLM、重跑 calibration、重算 chunk-level 统计、修改既有产物、混入 held-out。
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.distance import cdist

from ..config import data_root as default_data_root
from ..corpus.metadata import HELD_OUT, TRAIN, by_author_id as corpus_by_author_id
from ..profiles.style_profile import (
    AuthorStyleProfile, AuthorStyleProfileSynthesizer,
)
from ..schema.versions import AUTHOR_STYLE_PROFILE_SCHEMA_VERSION

STYLE_PROFILES_DIRNAME = "style_profiles"


def style_profile_layout(data_root_: Path | None = None) -> dict[str, Path]:
    base = Path(data_root_) if data_root_ is not None else default_data_root()
    return {"root": base / "analysis" / STYLE_PROFILES_DIRNAME}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_stylometry_validation_metadata(base: Path) -> dict[str, Any]:
    """从 stylometry baseline + index 提取**全局**验证元数据（可跨作者共享）。

    不重算矩阵，只读已持久化结果。这是"这个实验有多可信"的元数据，不是作者专属指纹。
    """
    stylo = base / "analysis" / "stylometry"
    baseline = _load_json(stylo / "baseline.json")
    index = _load_json(stylo / "index.json")

    family_counts: Counter[str] = Counter()
    for f in index["feature_names"]:
        family_counts[f.split(":", 1)[0]] += 1
    families = {f"n_{k}": v for k, v in sorted(family_counts.items())}

    cv = baseline["grouped_cv_accuracy"]
    return {
        "n_train_chunks": baseline["n_train_chunks"],
        "n_heldout_chunks": baseline["n_heldout_chunks"],
        "n_features": baseline["n_features"],
        "feature_families": families,
        "grouped_cv_accuracy": cv,
        "grouped_cv_mean": round(sum(cv) / len(cv), 6) if cv else None,
        "grouped_cv_leak_free": baseline["grouped_cv_leak_free"],
        "heldout_accuracy": baseline["heldout_eval"]["accuracy"],
        "stylometric_family_overlap": baseline["stylometric_family_overlap"],
        "stylometry_version": index.get("stylometry_version"),
        "source_artifact": "data/analysis/stylometry/{baseline,index}.json",
    }


_NORMALIZATION_DOC = (
    "family-normalized relative frequency：fw / word-unigram 按每文本词数归一化，"
    "char-3gram 按每文本字符数归一化（knowledge/stylometry/extract.py）")
_VECTORIZER_DOC = (
    "CountVectorizer：char-3gram top-400 + word-unigram top-400（stop=154 功能词）"
    " + 154 显式功能词；词汇表仅在 TRAIN 文本上 fit")


def _round_vec(v: Any, nd: int = 8) -> list[float]:
    """把 numpy 向量转成定长四舍五入的 Python 列表（确定性 + 压缩体积）。"""
    return [round(float(x), nd) for x in np.asarray(v)]


def _author_targets_from_matrix(
    X_train: np.ndarray,
    train_authors: list[str],
    train_works: list[str],
    stylometry_version: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """纯函数：从 TRAIN 矩阵计算作者专属文体学目标（质心 + 离散度）。

    只接受 TRAIN 侧的数据（X_train / train_authors / train_works）；**任何 held-out
    数据都不作为入参**，从签名上杜绝 held-out 参与作者目标。这是确定性聚合
    （均值/标准差 + 质心余弦距离），不是重新分析文本。

    返回 (full, compact)：
        - full    落盘到 stylometric_author_targets.json（含 954 维质心/离散度向量）
        - compact 写入画像 diagnostics.stylometry.author_target（紧凑标量 + 产物引用）
    """
    full: dict[str, Any] = {}
    compact: dict[str, Any] = {}
    for author in sorted(set(train_authors)):
        mask = np.array([a == author for a in train_authors])
        X_author = X_train[mask]
        centroid = X_author.mean(axis=0)
        dispersion = X_author.std(axis=0)
        cos_dists = cdist(X_author, centroid[None, :], metric="cosine").ravel()
        mean_within = float(np.mean(cos_dists))
        works = sorted({w for w, a in zip(train_works, train_authors) if a == author})

        compact[author] = {
            "author_id": author,
            "n_samples": int(mask.sum()),
            "source_work_ids": works,
            "stylometry_version": stylometry_version,
            "feature_dim": int(X_train.shape[1]),
            "fit_scope": "train_only",
            "normalization": _NORMALIZATION_DOC,
            "vectorizer_provenance": _VECTORIZER_DOC,
            "centroid_norm": round(float(np.linalg.norm(centroid)), 6),
            "mean_dispersion": round(float(np.mean(dispersion)), 6),
            "mean_within_author_cosine_distance": round(mean_within, 6),
            "artifact": "data/analysis/style_profiles/stylometric_author_targets.json",
            "artifact_keys": {"centroid": f"authors.{author}.centroid",
                              "dispersion": f"authors.{author}.dispersion"},
        }
        full[author] = {
            "author_id": author,
            "n_samples": int(mask.sum()),
            "source_work_ids": works,
            "stylometry_version": stylometry_version,
            "feature_dim": int(X_train.shape[1]),
            "fit_scope": "train_only",
            "normalization": _NORMALIZATION_DOC,
            "vectorizer_provenance": _VECTORIZER_DOC,
            "centroid": _round_vec(centroid),
            "dispersion": _round_vec(dispersion),
            "mean_within_author_cosine_distance": round(mean_within, 6),
        }
    return full, compact


def _compute_stylometric_author_targets(base: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """读回 Layer D TRAIN 矩阵 + index，委托纯函数计算作者专属文体学目标。

    只读 `matrix.npz` 的 `X_train` 与 `index.json` 的 `train_*` 字段；绝不触碰
    `X_heldout` / `heldout_*`。
    """
    stylo = base / "analysis" / "stylometry"
    with np.load(stylo / "matrix.npz") as mat:
        X_train = mat["X_train"]
    index = _load_json(stylo / "index.json")
    return _author_targets_from_matrix(
        X_train, index["train_author_ids"], index["train_work_ids"],
        index.get("stylometry_version"))


def _role_distribution(profile: AuthorStyleProfile) -> dict[str, int]:
    """direct / conditional / diagnostic / reference-only 计数。"""
    dist: Counter[str] = Counter()
    for g in profile.generation_controls.values():
        dist[g.control_role] += 1
    for n in profile.narrative_controls.values():
        dist[n.control_role] += 1
    for s in profile.strategy_controls:
        dist[s.control_role] += 1
    stylo = profile.diagnostics.get("stylometry", {})
    vm = stylo.get("validation_metadata", {})
    dist["diagnostic"] += len(vm.get("feature_families", {}))
    return dict(sorted(dist.items()))


def _author_summary(profile: AuthorStyleProfile) -> dict[str, Any]:
    direct = sum(1 for g in profile.generation_controls.values()
                 if g.control_role == "direct_control")
    ref = sum(1 for g in profile.generation_controls.values()
              if g.control_role == "reference_only")
    stylo = profile.diagnostics.get("stylometry", {})
    at = stylo.get("author_target", {})
    return {
        "author_id": profile.author_id,
        "n_generation_controls": len(profile.generation_controls),
        "n_generation_direct_control": direct,
        "n_generation_reference_only": ref,
        "n_narrative_controls": len(profile.narrative_controls),
        "n_strategy_controls": len(profile.strategy_controls),
        "strategy_support_status": profile.uncertainty.get("strategy_support", {}),
        "n_diagnostic_families": len(
            stylo.get("validation_metadata", {}).get("feature_families", {})),
        "stylometric_target": {
            "n_samples": at.get("n_samples"),
            "source_work_ids": at.get("source_work_ids"),
            "feature_dim": at.get("feature_dim"),
            "centroid_norm": at.get("centroid_norm"),
            "mean_within_author_cosine_distance": at.get("mean_within_author_cosine_distance"),
        },
        "role_distribution": _role_distribution(profile),
        "held_out_isolation_clean": profile.author_scope["held_out_isolation"]["clean"],
    }


def synthesize_style_profiles(data_root_: Path | None = None) -> dict[str, Any]:
    """读回 Phase 1–4.5 产物，合成并落盘作者风格画像 + 汇总 + 报告。

    fail-closed：任一作者的 held-out 隔离校验失败即抛 ProfileSynthesisError，且
    **不写出任何**画像/汇总/报告产物（先全部内存合成，成功后统一落盘）。
    """
    base = Path(data_root_) if data_root_ is not None else default_data_root()
    out_dir = style_profile_layout(base)
    out_dir["root"].mkdir(parents=True, exist_ok=True)

    full_profiles = _load_json(base / "analysis" / "profiles" / "author_profiles.json")
    sampled_profiles = _load_json(
        base / "analysis" / "calibration" / "profiles" / "author_profiles.json")
    validation_metadata = _build_stylometry_validation_metadata(base)
    full_targets, compact_targets = _compute_stylometric_author_targets(base)

    corpus = corpus_by_author_id()
    synthesizer = AuthorStyleProfileSynthesizer()

    # 先在内存中全部合成（任一失败则整体失败，不落盘）。
    profiles: dict[str, AuthorStyleProfile] = {}
    for author_id in sorted(full_profiles):
        works = corpus[author_id]
        train_ids = [m.work_id for m in works if m.role == TRAIN]
        held_ids = [m.work_id for m in works if m.role == HELD_OUT]

        full = full_profiles[author_id]
        sampled = sampled_profiles[author_id]
        canon_path = (base / "analysis" / "consolidation"
                      / f"{author_id}_canonical_strategies.json")
        canonicals = _load_json(canon_path)["canonical_strategies"]

        profile = synthesizer.synthesize(
            author_id=author_id,
            train_work_ids=train_ids,
            held_out_work_ids=held_ids,
            profile_work_ids=full.get("work_ids", []),
            full_corpus_features=full.get("features", {}),
            sampled_features=sampled.get("features", {}),
            sampled_narrative=sampled.get("narrative", {}),
            canonical_strategies=canonicals,
            stylometry_author_target=compact_targets[author_id],
            stylometry_validation_metadata=validation_metadata,
        )
        profiles[author_id] = profile

    # 全部合成成功后才统一落盘。
    targets_artifact = {
        "stage": "stylometric_author_targets",
        "schema_version": AUTHOR_STYLE_PROFILE_SCHEMA_VERSION,
        "stylometry_version": validation_metadata.get("stylometry_version"),
        "fit_scope": "train_only",
        "deterministic": True,
        "no_llm": True,
        "no_held_out": True,
        "authors": full_targets,
    }
    (out_dir["root"] / "stylometric_author_targets.json").write_text(
        json.dumps(targets_artifact, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8")
    for author_id, profile in profiles.items():
        (out_dir["root"] / f"{author_id}_style_profile.json").write_text(
            json.dumps(profile.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8")

    summary = {
        "stage": "author_style_profile_synthesis",
        "schema_version": AUTHOR_STYLE_PROFILE_SCHEMA_VERSION,
        "deterministic": True,
        "no_llm": True,
        "authors": {aid: _author_summary(p) for aid, p in profiles.items()},
    }
    (out_dir["root"] / "style_profile_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir["root"] / "style_profile_report.md").write_text(
        _render_markdown(summary, profiles), encoding="utf-8")
    return summary


def _render_markdown(summary: dict[str, Any],
                     profiles: dict[str, AuthorStyleProfile]) -> str:
    lines = [
        "# Weaver Style Engine — 作者风格画像合成（Phase 5）报告",
        "",
        f"- schema：`{summary['schema_version']}`  确定性合成：`True`  无 LLM：`True`",
        "",
        "## 角色分布（direct / conditional / diagnostic / reference-only）",
        "",
        "| 作者 | generation | direct | reference | narrative | strategy | diagnostic |",
        "|---|---|---|---|---|---|---|",
    ]
    for aid in sorted(profiles):
        s = summary["authors"][aid]
        rd = s["role_distribution"]
        lines.append(
            f"| {aid} | {s['n_generation_controls']} | {rd.get('direct_control', 0)} | "
            f"{rd.get('reference_only', 0)} | {s['n_narrative_controls']} | "
            f"{s['n_strategy_controls']} | {rd.get('diagnostic', 0)} |")
    lines += ["", "## 策略优先级（deterministic）", "",
              "排序键（降序）：support_status tier（validated>candidate>discovered）→ "
              "跨作品数 → 跨 chunk 数 → confidence → raw observations → canonical id（稳定兜底）。",
              ""]
    for aid in sorted(profiles):
        p = profiles[aid]
        lines += [f"### {aid}", "",
                  "| priority | canonical_strategy_id | status | works | chunks | confidence |",
                  "|---|---|---|---|---|---|"]
        for s in p.strategy_controls[:10]:
            lines.append(
                f"| {s.control_priority} | `{s.canonical_strategy_id}` | {s.support_status} | "
                f"{len(s.supporting_work_ids)} | {len(s.supporting_chunk_ids)} | {s.confidence} |")
        if len(p.strategy_controls) > 10:
            lines.append(f"| … | （其余 {len(p.strategy_controls) - 10} 项略） | | | | |")
        lines.append("")
    lines += ["## Held-out 隔离", ""]
    for aid in sorted(profiles):
        p = profiles[aid]
        iso = p.author_scope["held_out_isolation"]
        lines.append(
            f"- **{aid}**：clean=`{iso['clean']}`  profile 污染=`{iso['profile_held_out_contamination']}`  "
            f"strategy 污染=`{iso['strategy_held_out_contamination']}`  "
            f"held-out 作品=`{p.author_scope['held_out_work_ids']}`")
    lines.append("")
    lines += ["## 确定性复现", "",
              "- 每份画像含 `reproducibility_hash`（sha256，覆盖除 hash 外的全部内容）。",
              "- 同输入重复合成 → 结构与 hash 完全一致。"]
    return "\n".join(lines) + "\n"


def main() -> None:
    summary = synthesize_style_profiles()
    for aid, s in summary["authors"].items():
        st = s["stylometric_target"]
        print(f"{aid}: generation={s['n_generation_controls']} "
              f"direct={s['n_generation_direct_control']} "
              f"reference={s['n_generation_reference_only']} "
              f"narrative={s['n_narrative_controls']} "
              f"strategy={s['n_strategy_controls']} "
              f"diagnostic={s['n_diagnostic_families']} "
              f"heldout_clean={s['held_out_isolation_clean']} "
              f"stylo(n={st['n_samples']},centroid_norm={st['centroid_norm']})")
    print("artifacts: data/analysis/style_profiles/"
          "{author_id}_style_profile.json + stylometric_author_targets.json + "
          "style_profile_summary.json + style_profile_report.md")


if __name__ == "__main__":
    main()
