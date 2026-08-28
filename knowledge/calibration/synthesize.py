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


def _build_stylometry_diagnostics(base: Path) -> dict[str, Any]:
    """从 stylometry baseline + index 提取诊断摘要（不重算矩阵，只读已持久化结果）。"""
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


def _role_distribution(profile: AuthorStyleProfile) -> dict[str, int]:
    """direct / conditional / diagnostic / reference-only 计数。"""
    dist: Counter[str] = Counter()
    for g in profile.generation_controls.values():
        dist[g.control_role] += 1
    for n in profile.narrative_controls.values():
        dist[n.control_role] += 1
    for s in profile.strategy_controls:
        dist[s.control_role] += 1
    diag = profile.diagnostics.get("stylometry", {})
    dist["diagnostic"] += len(diag.get("feature_families", {}))
    return dict(sorted(dist.items()))


def _author_summary(profile: AuthorStyleProfile) -> dict[str, Any]:
    direct = sum(1 for g in profile.generation_controls.values()
                 if g.control_role == "direct_control")
    ref = sum(1 for g in profile.generation_controls.values()
              if g.control_role == "reference_only")
    return {
        "author_id": profile.author_id,
        "n_generation_controls": len(profile.generation_controls),
        "n_generation_direct_control": direct,
        "n_generation_reference_only": ref,
        "n_narrative_controls": len(profile.narrative_controls),
        "n_strategy_controls": len(profile.strategy_controls),
        "strategy_support_status": profile.uncertainty.get("strategy_support", {}),
        "n_diagnostic_families": len(
            profile.diagnostics.get("stylometry", {}).get("feature_families", {})),
        "role_distribution": _role_distribution(profile),
        "held_out_isolation_clean": profile.author_scope["held_out_isolation"]["clean"],
    }


def synthesize_style_profiles(data_root_: Path | None = None) -> dict[str, Any]:
    """读回 Phase 1–4.5 产物，合成并落盘作者风格画像 + 汇总 + 报告。"""
    base = Path(data_root_) if data_root_ is not None else default_data_root()
    out_dir = style_profile_layout(base)
    out_dir["root"].mkdir(parents=True, exist_ok=True)

    full_profiles = _load_json(base / "analysis" / "profiles" / "author_profiles.json")
    sampled_profiles = _load_json(
        base / "analysis" / "calibration" / "profiles" / "author_profiles.json")
    stylometry = _build_stylometry_diagnostics(base)

    corpus = corpus_by_author_id()
    synthesizer = AuthorStyleProfileSynthesizer()

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
            stylometry_diagnostics=stylometry,
        )
        profiles[author_id] = profile
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
        print(f"{aid}: generation={s['n_generation_controls']} "
              f"direct={s['n_generation_direct_control']} "
              f"reference={s['n_generation_reference_only']} "
              f"narrative={s['n_narrative_controls']} "
              f"strategy={s['n_strategy_controls']} "
              f"diagnostic={s['n_diagnostic_families']} "
              f"heldout_clean={s['held_out_isolation_clean']}")
    print("artifacts: data/analysis/style_profiles/"
          "{author_id}_style_profile.json + style_profile_summary.json + style_profile_report.md")


if __name__ == "__main__":
    main()
