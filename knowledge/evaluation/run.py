# knowledge/evaluation/run.py
"""Phase 8 执行：对 Phase 7 生成正文跑一次完整的风格反馈闭环 + 独立文学评价。

流程（spec §15，V0.1 单轮 max_iterations=2）：
    对每位作者（austen/dickens）：
        1. 加载 GeneratedPassage / StylePlan / AuthorStyleProfile / band 阈值；
        2. 再测量 → ActualStyleProfile（Layer A 统计+判断 / B / C / D）；
        3. 独立 LLM 文学评价（6 维）；
        4. 目标 vs 实际 → ComparisonResult（语言 band / 叙事 / 策略覆盖）；
        5. 优先化改写计划（P0–P4）；
        6. 最小编辑改写 → RevisionResult；
        7. 改写后再测量 + 再评价（是否朝目标移动 / 文学分是否守住）；
        8. stylometric 诊断 + 确定性决策 Accept / Continue / Roll Back；
        9. 落盘 machine-readable JSON + human markdown（版本化 + provenance）。

铁律：
    - 绝不覆盖 data/analysis/generation/ 既有产物；全部写入新的
      data/analysis/evaluation/；
    - 文学评价与改写器盲测；改写指令绝不含作者名 / 原始数值 / 微观 stylometric 指纹；
    - stylometric 距离只诊断，绝不进改写指令；密钥只读（DEEPSEEK_API_KEY）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..analysis.base import AnalysisUnavailable
from ..config import data_root as default_data_root
from ..corpus.metadata import author_display_names
from ..generation.schema import GeneratedPassage
from ..planning.run import AUTHOR_IDS, _band_thresholds, _load_profile
from ..planning.schema import StylePlan
from ..providers.llm_provider import (
    CacheBackedLLMProvider, DeepSeekProvider, LLMCache, LLMProvider,
)
from ..schema.versions import EVALUATION_SCHEMA_VERSION
from .analyze import measure_actual_profile
from .compare import compare_target_actual
from .literary import LiteraryEvaluator
from .revision import RevisionRewriter, build_revision_plan
from .schema import (
    ComparisonResult, EvalError, LiteraryEvaluation, RevisionResult,
)

EVALUATION_DIRNAME = "evaluation"
MAX_ITERATIONS = 2


# --------------------------------------------------------------------------- #
# 布局 / provider
# --------------------------------------------------------------------------- #
def evaluation_layout(data_root_: Path | None = None) -> dict[str, Path]:
    base = Path(data_root_) if data_root_ is not None else default_data_root()
    return {"root": base / "analysis" / EVALUATION_DIRNAME,
            "cache": base / "analysis" / EVALUATION_DIRNAME / "llm_cache"}


def build_provider(data_root_: Path | None = None) -> CacheBackedLLMProvider:
    """真实后端（DeepSeek）+ 磁盘缓存，与 analysis 各层共用同一 provider。"""
    out = evaluation_layout(data_root_)
    out["cache"].mkdir(parents=True, exist_ok=True)
    return CacheBackedLLMProvider(DeepSeekProvider(), LLMCache(out["cache"]))


# --------------------------------------------------------------------------- #
# 决策（纯函数，确定性）
# --------------------------------------------------------------------------- #
def _count_high_priority_deviations(c: ComparisonResult) -> int:
    """高优先级偏差计数：P1（叙事 off_target）+ P2（策略未命中）+ P3（语言 band 偏离）。

    不含 P0（情节/语义）与 P4（stylometric 仅诊断）。stylometric 距离绝不影响决策。
    """
    n = sum(1 for d in c.language_deviations if d.status in ("above", "below"))
    n += sum(1 for d in c.narrative_deviations if d.status == "off_target")
    n += sum(1 for s in c.strategy_coverage if s.active and not s.matched)
    return n


def decide_feedback_outcome(before: ComparisonResult | None,
                            after: ComparisonResult | None, *,
                            iteration: int = 1,
                            max_iterations: int = MAX_ITERATIONS) -> tuple[str, str]:
    """确定性 Accept / Continue / Roll Back（spec §15.5）。

    规则（优先级从高到低）：
        - after 为 None（改写/再测量失败）→ roll_back；
        - 高优先级偏差归零 → accept；
        - 高优先级偏差减少且未达 max_iterations → continue；
        - 高优先级偏差减少但已达 max_iterations → accept（接受最优可用）；
        - 未改善（甚至变差）→ roll_back。
    stylometric 距离绝不进入决策（仅诊断），故 stylometric 改善不会掩盖高层回归。
    """
    if after is None:
        return "roll_back", "re-analysis unavailable (rewrite failed or unconfigured)"
    before_n = _count_high_priority_deviations(before) if before else 0
    after_n = _count_high_priority_deviations(after)
    improved = after_n < before_n
    if after_n == 0:
        return "accept", f"all high-priority deviations resolved ({before_n} -> 0)"
    if improved:
        if iteration >= max_iterations:
            return "accept", (f"improved {before_n} -> {after_n} but max_iterations="
                              f"{max_iterations} reached")
        return "continue", f"improved {before_n} -> {after_n}; another iteration possible"
    return "roll_back", f"no improvement ({before_n} -> {after_n}); keep the original"


# --------------------------------------------------------------------------- #
# I/O 辅助
# --------------------------------------------------------------------------- #
def _write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _author_names(author_ids) -> list[str]:
    names = author_display_names()
    return [names[aid] for aid in author_ids if aid in names]


# --------------------------------------------------------------------------- #
# 编排
# --------------------------------------------------------------------------- #
def run_evaluation(data_root_: Path | None = None,
                   provider: LLMProvider | None = None) -> dict[str, Any]:
    base = Path(data_root_) if data_root_ is not None else default_data_root()
    out_dir = evaluation_layout(base)["root"]
    out_dir.mkdir(parents=True, exist_ok=True)

    provider = provider or build_provider(base)
    if not provider.is_configured():
        raise EvalError("未配置 LLM provider（缺 DEEPSEEK_API_KEY）")

    profiles = {aid: _load_profile(base, aid) for aid in AUTHOR_IDS}
    train_work_ids = sorted(
        {w for p in profiles.values() for w in p.author_scope.get("train_work_ids", [])})
    band_thresholds = _band_thresholds(base, train_work_ids)

    evaluator = LiteraryEvaluator(provider, blind=True)
    rewriter = RevisionRewriter(provider, blind=True)
    names = _author_names(AUTHOR_IDS)

    authors: dict[str, Any] = {}
    for author_id in AUTHOR_IDS:
        passage = _load_passage(base, author_id)
        plan = _load_plan(base, author_id)
        profile = profiles[author_id]

        # 2. 再测量（原正文）
        actual = measure_actual_profile(
            passage.generated_text, author_id=author_id,
            passage_id=passage.generation_id, style_plan_id=plan.style_plan_id,
            profile=profile, provider=provider, data_root_=base)

        # 3. 独立文学评价（原正文）
        eval_before = _evaluate(evaluator, passage.generated_text, author_id,
                                passage.generation_id)

        # 4. 目标 vs 实际
        comparison_before = compare_target_actual(plan, profile, actual, band_thresholds)

        # 5. 改写计划
        rev_plan = build_revision_plan(comparison_before, plan, evaluation=eval_before)

        # 6. 最小编辑改写
        rev_result = rewriter.rewrite(
            passage.generated_text, rev_plan, author_names=names)

        # 7–8. 改写后再测量 + 再评价 + 决策
        if isinstance(rev_result, AnalysisUnavailable):
            comparison_after = None
            eval_after = None
            actual_after = None
            decision, reason = decide_feedback_outcome(
                comparison_before, None, iteration=1, max_iterations=MAX_ITERATIONS)
        else:
            rev_passage_id = f"{passage.generation_id}:rev1"
            actual_after = measure_actual_profile(
                rev_result.revised_text, author_id=author_id,
                passage_id=rev_passage_id, style_plan_id=plan.style_plan_id,
                profile=profile, provider=provider, data_root_=base)
            eval_after = _evaluate(evaluator, rev_result.revised_text, author_id,
                                   rev_passage_id)
            comparison_after = compare_target_actual(
                plan, profile, actual_after, band_thresholds)
            decision, reason = decide_feedback_outcome(
                comparison_before, comparison_after, iteration=1,
                max_iterations=MAX_ITERATIONS)

        # 9. 落盘
        _write_json(out_dir / f"{author_id}_actual_profile.json", actual.to_dict())
        _write_json(out_dir / f"{author_id}_revision_plan.json", rev_plan.to_dict())
        if eval_before is not None:
            _write_json(out_dir / f"{author_id}_literary_evaluation.json",
                        eval_before.to_dict())
        if isinstance(rev_result, RevisionResult):
            _write_json(out_dir / f"{author_id}_revision_result.json",
                        rev_result.to_dict())
        if actual_after is not None:
            _write_json(out_dir / f"{author_id}_revised_actual_profile.json",
                        actual_after.to_dict())
        if eval_after is not None:
            _write_json(out_dir / f"{author_id}_revised_literary_evaluation.json",
                        eval_after.to_dict())

        authors[author_id] = {
            "generation_id": passage.generation_id,
            "style_plan_id": plan.style_plan_id,
            "comparison_before": comparison_before.summary,
            "literary_total_before": (eval_before.total_score if eval_before else None),
            "n_revision_items": len(rev_plan.revision_items),
            "revision_items_by_priority": rev_plan.metadata.get("by_priority"),
            "revision_items": [i.to_dict() for i in rev_plan.revision_items],
            "change_descriptions": (rev_result.change_descriptions
                                    if isinstance(rev_result, RevisionResult) else []),
            "comparison_after": (comparison_after.summary
                                 if comparison_after is not None else None),
            "literary_total_after": (eval_after.total_score if eval_after else None),
            "stylometric_before": actual.layer_d_stylometric,
            "stylometric_after": (actual_after.layer_d_stylometric
                                  if actual_after is not None else None),
            "decision": decision,
            "decision_reason": reason,
        }

    summary = _build_summary(provider, authors)
    _write_json(out_dir / "evaluation_summary.json", summary)
    (out_dir / "evaluation_report.md").write_text(
        _render_report(summary, authors), encoding="utf-8")
    return summary


def _load_passage(base: Path, author_id: str) -> GeneratedPassage:
    path = base / "analysis" / "generation" / f"{author_id}_generation.json"
    if not path.exists():
        raise EvalError(f"缺 Phase 7 生成产物: {path}")
    return GeneratedPassage.from_dict(_load_json(path))


def _load_plan(base: Path, author_id: str) -> StylePlan:
    path = base / "analysis" / "planning" / f"{author_id}_style_plan.json"
    if not path.exists():
        raise EvalError(f"缺 Phase 6 计划产物: {path}")
    return StylePlan.from_dict(_load_json(path))


def _evaluate(evaluator: LiteraryEvaluator, text: str, author_id: str,
              passage_id: str) -> LiteraryEvaluation | None:
    res = evaluator.evaluate(text, author_id=author_id, passage_id=passage_id)
    return None if isinstance(res, AnalysisUnavailable) else res


def _build_summary(provider: LLMProvider, authors: dict[str, Any]) -> dict[str, Any]:
    inner = getattr(provider, "_inner", None)
    usage = getattr(inner, "usage", {})
    return {
        "stage": "style_feedback_loop_and_literary_evaluation",
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "max_iterations": MAX_ITERATIONS,
        "provider": getattr(provider, "provider_id", ""),
        "model": getattr(provider, "model", ""),
        "blind": True,
        "authors": authors,
        "token_usage": dict(usage) if usage else None,
    }


# --------------------------------------------------------------------------- #
# 渲染（人类可读）
# --------------------------------------------------------------------------- #
def _render_report(summary: dict[str, Any], authors: dict[str, Any]) -> str:
    lines = [
        "# Weaver Style Engine — Style Feedback Loop + 文学评价报告（Phase 8）",
        "",
        f"- provider / model：`{summary.get('provider')}` / `{summary.get('model')}`",
        f"- blind：`{summary.get('blind')}`　max_iterations：`{summary.get('max_iterations')}`",
        "",
        "本阶段对 Phase 7 生成正文再测量 → 目标 vs 实际 → 改写计划 → 最小编辑改写 → "
        "再分析 → 决策。文学评价为独立 LLM 判定（6 维 1–10）；stylometric 距离仅诊断。",
        "",
    ]
    for author_id in AUTHOR_IDS:
        a = authors[author_id]
        lines += [
            f"## {author_id.capitalize()}",
            "",
            f"- generation_id：`{a['generation_id']}`　style_plan_id：`{a['style_plan_id']}`",
            f"- 文学评价总分（改写前）：`{a['literary_total_before']}`",
            f"- 改写项数：`{a['n_revision_items']}`（{a['revision_items_by_priority']}）",
            "",
            "### 改写前偏差汇总",
            "",
            f"```json\n{json.dumps(a['comparison_before'], ensure_ascii=False, indent=2)}\n```",
            "",
            "### 改写项（优先级 P0→P4）",
            "",
        ]
        for item in a["revision_items"]:
            lines.append(f"- **[{item['priority']}]**（{item['category']} / {item['target']}）"
                         f"{item['instruction']}")
        lines += [
            "",
            "### 改写变更说明",
            "",
        ]
        for d in a["change_descriptions"]:
            lines.append(f"- {d}")
        lines += [
            "",
            "### 改写后",
            "",
            f"- 文学评价总分（改写后）：`{a['literary_total_after']}`",
            f"- stylometric 余弦距离：before=`{(a['stylometric_before'] or {}).get('cosine_distance')}`"
            f"　after=`{(a['stylometric_after'] or {}).get('cosine_distance')}`",
            f"- **决策**：`{a['decision']}` — {a['decision_reason']}",
            "",
        ]
    lines += [
        "---",
        "",
        "> 机器可读产物见 `data/analysis/evaluation/`（actual_profile / literary_evaluation / "
        "revision_plan / revision_result / revised_actual_profile / revised_literary_evaluation / "
        "evaluation_summary.json）。stylometric 距离仅为诊断，从未进入改写指令。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    summary = run_evaluation()
    for aid in AUTHOR_IDS:
        a = summary["authors"][aid]
        print(f"{aid}: eval_before={a['literary_total_before']} "
              f"eval_after={a['literary_total_after']} "
              f"rev_items={a['n_revision_items']} decision={a['decision']}")
    print(f"token_usage: {summary['token_usage']}")
    print("artifacts: data/analysis/evaluation/evaluation_summary.json + "
          "evaluation_report.md + {author_id}_{actual_profile,literary_evaluation,"
          "revision_plan,revision_result,revised_actual_profile,"
          "revised_literary_evaluation}.json")


if __name__ == "__main__":
    main()
