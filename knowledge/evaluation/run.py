# knowledge/evaluation/run.py
"""Phase 8.2 执行：改写有效性 + 测量有效性（对 Phase 7 生成正文跑反馈闭环）。

与 Phase 8.1 的区别（spec §三/§十四/§十六）：
    - 决策改为四阶 gate，在 Content Integrity 之前新增 **Revision Effect 门**（确定性，
      零 token）：改写无实质词级变化 → `no_effect`，短路后续一切昂贵步骤；
        0. Revision Effect（改写是否实质变化：identical/formatting_only/punctuation_only
           /substantive；non-substantive → no_effect 短路）；
        1. Content Integrity（最高）：改写是否破坏情节/角色/关系/约束/事件 → roll_back；
        2. Literary Quality guard：文学总分下降超过可配置容忍度 → roll_back；
        3. Style Fidelity：偏差是否减少 → accept / continue / roll_back。
    - 只有 `substantive_edit == True` 才允许 after-measurement 参与比较，杜绝 LLM 测量
      噪声（对等价文本的评分漂移）被记为真实改善；
    - 改写器自报字段降级为 `claimed_*`（best-effort），真实有效性由 deterministic
      `revision_effect` 给出；
    - no_effect 独立于 no_action（空计划）与 roll_back（实质改写被拒）。

铁律：
    - 绝不覆盖 data/analysis/evaluation/（Phase 8 v1）或 evaluation_v2/（Phase 8.1）；
      Phase 8.2 未来真实运行时写入新的 evaluation_v3/（本轮不运行真实 LLM）；
    - 文学评价与改写器、内容完整性检查器均盲测；改写指令与完整性 prompt 绝不含作者名
      或模仿令牌；stylometric 距离只诊断；P4 恒无改写指令；密钥只读（DEEPSEEK_API_KEY）。
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
from ..planning.schema import StylePlan, WritingRequest
from ..providers.llm_provider import (
    CacheBackedLLMProvider, DeepSeekProvider, LLMCache, LLMProvider,
)
from ..schema.versions import EVALUATION_SCHEMA_VERSION, FEEDBACK_DECISION_SCHEMA_VERSION
from .analyze import measure_actual_profile
from .compare import compare_target_actual
from .effect import RevisionEffectAnalyzer
from .integrity import ContentIntegrityChecker
from .literary import LiteraryEvaluator
from .revision import RevisionRewriter, build_revision_plan
from .schema import (
    ComparisonResult, ContentIntegrityResult, EvalError, EvaluationPolicy,
    FeedbackDecision, LiteraryEvaluation, RevisionEffectResult, RevisionResult,
    FEEDBACK_ACCEPT, FEEDBACK_CONTINUE, FEEDBACK_NO_ACTION, FEEDBACK_NO_EFFECT,
    FEEDBACK_ROLL_BACK, GUARD_NOT_APPLICABLE_NO_EFFECT,
)

EVALUATION_DIRNAME = "evaluation"          # Phase 8 v1（保留，绝不覆盖）
EVALUATION_DIRNAME_V2 = "evaluation_v2"    # Phase 8.1（保留，绝不覆盖）
EVALUATION_DIRNAME_V3 = "evaluation_v3"    # Phase 8.2 新产物
MAX_ITERATIONS = 2


# --------------------------------------------------------------------------- #
# 布局 / provider
# --------------------------------------------------------------------------- #
def evaluation_layout(data_root_: Path | None = None) -> dict[str, Path]:
    """Phase 8.2 布局：产物写入 evaluation_v3；LLM cache 复用 v1（省 token）。"""
    base = Path(data_root_) if data_root_ is not None else default_data_root()
    return {"root": base / "analysis" / EVALUATION_DIRNAME_V3,
            "cache": base / "analysis" / EVALUATION_DIRNAME / "llm_cache"}


def build_provider(data_root_: Path | None = None) -> CacheBackedLLMProvider:
    """真实后端（DeepSeek）+ 磁盘缓存，与 analysis 各层共用同一 provider。"""
    out = evaluation_layout(data_root_)
    out["cache"].mkdir(parents=True, exist_ok=True)
    return CacheBackedLLMProvider(DeepSeekProvider(), LLMCache(out["cache"]))


# --------------------------------------------------------------------------- #
# 决策（纯函数，确定性；三阶 gate）
# --------------------------------------------------------------------------- #
def _count_high_priority_deviations(c: ComparisonResult) -> int:
    """高优先级偏差计数：P1（叙事 off_target）+ P2（策略未命中）+ P3（语言 band 偏离）。

    不含 P0（情节/语义）与 P4（stylometric 仅诊断）。stylometric 距离绝不影响决策。
    """
    n = sum(1 for d in c.language_deviations if d.status in ("above", "below"))
    n += sum(1 for d in c.narrative_deviations if d.status == "off_target")
    n += sum(1 for s in c.strategy_coverage if s.active and not s.matched)
    return n


def _style_fidelity(before_n: int | None, after_n: int | None) -> dict[str, Any]:
    return {
        "high_priority_deviations_before": before_n,
        "high_priority_deviations_after": after_n,
        "improved": bool(after_n is not None and before_n is not None
                         and after_n < before_n),
    }


def _literary_quality(before: float | None, after: float | None,
                      policy: EvaluationPolicy, *, guard: str) -> dict[str, Any]:
    """文学质量护栏报告（绝不把 unavailable 的分数当 0）。

    `guard` 状态：
        - "applied"：before/after 均有分数，执行下降容忍度检查；
        - "unavailable"：before 或 after 缺失，护栏无法评估（绝不伪造基线/结果）；
        - "not_applicable"：no_action 或 integrity 失败（未进入文学护栏）；
        - "not_applicable_no_effect"：改写无实质变化（Phase 8.2 no_effect，不比较）。
    """
    drop = round(before - after, 4) if (before is not None and after is not None) else None
    evaluated = guard == "applied" and before is not None and after is not None
    return {
        "before": before,
        "after": after,
        "drop": drop,
        "max_literary_drop": policy.max_literary_drop,
        "drop_exceeded": bool(drop is not None and drop > policy.max_literary_drop),
        "evaluated": evaluated,
        "guard": guard,
    }


def _make_decision(outcome: str, reason: str, *,
                   iteration: int, max_iterations: int,
                   before_n: int | None, after_n: int | None,
                   lit: dict[str, Any],
                   content_integrity: ContentIntegrityResult | None,
                   author_id: str, passage_id: str,
                   revision_effect: dict[str, Any] | None = None,
                   style_comparison_performed: bool = True) -> FeedbackDecision:
    return FeedbackDecision(
        schema_version=FEEDBACK_DECISION_SCHEMA_VERSION,
        outcome=outcome, reason=reason,
        content_integrity_passed=(content_integrity.passed
                                  if content_integrity is not None else None),
        content_integrity=(content_integrity.to_dict()
                           if content_integrity is not None else None),
        style_fidelity=_style_fidelity(before_n, after_n),
        literary_quality=lit,
        iteration=iteration, max_iterations=max_iterations,
        author_id=author_id, passage_id=passage_id,
        revision_effect=revision_effect,
        literary_guard_status=lit["guard"],
        style_comparison_performed=style_comparison_performed,
    )


def decide_feedback_outcome(
    before: ComparisonResult | None,
    after: ComparisonResult | None,
    *,
    iteration: int = 1,
    max_iterations: int = MAX_ITERATIONS,
    literary_before: float | None = None,
    literary_after: float | None = None,
    content_integrity: ContentIntegrityResult | None = None,
    no_revision: bool = False,
    revision_effect: RevisionEffectResult | None = None,
    policy: EvaluationPolicy | None = None,
    author_id: str = "",
    passage_id: str = "",
) -> FeedbackDecision:
    """四阶 gate（spec §三）：Revision Effect → Content Integrity → Literary Quality
    → Style Fidelity。

    Style Fidelity 与 Literary Quality 分别报告，绝不合并成单一加权总分；决策基于
    gate/规则，而非神秘加权分。`revision_effect` 非空且 non-substantive → no_effect
    （短路，绝不进入后续 gate；杜绝 LLM 测量噪声被记为改善）。
    """
    policy = policy or EvaluationPolicy()
    before_n = _count_high_priority_deviations(before) if before is not None else 0
    after_n = _count_high_priority_deviations(after) if after is not None else None

    # STEP 0：no_action（改写计划为空，未执行任何改写，独立于 roll_back）
    if no_revision:
        return _make_decision(
            FEEDBACK_NO_ACTION, "revision plan empty; no revision performed",
            iteration=iteration, max_iterations=max_iterations,
            before_n=before_n, after_n=None,
            lit=_literary_quality(literary_before, literary_after, policy,
                                  guard="not_applicable"),
            content_integrity=None, author_id=author_id, passage_id=passage_id)

    # STEP 0.5：Revision Effect 门（Phase 8.2，确定性零 token）：改写无实质词级变化
    # → no_effect。独立于 no_action（空计划）与 roll_back（实质改写被拒）。短路后续
    # 一切昂贵步骤（完整性 / 重测 / 文学评价 / 策略匹配），绝不把 LLM 测量噪声记为改善。
    if revision_effect is not None and not revision_effect.substantive_edit:
        return _make_decision(
            FEEDBACK_NO_EFFECT,
            f"revision produced no substantive textual change "
            f"(effect_status={revision_effect.effect_status})",
            iteration=iteration, max_iterations=max_iterations,
            before_n=before_n, after_n=None,
            lit=_literary_quality(literary_before, literary_after, policy,
                                  guard=GUARD_NOT_APPLICABLE_NO_EFFECT),
            content_integrity=None, author_id=author_id, passage_id=passage_id,
            revision_effect=revision_effect.to_dict(),
            style_comparison_performed=False)

    # STEP 1：Content Integrity gate（最高优先级；破坏内容 → roll_back）
    if content_integrity is not None and not content_integrity.passed:
        return _make_decision(
            FEEDBACK_ROLL_BACK,
            f"content integrity violation: "
            f"{content_integrity.reasoning_summary or 'rewrite broke user content'}",
            iteration=iteration, max_iterations=max_iterations,
            before_n=before_n, after_n=after_n,
            lit=_literary_quality(literary_before, literary_after, policy,
                                  guard="not_applicable"),
            content_integrity=content_integrity, author_id=author_id,
            passage_id=passage_id)

    # STEP 2：Literary Quality guard（文学分明显下降超过容忍度 → roll_back）
    guard = ("applied" if (literary_before is not None and literary_after is not None)
             else "unavailable")

    # fail-closed：基线有效但改写后文学评价 unavailable（如 6 维证据契约全失败）→
    # 无法验证文学质量是否保留，绝不单凭 Style Fidelity 接受（spec 决策完整性）。
    if literary_before is not None and literary_after is None and after is not None:
        return _make_decision(
            FEEDBACK_ROLL_BACK,
            "post-revision literary evaluation unavailable; literary quality "
            "preservation cannot be verified",
            iteration=iteration, max_iterations=max_iterations,
            before_n=before_n, after_n=after_n,
            lit=_literary_quality(literary_before, literary_after, policy,
                                  guard=guard),
            content_integrity=content_integrity, author_id=author_id,
            passage_id=passage_id)

    lit = _literary_quality(literary_before, literary_after, policy, guard=guard)
    if guard == "applied" and lit["drop_exceeded"]:
        return _make_decision(
            FEEDBACK_ROLL_BACK,
            f"literary quality dropped {lit['drop']} > tolerance "
            f"{policy.max_literary_drop} (style gain not worth the literary loss)",
            iteration=iteration, max_iterations=max_iterations,
            before_n=before_n, after_n=after_n, lit=lit,
            content_integrity=content_integrity, author_id=author_id,
            passage_id=passage_id)

    # STEP 3：Style Fidelity
    if after is None:
        return _make_decision(
            FEEDBACK_ROLL_BACK, "re-analysis unavailable (rewrite failed or unconfigured)",
            iteration=iteration, max_iterations=max_iterations,
            before_n=before_n, after_n=None, lit=lit,
            content_integrity=content_integrity, author_id=author_id,
            passage_id=passage_id)
    if after_n == 0:
        return _make_decision(
            FEEDBACK_ACCEPT, f"all high-priority deviations resolved ({before_n} -> 0)",
            iteration=iteration, max_iterations=max_iterations,
            before_n=before_n, after_n=after_n, lit=lit,
            content_integrity=content_integrity, author_id=author_id,
            passage_id=passage_id)
    if after_n < before_n:   # improved
        if iteration >= max_iterations:
            return _make_decision(
                FEEDBACK_ACCEPT, f"improved {before_n} -> {after_n} but max_iterations="
                f"{max_iterations} reached", iteration=iteration,
                max_iterations=max_iterations, before_n=before_n, after_n=after_n,
                lit=lit, content_integrity=content_integrity,
                author_id=author_id, passage_id=passage_id)
        return _make_decision(
            FEEDBACK_CONTINUE, f"improved {before_n} -> {after_n}; another iteration "
            "possible", iteration=iteration, max_iterations=max_iterations,
            before_n=before_n, after_n=after_n, lit=lit,
            content_integrity=content_integrity, author_id=author_id,
            passage_id=passage_id)
    return _make_decision(
        FEEDBACK_ROLL_BACK, f"no improvement ({before_n} -> {after_n}); keep the original",
        iteration=iteration, max_iterations=max_iterations,
        before_n=before_n, after_n=after_n, lit=lit,
        content_integrity=content_integrity, author_id=author_id,
        passage_id=passage_id)


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
                   provider: LLMProvider | None = None,
                   policy: EvaluationPolicy | None = None) -> dict[str, Any]:
    base = Path(data_root_) if data_root_ is not None else default_data_root()
    out_dir = evaluation_layout(base)["root"]
    out_dir.mkdir(parents=True, exist_ok=True)
    policy = policy or EvaluationPolicy()

    provider = provider or build_provider(base)
    if not provider.is_configured():
        raise EvalError("未配置 LLM provider（缺 DEEPSEEK_API_KEY）")

    profiles = {aid: _load_profile(base, aid) for aid in AUTHOR_IDS}
    train_work_ids = sorted(
        {w for p in profiles.values() for w in p.author_scope.get("train_work_ids", [])})
    band_thresholds = _band_thresholds(base, train_work_ids)

    evaluator = LiteraryEvaluator(provider, blind=True)
    rewriter = RevisionRewriter(provider, blind=True)
    checker = ContentIntegrityChecker(provider, blind=True)
    names = _author_names(AUTHOR_IDS)

    authors: dict[str, Any] = {}
    for author_id in AUTHOR_IDS:
        passage = _load_passage(base, author_id)
        plan = _load_plan(base, author_id)
        profile = profiles[author_id]
        request = WritingRequest.from_dict(plan.writing_request)

        # 2. 再测量（原正文）
        actual = measure_actual_profile(
            passage.generated_text, author_id=author_id,
            passage_id=passage.generation_id, style_plan_id=plan.style_plan_id,
            profile=profile, provider=provider, data_root_=base)

        # 3. 独立文学评价（原正文，Phase 8.1 evidence contract）
        eval_before = _evaluate(evaluator, passage.generated_text, author_id,
                                passage.generation_id)

        # 4. 目标 vs 实际
        comparison_before = compare_target_actual(plan, profile, actual, band_thresholds)

        # 5. 改写计划
        rev_plan = build_revision_plan(
            comparison_before, plan, evaluation=eval_before,
            weak_score_threshold=policy.weak_score_threshold)

        lit_before = eval_before.total_score if eval_before else None

        # 6–8. 改写 → 完整性 gate → 再分析 → 决策
        if not rev_plan.revision_items:
            decision = decide_feedback_outcome(
                comparison_before, None, iteration=1, max_iterations=MAX_ITERATIONS,
                literary_before=lit_before, literary_after=None,
                content_integrity=None, no_revision=True, policy=policy,
                author_id=author_id, passage_id=passage.generation_id)
            integrity = None
            rev_result = None
            actual_after = None
            eval_after = None
            comparison_after = None
        else:
            rev_result = rewriter.rewrite(
                passage.generated_text, rev_plan, author_names=names)

            if isinstance(rev_result, AnalysisUnavailable):
                decision = decide_feedback_outcome(
                    comparison_before, None, iteration=1, max_iterations=MAX_ITERATIONS,
                    literary_before=lit_before, literary_after=None,
                    content_integrity=None, no_revision=False, policy=policy,
                    author_id=author_id, passage_id=passage.generation_id)
                integrity = None
                actual_after = None
                eval_after = None
                comparison_after = None
            else:
                # Gate 0：Revision Effect（确定性，零 token）——改写是否产生实质词级变化。
                effect = RevisionEffectAnalyzer().analyze(
                    passage.generated_text, rev_result.revised_text)
                rev_result.revision_effect = effect.to_dict()

                if not effect.substantive_edit:
                    # no_effect：改写无实质变化 → 短路，绝不调 integrity / 重测 /
                    # 文学评价 / 策略匹配（杜绝 LLM 测量噪声被记为改善；spec §十四）。
                    decision = decide_feedback_outcome(
                        comparison_before, None, iteration=1,
                        max_iterations=MAX_ITERATIONS, literary_before=lit_before,
                        literary_after=None, content_integrity=None,
                        no_revision=False, revision_effect=effect, policy=policy,
                        author_id=author_id, passage_id=passage.generation_id)
                    integrity = None
                    actual_after = None
                    eval_after = None
                    comparison_after = None
                else:
                    # Content Integrity gate（先于风格重测，省 token；spec §十一）
                    integrity = checker.check(
                        passage.generated_text, rev_result.revised_text, request,
                        author_names=names)

                    if isinstance(integrity, AnalysisUnavailable):
                        decision = _make_decision(
                            FEEDBACK_ROLL_BACK,
                            "content integrity check unavailable; fail-closed",
                            iteration=1, max_iterations=MAX_ITERATIONS,
                            before_n=_count_high_priority_deviations(comparison_before),
                            after_n=None,
                            lit=_literary_quality(lit_before, None, policy,
                                                  guard="unavailable"),
                            content_integrity=None, author_id=author_id,
                            passage_id=passage.generation_id)
                        integrity = None
                        actual_after = None
                        eval_after = None
                        comparison_after = None
                    elif not integrity.passed:
                        decision = decide_feedback_outcome(
                            comparison_before, None, iteration=1,
                            max_iterations=MAX_ITERATIONS, literary_before=lit_before,
                            literary_after=None, content_integrity=integrity,
                            no_revision=False, policy=policy, author_id=author_id,
                            passage_id=passage.generation_id)
                        actual_after = None
                        eval_after = None
                        comparison_after = None
                    else:
                        rev_passage_id = f"{passage.generation_id}:rev1"
                        actual_after = measure_actual_profile(
                            rev_result.revised_text, author_id=author_id,
                            passage_id=rev_passage_id, style_plan_id=plan.style_plan_id,
                            profile=profile, provider=provider, data_root_=base)
                        eval_after = _evaluate(evaluator, rev_result.revised_text,
                                               author_id, rev_passage_id)
                        comparison_after = compare_target_actual(
                            plan, profile, actual_after, band_thresholds)
                        decision = decide_feedback_outcome(
                            comparison_before, comparison_after, iteration=1,
                            max_iterations=MAX_ITERATIONS, literary_before=lit_before,
                            literary_after=(eval_after.total_score if eval_after else None),
                            content_integrity=integrity, no_revision=False, policy=policy,
                            author_id=author_id, passage_id=passage.generation_id)

        # 9. 落盘
        _write_json(out_dir / f"{author_id}_actual_profile.json", actual.to_dict())
        _write_json(out_dir / f"{author_id}_revision_plan.json", rev_plan.to_dict())
        if eval_before is not None:
            _write_json(out_dir / f"{author_id}_literary_evaluation.json",
                        eval_before.to_dict())
        if isinstance(rev_result, RevisionResult):
            _write_json(out_dir / f"{author_id}_revision_result.json",
                        rev_result.to_dict())
        if isinstance(integrity, ContentIntegrityResult):
            _write_json(out_dir / f"{author_id}_content_integrity.json",
                        integrity.to_dict())
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
            "literary_total_before": lit_before,
            "n_revision_items": len(rev_plan.revision_items),
            "revision_items_by_priority": rev_plan.metadata.get("by_priority"),
            "revision_items": [i.to_dict() for i in rev_plan.revision_items],
            "claimed_change_descriptions": (rev_result.claimed_change_descriptions
                                            if isinstance(rev_result, RevisionResult)
                                            else []),
            "revision_effect": (rev_result.revision_effect
                                if isinstance(rev_result, RevisionResult) else None),
            "comparison_after": (comparison_after.summary
                                 if comparison_after is not None else None),
            "literary_total_after": (eval_after.total_score if eval_after else None),
            "stylometric_before": actual.layer_d_stylometric,
            "stylometric_after": (actual_after.layer_d_stylometric
                                  if actual_after is not None else None),
            "content_integrity": (integrity.to_dict()
                                  if isinstance(integrity, ContentIntegrityResult)
                                  else None),
            "decision": decision.to_dict(),
        }

    summary = _build_summary(provider, policy, authors)
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


def _build_summary(provider: LLMProvider, policy: EvaluationPolicy,
                   authors: dict[str, Any]) -> dict[str, Any]:
    inner = getattr(provider, "_inner", None)
    usage = getattr(inner, "usage", {})
    return {
        "stage": "style_feedback_loop_and_literary_evaluation",
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "decision_schema_version": FEEDBACK_DECISION_SCHEMA_VERSION,
        "max_iterations": MAX_ITERATIONS,
        "evaluation_policy": policy.to_dict(),
        "provider": getattr(provider, "provider_id", ""),
        "model": getattr(provider, "model", ""),
        "blind": True,
        "authors": authors,
        "token_usage": dict(usage) if usage else None,
        "cache_hits": getattr(provider, "cache_hits", None),
        "cache_misses": getattr(provider, "cache_misses", None),
    }


# --------------------------------------------------------------------------- #
# 渲染（人类可读）
# --------------------------------------------------------------------------- #
def _render_report(summary: dict[str, Any], authors: dict[str, Any]) -> str:
    lines = [
        "# Weaver Style Engine — Style Feedback Loop + 文学评价报告（Phase 8.2）",
        "",
        f"- provider / model：`{summary.get('provider')}` / `{summary.get('model')}`",
        f"- blind：`{summary.get('blind')}`　max_iterations：`{summary.get('max_iterations')}`",
        f"- evaluation policy：`{json.dumps(summary.get('evaluation_policy'), ensure_ascii=False)}`",
        "",
        "决策为四阶 gate：Revision Effect（确定性零 token）→ Content Integrity（最高）→ "
        "Literary Quality guard → Style Fidelity。Style 与 Literary **分别报告**，绝不"
        "合并成单一加权分；stylometric 距离仅诊断；no_effect 独立于 no_action 与 roll_back。",
        "",
    ]
    for author_id in AUTHOR_IDS:
        a = authors[author_id]
        d = a["decision"]
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
        lines += ["", "### 改写变更说明（自报，best-effort）", ""]
        for ch in a["claimed_change_descriptions"]:
            lines.append(f"- {ch}")
        eff = a.get("revision_effect")
        if eff is not None:
            lines += [
                "",
                "### 改写有效性（确定性，零 LLM）",
                "",
                f"- effect_status：`{eff['effect_status']}`　"
                f"substantive_edit：`{eff['substantive_edit']}`　"
                f"word_change_count：`{eff['word_change_count']}`　"
                f"word_change_ratio：`{eff['word_change_ratio']}`",
                f"- {eff['reason']}",
            ]
        lines += ["", "### 内容完整性检查", ""]
        ci = a.get("content_integrity")
        if ci is None:
            lines.append("- 未运行（改写计划为空或改写失败）。")
        else:
            lines.append(
                f"- passed：`{ci['passed']}`（plot_facts=`{ci['plot_facts_preserved']}` "
                f"characters=`{ci['characters_preserved']}` "
                f"relationships=`{ci['relationships_preserved']}` "
                f"constraints=`{ci['constraints_preserved']}` "
                f"new_events=`{ci['new_major_events']}` "
                f"removed_events=`{ci['removed_major_events']}`）")
            for v in ci["violations"]:
                lines.append(f"  - [{v['severity']}] {v['kind']}: {v['description']}")
        lines += [
            "",
            "### 决策（三阶 gate）",
            "",
            f"```json\n{json.dumps(d, ensure_ascii=False, indent=2)}\n```",
            "",
            f"- 文学评价总分（改写后）：`{a['literary_total_after']}`",
            f"- stylometric 余弦距离：before=`{(a['stylometric_before'] or {}).get('cosine_distance')}`"
            f"　after=`{(a['stylometric_after'] or {}).get('cosine_distance')}`",
            f"- **决策**：`{d['outcome']}` — {d['reason']}",
            "",
        ]
    lines += [
        "---",
        "",
        "> 机器可读产物见 `data/analysis/evaluation_v3/`（actual_profile / "
        "literary_evaluation / revision_plan / revision_result / content_integrity / "
        "revised_actual_profile / revised_literary_evaluation / evaluation_summary.json）。"
        " Phase 8 v1 原始产物见 `data/analysis/evaluation/`，Phase 8.1 见 "
        "`data/analysis/evaluation_v2/`（均未覆盖）。stylometric 距离仅为诊断，从未进入"
        "改写指令或决策。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    summary = run_evaluation()
    for aid in AUTHOR_IDS:
        a = summary["authors"][aid]
        d = a["decision"]
        sf = d["style_fidelity"]
        lq = d["literary_quality"]
        print(f"{aid}: eval_before={a['literary_total_before']} "
              f"eval_after={a['literary_total_after']} "
              f"rev_items={a['n_revision_items']} "
              f"style_dev={sf['high_priority_deviations_before']}->"
              f"{sf['high_priority_deviations_after']} "
              f"lit_drop={lq['drop']} integrity={d['content_integrity_passed']} "
              f"outcome={d['outcome']}")
    print(f"token_usage: {summary['token_usage']}")
    print("artifacts: data/analysis/evaluation_v3/evaluation_summary.json + "
          "evaluation_report.md + {author_id}_{actual_profile,literary_evaluation,"
          "revision_plan,revision_result,content_integrity,revised_actual_profile,"
          "revised_literary_evaluation}.json")


if __name__ == "__main__":
    main()
