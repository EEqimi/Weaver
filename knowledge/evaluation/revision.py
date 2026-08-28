# knowledge/evaluation/revision.py
"""Phase 8 改写计划（确定性优先级 P0–P4）+ 最小编辑改写器（LLM，盲测）。

`build_revision_plan` 是纯函数：把偏差（compare）与文学评价（literary）映射为
优先化的 RevisionItem，指令只含**可解释**自然语言（来自 plan guidance / 评价
weakness / 策略 trigger→operation→effect），绝不含作者名、原始数值、或微观
stylometric 指纹。

`RevisionRewriter` 是最小编辑改写：给定原文 + 改写计划，产出修订正文 + 变更说明
（局部性映射）。铁律：
    - P0（故事情节 / 语义连贯）绝不因低优先级风格编辑而被破坏：改写指令显式禁止
      改动情节 / 事实 / 人物 / 中性 brief 约束（测试强制断言此约束存在）；
    - 盲测：prompt 绝不含作者名或 "write like"/"imitate"/"in the style of"
      （复用 generation/schema.py 的 A/B 泄露守卫，fail-closed）；
    - stylometric 距离只作诊断，绝不生成改写指令（故 P4 恒无 RevisionItem）。
"""
from __future__ import annotations

from typing import Any, Iterable

from ..analysis.base import AnalysisUnavailable, LLMResponseError, parse_json_response
from ..generation.schema import (
    GenerationError, assert_no_author_identity, assert_no_imitation_instruction,
    output_hash,
)
from ..planning.schema import StylePlan
from ..providers.llm_provider import LLMProvider, cache_key
from ..schema.versions import EVALUATION_SCHEMA_VERSION, REVISION_REWRITER_VERSION
from .schema import (
    ComparisonResult, EvalError, EvaluationPolicy, LiteraryEvaluation,
    RevisionItem, RevisionPlan, RevisionResult, priority_rank,
)

ANALYZER_ID = "RevisionRewriter"
ANALYZER_VERSION = REVISION_REWRITER_VERSION

# 文学评价维度 → 改写优先级（确定性映射）。P0 仅 plot_logic（语义/情节连贯）。
_EVAL_DIM_PRIORITY: dict[str, str] = {
    "plot_logic": "P0",
    "characterization": "P1",
    "theme_expression": "P1",
    "pacing": "P1",
    "language_texture": "P3",
    "emotional_resonance": "P3",
}
_EVAL_DIM_CATEGORY: dict[str, str] = {
    "plot_logic": "story_coherence",
    "characterization": "narrative",
    "theme_expression": "narrative",
    "pacing": "narrative",
    "language_texture": "language",
    "emotional_resonance": "language",
}

# 低于此分的文学维度产生一条改写项（1–10 量表，5 = 中位）。统一从 EvaluationPolicy 取，
# 不散落硬编码常数（spec §二 STEP 2）。
DEFAULT_WEAK_SCORE_THRESHOLD = EvaluationPolicy().weak_score_threshold


# --------------------------------------------------------------------------- #
# 改写计划（纯函数）
# --------------------------------------------------------------------------- #
def _evaluation_items(evaluation: LiteraryEvaluation | None,
                      threshold: float) -> list[RevisionItem]:
    if evaluation is None:
        return []
    out: list[RevisionItem] = []
    for dim_id, dim in evaluation.dimensions.items():
        if dim.score >= threshold:
            continue
        priority = _EVAL_DIM_PRIORITY.get(dim_id, "P3")
        category = _EVAL_DIM_CATEGORY.get(dim_id, "language")
        out.append(RevisionItem(
            priority=priority, category=category, target=dim_id,
            instruction=f"Strengthen this dimension: {dim.weakness}",
            reason=f"literary evaluation: {dim.label} below threshold",
        ))
    return out


def _narrative_items(comparison: ComparisonResult,
                     plan: StylePlan) -> list[RevisionItem]:
    guidance = {nc.field: nc.guidance for nc in plan.narrative_controls}
    out: list[RevisionItem] = []
    for d in comparison.narrative_deviations:
        if d.status != "off_target":
            continue
        instr = guidance.get(d.field) or f"Adjust narration toward {d.target_value}."
        out.append(RevisionItem(
            priority="P1", category="narrative", target=d.field,
            instruction=instr,
            reason=f"measured {d.actual_value!r}, target {d.target_value!r}",
        ))
    return out


def _strategy_items(comparison: ComparisonResult,
                    plan: StylePlan) -> list[RevisionItem]:
    controls = {s.canonical_strategy_id: s for s in plan.strategy_controls}
    out: list[RevisionItem] = []
    for c in comparison.strategy_coverage:
        if not c.active or c.matched:
            continue
        s = controls.get(c.strategy_id)
        if s is None:
            continue
        out.append(RevisionItem(
            priority="P2", category="strategy", target=c.strategy_id,
            instruction=(
                f"When the scene involves: {s.trigger}. Then {s.operation} "
                f"to {s.effect}."),
            reason="active strategy not detected in the passage",
        ))
    return out


def _language_items(comparison: ComparisonResult,
                    plan: StylePlan) -> list[RevisionItem]:
    guidance = {c.feature_id: c.guidance for c in plan.language_controls}
    out: list[RevisionItem] = []
    for d in comparison.language_deviations:
        if d.status not in ("above", "below"):
            continue
        instr = guidance.get(d.feature_id) or f"Move this feature toward {d.target_band}."
        out.append(RevisionItem(
            priority="P3", category="language", target=d.feature_id,
            instruction=instr,
            reason=f"measured {d.actual_band}, target {d.target_band}",
        ))
    return out


def build_revision_plan(comparison: ComparisonResult, plan: StylePlan,
                        evaluation: LiteraryEvaluation | None = None,
                        *, weak_score_threshold: float = DEFAULT_WEAK_SCORE_THRESHOLD
                        ) -> RevisionPlan:
    """偏差 + 文学评价 → 优先化改写计划（P0→P4 有序，纯函数）。

    P4（stylometric）恒无改写指令：stylometric 距离只进入 comparison.summary 作诊断。
    """
    items: list[RevisionItem] = []
    items += _evaluation_items(evaluation, weak_score_threshold)
    items += _narrative_items(comparison, plan)
    items += _strategy_items(comparison, plan)
    items += _language_items(comparison, plan)

    # 稳定排序：先优先级（P0 最前），同优先级按 target 字典序。
    items.sort(key=lambda i: (priority_rank(i.priority), i.target))

    metadata = {
        "weak_score_threshold": weak_score_threshold,
        "n_items": len(items),
        "by_priority": {
            p: sum(1 for i in items if i.priority == p)
            for p in ("P0", "P1", "P2", "P3", "P4")
        },
        "stylometric": comparison.summary.get("stylometric"),
        "stylometric_note": "stylometric 距离仅诊断，绝不生成改写指令",
    }
    return RevisionPlan(
        schema_version=EVALUATION_SCHEMA_VERSION,
        author_id=plan.author_id,
        passage_id=comparison.passage_id,
        style_plan_id=plan.style_plan_id,
        revision_items=items,
        metadata=metadata,
    )


# --------------------------------------------------------------------------- #
# 最小编辑改写器（LLM，盲测）
# --------------------------------------------------------------------------- #
def _render_items(items: Iterable[RevisionItem]) -> str:
    lines = [f"- [{i.priority}] {i.instruction}" for i in items]
    return "\n".join(lines) if lines else "(none)"


def _build_system_prompt(items: list[RevisionItem]) -> str:
    return (
        "You are a careful literary editor revising a single prose passage. "
        "Make only the minimal edits required by the revision list below.\n"
        "Hard constraints (do NOT violate them under any circumstances):\n"
        "- Do NOT change the plot, story facts, characters, relationships, or the "
        "scene's premise, and do NOT add or remove named characters or events.\n"
        "- Do NOT copy or reproduce wording from any source text; write original "
        "sentences.\n"
        "- Do NOT name any author and do NOT try to reproduce a particular author's "
        "manner.\n"
        "- Apply only the listed revisions; leave everything else untouched.\n"
        "Return ONLY a JSON object (no prose, no markdown fences) with exactly:\n"
        '  "revised_text": the full revised passage,\n'
        '  "change_descriptions": an array of short strings, each describing one '
        "change you made and where in the passage it occurs.\n\n"
        "REVISION LIST (in priority order):\n"
        f"{_render_items(items)}"
    )


class RevisionRewriter:
    """最小编辑改写器（LLM，盲测，P0 保护强制入 prompt）。"""

    def __init__(self, provider: LLMProvider, blind: bool = True):
        self._provider = provider
        self.blind = blind

    def rewrite(self, original_text: str, plan: RevisionPlan,
                author_names: Iterable[str] = ()) -> RevisionResult | AnalysisUnavailable:
        if not self._provider.is_configured():
            return AnalysisUnavailable("revision", ANALYZER_ID, ANALYZER_VERSION,
                                       "未配置 LLM provider")

        # 空改写计划 → 确定性短路，绝不烧 token（原样返回，变更说明为空）。
        if not plan.revision_items:
            return RevisionResult(
                schema_version=EVALUATION_SCHEMA_VERSION,
                author_id=plan.author_id,
                passage_id=plan.passage_id,
                original_passage_hash=output_hash(original_text),
                revised_passage_hash=output_hash(original_text),
                revised_text=original_text,
                change_descriptions=[],
                revision_items_applied=[],
                blind=self.blind,
                rewriter_version=ANALYZER_VERSION,
            )

        system = _build_system_prompt(plan.revision_items)
        user = f'PASSAGE:\n"""{original_text}"""'

        # A/B 泄露守卫（fail-closed）：风格/改写指令不含模仿令牌；全文不含作者名。
        try:
            assert_no_imitation_instruction(system)
            assert_no_author_identity(system + "\n" + user, author_names)
        except GenerationError as e:
            raise EvalError(f"revision prompt 泄露守卫触发: {e}") from e

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        key = cache_key(
            text=original_text, analyzer_id=ANALYZER_ID,
            analyzer_version=ANALYZER_VERSION, schema_version=EVALUATION_SCHEMA_VERSION,
            model=self._provider.model, provider_id=self._provider.provider_id,
            prompt_name=f"revision:blind={self.blind}:n_items={len(plan.revision_items)}",
        )
        raw = self._provider.complete(messages, cache_hint=key)
        data = parse_json_response(raw)
        return self._to_result(plan, original_text, data)

    def _to_result(self, plan: RevisionPlan, original_text: str,
                   data: dict) -> RevisionResult:
        revised = data.get("revised_text")
        if not isinstance(revised, str) or not revised.strip():
            raise LLMResponseError("revision 的 revised_text 必须是非空字符串")
        descs = data.get("change_descriptions", [])
        if not isinstance(descs, list) or not all(isinstance(x, str) for x in descs):
            raise LLMResponseError("revision 的 change_descriptions 必须是字符串列表")
        return RevisionResult(
            schema_version=EVALUATION_SCHEMA_VERSION,
            author_id=plan.author_id,
            passage_id=plan.passage_id,
            original_passage_hash=output_hash(original_text),
            revised_passage_hash=output_hash(revised),
            revised_text=revised,
            change_descriptions=list(descs),
            revision_items_applied=[f"{i.priority}:{i.target}" for i in plan.revision_items],
            blind=self.blind,
            rewriter_version=ANALYZER_VERSION,
        )
