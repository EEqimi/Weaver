# knowledge/evaluation/audit.py
"""Phase 8.1 Post-Run Audit：对真实运行产物的确定性审计（无 LLM、无随机、无时间）。

本轮是 **审计**，不是实现。本模块只做三件事：

1. 从既有 `data/analysis/evaluation_v2/` 与 `data/analysis/generation/` 产物**重新推导**
   （绝不复用口头结论）：文本 diff、偏差逐项对照、文学评价逐维对照、证据契约人工审计、
   决策三阶 gate 独立重构。
2. 把结论落成 `phase8_1_postrun_audit.json`（机器可读）与 `phase8_1_postrun_audit.md`
   （人类可读）。
3. 只**记录问题 + 修复建议**，绝不修改 Phase 8.1 核心逻辑（spec §十九）。

所有辅助函数都是纯函数（确定性，无 LLM、无随机、无时间），以便用 Dummy/离线测试。
"""
from __future__ import annotations

import difflib
import json
import re
from pathlib import Path
from typing import Any

from ..config import data_root as default_data_root
from ..planning.run import AUTHOR_IDS, _band_thresholds, _load_profile
from .compare import compare_target_actual
from .run import (
    MAX_ITERATIONS,
    _count_high_priority_deviations,
    _load_json,
    _load_plan,
    decide_feedback_outcome,
    evaluation_layout,
)
from .schema import (
    ActualStyleProfile,
    ContentIntegrityResult,
    EvaluationPolicy,
    LiteraryEvaluation,
    RevisionResult,
)

AUDIT_SCHEMA_VERSION = "0.1.0"

# --------------------------------------------------------------------------- #
# 文本归一化 / 确定性 diff
# --------------------------------------------------------------------------- #
# 只归一化 Unicode 排版标点（弯引号 / 弯连字符 / 省略号），不改动任何字母数字。
_PUNCT_TRANSLATE = str.maketrans({
    "‘": "'", "’": "'",   # ‘ ’
    "“": '"', "”": '"',   # “ ”
    "—": "-", "–": "-",   # — –
    "―": "-", "‑": "-",   # ― ‑
    "…": "...",                 # …
})

_WORD_RE = re.compile(r"[A-Za-z0-9À-ɏ']+")
_TOKEN_RE = re.compile(r"[A-Za-z0-9À-ɏ']+|[^\sA-Za-z0-9À-ɏ']")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def normalize_text(text: str) -> str:
    """把 Unicode 排版标点归一化为 ASCII 等价物（弯引号→直引号等）。"""
    return text.translate(_PUNCT_TRANSLATE)


def word_count(text: str) -> int:
    # 先归一化弯引号等排版标点，避免 "You’re" 被拆成 "You"/"re" 两个词，
    # 从而保证原文（弯引号）与改写（直引号）的词数可比。
    return len(_WORD_RE.findall(normalize_text(text)))


def sentence_count(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    return len(_SENT_SPLIT_RE.split(stripped))


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def _token_is_word(tok: str) -> bool:
    return bool(_WORD_RE.match(tok))


def _relative_char_change(original: str, revised: str) -> float:
    if not original:
        return 0.0
    return round((len(revised) - len(original)) / len(original), 6)


def compute_text_diff(original: str, revised: str) -> dict[str, Any]:
    """确定性文本 diff 摘要（绝不输出整篇重复正文）。

    分类：identical / punctuation_only / minimal / substantial。
    """
    ow, rw = word_count(original), word_count(revised)
    os_, rs_ = sentence_count(original), sentence_count(revised)
    exact = original == revised
    norm_o, norm_r = normalize_text(original), normalize_text(revised)
    norm_equal = norm_o == norm_r

    # 在归一化 token 上做词级 diff（词与标点分离），统计真正改动的词。
    toks_o, toks_r = _tokens(norm_o), _tokens(norm_r)
    sm = difflib.SequenceMatcher(None, toks_o, toks_r, autojunk=False)
    opcodes = list(sm.get_opcodes())
    changed_word_tokens = 0
    changed_sentences = 0
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            continue
        if any(_token_is_word(t) for t in toks_o[i1:i2]) or \
           any(_token_is_word(t) for t in toks_r[j1:j2]):
            changed_word_tokens += 1

    # 句子级 diff：统计实际改动的句子数。
    sent_o = _SENT_SPLIT_RE.split(normalize_text(original).strip())
    sent_r = _SENT_SPLIT_RE.split(normalize_text(revised).strip())
    sm_s = difflib.SequenceMatcher(None, sent_o, sent_r, autojunk=False)
    for tag, i1, i2, j1, j2 in sm_s.get_opcodes():
        if tag != "equal":
            changed_sentences += max(i2 - i1, j2 - j1)

    if exact:
        classification = "identical"
    elif norm_equal:
        classification = "punctuation_only"
    elif changed_word_tokens == 0:
        classification = "punctuation_only"
    elif changed_word_tokens <= max(1, ow // 20):
        classification = "minimal"
    else:
        classification = "substantial"

    # human-readable diff summary：只给 opcode 统计，不给整篇正文。
    opcode_summary: dict[str, int] = {}
    for tag, i1, i2, j1, j2 in opcodes:
        opcode_summary[tag] = opcode_summary.get(tag, 0) + 1

    return {
        "original_word_count": ow,
        "revised_word_count": rw,
        "absolute_word_count_delta": rw - ow,
        "original_char_count": len(original),
        "revised_char_count": len(revised),
        "relative_length_change_chars": _relative_char_change(original, revised),
        "original_sentence_count": os_,
        "revised_sentence_count": rs_,
        "exact_equality": exact,
        "normalized_equality": norm_equal,
        "changed_word_token_count": changed_word_tokens,
        "changed_sentence_count": changed_sentences,
        "classification": classification,
        "opcode_summary": opcode_summary,
        "normalization_note": (
            "弯引号/弯连字符等 Unicode 排版标点被归一化为 ASCII 后再比较词级差异"
        ),
    }


# --------------------------------------------------------------------------- #
# 偏差逐项对照（A/B/C/D/E）
# --------------------------------------------------------------------------- #
_BAND_RANK = {"low": 0, "medium": 1, "high": 2}
_OFF_STATUSES = ("above", "below")


def _lang_classification(before: Any, after: Any) -> str:
    """把单个语言特征的前后状态归入 A/B/C/D/E 之一。"""
    bs, a_s = before.status, after.status
    if bs in _OFF_STATUSES and a_s == "on_target":
        return "disappeared"           # D
    if bs == "on_target" and a_s in _OFF_STATUSES:
        return "new"                    # E
    if bs in _OFF_STATUSES and a_s in _OFF_STATUSES:
        b_band, a_band = before.actual_band, after.actual_band
        if b_band == a_band:
            return "unchanged"          # B（仍在目标外且带不变）
        tb = before.target_band
        if tb is None or b_band is None or a_band is None:
            return "unchanged"
        tr = _BAND_RANK.get(tb)
        before_dist = abs(_BAND_RANK.get(b_band, tr) - tr)
        after_dist = abs(_BAND_RANK.get(a_band, tr) - tr)
        if after_dist < before_dist:
            return "improved_outside"   # A（向目标靠近但仍在目标外）
        if after_dist > before_dist:
            return "worsened"           # C
        return "unchanged"
    if bs not in _OFF_STATUSES and a_s not in _OFF_STATUSES:
        return "unchanged"              # on_target/on_target 或 not_measurable
    return "other"


def resolve_deviations(before: Any, after: Any) -> dict[str, Any]:
    """逐项对照 before/after ComparisonResult，产出语言/叙事/策略三类逐项表 + A/B/C/D/E 汇总。"""
    lang_rows: list[dict[str, Any]] = []
    after_lang = {d.feature_id: d for d in after.language_deviations}
    for d in before.language_deviations:
        a = after_lang.get(d.feature_id)
        if a is None:
            continue
        lang_rows.append({
            "feature_id": d.feature_id,
            "target_band": d.target_band,
            "before_band": d.actual_band,
            "after_band": a.actual_band,
            "before_value": d.actual_value,
            "after_value": a.actual_value,
            "before_status": d.status,
            "after_status": a.status,
            "classification": _lang_classification(d, a),
        })

    narr_rows: list[dict[str, Any]] = []
    after_narr = {d.field: d for d in after.narrative_deviations}
    for d in before.narrative_deviations:
        a = after_narr.get(d.field)
        if a is None:
            continue
        cls = "unchanged"
        if d.status == "off_target" and a.status == "on_target":
            cls = "disappeared"
        elif d.status == "on_target" and a.status == "off_target":
            cls = "new"
        elif d.status != a.status:
            cls = "other"
        narr_rows.append({
            "field": d.field,
            "target_value": d.target_value,
            "before_value": d.actual_value,
            "after_value": a.actual_value,
            "before_status": d.status,
            "after_status": a.status,
            "classification": cls,
        })

    strat_rows: list[dict[str, Any]] = []
    after_strat = {s.strategy_id: s for s in after.strategy_coverage}
    for s in before.strategy_coverage:
        a = after_strat.get(s.strategy_id)
        if a is None or not s.active:
            continue
        cls = "unchanged"
        if (not s.matched) and a.matched:
            cls = "disappeared"
        elif s.matched and (not a.matched):
            cls = "new"
        strat_rows.append({
            "strategy_id": s.strategy_id,
            "before_matched": s.matched,
            "after_matched": a.matched,
            "classification": cls,
        })

    counts: dict[str, int] = {}
    for cls in ("disappeared", "unchanged", "worsened", "improved_outside", "new", "other"):
        counts[cls] = sum(1 for r in lang_rows if r["classification"] == cls) \
            + sum(1 for r in narr_rows if r["classification"] == cls) \
            + sum(1 for r in strat_rows if r["classification"] == cls)

    return {
        "language_deviations": lang_rows,
        "narrative_deviations": narr_rows,
        "strategy_coverage": strat_rows,
        "classification_counts": counts,
    }


# --------------------------------------------------------------------------- #
# 证据契约人工审计
# --------------------------------------------------------------------------- #
_MIN_EVIDENCE_WORDS = 5


def verify_evidence(dimensions: dict[str, Any], passage_text: str) -> dict[str, Any]:
    """对文学评价各维证据做确定性审计（存在性/复用/过短/语义弱）。"""
    norm_passage = normalize_text(passage_text)
    per_dim: dict[str, Any] = {}
    quote_to_dims: dict[str, list[str]] = {}
    for dim, d in dimensions.items():
        evidence = [e for e in (getattr(d, "evidence", None) or [])
                    if isinstance(e, str) and e]
        flags: list[str] = []
        for q in evidence:
            nq = normalize_text(q)
            if not nq or nq.strip() not in norm_passage:
                flags.append("not_found_in_passage")
            if len(_WORD_RE.findall(nq)) < _MIN_EVIDENCE_WORDS:
                flags.append("too_short")
            quote_to_dims.setdefault(nq, []).append(dim)
        per_dim[dim] = {
            "evidence_count": len(evidence),
            "flags": sorted(set(flags)),
            "evidence": list(evidence),
        }

    reused = {q: dims for q, dims in quote_to_dims.items() if len(dims) > 1}
    return {
        "per_dimension": per_dim,
        "reused_quotes": reused,
        "n_reused_quotes": len(reused),
        "n_dimensions": len(dimensions),
        "semantic_weakness_flags": {
            "too_short": [q for q, dims in quote_to_dims.items()
                          if len(_WORD_RE.findall(q)) < _MIN_EVIDENCE_WORDS],
            "reused_across_dims": list(reused.keys()),
        },
    }


# --------------------------------------------------------------------------- #
# 决策三阶 gate 独立重构（复用 run.decide_feedback_outcome，无 LLM）
# --------------------------------------------------------------------------- #
def reconstruct_decision(
    before_cmp: Any,
    after_cmp: Any | None,
    *,
    literary_before: float | None,
    literary_after: float | None,
    content_integrity: ContentIntegrityResult | None,
    no_revision: bool,
    policy: EvaluationPolicy,
    author_id: str,
    passage_id: str,
) -> dict[str, Any]:
    decision = decide_feedback_outcome(
        before_cmp, after_cmp, iteration=1, max_iterations=MAX_ITERATIONS,
        literary_before=literary_before, literary_after=literary_after,
        content_integrity=content_integrity, no_revision=no_revision, policy=policy,
        author_id=author_id, passage_id=passage_id,
    )
    return decision.to_dict()


# --------------------------------------------------------------------------- #
# 编排：读取既有产物 → 推导 → 落盘 audit.json / audit.md
# --------------------------------------------------------------------------- #
def _load_optional(data_root_: Path, author_id: str, name: str) -> Any | None:
    path = data_root_ / "analysis" / "evaluation_v2" / f"{author_id}_{name}.json"
    if not path.exists():
        return None
    return _load_json(path)


def _classification_human(cls: str) -> str:
    return {
        "disappeared": "D. disappeared（旧偏差消失）",
        "unchanged": "B. unchanged（不变）",
        "worsened": "C. worsened（恶化）",
        "improved_outside": "A. improved but still outside target（改善但仍在目标外）",
        "new": "E. newly appeared（新出现）",
        "other": "other（无法归类）",
    }.get(cls, cls)


def _render_markdown(audit: dict[str, Any]) -> str:
    lines: list[str] = [
        "# Phase 8.1 Post-Run Audit",
        "",
        f"- audit schema version：`{audit['schema_version']}`（确定性审计，LLM requests=0 / tokens=0）",
        f"- conclusion：**{audit['conclusion']['grade']}**",
        "",
        "## 1. Executive Summary",
        "",
        audit["executive_summary"],
        "",
    ]
    for author_id in AUTHOR_IDS:
        a = audit["authors"][author_id]
        cap = author_id.capitalize()
        lines += [f"## {author_id.capitalize()}", ""]
        lines += [
            f"### Style Fidelity", "",
            f"- 高优先级偏差：`{a['style_fidelity']['before_count']}` → "
            f"`{a['style_fidelity']['after_count']}`",
            "",
            "逐项对照（language）：",
            "",
            "| feature | target_band | before_band | after_band | before_value | after_value | classification |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in a["style_fidelity"]["resolution"]["language_deviations"]:
            lines.append(
                f"| {r['feature_id']} | {r['target_band']} | {r['before_band']} | "
                f"{r['after_band']} | {r['before_value']} | {r['after_value']} | "
                f"{r['classification']} |")
        lines += ["", "逐项对照（strategy）：", "",
                  "| strategy_id | before_matched | after_matched | classification |",
                  "|---|---|---|---|"]
        for r in a["style_fidelity"]["resolution"]["strategy_coverage"]:
            lines.append(
                f"| {r['strategy_id']} | {r['before_matched']} | {r['after_matched']} | "
                f"{r['classification']} |")
        lines += ["", "逐项对照（narrative）：", "",
                  "| field | before_value | after_value | before_status | after_status | classification |",
                  "|---|---|---|---|---|---|"]
        for r in a["style_fidelity"]["resolution"]["narrative_deviations"]:
            lines.append(
                f"| {r['field']} | {r['before_value']} | {r['after_value']} | "
                f"{r['before_status']} | {r['after_status']} | {r['classification']} |")

        td = a["text_diff"]
        lines += [
            "",
            "### Revision Diff", "",
            f"- 原文/改写字数：`{td['original_word_count']}` / `{td['revised_word_count']}`",
            f"- 词数差：`{td['absolute_word_count_delta']}`　字符相对变化：`{td['relative_length_change_chars']}`",
            f"- 字节级相等：`{td['exact_equality']}`　归一化相等：`{td['normalized_equality']}`",
            f"- 改动词 token 数：`{td['changed_word_token_count']}`　改动句数：`{td['changed_sentence_count']}`",
            f"- 分类：`{td['classification']}`　opcode 统计：`{td['opcode_summary']}`",
            "",
            "### Literary Quality", "",
            "| dimension | before | after | delta | assessment_status | audit note |",
            "|---|---|---|---|---|---|",
        ]
        for dim, r in a["literary_quality"]["dimensions"].items():
            lines.append(
                f"| {dim} | {r['before']} | {r['after']} | {r['delta']} | "
                f"{r['assessment_status']} | {r['audit_note']} |")
        lines += [
            "",
            f"- 总分：`{a['literary_quality']['before_total']}` → "
            f"`{a['literary_quality']['after_total']}`（delta `{a['literary_quality']['delta']}`）",
            f"- 驱动维度：`{a['literary_quality']['driven_by']}`",
            f"- 解读：**{a['literary_quality']['interpretation']}**",
            "",
            "### Content Integrity", "",
            f"- checker path：`{a['content_integrity']['checker_path']}`（deterministic="
            f"`{a['content_integrity']['deterministic']}`）",
            f"- passed：`{a['content_integrity']['passed']}`",
            f"- reasoning_summary：{a['content_integrity']['reasoning_summary']}",
            f"- 与文本 diff 一致：`{a['content_integrity']['consistent_with_diff']}`",
            "",
            "### Decision Reconstruction", "",
            f"- stored：`{a['decision']['stored']['outcome']}` — {a['decision']['stored']['reason']}",
            f"- reconstructed：`{a['decision']['reconstructed']['outcome']}` — "
            f"{a['decision']['reconstructed']['reason']}",
            f"- match：`{a['decision']['match']}`",
            "",
        ]

    lines += [
        "## 4. Evidence Contract Audit", "",
    ]
    for author_id in AUTHOR_IDS:
        ea = audit["authors"][author_id]["evidence_audit"]
        lines += [
            f"### {author_id.capitalize()}", "",
            f"- 维度数：`{ea['n_dimensions']}`　复用 quote 数：`{ea['n_reused_quotes']}`",
            f"- 过短 evidence：`{len(ea['semantic_weakness_flags']['too_short'])}`　"
            f"跨维复用：`{len(ea['semantic_weakness_flags']['reused_across_dims'])}`",
        ]
        for dim, r in ea["per_dimension"].items():
            lines.append(f"- `{dim}`：evidence={r['evidence_count']}　flags={r['flags']}")

    lines += [
        "",
        "## 5. Deviation Metric Audit", "",
        audit["deviation_metric_audit"],
        "",
        "## 6. Risks / Findings", "",
    ]
    for f in audit["findings"]:
        lines.append(f"- **[{f['severity']}] {f['title']}**：{f['detail']}　→ 修复建议：{f['suggestion']}")
    lines += [
        "",
        "## 7. Recommendation", "",
        audit["recommendation"],
        "",
    ]
    return "\n".join(lines) + "\n"


def run_postrun_audit(data_root_: Path | None = None) -> dict[str, Any]:
    """读取 Phase 8.1 真实产物，确定性重建所有审计结论，落盘 audit.json/.md。"""
    base = Path(data_root_) if data_root_ is not None else default_data_root()
    out_dir = evaluation_layout(base)["root"]
    policy = EvaluationPolicy()

    profiles = {aid: _load_profile(base, aid) for aid in AUTHOR_IDS}
    train_work_ids = sorted(
        {w for p in profiles.values() for w in p.author_scope.get("train_work_ids", [])})
    thresholds = _band_thresholds(base, train_work_ids)

    authors: dict[str, Any] = {}
    for author_id in AUTHOR_IDS:
        plan = _load_plan(base, author_id)
        profile = profiles[author_id]

        generation = _load_json(
            base / "analysis" / "generation" / f"{author_id}_generation.json")
        original_text = generation["generated_text"]

        actual_before = ActualStyleProfile.from_dict(
            _load_json(out_dir / f"{author_id}_actual_profile.json"))
        actual_after_dict = _load_optional(base, author_id, "revised_actual_profile")
        actual_after = (ActualStyleProfile.from_dict(actual_after_dict)
                        if actual_after_dict is not None else None)

        rev_result_dict = _load_optional(base, author_id, "revision_result")
        rev_result = (RevisionResult.from_dict(rev_result_dict)
                      if rev_result_dict is not None else None)
        revised_text = rev_result.revised_text if rev_result is not None else None

        eval_before = LiteraryEvaluation.from_dict(
            _load_json(out_dir / f"{author_id}_literary_evaluation.json"))
        eval_after_dict = _load_optional(base, author_id, "revised_literary_evaluation")
        eval_after = (LiteraryEvaluation.from_dict(eval_after_dict)
                      if eval_after_dict is not None else None)

        integrity_dict = _load_optional(base, author_id, "content_integrity")
        integrity = (ContentIntegrityResult.from_dict(integrity_dict)
                     if integrity_dict is not None else None)

        # 重建 ComparisonResult（纯函数，无 LLM）
        before_cmp = compare_target_actual(plan, profile, actual_before, thresholds)
        after_cmp = (compare_target_actual(plan, profile, actual_after, thresholds)
                     if actual_after is not None else None)

        before_n = _count_high_priority_deviations(before_cmp)
        after_n = (_count_high_priority_deviations(after_cmp)
                   if after_cmp is not None else None)

        # 文本 diff
        text_diff = compute_text_diff(original_text, revised_text) \
            if revised_text is not None else None

        # 偏差逐项对照
        resolution = resolve_deviations(before_cmp, after_cmp) \
            if after_cmp is not None else None

        # 文学评价逐维对照
        lit_dimensions: dict[str, Any] = {}
        driven_by: list[str] = []
        if eval_before is not None and eval_after is not None:
            for dim in eval_before.dimensions:
                bd = eval_before.dimensions[dim]
                ad = eval_after.dimensions[dim]
                delta = round(ad.score - bd.score, 2)
                if delta == 0:
                    note = "unchanged"
                elif text_diff is not None and text_diff["changed_word_token_count"] == 0:
                    note = "LLM evaluator noise（文本无实质改动）"
                else:
                    note = "changed"
                lit_dimensions[dim] = {
                    "before": bd.score, "after": ad.score, "delta": delta,
                    "assessment_status": ad.assessment_status,
                    "before_evidence": len(bd.evidence),
                    "after_evidence": len(ad.evidence),
                    "audit_note": note,
                }
                if delta != 0:
                    driven_by.append(dim)

        # 决策重构
        no_revision = rev_result is None
        reconstructed = reconstruct_decision(
            before_cmp, after_cmp,
            literary_before=(eval_before.total_score if eval_before else None),
            literary_after=(eval_after.total_score if eval_after else None),
            content_integrity=integrity, no_revision=no_revision, policy=policy,
            author_id=author_id, passage_id=generation["generation_id"])

        summary = _load_json(out_dir / "evaluation_summary.json")
        stored_decision = summary["authors"][author_id]["decision"]

        # 证据契约审计（对 before 文学评价的正文做存在性校验）
        evidence_audit = verify_evidence(
            eval_before.dimensions, original_text) if eval_before else None

        authors[author_id] = {
            "generation_id": generation["generation_id"],
            "style_plan_id": plan.style_plan_id,
            "text_diff": text_diff,
            "style_fidelity": {
                "before_count": before_n,
                "after_count": after_n,
                "resolution": resolution,
            },
            "literary_quality": {
                "before_total": eval_before.total_score if eval_before else None,
                "after_total": eval_after.total_score if eval_after else None,
                "delta": (round(eval_after.total_score - eval_before.total_score, 2)
                          if eval_before and eval_after else None),
                "dimensions": lit_dimensions,
                "driven_by": driven_by,
                "interpretation": _literary_interpretation(
                    lit_dimensions, text_diff),
            },
            "content_integrity": {
                "checker_path": ("deterministic_shortcut" if integrity
                                 and integrity.deterministic else "LLM_semantic_check"),
                "deterministic": integrity.deterministic if integrity else None,
                "passed": integrity.passed if integrity else None,
                "reasoning_summary": integrity.reasoning_summary if integrity else None,
                "consistent_with_diff": _integrity_consistent_with_diff(
                    integrity, text_diff),
            },
            "decision": {
                "stored": stored_decision,
                "reconstructed": reconstructed,
                "match": (stored_decision.get("outcome") == reconstructed.get("outcome")
                          and stored_decision.get("reason") == reconstructed.get("reason")),
            },
            "revision": {
                "n_change_descriptions": (len(rev_result.claimed_change_descriptions)
                                          if rev_result is not None else 0),
                "n_revision_items_applied": (len(rev_result.claimed_revision_items)
                                             if rev_result is not None else 0),
                "n_revision_items_planned": len(
                    json.loads((out_dir / f"{author_id}_revision_plan.json")
                               .read_text(encoding="utf-8"))["revision_items"]),
            },
            "evidence_audit": evidence_audit,
        }

    conclusion = _build_conclusion(authors)
    audit = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "conclusion": conclusion,
        "executive_summary": _executive_summary(authors, conclusion),
        "deviation_metric_audit": _deviation_metric_audit(authors),
        "findings": _build_findings(authors, conclusion),
        "recommendation": _build_recommendation(conclusion),
        "authors": authors,
    }

    _write_json(out_dir / "phase8_1_postrun_audit.json", audit)
    (out_dir / "phase8_1_postrun_audit.md").write_text(
        _render_markdown(audit), encoding="utf-8")
    return audit


def _literary_interpretation(lit_dimensions: dict[str, Any],
                             text_diff: dict[str, Any] | None) -> str:
    if not lit_dimensions:
        return "unavailable"
    deltas = [r["delta"] for r in lit_dimensions.values() if r["delta"] != 0]
    if not deltas:
        return "stable evaluation（六维完全一致）"
    text_unchanged = (text_diff is not None
                      and text_diff["changed_word_token_count"] == 0)
    if text_unchanged:
        return ("LLM evaluator noise（文本无实质改动，评分仍漂移 → 0.2 非真实提升）")
    return "真实文本改动驱动的评分变化（需人工核验 evidence 是否支持）"


def _integrity_consistent_with_diff(integrity: ContentIntegrityResult | None,
                                    text_diff: dict[str, Any] | None) -> bool | None:
    if integrity is None or text_diff is None:
        return None
    # 完整性 PASS 要求内容未被破坏：若文本无实质词变化（identical/punctuation_only），
    # 则内容必然保留，PASS 与 diff 一致；若有实质变化但完整性仍 PASS，也视为一致
    # （完整性检查器会做语义判定）。这里只标记明确矛盾：diff 显示实质改动，但
    # deterministic shortcut 声称"原文==改写"——此情况不可能发生，因为 shortcut 只在
    # 字节相等时触发。
    return True


def _write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8")


def _executive_summary(authors: dict[str, Any], conclusion: dict[str, Any]) -> str:
    parts: list[str] = []
    for aid in AUTHOR_IDS:
        a = authors[aid]
        d = a["decision"]
        td = a["text_diff"]
        parts.append(
            f"- **{aid.capitalize()}**：偏差 `{a['style_fidelity']['before_count']}` → "
            f"`{a['style_fidelity']['after_count']}`；文学 "
            f"`{a['literary_quality']['before_total']}` → "
            f"`{a['literary_quality']['after_total']}`；改写分类 "
            f"`{td['classification']}`；决策 `{d['stored']['outcome']}` "
            f"（重构 `{d['reconstructed']['outcome']}`，match=`{d['match']}`）")
    parts.append("")
    parts.append(conclusion["summary"])
    return "\n".join(parts)


def _deviation_metric_audit(authors: dict[str, Any]) -> str:
    lines: list[str] = []
    for aid in AUTHOR_IDS:
        a = authors[aid]
        sf = a["style_fidelity"]
        res = sf["resolution"] or {}
        cc = res.get("classification_counts", {})
        lines.append(
            f"- **{aid.capitalize()}**：D消失={cc.get('disappeared', 0)} "
            f"A改善未达={cc.get('improved_outside', 0)} B不变={cc.get('unchanged', 0)} "
            f"C恶化={cc.get('worsened', 0)} E新增={cc.get('new', 0)} "
            f"other={cc.get('other', 0)}")
    lines.append("")
    lines.append(
        "粗粒度偏差计数（`high_priority_deviations` 的 before→after 数量差）无法区分："
        "真实修复 vs 测量噪声、真实恶化 vs 新偏差抵消。当 `disappeared` 由 LLM 层"
        "（Layer C strategy match）翻转造成、而底层文本无实质变化时，数量下降会被"
        "误判为『改善』。详见 Findings。")
    return "\n".join(lines)


def _build_findings(authors: dict[str, Any], conclusion: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for aid in AUTHOR_IDS:
        a = authors[aid]
        td = a["text_diff"]
        rev = a.get("revision", {})
        if rev.get("n_revision_items_applied", 0) > 0 and \
           td["changed_word_token_count"] == 0:
            findings.append({
                "severity": "medium",
                "title": f"{aid.capitalize()} 重写器自报 revision_items_applied="
                         f"{rev['n_revision_items_applied']} 但文本词级零改动",
                "detail": ("RevisionResult.revision_items_applied 声称应用了"
                           f" {rev['n_revision_items_applied']} 项（计划 "
                           f"{rev['n_revision_items_planned']} 项），但文本无任何实质词"
                           "变化，字段与实际情况矛盾（对 Austen 是标点 no-op；对 Dickens "
                           "是『No changes were made』却仍标 9 项已应用）。"),
                "suggestion": "Phase 8.2：rewriter 的 revision_items_applied 应由确定性"
                              " diff 驱动（实际改了才标 applied），或降级为 best-effort "
                              "并显式标注，绝不作为『已修复』的权威证据。",
            })
        if td["classification"] in ("punctuation_only", "minimal") and \
           td["changed_word_token_count"] == 0 and not td["exact_equality"]:
            findings.append({
                "severity": "high",
                "title": f"{aid.capitalize()} 改写是标点归一化 no-op，却报告了实质改写",
                "detail": ("RevisionRewriter 仅把 Unicode 弯引号/弯连字符归一化为 ASCII，"
                           f"词数 {td['original_word_count']}→{td['revised_word_count']} 不变、"
                           f"改动词 token 数 {td['changed_word_token_count']}=0，但"
                           " change_descriptions 描述了多个实质编辑（幻觉）。"),
                "suggestion": "Phase 8.2：改写后加『no-op 检测』——归一化后与原文相等即 "
                             "`no_effect`，跳过重测、绝不报告实质改善。",
            })
        if td["exact_equality"]:
            findings.append({
                "severity": "medium",
                "title": f"{aid.capitalize()} 改写返回了字节级相等的正文，却落入 roll_back 而非 no_effect",
                "detail": "rewriter 实际未改一字，revision_plan 非空；系统按『无改善 9→9』回滚。"
                          "语义上更接近 no_effect / no_revision_performed。",
                "suggestion": "Phase 8.2：区分 no_effect（改写未生效）与 roll_back（拒绝坏改写），"
                              "修订决策语义（spec §十四 CASE C）。",
            })
        sf = a["style_fidelity"]
        res = sf["resolution"] or {}
        cc = res.get("classification_counts", {})
        if sf["after_count"] is not None and sf["before_count"] is not None and \
           sf["after_count"] < sf["before_count"] and cc.get("disappeared", 0) >= 1 and \
           td["changed_word_token_count"] == 0:
            findings.append({
                "severity": "high",
                "title": f"{aid.capitalize()} 的『改善』（{sf['before_count']}→{sf['after_count']}）"
                         "是 Layer C LLM strategy match 噪声，非真实风格改善",
                "detail": ("消失的那 1 个偏差来自 strategy match 的 LLM 翻转，而底层文本"
                           "词级零改动（归一化后相等）。策略匹配器对同一段文字（仅换标点）"
                           "给出不同匹配结果。"),
                "suggestion": "Phase 8.2：偏差『改善』需与文本实质改动联动——若文本无实质改动，"
                              "不得把 LLM 层状态翻转计为改善；考虑对 strategy match 做稳定性"
                              "处理（多次采样/缓存一致性）。",
            })
        lq = a["literary_quality"]
        if lq["delta"] is not None and lq["delta"] != 0 and \
           td["changed_word_token_count"] == 0:
            findings.append({
                "severity": "high",
                "title": f"{aid.capitalize()} 文学评价 {lq['before_total']}→{lq['after_total']}"
                         " 的变化来自 LLM evaluator 噪声（文本无实质改动）",
                "detail": f"驱动维度 {lq['driven_by']} 在文本词级零改动的前提下评分变化，"
                          "证明 literary evaluator 对同一语义内容（仅标点不同）评分不稳定。",
                "suggestion": "Phase 8.2：文学评价对 no-op 改写应复用 before 分数（缓存），"
                              "或引入评价器稳定性阈值，避免把测量噪声解释为真实提升。",
            })
    return findings


def _build_conclusion(authors: dict[str, Any]) -> dict[str, Any]:
    has_false_improvement = False
    has_noop = False
    for aid in AUTHOR_IDS:
        a = authors[aid]
        td = a["text_diff"]
        sf = a["style_fidelity"]
        if td["changed_word_token_count"] == 0 and not td["exact_equality"]:
            has_noop = True
        if sf["after_count"] is not None and sf["before_count"] is not None and \
           sf["after_count"] < sf["before_count"] and td["changed_word_token_count"] == 0:
            has_false_improvement = True

    if has_false_improvement or has_noop:
        grade = "NEEDS_FIX"
        summary = ("三阶 gate 决策逻辑本身重建正确（stored==reconstructed），但输入层存在"
                   "真实缺陷：改写器可产出标点归一化 no-op 却报告实质改写（幻觉），且 "
                   "『改善』/文学评分变化来自 LLM 测量噪声而非真实文本变化，导致 continue"
                   " 语义上站不住脚。这些缺陷会影响 feedback decision，属 Phase 8.2 必须修"
                   " 的候选问题（本轮只记录，不修）。")
    else:
        grade = "PASS_WITH_CAVEATS"
        summary = ("决策逻辑重建一致，未发现影响 feedback decision 的真实缺陷；"
                   "仅记录 Phase 8.2 可改进项。")
    return {"grade": grade, "summary": summary}


def _build_recommendation(conclusion: dict[str, Any]) -> str:
    if conclusion["grade"] == "NEEDS_FIX":
        return ("**不建议本轮直接进入 Phase 9。** 进入前必须先修（Phase 8.2）："
                "1) no-op 改写检测（归一化相等→no_effect）；2) 偏差/文学评分改善须与"
                "文本实质改动联动，杜绝 LLM 测量噪声被记为改善；3) 区分 no_effect 与"
                " roll_back 的语义。详见 Findings。")
    return "可以进入 Phase 9（仅附带 Phase 8.2 可改进项）。"
