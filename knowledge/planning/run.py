# knowledge/planning/run.py
"""Phase 6 执行：加载画像 → StylePlanner → PromptCompiler → 落盘计划与提示词 + 对比报告。

确定性，无 LLM，无随机，无时间戳内容。同一中性写作需求（WritingRequest）分别作用于
Austen / Dickens 画像，产出各自 StylePlan 与编译提示词，用于展示"同一 brief、不同画像
→ 不同风格控制"的对比。

铁律：绝不生成正文；绝不调用 LLM；绝不改写用户 brief / 约束；绝不在提示词中提作者名。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import data_root as default_data_root
from ..profiles.style_profile import AuthorStyleProfile
from .compiler import CompiledPrompt, PromptCompiler
from .planner import StylePlanner
from .schema import WritingRequest

PLANNING_DIRNAME = "planning"
AUTHOR_IDS = ("austen", "dickens")

# 中性写作需求：同一 brief 给两位作者，观察画像如何产生不同的风格控制。
NEUTRAL_REQUEST = WritingRequest(
    content=(
        "A young woman returns to her family's country house after several years away "
        "and, walking alone in the garden at dusk, meets someone she thought she had "
        "lost. Write the scene of their conversation and her inner response."
    ),
    desired_length="short_scene",
    target_words=400,
    language="english",
    pov=None,
    constraints=[
        "Do not introduce new named characters.",
        "Leave ambiguous whether the meeting truly happened or was imagined.",
    ],
)


def planning_layout(data_root_: Path | None = None) -> dict[str, Path]:
    base = Path(data_root_) if data_root_ is not None else default_data_root()
    return {"root": base / "analysis" / PLANNING_DIRNAME}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8")


def _load_profile(base: Path, author_id: str) -> AuthorStyleProfile:
    path = base / "analysis" / "style_profiles" / f"{author_id}_style_profile.json"
    profile = AuthorStyleProfile.from_dict(_load_json(path))
    if not profile.verify_reproducibility_hash():
        raise RuntimeError(f"{author_id}: reproducibility hash 校验失败，拒绝规划。")
    iso = profile.author_scope.get("held_out_isolation", {})
    if not iso.get("clean", False):
        raise RuntimeError(f"{author_id}: held-out 隔离不干净，拒绝规划。")
    return profile


def run_planning(data_root_: Path | None = None) -> dict[str, Any]:
    base = Path(data_root_) if data_root_ is not None else default_data_root()
    out_dir = planning_layout(base)
    out_dir["root"].mkdir(parents=True, exist_ok=True)

    planner = StylePlanner()
    compiler = PromptCompiler()

    plans: dict[str, Any] = {}
    prompts: dict[str, CompiledPrompt] = {}
    for author_id in AUTHOR_IDS:
        profile = _load_profile(base, author_id)
        plan = planner.plan(profile, NEUTRAL_REQUEST)
        prompt = compiler.compile(plan)
        plans[author_id] = plan
        prompts[author_id] = prompt
        _write_json(out_dir["root"] / f"{author_id}_style_plan.json", plan.to_dict())
        _write_json(out_dir["root"] / f"{author_id}_compiled_prompt.json", prompt.to_dict())
        (out_dir["root"] / f"{author_id}_compiled_prompt.md").write_text(
            prompt.text, encoding="utf-8")

    (out_dir["root"] / "planning_comparison_report.md").write_text(
        _render_comparison(plans, prompts, NEUTRAL_REQUEST), encoding="utf-8")

    summary = {
        "stage": "style_planning_and_prompt_compilation",
        "deterministic": True,
        "no_llm": True,
        "no_generated_prose": True,
        "writing_request": NEUTRAL_REQUEST.to_dict(),
        "authors": {
            aid: _author_summary(plans[aid], prompts[aid])
            for aid in AUTHOR_IDS
        },
    }
    _write_json(out_dir["root"] / "planning_summary.json", summary)
    return summary


def _author_summary(plan: Any, prompt: CompiledPrompt) -> dict[str, Any]:
    return {
        "style_plan_id": plan.style_plan_id,
        "source_profile_hash": plan.source_profile_hash,
        "n_language_controls": len(plan.language_controls),
        "n_narrative_controls_active": sum(
            1 for n in plan.narrative_controls if n.activation == "medium"),
        "n_strategies_active": len(plan.strategy_controls),
        "n_reference_controls": len(plan.reference_controls),
        "n_suppressed_controls": len(plan.suppressed_controls),
        "warnings": list(plan.warnings),
        "prompt_char_count": prompt.char_count,
        "prompt_truncated": prompt.truncated,
    }


def _render_comparison(plans: dict[str, Any], prompts: dict[str, CompiledPrompt],
                       request: WritingRequest) -> str:
    lines = [
        "# Weaver Style Engine — Style Planner & Prompt Compiler 对比报告（Phase 6）",
        "",
        "确定性合成：`True`  无 LLM：`True`  无生成正文：`True`。",
        "同一中性写作需求分别作用于 Austen / Dickens 画像，产出各自 StylePlan 与编译提示词。",
        "",
        "## 中性写作需求（两者相同）",
        "",
        f"- **内容**：{request.content}",
        f"- **长度**：{request.desired_length}（约 {request.target_words} 词）",
        f"- **语言**：{request.language}",
        f"- **约束**：{request.constraints}",
        "",
    ]

    for author_id in AUTHOR_IDS:
        plan = plans[author_id]
        prompt = prompts[author_id]
        lines += [
            f"## {author_id.capitalize()}",
            "",
            f"- style_plan_id：`{plan.style_plan_id}`",
            f"- source_profile_hash：`{plan.source_profile_hash}`",
            f"- 激活语言控制：{len(plan.language_controls)}  参考：{len(plan.reference_controls)}  "
            f"抑制：{len(plan.suppressed_controls)}  激活策略：{len(plan.strategy_controls)}  "
            f"参考策略：{len(plan.reference_strategy_controls)}",
            "",
            "### 语言控制（primary → secondary）",
            "",
            "| feature | role | activation | guidance |",
            "|---|---|---|---|",
        ]
        for c in plan.language_controls:
            lines.append(
                f"| {c.feature_id} | {c.registry_control_role} | {c.activation} | {c.guidance} |")
        lines += ["", "### 叙事控制（activated）", ""]
        active_narrative = [n for n in plan.narrative_controls if n.activation == "medium"]
        for n in active_narrative:
            lines.append(f"- **{n.field}**（{n.value_type}）：{n.guidance}")
        lines += ["", "### 条件策略（active，按优先级）", ""]
        for s in plan.strategy_controls[:6]:
            lines.append(
                f"- **{s.canonical_name}**（{s.support_status}, works={s.n_supporting_works}, "
                f"chunks={s.n_supporting_chunks}）\n"
                f"  - 触发：{s.trigger}\n  - 操作：{s.operation}\n  - 效果：{s.effect}")
        lines += ["", "### warnings", ""]
        for w in plan.warnings:
            lines.append(f"- {w}")
        lines += [
            "",
            "### 编译提示词（关键数字）",
            "",
            f"- 字符数：{prompt.char_count}  截断：`{prompt.truncated}`  "
            f"（{prompt.truncation_note or '无'}）",
            "",
        ]

    lines += [
        "## 对比要点（同一 brief，不同画像）",
        "",
        "| 维度 | Austen | Dickens |",
        "|---|---|---|",
    ]
    rows: list[tuple[str, str, str]] = []
    # 语言控制 guidance 对比
    for fid in ["dialogue_ratio", "mean_sentence_length", "mean_paragraph_length",
                "lexical_diversity"]:
        g = {}
        for aid in AUTHOR_IDS:
            g[aid] = next(
                (c.guidance for c in plans[aid].language_controls if c.feature_id == fid),
                "—")
        rows.append((fid, g["austen"], g["dickens"]))
    for label, a, d in rows:
        lines.append(f"| {label} | {a} | {d} |")
    lines += [
        "",
        "> 注：上述 guidance 为自然语言描述（banding 结果），不含任何原始数值；",
        "> 微观 stylometric 指纹（功能词 / 字符 n-gram / PCA / centroid）永不进入提示词。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    summary = run_planning()
    for aid, s in summary["authors"].items():
        print(f"{aid}: plan={s['style_plan_id']} "
              f"lang={s['n_language_controls']} "
              f"narrative_active={s['n_narrative_controls_active']} "
              f"strategies_active={s['n_strategies_active']} "
              f"ref={s['n_reference_controls']} supp={s['n_suppressed_controls']} "
              f"prompt_chars={s['prompt_char_count']} truncated={s['prompt_truncated']}")
    print("artifacts: data/analysis/planning/"
          "{author_id}_style_plan.json + {author_id}_compiled_prompt.{json,md} + "
          "planning_comparison_report.md + planning_summary.json")


if __name__ == "__main__":
    main()
