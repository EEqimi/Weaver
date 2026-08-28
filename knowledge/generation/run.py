# knowledge/generation/run.py
"""Phase 7 执行：CompiledPrompt → 真实生成模型 → GeneratedPassage（+ 产物）。

两次独立入口，严格按 spec §四/§十八的顺序：
    1. `run_plumbing`   —— 先做 exactly ONE Austen plumbing request，验证
       HTTP/auth/prompt/response/token/stop-reason，不产出正式正文产物。
    2. `run_generation` —— plumbing OK 后，正式生成 Austen + Dickens（fresh request），
       落盘 JSON + Markdown + 对比报告 + 汇总。

铁律（spec）：
    - 同一 WritingRequest、同一模型、同一生成参数；唯一变量是画像导出的风格控制。
    - 实际 prompt 绝不含作者名 / "write like" / "imitate" / "in the style of"
      （`assert_no_author_leakage` fail-closed）。
    - 复用 DeepSeekProvider（OpenAI 兼容）的 HTTP 传输，绝不另写第二套 client。
    - 绝不自动评价（Phase 8）；绝不自动改写正文。
    - 密钥只读（DEEPSEEK_API_KEY），绝不打印 / 保存 / 提交。
    - 独立 experiment_id / 无缓存：每次 generate 都是 fresh request（不藏 cache）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import data_root as default_data_root
from ..planning.compiler import PromptCompiler
from ..planning.planner import StylePlanner
from ..planning.run import (
    AUTHOR_IDS, NEUTRAL_REQUEST, _band_thresholds, _load_profile,
)
from ..providers.llm_provider import DeepSeekProvider
from ..schema.versions import (
    GENERATION_SCHEMA_VERSION, GENERATION_VERSION, PROMPT_COMPILER_VERSION,
    STYLE_PLANNER_VERSION, WRITING_REQUEST_SCHEMA_VERSION,
)
from .provider import GenerationProvider
from .schema import (
    GeneratedPassage, GenerationError, GenerationParameters,
    assert_no_author_leakage, compiled_prompt_hash, make_generation_id,
)

GENERATION_DIRNAME = "generation"
EXPERIMENT_ID = "phase7-generation-v0.1"

# 两位作者严格一致（spec §二）；唯一变量是画像导出的风格控制。
GENERATION_PARAMETERS = GenerationParameters(temperature=0.8, top_p=0.9, max_tokens=2048)

# 首轮目标长度（spec §十：500–800 English words）。
TARGET_MIN_WORDS = 500
TARGET_MAX_WORDS = 800

# 生成 prompt 的字符预算由 Phase 6.1 编译器保证（max_prompt_chars=6000），此处仅作
# 记账级保护：若异常超出，拒绝发送（spec §十四 token 成本保护）。
MAX_PROMPT_CHARS_GUARD = 6000


def generation_layout(data_root_: Path | None = None) -> dict[str, Path]:
    base = Path(data_root_) if data_root_ is not None else default_data_root()
    return {"root": base / "analysis" / GENERATION_DIRNAME}


def _write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build_generation_provider() -> GenerationProvider:
    """真实后端：DeepSeek 官方 API（OpenAI 兼容），model = deepseek-chat。"""
    return GenerationProvider(DeepSeekProvider())


def _plan_and_prompt(profile: Any, band_thresholds: dict[str, Any]):
    """画像 → StylePlan → CompiledPrompt（确定性），并 fail-closed 校验无作者泄露。"""
    planner = StylePlanner(band_thresholds=band_thresholds)
    compiler = PromptCompiler()
    plan = planner.plan(profile, NEUTRAL_REQUEST)
    prompt = compiler.compile(plan)
    assert_no_author_leakage(prompt.text)
    if len(prompt.text) > MAX_PROMPT_CHARS_GUARD:
        raise GenerationError(
            f"{profile.author_id}: prompt {len(prompt.text)} chars 超保护上限 "
            f"{MAX_PROMPT_CHARS_GUARD}，拒绝发送")
    return plan, prompt


def _provenance(plan: Any, band_thresholds: dict[str, Any], profile: Any) -> dict[str, Any]:
    bt = band_thresholds or {}
    return {
        "prompt_compiler_version": PROMPT_COMPILER_VERSION,
        "style_planner_version": STYLE_PLANNER_VERSION,
        "writing_request_schema_version": WRITING_REQUEST_SCHEMA_VERSION,
        "band_schema_version": bt.get("schema_version"),
        "generation_schema_version": GENERATION_SCHEMA_VERSION,
        "author_scope": profile.author_scope,
    }


def _build_passage(author_id: str, plan: Any, prompt: Any, result: Any,
                   provider: GenerationProvider, parameters: GenerationParameters,
                   provenance: dict[str, Any], experiment_id: str,
                   fresh_request: bool) -> GeneratedPassage:
    prompt_hash = compiled_prompt_hash(prompt.text)
    generation_id = make_generation_id(
        author_id, plan.style_plan_id, prompt_hash, parameters.to_dict())
    return GeneratedPassage(
        generation_id=generation_id,
        schema_version=GENERATION_SCHEMA_VERSION,
        author_id=author_id,
        style_plan_id=plan.style_plan_id,
        source_profile_hash=plan.source_profile_hash,
        writing_request=plan.writing_request,
        provider=provider.provider_id,
        model=provider.model,
        generation_parameters=parameters.to_dict(),
        compiled_prompt_hash=prompt_hash,
        compiled_prompt=prompt.text,
        generated_text=result.content,
        finish_reason=result.finish_reason,
        usage=result.usage,
        generation_version=GENERATION_VERSION,
        cache_hit=result.cache_hit,
        n_retries=result.n_retries,
        provenance=provenance,
        experiment_id=experiment_id,
        fresh_request=fresh_request,
    )


def _word_count(text: str) -> int:
    return len(text.split())


# --------------------------------------------------------------------------- #
# plumbing：exactly ONE Austen request
# --------------------------------------------------------------------------- #
def run_plumbing(data_root_: Path | None = None,
                 provider: GenerationProvider | None = None) -> dict[str, Any]:
    base = Path(data_root_) if data_root_ is not None else default_data_root()
    out_dir = generation_layout(base)["root"]
    out_dir.mkdir(parents=True, exist_ok=True)

    provider = provider or build_generation_provider()
    if not provider.is_configured():
        raise GenerationError("未配置 LLM provider（缺 DEEPSEEK_API_KEY）")

    profiles = {aid: _load_profile(base, aid) for aid in AUTHOR_IDS}
    train_work_ids = sorted(
        {w for p in profiles.values() for w in p.author_scope.get("train_work_ids", [])})
    band_thresholds = _band_thresholds(base, train_work_ids)

    author_id = "austen"  # 只做一次，且是 Austen（spec §四）
    plan, prompt = _plan_and_prompt(profiles[author_id], band_thresholds)
    result = provider.generate(prompt.text, GENERATION_PARAMETERS)
    if not result.content.strip():
        raise GenerationError(f"{author_id}: plumbing 生成正文为空")

    record: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "plumbing": True,
        "author_id": author_id,
        "style_plan_id": plan.style_plan_id,
        "source_profile_hash": plan.source_profile_hash,
        "compiled_prompt_hash": compiled_prompt_hash(prompt.text),
        "provider": provider.provider_id,
        "model": provider.model,
        "endpoint": f"{provider.base_url}/chat/completions",
        "generation_parameters": GENERATION_PARAMETERS.to_dict(),
        "finish_reason": result.finish_reason,
        "usage": result.usage.to_dict(),
        "content_char_count": len(result.content),
        "content_word_count": _word_count(result.content),
        "content_preview": result.content[:200],
        "n_retries": result.n_retries,
        "cache_hit": result.cache_hit,
        "fresh_request": True,
    }
    _write_json(out_dir / "generation_plumbing.json", record)
    return record


# --------------------------------------------------------------------------- #
# formal generation：Austen + Dickens（fresh request，不藏 cache）
# --------------------------------------------------------------------------- #
def run_generation(data_root_: Path | None = None,
                   provider: GenerationProvider | None = None) -> dict[str, Any]:
    base = Path(data_root_) if data_root_ is not None else default_data_root()
    out_dir = generation_layout(base)["root"]
    out_dir.mkdir(parents=True, exist_ok=True)

    provider = provider or build_generation_provider()
    if not provider.is_configured():
        raise GenerationError("未配置 LLM provider（缺 DEEPSEEK_API_KEY）")

    profiles = {aid: _load_profile(base, aid) for aid in AUTHOR_IDS}
    train_work_ids = sorted(
        {w for p in profiles.values() for w in p.author_scope.get("train_work_ids", [])})
    band_thresholds = _band_thresholds(base, train_work_ids)

    passages: dict[str, GeneratedPassage] = {}
    for author_id in AUTHOR_IDS:
        profile = profiles[author_id]
        plan, prompt = _plan_and_prompt(profile, band_thresholds)
        provenance = _provenance(plan, band_thresholds, profile)
        result = provider.generate(prompt.text, GENERATION_PARAMETERS)
        passage = _build_passage(
            author_id, plan, prompt, result, provider, GENERATION_PARAMETERS,
            provenance, EXPERIMENT_ID, fresh_request=True)
        passages[author_id] = passage

        _write_json(out_dir / f"{author_id}_generation.json", passage.to_dict())
        (out_dir / f"{author_id}_passage.md").write_text(
            _render_passage_md(passage), encoding="utf-8")

    plumbing = _read_plumbing(out_dir)

    experiment = _build_experiment(passages, provider, plumbing)
    _write_json(out_dir / "generation_experiment.json", experiment)
    (out_dir / "generation_comparison_report.md").write_text(
        _render_comparison(passages, plumbing), encoding="utf-8")

    summary = _build_summary(passages, plumbing, experiment)
    _write_json(out_dir / "generation_summary.json", summary)
    return summary


def _read_plumbing(out_dir: Path) -> dict[str, Any] | None:
    p = out_dir / "generation_plumbing.json"
    if p.exists():
        return _load_json(p)
    return None


def _build_experiment(passages: dict[str, GeneratedPassage],
                      provider: GenerationProvider,
                      plumbing: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "stage": "style_conditioned_generation",
        "generation_version": GENERATION_VERSION,
        "generation_schema_version": GENERATION_SCHEMA_VERSION,
        "provider": provider.provider_id,
        "model": provider.model,
        "endpoint": f"{provider.base_url}/chat/completions",
        "generation_parameters": GENERATION_PARAMETERS.to_dict(),
        "writing_request": NEUTRAL_REQUEST.to_dict(),
        "independent_cache_namespace": True,   # 无 LLMCache，每次都是 fresh request
        "plumbing": plumbing,
        "authors": {
            aid: _author_generation_summary(passages[aid])
            for aid in AUTHOR_IDS
        },
    }


def _author_generation_summary(p: GeneratedPassage) -> dict[str, Any]:
    return {
        "generation_id": p.generation_id,
        "author_id": p.author_id,
        "style_plan_id": p.style_plan_id,
        "source_profile_hash": p.source_profile_hash,
        "compiled_prompt_hash": p.compiled_prompt_hash,
        "finish_reason": p.finish_reason,
        "usage": p.usage.to_dict(),
        "word_count": _word_count(p.generated_text),
        "char_count": len(p.generated_text),
        "cache_hit": p.cache_hit,
        "fresh_request": p.fresh_request,
        "n_retries": p.n_retries,
    }


def _build_summary(passages: dict[str, GeneratedPassage],
                   plumbing: dict[str, Any] | None,
                   experiment: dict[str, Any]) -> dict[str, Any]:
    totals = _totals(passages)
    return {
        "stage": "style_conditioned_generation",
        "experiment_id": EXPERIMENT_ID,
        "deterministic_plan_and_prompt": True,
        "real_generation": True,
        "no_auto_evaluation": True,          # Phase 8 才做
        "no_auto_revision": True,
        "provider": experiment["provider"],
        "model": experiment["model"],
        "endpoint": experiment["endpoint"],
        "generation_parameters": GENERATION_PARAMETERS.to_dict(),
        "writing_request": NEUTRAL_REQUEST.to_dict(),
        "plumbing": plumbing,
        "authors": {aid: _author_generation_summary(passages[aid]) for aid in AUTHOR_IDS},
        "total_tokens": totals,
        "target_word_range": [TARGET_MIN_WORDS, TARGET_MAX_WORDS],
    }


def _totals(passages: dict[str, GeneratedPassage]) -> dict[str, int]:
    return {
        "prompt_tokens": sum(p.usage.prompt_tokens for p in passages.values()),
        "completion_tokens": sum(p.usage.completion_tokens for p in passages.values()),
        "total_tokens": sum(p.usage.total_tokens for p in passages.values()),
    }


# --------------------------------------------------------------------------- #
# 渲染（人类可读）
# --------------------------------------------------------------------------- #
def _render_passage_md(p: GeneratedPassage) -> str:
    lines = [
        f"# {p.author_id.capitalize()} — Style-Conditioned Generated Passage",
        "",
        "- **experiment_id**: `{p.experiment_id}`",
        "- **generation_id**: `{p.generation_id}`",
        "- **schema_version**: `{p.schema_version}`  **generation_version**: `{p.generation_version}`",
        f"- **author_id**: `{p.author_id}`",
        f"- **style_plan_id**: `{p.style_plan_id}`",
        f"- **source_profile_hash**: `{p.source_profile_hash}`",
        f"- **compiled_prompt_hash**: `{p.compiled_prompt_hash}`",
        f"- **provider**: `{p.provider}`  **model**: `{p.model}`",
        f"- **generation_parameters**: {p.generation_parameters}",
        f"- **finish_reason**: `{p.finish_reason}`",
        f"- **usage**: {p.usage.to_dict()}",
        f"- **fresh_request**: `{p.fresh_request}`  **cache_hit**: `{p.cache_hit}`  "
        f"**n_retries**: `{p.n_retries}`",
        "",
        "---",
        "",
        p.generated_text.strip(),
        "",
    ]
    return "\n".join(lines)


def _render_comparison(passages: dict[str, GeneratedPassage],
                       plumbing: dict[str, Any] | None) -> str:
    lines = [
        "# Weaver Style Engine — Generation 对比报告（Phase 7）",
        "",
        "同一中性写作需求、同一模型、同一生成参数；**唯一变量**是画像导出的风格控制。",
        "本报告只做基本检查（词数 / token / finish_reason / prompt hash），**不做文学评价**",
        "（Phase 8）。",
        "",
        f"- **experiment_id**: `{EXPERIMENT_ID}`",
        f"- **provider / model**: `{passages[AUTHOR_IDS[0]].provider}` / "
        f"`{passages[AUTHOR_IDS[0]].model}`",
        f"- **生成参数**: {GENERATION_PARAMETERS.to_dict()}",
        f"- **目标长度**: {TARGET_MIN_WORDS}–{TARGET_MAX_WORDS} words（首轮）",
        "",
        "## 同一中性写作需求（两位作者相同）",
        "",
        "```text",
        NEUTRAL_REQUEST.content,
        "```",
        "",
        "## 对照表",
        "",
        "| 维度 | Austen | Dickens |",
        "|---|---|---|",
    ]
    rows: list[tuple[str, str, str]] = [
        ("style_plan_id", passages["austen"].style_plan_id, passages["dickens"].style_plan_id),
        ("source_profile_hash", passages["austen"].source_profile_hash,
         passages["dickens"].source_profile_hash),
        ("compiled_prompt_hash", passages["austen"].compiled_prompt_hash,
         passages["dickens"].compiled_prompt_hash),
        ("generation_id", passages["austen"].generation_id, passages["dickens"].generation_id),
        ("finish_reason", passages["austen"].finish_reason, passages["dickens"].finish_reason),
        ("word_count", str(_word_count(passages["austen"].generated_text)),
         str(_word_count(passages["dickens"].generated_text))),
        ("char_count", str(len(passages["austen"].generated_text)),
         str(len(passages["dickens"].generated_text))),
        ("prompt_tokens", str(passages["austen"].usage.prompt_tokens),
         str(passages["dickens"].usage.prompt_tokens)),
        ("completion_tokens", str(passages["austen"].usage.completion_tokens),
         str(passages["dickens"].usage.completion_tokens)),
        ("total_tokens", str(passages["austen"].usage.total_tokens),
         str(passages["dickens"].usage.total_tokens)),
        ("fresh_request / cache_hit",
         f"{passages['austen'].fresh_request} / {passages['austen'].cache_hit}",
         f"{passages['dickens'].fresh_request} / {passages['dickens'].cache_hit}"),
    ]
    for label, a, d in rows:
        lines.append(f"| {label} | `{a}` | `{d}` |")

    lines += ["", "## 无作者泄露校验（fail-closed）", ""]
    for aid in AUTHOR_IDS:
        try:
            assert_no_author_leakage(passages[aid].compiled_prompt)
            lines.append(f"- {aid}: prompt **不含**作者名 / `write like` / `imitate` / "
                         f"`in the style of`")
        except GenerationError as e:
            lines.append(f"- {aid}: **泄露检出 → 已拒绝发送**: {e}")

    if plumbing:
        lines += [
            "",
            "## Plumbing request（真实请求前的单次验证）",
            "",
            f"- author: `{plumbing['author_id']}`  finish_reason: `{plumbing['finish_reason']}`  "
            f"n_retries: `{plumbing['n_retries']}`",
            f"- usage: {plumbing['usage']}  word_count: {plumbing['content_word_count']}",
        ]

    lines += [
        "",
        "> 注：正文全文见 `{austen,dickens}_passage.md`；机器可读产物见 "
        "`{austen,dickens}_generation.json`；实验清单见 `generation_experiment.json`。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    summary = run_generation()
    print(f"experiment_id: {summary['experiment_id']}")
    print(f"provider/model: {summary['provider']} / {summary['model']}")
    print(f"parameters: {summary['generation_parameters']}")
    for aid in AUTHOR_IDS:
        s = summary["authors"][aid]
        print(f"{aid}: words={s['word_count']} finish={s['finish_reason']} "
              f"tokens={s['usage']['total_tokens']} "
              f"(in {s['usage']['prompt_tokens']} / out {s['usage']['completion_tokens']}) "
              f"prompt_hash={s['compiled_prompt_hash']}")
    print(f"total_tokens: {summary['total_tokens']}")
    print("artifacts: data/analysis/generation/"
          "generation_experiment.json + {author_id}_generation.json + "
          "{author_id}_passage.md + generation_comparison_report.md + generation_summary.json")


if __name__ == "__main__":
    main()
