# knowledge/generation/run.py
"""Phase 7 执行：CompiledPrompt → 真实生成模型 → GeneratedPassage（+ 产物）。

两次独立入口，严格按 spec §四/§十八的顺序：
    1. `run_plumbing`   —— 先做 exactly ONE Austen plumbing request，验证
       HTTP/auth/prompt/response/token/stop-reason，不产出正式正文产物。
    2. `run_generation` —— plumbing OK 后，正式生成 Austen + Dickens（fresh request），
       落盘 JSON + Markdown + 对比报告 + 汇总。

铁律（spec）：
    - 同一 WritingRequest、同一模型、同一生成参数；唯一变量是画像导出的风格控制。
    - 实际 prompt 绝不含作者名 / "write like" / "imitate" / "in the style of"：
      作者身份名单来自 author metadata（`assert_no_author_identity`），模仿指令只查
      我们生成的风格控制指令（`assert_no_imitation_instruction`），均 fail-closed。
    - 复用 DeepSeekProvider（OpenAI 兼容）的 HTTP 传输，绝不另写第二套 client。
    - 绝不自动评价（Phase 8）；绝不自动改写正文。
    - 密钥只读（DEEPSEEK_API_KEY），绝不打印 / 保存 / 提交。
    - 独立 experiment_id / 无缓存：每次 generate 都是 fresh request（不藏 cache）。
    - plumbing gate：`run_generation` 必须有合法 plumbing 记录，否则 fail-closed。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import data_root as default_data_root
from ..corpus.metadata import author_display_names
from ..planning.compiler import PromptCompiler
from ..planning.planner import StylePlanner
from ..planning.run import (
    AUTHOR_IDS, NEUTRAL_REQUEST, _band_thresholds, _load_profile,
)
from ..planning.schema import WritingRequest
from ..providers.llm_provider import DeepSeekProvider
from ..schema.versions import (
    GENERATION_SCHEMA_VERSION, GENERATION_VERSION, PROMPT_COMPILER_VERSION,
    STYLE_PLANNER_VERSION, WRITING_REQUEST_SCHEMA_VERSION,
)
from .provider import GenerationProvider
from .schema import (
    GeneratedPassage, GenerationError, GenerationParameters,
    assert_no_author_identity, assert_no_imitation_instruction,
    compiled_prompt_hash, make_generation_condition_id, make_generation_id,
    output_hash,
)

GENERATION_DIRNAME = "generation"
EXPERIMENT_ID = "phase7-generation-v0.1"
EXPERIMENT_ID_82 = "phase8_2-generation-v0.1"  # Phase 8.2 真实验证的独立实验身份（austen_02/dickens_02）

# 两位作者严格一致（spec §二）；唯一变量是画像导出的风格控制。
GENERATION_PARAMETERS = GenerationParameters(temperature=0.8, top_p=0.9, max_tokens=2048)

# 首轮目标长度（spec §十：500–800 English words）。
TARGET_MIN_WORDS = 500
TARGET_MAX_WORDS = 800

# 生成 prompt 的字符预算由 Phase 6.1 编译器保证（max_prompt_chars=6000），此处仅作
# 记账级保护：若异常超出，拒绝发送（spec §十四 token 成本保护）。
MAX_PROMPT_CHARS_GUARD = 6000

# plumbing gate 认为"可接受"的 finish_reason（正常完成；"length" 表示截断，不视为合格）。
ACCEPTABLE_FINISH_REASONS = frozenset({"stop"})


def generation_layout(data_root_: Path | None = None,
                      experiment_id: str | None = None) -> dict[str, Path]:
    """生成产物根目录。

    默认（experiment_id == EXPERIMENT_ID 或 None）写到扁平的 `generation/`；指定其它
    experiment_id 时写到 `generation/{experiment_id}/` 子目录，与 Phase 7 产物物理隔离
    （绝不覆盖 austen_01 / dickens_01）。plumbing 记录恒在默认根目录（一次性传输验证）。
    """
    base = Path(data_root_) if data_root_ is not None else default_data_root()
    root = base / "analysis" / GENERATION_DIRNAME
    if experiment_id and experiment_id != EXPERIMENT_ID:
        root = root / experiment_id
    return {"root": root}


def _write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build_generation_provider() -> GenerationProvider:
    """真实后端：DeepSeek 官方 API（OpenAI 兼容），model = deepseek-chat。"""
    return GenerationProvider(DeepSeekProvider())


def _style_control_text(prompt: Any) -> str:
    """编译提示词里"我们生成的风格控制指令"（除 CONTENT 外的所有 section）。

    模仿指令守卫只查这一段——绝不把用户 brief 正文里合法的 "imitate" 当成泄露。
    """
    return "\n\n".join(
        f"## {s['heading']}\n{s['body']}" for s in prompt.sections
        if s.get("heading") != "CONTENT")


def _author_names_for(author_ids) -> list[str]:
    """当前作者身份名单（来自语料 metadata，非硬编码）。"""
    names = author_display_names()
    return [names[aid] for aid in author_ids if aid in names]


def _assert_prompt_safe(prompt: Any, author_names: list[str]) -> None:
    """prompt 泄露守卫（Phase 7.1 §4，A/B 分离）：
    B. 全文绝不含当前作者显示名；A. 风格控制指令（非 CONTENT）绝不含模仿令牌。
    """
    assert_no_author_identity(prompt.text, author_names)
    assert_no_imitation_instruction(_style_control_text(prompt))


def _plan_and_prompt(profile: Any, band_thresholds: dict[str, Any],
                     author_names: list[str] | None = None,
                     request: WritingRequest | None = None):
    """画像 → StylePlan → CompiledPrompt（确定性），并 fail-closed 校验无作者泄露。

    `request` 缺省为中性写作需求（NEUTRAL_REQUEST，批量生成对比用）；Writer 服务层
    传入用户自定义 WritingRequest 时沿用同一条规划/编译/泄露守卫路径（向后兼容）。
    """
    planner = StylePlanner(band_thresholds=band_thresholds)
    compiler = PromptCompiler()
    plan = planner.plan(profile, request or NEUTRAL_REQUEST)
    prompt = compiler.compile(plan)
    _assert_prompt_safe(prompt, author_names or _author_names_for([profile.author_id]))
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
    # 条件身份：作者/计划/prompt/provider/model/参数（确定性，与正文无关）。
    condition_id = make_generation_condition_id(
        author_id, plan.style_plan_id, prompt_hash,
        provider.provider_id, provider.model, parameters.to_dict())
    # 具体结果身份：条件 + 实验 + 正文 hash（+ provider request id）。正文不同 → id 不同。
    generation_id = make_generation_id(
        condition_id, experiment_id, output_hash(result.content),
        getattr(result, "request_id", "") or "")
    # 风格控制指令（非 CONTENT）一并入 provenance，供渲染阶段复检 A/B 泄露守卫，
    # 而无需在渲染时重跑 planner/compiler。
    provenance = dict(provenance)
    provenance["style_control_text"] = _style_control_text(prompt)
    return GeneratedPassage(
        generation_id=generation_id,
        generation_condition_id=condition_id,
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
        request_id=getattr(result, "request_id", "") or "",
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
        "success": True,
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
                   provider: GenerationProvider | None = None,
                   experiment_id: str = EXPERIMENT_ID) -> dict[str, Any]:
    base = Path(data_root_) if data_root_ is not None else default_data_root()
    out_dir = generation_layout(base, experiment_id)["root"]
    out_dir.mkdir(parents=True, exist_ok=True)

    provider = provider or build_generation_provider()
    if not provider.is_configured():
        raise GenerationError("未配置 LLM provider（缺 DEEPSEEK_API_KEY）")

    profiles = {aid: _load_profile(base, aid) for aid in AUTHOR_IDS}
    train_work_ids = sorted(
        {w for p in profiles.values() for w in p.author_scope.get("train_work_ids", [])})
    band_thresholds = _band_thresholds(base, train_work_ids)

    # plumbing gate（Phase 7.1 §2）：正式生成前必须有合法 plumbing 记录，否则 fail-closed。
    # plumbing 恒在默认根目录（一次性传输验证，同 provider/model/params 复用，不随实验重发）。
    plumbing = _read_plumbing(generation_layout(base)["root"])
    _require_valid_plumbing(plumbing, provider)

    passages: dict[str, GeneratedPassage] = {}
    for author_id in AUTHOR_IDS:
        profile = profiles[author_id]
        plan, prompt = _plan_and_prompt(profile, band_thresholds)
        provenance = _provenance(plan, band_thresholds, profile)
        result = provider.generate(prompt.text, GENERATION_PARAMETERS)
        passage = _build_passage(
            author_id, plan, prompt, result, provider, GENERATION_PARAMETERS,
            provenance, experiment_id, fresh_request=True)
        passages[author_id] = passage

        _write_json(out_dir / f"{author_id}_generation.json", passage.to_dict())
        (out_dir / f"{author_id}_passage.md").write_text(
            _render_passage_md(passage), encoding="utf-8")

    experiment = _build_experiment(passages, provider, plumbing, experiment_id)
    _write_json(out_dir / "generation_experiment.json", experiment)
    (out_dir / "generation_comparison_report.md").write_text(
        _render_comparison(passages, plumbing, experiment_id), encoding="utf-8")

    summary = _build_summary(passages, plumbing, experiment, experiment_id)
    _write_json(out_dir / "generation_summary.json", summary)
    return summary


def _read_plumbing(out_dir: Path) -> dict[str, Any] | None:
    p = out_dir / "generation_plumbing.json"
    if p.exists():
        return _load_json(p)
    return None


def _require_valid_plumbing(plumbing: dict[str, Any] | None,
                            provider: GenerationProvider) -> None:
    """plumbing gate（Phase 7.1 §2）：缺 / 失败 / 不匹配的 plumbing 一律 fail-closed。

    绝不 "merely record plumbing=None"——正式生成前必须验证 plumbing 记录存在且合格。
    """
    if not plumbing:
        raise GenerationError(
            "缺 plumbing 记录：正式生成前必须先跑 `run_plumbing`（fail-closed）")
    if plumbing.get("plumbing") is not True:
        raise GenerationError("plumbing 记录不是合法 plumbing（plumbing!=True）")
    if plumbing.get("success") is not True:
        raise GenerationError("plumbing 未成功（success!=True）")
    if not plumbing.get("content_char_count"):
        raise GenerationError("plumbing 生成的正文为空（content_char_count=0）")
    if plumbing.get("finish_reason") not in ACCEPTABLE_FINISH_REASONS:
        raise GenerationError(
            f"plumbing finish_reason 不合格: {plumbing.get('finish_reason')!r}")
    if plumbing.get("provider") != provider.provider_id:
        raise GenerationError(
            f"plumbing provider 不匹配: {plumbing.get('provider')!r} != "
            f"{provider.provider_id!r}")
    if plumbing.get("model") != provider.model:
        raise GenerationError(
            f"plumbing model 不匹配: {plumbing.get('model')!r} != {provider.model!r}")
    if plumbing.get("generation_parameters") != GENERATION_PARAMETERS.to_dict():
        raise GenerationError("plumbing generation_parameters 与本次不一致")
    if plumbing.get("fresh_request") is not True:
        raise GenerationError("plumbing 不是 fresh_request（可能藏 cache）")
    if plumbing.get("cache_hit") is not False:
        raise GenerationError("plumbing 命中 cache（cache_hit!=False）")


def _build_experiment(passages: dict[str, GeneratedPassage],
                      provider: GenerationProvider,
                      plumbing: dict[str, Any] | None,
                      experiment_id: str = EXPERIMENT_ID) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
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
                   experiment: dict[str, Any],
                   experiment_id: str = EXPERIMENT_ID) -> dict[str, Any]:
    totals = _totals(passages)
    return {
        "stage": "style_conditioned_generation",
        "experiment_id": experiment_id,
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
        f"- **experiment_id**: `{p.experiment_id}`",
        f"- **generation_condition_id**: `{p.generation_condition_id}`",
        f"- **generation_id**: `{p.generation_id}`",
        f"- **schema_version**: `{p.schema_version}`  **generation_version**: `{p.generation_version}`",
        f"- **request_id**: `{p.request_id}`",
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
                       plumbing: dict[str, Any] | None,
                       experiment_id: str = EXPERIMENT_ID) -> str:
    lines = [
        "# Weaver Style Engine — Generation 对比报告（Phase 7）",
        "",
        "同一中性写作需求、同一模型、同一生成参数；**唯一变量**是画像导出的风格控制。",
        "本报告只做基本检查（词数 / token / finish_reason / prompt hash），**不做文学评价**",
        "（Phase 8）。",
        "",
        f"- **experiment_id**: `{experiment_id}`",
        f"- **provider / model**: `{passages[AUTHOR_IDS[0]].provider}` / "
        f"`{passages[AUTHOR_IDS[0]].model}`",
        f"- **生成参数**: {GENERATION_PARAMETERS.to_dict()}",
        f"- **目标长度**: {TARGET_MIN_WORDS}–{TARGET_MAX_WORDS} words（首轮）",
        "",
        "## 同一中性写作需求（各注册作者相同）",
        "",
        "```text",
        NEUTRAL_REQUEST.content,
        "```",
        "",
        "## 对照表",
        "",
    ]
    _display = author_display_names()
    header = "| 维度 | " + " | ".join(_display.get(aid, aid) for aid in AUTHOR_IDS) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(AUTHOR_IDS) + 1))
    rows: list[list[str]] = []
    for label, fn in [
        ("style_plan_id", lambda p: p.style_plan_id),
        ("source_profile_hash", lambda p: p.source_profile_hash),
        ("compiled_prompt_hash", lambda p: p.compiled_prompt_hash),
        ("generation_id", lambda p: p.generation_id),
        ("finish_reason", lambda p: p.finish_reason),
        ("word_count", lambda p: str(_word_count(p.generated_text))),
        ("char_count", lambda p: str(len(p.generated_text))),
        ("prompt_tokens", lambda p: str(p.usage.prompt_tokens)),
        ("completion_tokens", lambda p: str(p.usage.completion_tokens)),
        ("total_tokens", lambda p: str(p.usage.total_tokens)),
        ("fresh_request / cache_hit",
         lambda p: f"{p.fresh_request} / {p.cache_hit}"),
    ]:
        row = [label]
        for aid in AUTHOR_IDS:
            row.append(fn(passages[aid]))
        rows.append(row)
    for row in rows:
        lines.append("| " + row[0] + " | " + " | ".join(f"`{c}`" for c in row[1:]) + " |")

    lines += ["", "## 无作者泄露校验（fail-closed）", ""]
    for aid in AUTHOR_IDS:
        p = passages[aid]
        try:
            # B. 全文绝不含当前作者显示名（名单来自 author metadata）；A. 风格控制
            # 指令（非 CONTENT）绝不含模仿令牌。二者任一违反即视为泄露、拒绝发送。
            assert_no_author_identity(p.compiled_prompt, _author_names_for([aid]))
            assert_no_imitation_instruction(p.provenance.get("style_control_text", ""))
            lines.append(f"- {aid}: prompt **不含**作者名（来自 metadata）或模仿指令 "
                         f"`write like` / `imitate` / `in the style of`")
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
