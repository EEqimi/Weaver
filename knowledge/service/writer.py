# knowledge/service/writer.py
"""Writer 服务层（最小可用 Writer 的单一业务入口，UI / CLI / Web API 共享）。

理想结构（spec Request C）：
    UI input → WritingRequest → 既有 StylePlanner → 既有 PromptCompiler →
    既有 Generation provider → 可选既有 Evaluation/Revision 单轮反馈 → UI output。

本模块只做**编排**，绝不重实现核心分析/生成逻辑。任何前端（本次的 stdlib http.server
Web UI，未来可能的 Streamlit / Web API）只负责收集表单字段、调用 `list_authors` /
`generate`、渲染结果。

铁律（沿用全仓）：
    - 作者全集来自 Generic Author Registry（`author_ids()` / `author_display_names()`），
      绝不硬编码 Austen/Dickens；只有已建成 AuthorStyleProfile 的作者才可生成。
    - Generate = 真实 LLM 调用（复用 DeepSeekProvider / GenerationProvider，绝不另写
      HTTP client）；feedback_iterations 只允许 0 或 1（Phase 9.1 多轮仍留在引擎）。
    - 密钥只读（DEEPSEEK_API_KEY），绝不打印 / 暴露 / 保存 / 提交（本模块任何返回
      dict 与错误消息都不含密钥）。
    - 生成正文绝不提交 Git（本服务只返回内存中的结果；Web UI 亦只存会话内存）。
    - prompt 泄露守卫（A/B 分离）沿用 generation 的 `_plan_and_prompt`。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..analysis.base import AnalysisUnavailable, LLMNotConfiguredError
from ..config import data_root as default_data_root
from ..corpus.metadata import author_display_names, author_ids
from ..evaluation.integrity import ContentIntegrityChecker
from ..evaluation.literary import LiteraryEvaluator
from ..evaluation.revision import RevisionRewriter
from ..evaluation.run import (
    build_provider,
    compare_target_actual,
    measure_actual_profile,
    _run_feedback_loop,
)
from ..evaluation.schema import (
    FEEDBACK_ACCEPT,
    EvaluationPolicy,
    RevisionResult,
)
from ..generation.provider import GenerationProvider
from ..generation.run import (
    GENERATION_PARAMETERS,
    _band_thresholds,
    _build_passage,
    _plan_and_prompt,
    _provenance,
    build_generation_provider,
)
from ..generation.schema import GenerationError
from ..planning.run import _load_profile
from ..planning.schema import WritingRequest
from ..providers.llm_provider import LLMProvider, LLMTransportError

# 独立 experiment_id：Writer 产物与 Phase 7 正式实验物理/语义隔离（仅内存，不落盘）。
WRITER_EXPERIMENT_ID = "writer-ui-v0.1"


class WriterError(Exception):
    """Writer 服务的用户可读错误（缺 API key / 作者未就绪 / 传输失败 / 生成失败等）。

    消息绝不包含密钥；调用方（Web UI / CLI）可直接向用户展示 `str(e)`。
    """


def _base(data_root_: Path | None) -> Path:
    return Path(data_root_) if data_root_ is not None else default_data_root()


def _profile_path(base: Path, author_id: str) -> Path:
    return base / "analysis" / "style_profiles" / f"{author_id}_style_profile.json"


def _is_ready(base: Path, author_id: str) -> tuple[bool, str]:
    """一位作者是否已就绪（画像已建成且可加载），返回 (ready, reason)。"""
    if not _profile_path(base, author_id).exists():
        return False, "author profile has not been built"
    try:
        _load_profile(base, author_id)
    except Exception as e:  # noqa: BLE001 — 统一转成友好原因
        return False, f"author profile failed to load: {e}"
    return True, "ready"


def _profile_train_work_ids(base: Path, author_id: str) -> list[str]:
    try:
        return _load_profile(base, author_id).author_scope.get("train_work_ids", [])
    except Exception:  # noqa: BLE001 — 未就绪作者跳过，不阻断其它作者
        return []


def list_authors(data_root_: Path | None = None) -> list[dict[str, Any]]:
    """列出所有已注册作者 + 是否 ready（画像已建成）。

    作者全集来自 Generic Author Registry（`author_ids()` / `author_display_names()`），
    非硬编码。UI 据此渲染下拉框：ready 可生成；not ready 显示
    "Not ready — author profile has not been built" 并禁止生成。
    """
    base = _base(data_root_)
    names = author_display_names()
    out: list[dict[str, Any]] = []
    for aid in author_ids():
        ready, reason = _is_ready(base, aid)
        out.append({
            "author_id": aid,
            "display_name": names.get(aid, aid),
            "ready": ready,
            "reason": reason,
        })
    return out


def build_request(content: str, *, desired_length: str = "short_scene",
                  target_words: int | None = None, language: str = "english",
                  pov: str | None = None,
                  constraints: list[str] | None = None) -> WritingRequest:
    """UI 输入 → WritingRequest（复用 planning/schema.py 的字段与校验）。"""
    return WritingRequest(
        content=content,
        desired_length=desired_length,
        target_words=target_words,
        language=language,
        pov=(pov.strip() if isinstance(pov, str) and pov.strip() else None),
        constraints=list(constraints or []),
    )


def _friendly_llm_error(e: Exception) -> str:
    """把底层 LLM 异常映射为用户可读消息（绝不携带密钥 / 请求体）。"""
    if isinstance(e, LLMNotConfiguredError):
        return "未配置 LLM provider：请设置 DEEPSEEK_API_KEY 环境变量"
    if isinstance(e, LLMTransportError):
        return f"与 LLM 服务通信失败（网络/认证错误）：{e}"
    if isinstance(e, GenerationError):
        return f"生成失败：{e}"
    return f"生成失败：{e}"


def generate(author_id: str, request: WritingRequest, *,
             provider: GenerationProvider | None = None,
             feedback_iterations: int = 0,
             data_root_: Path | None = None,
             evaluation_provider: LLMProvider | None = None,
             experiment_id: str = WRITER_EXPERIMENT_ID) -> dict[str, Any]:
    """对一位已就绪作者执行一次风格化生成（可选单轮反馈）。

    UI input（author_id + WritingRequest）→ StylePlanner → PromptCompiler →
    Generation provider（真实 LLM）→ 可选 Evaluation/Revision 单轮反馈 → 返回结果 dict。

    - feedback_iterations 只允许 0 或 1；1 时走既有反馈闭环（max_iterations=1，保守）。
    - 绝不硬编码作者；未就绪作者直接 WriterError（"Not ready — ..."）。
    - 返回 dict 不含任何密钥字段；生成的正文只在此返回、不落盘（会话内）。
    """
    if feedback_iterations not in (0, 1):
        raise WriterError(
            f"feedback_iterations 只允许 0 或 1，得到 {feedback_iterations!r}")
    if not isinstance(request, WritingRequest):
        raise WriterError("request 必须是 WritingRequest")
    if not request.content.strip():
        raise WriterError("写作需求（content）不能为空")

    base = _base(data_root_)

    ready, reason = _is_ready(base, author_id)
    if not ready:
        raise WriterError(f"作者 {author_id!r} 未就绪：{reason}")

    names = author_display_names()
    author_names = [names[author_id]] if author_id in names else []
    profile = _load_profile(base, author_id)

    # band 阈值：与既有生成管线一致，取所有已就绪作者的 TRAIN work_id 并集
    # （当 Austen+Dickens 均已就绪时与 run_generation 的并集完全相同，不重建阈值）。
    train_work_ids = sorted(
        {w for a in author_ids() for w in _profile_train_work_ids(base, a)})
    band_thresholds = _band_thresholds(base, train_work_ids)

    provider = provider or build_generation_provider()
    if not provider.is_configured():
        raise WriterError("未配置 LLM provider：请设置 DEEPSEEK_API_KEY 环境变量")

    plan, prompt = _plan_and_prompt(
        profile, band_thresholds, author_names, request=request)
    provenance = _provenance(plan, band_thresholds, profile)

    try:
        result = provider.generate(prompt.text, GENERATION_PARAMETERS)
    except (GenerationError, LLMTransportError, LLMNotConfiguredError) as e:
        raise WriterError(_friendly_llm_error(e)) from e

    passage = _build_passage(
        author_id, plan, prompt, result, provider, GENERATION_PARAMETERS,
        provenance, experiment_id, fresh_request=True)

    out: dict[str, Any] = {
        "author_id": author_id,
        "display_name": names.get(author_id, author_id),
        "style_plan_id": plan.style_plan_id,
        "compiled_prompt_hash": passage.compiled_prompt_hash,
        "generation_id": passage.generation_id,
        "generation_condition_id": passage.generation_condition_id,
        "provider": passage.provider,
        "model": passage.model,
        "finish_reason": passage.finish_reason,
        "word_count": len(passage.generated_text.split()),
        "char_count": len(passage.generated_text),
        "usage": passage.usage.to_dict(),
        "generated_text": passage.generated_text,
        "feedback": None,
    }

    if feedback_iterations == 1:
        fb = _run_feedback(
            base, author_id, author_names, plan, profile, request, passage,
            band_thresholds, evaluation_provider)
        out["feedback"] = fb
        out["generated_text"] = fb["final_text"]
        out["word_count"] = len(fb["final_text"].split())
        out["char_count"] = len(fb["final_text"])

    return out


def _run_feedback(base: Path, author_id: str, author_names: list[str],
                  plan: Any, profile: Any, request: WritingRequest,
                  passage: Any, band_thresholds: dict[str, Any],
                  evaluation_provider: LLMProvider | None) -> dict[str, Any]:
    """对一段新鲜生成正文跑单轮既有反馈闭环（revise → measure → decide，max_iterations=1）。

    复用 evaluation/run.py 的 `measure_actual_profile` / `compare_target_actual` /
    `_run_feedback_loop`（Phase 8.2/9.1 的同一闭环），绝不重实现。feedback 是额外 API
    消耗，故 UI 保守地只用 0 或 1。
    """
    eval_provider = evaluation_provider or build_provider(base)
    if not eval_provider.is_configured():
        raise WriterError("未配置 LLM provider：反馈评估需要 DEEPSEEK_API_KEY 环境变量")

    evaluator = LiteraryEvaluator(eval_provider, blind=True)
    rewriter = RevisionRewriter(eval_provider, blind=True)
    checker = ContentIntegrityChecker(eval_provider, blind=True)
    policy = EvaluationPolicy()

    original_text = passage.generated_text
    actual = measure_actual_profile(
        original_text, author_id=author_id, passage_id=passage.generation_id,
        style_plan_id=plan.style_plan_id, profile=profile, provider=eval_provider,
        data_root_=base)

    eval_before = evaluator.evaluate(original_text, author_id=author_id,
                                     passage_id=passage.generation_id)
    eval_before = None if isinstance(eval_before, AnalysisUnavailable) else eval_before

    comparison_before = compare_target_actual(plan, profile, actual, band_thresholds)
    lit_before = eval_before.total_score if eval_before is not None else None

    loop = _run_feedback_loop(
        original_text=original_text,
        comparison_before=comparison_before, lit_before=lit_before,
        plan=plan, profile=profile, request=request,
        author_id=author_id, passage_id=passage.generation_id,
        names=author_names, rewriter=rewriter, checker=checker, evaluator=evaluator,
        band_thresholds=band_thresholds, provider=eval_provider, policy=policy,
        max_iterations=1, base=base)

    # 最终正文（镜像 _run_feedback_loop 的 final_text_hash 语义）：
    # accept → 采纳改写后文本；roll_back / no_effect / no_action → 保留当前最佳（原文）。
    final_text = original_text
    if (loop.get("final_outcome") == FEEDBACK_ACCEPT
            and isinstance(loop.get("rev_result"), RevisionResult)):
        final_text = loop["rev_result"].revised_text

    decision = loop.get("decision")
    return {
        "outcome": loop.get("final_outcome"),
        "iteration": loop.get("final_iteration"),
        "final_text": final_text,
        "decision": decision.to_dict() if decision is not None else None,
        "iterations": loop.get("iterations"),
    }
