# tests/test_writer_service.py
"""Writer 服务层 + 最小 Web UI 的确定性测试（Dummy/fake，零 LLM，零 token）。

覆盖 Request C 的测试契约：
    1. 作者下拉框来自 registry（非硬编码）。
    2. 未就绪作者不可生成（"Not ready — author profile has not been built"）。
    3. WritingRequest 由 UI 输入正确构造。
    4. 服务层复用既有 planner/compiler 并调用生成 provider。
    5. feedback=0 绝不进入评价/改写。
    6. feedback=1 进入既有反馈闭环（max_iterations=1）。
    7. API 失败 → 用户可读错误。
    8. UI/服务层绝不暴露 API key。
    9. `_plan_and_prompt` 缺省仍用 NEUTRAL_REQUEST（Austen/Dickens 生成链不回归）。

绝不调用真实模型；绝不读 DEEPSEEK_API_KEY 之外；绝不写真实 data/。
"""
import json
from types import SimpleNamespace

import pytest

from knowledge.analysis.base import LLMResponseError
from knowledge.evaluation.schema import RevisionResult
from knowledge.generation.provider import DummyGenerationProvider
from knowledge.planning.schema import WritingRequest
from knowledge.providers.llm_provider import LLMTransportError
from knowledge.service import writer


def _req(content="A young woman in a garden at dusk."):
    return WritingRequest(content=content)


class _NoCallProvider:
    """is_configured=True，但任何 complete 都失败（反馈评估绝不应真实调用）。"""

    def is_configured(self):
        return True

    def complete(self, *a, **k):
        raise AssertionError("测试绝不应调用真实 provider")


def _install_ready(monkeypatch, author_id="austen"):
    """把作者注册为 ready，并 stub 掉 service 层对磁盘/规划的依赖。"""
    monkeypatch.setattr(writer, "author_ids", lambda: (author_id,))
    monkeypatch.setattr(writer, "author_display_names",
                        lambda: {author_id: "Jane Austen"})
    monkeypatch.setattr(writer, "_is_ready", lambda base, aid: (True, "ready"))
    profile = SimpleNamespace(author_id=author_id,
                              author_scope={"train_work_ids": ["w1"]})
    monkeypatch.setattr(writer, "_load_profile", lambda base, aid: profile)
    monkeypatch.setattr(writer, "_band_thresholds", lambda base, ids: {})
    monkeypatch.setattr(writer, "_provenance", lambda plan, bt, prof: {})
    return profile


def _install_plan_prompt_passage(monkeypatch, text="Generated prose."):
    """stub _plan_and_prompt / _build_passage，记录 request 是否透传 + 调用计数。"""
    plan = SimpleNamespace(style_plan_id="sp1", source_profile_hash="sh")
    prompt = SimpleNamespace(text="PROMPT")
    calls = {"plan_and_prompt_request": None, "build_passage": 0}

    def _pap(profile, bt, names=None, request=None):
        calls["plan_and_prompt_request"] = request
        return plan, prompt
    monkeypatch.setattr(writer, "_plan_and_prompt", _pap)

    def _bp(author_id, plan, prompt, result, provider, params, provenance,
            experiment_id, fresh_request):
        calls["build_passage"] += 1
        return SimpleNamespace(
            generated_text=result.content,
            compiled_prompt_hash="ph",
            generation_id="gid",
            generation_condition_id="cid",
            provider=provider.provider_id,
            model=provider.model,
            finish_reason=result.finish_reason,
            usage=result.usage,
        )
    monkeypatch.setattr(writer, "_build_passage", _bp)
    return plan, prompt, calls


# --------------------------------------------------------------------------- #
# 1. 作者下拉框来自 registry
# --------------------------------------------------------------------------- #
def test_list_authors_comes_from_registry(monkeypatch, tmp_path):
    monkeypatch.setattr(writer, "author_ids",
                        lambda: ("austen", "bronte", "dickens"))
    monkeypatch.setattr(writer, "author_display_names", lambda: {
        "austen": "Jane Austen", "bronte": "Charlotte Bronte",
        "dickens": "Charles Dickens"})

    authors = writer.list_authors(data_root_=tmp_path)

    assert [a["author_id"] for a in authors] == ["austen", "bronte", "dickens"]
    assert [a["display_name"] for a in authors] == \
        ["Jane Austen", "Charlotte Bronte", "Charles Dickens"]
    # 无 profile → 全部 not ready（registry 派生，非硬编码两位）。
    assert all(not a["ready"] for a in authors)
    assert all(a["reason"] == "author profile has not been built" for a in authors)


# --------------------------------------------------------------------------- #
# 2. 未就绪作者不可生成
# --------------------------------------------------------------------------- #
def test_unavailable_author_cannot_generate(monkeypatch, tmp_path):
    monkeypatch.setattr(writer, "author_ids", lambda: ("austen",))
    monkeypatch.setattr(writer, "author_display_names",
                        lambda: {"austen": "Jane Austen"})

    with pytest.raises(writer.WriterError, match="profile has not been built"):
        writer.generate("austen", _req(), data_root_=tmp_path)


# --------------------------------------------------------------------------- #
# 3. WritingRequest 构造
# --------------------------------------------------------------------------- #
def test_build_request_constructs_writing_request():
    req = writer.build_request(
        "Hello world", desired_length="scene", target_words=123,
        language="english", pov="  third  ", constraints=["a", "b"])
    assert isinstance(req, WritingRequest)
    assert req.content == "Hello world"
    assert req.desired_length == "scene"
    assert req.target_words == 123
    assert req.pov == "third"               # 去空白
    assert req.constraints == ["a", "b"]
    assert writer.build_request("x", pov="   ").pov is None   # 空白 POV → None


# --------------------------------------------------------------------------- #
# 4. 服务层复用 planner/compiler 并调用生成 provider
# --------------------------------------------------------------------------- #
def test_generate_calls_provider_and_reuses_planner(monkeypatch, tmp_path):
    _install_ready(monkeypatch)
    _, _, calls = _install_plan_prompt_passage(monkeypatch, text="Hello world prose.")
    provider = DummyGenerationProvider(content="Hello world prose.",
                                       provider_id="dummy")
    req = _req("A young woman in a garden.")

    result = writer.generate("austen", req, provider=provider, data_root_=tmp_path)

    assert result["author_id"] == "austen"
    assert result["display_name"] == "Jane Austen"
    assert result["generated_text"] == "Hello world prose."
    assert result["word_count"] == 3
    assert result["provider"] == "dummy"
    assert result["feedback"] is None
    # 自定义 request 透传给 planner（非 NEUTRAL_REQUEST）。
    assert calls["plan_and_prompt_request"] is req
    assert calls["build_passage"] == 1


# --------------------------------------------------------------------------- #
# 5. feedback=0 绝不进入评价/改写
# --------------------------------------------------------------------------- #
def test_feedback_zero_skips_evaluation(monkeypatch, tmp_path):
    _install_ready(monkeypatch)
    _install_plan_prompt_passage(monkeypatch)

    def _boom(*a, **k):
        raise AssertionError("feedback=0 绝不应进入评价/改写闭环")
    monkeypatch.setattr(writer, "measure_actual_profile", _boom)
    monkeypatch.setattr(writer, "_run_feedback_loop", _boom)

    provider = DummyGenerationProvider()
    result = writer.generate("austen", _req(), provider=provider,
                             data_root_=tmp_path, feedback_iterations=0)
    assert result["feedback"] is None


# --------------------------------------------------------------------------- #
# 6. feedback=1 进入既有反馈闭环（max_iterations=1）
# --------------------------------------------------------------------------- #
def test_feedback_one_enters_feedback_pipeline(monkeypatch, tmp_path):
    _install_ready(monkeypatch)
    _, _, calls = _install_plan_prompt_passage(monkeypatch, text="Original text.")

    class _Eval:
        def evaluate(self, text, author_id="", passage_id=""):
            return SimpleNamespace(total_score=7.0)

    seen = {"run_feedback_loop": None, "measure": 0, "compare": 0}

    def _measure(text, **kw):
        seen["measure"] += 1
        return SimpleNamespace()

    def _compare(plan, profile, actual, thresholds):
        seen["compare"] += 1
        return SimpleNamespace(summary={})

    def _rfl(**kw):
        seen["run_feedback_loop"] = kw
        return {
            "rounds": [], "iterations": [],
            "decision": SimpleNamespace(to_dict=lambda: {"outcome": "accept"}),
            "rev_result": RevisionResult(
                schema_version="0.2.0", author_id="austen", passage_id="gid",
                original_passage_hash="oh", revised_passage_hash="rh",
                revised_text="Revised text."),
            "final_outcome": "accept", "final_iteration": 1,
            "final_text_hash": "x",
        }

    monkeypatch.setattr(writer, "measure_actual_profile", _measure)
    monkeypatch.setattr(writer, "compare_target_actual", _compare)
    monkeypatch.setattr(writer, "_run_feedback_loop", _rfl)
    monkeypatch.setattr(writer, "LiteraryEvaluator", lambda p, blind=True: _Eval())
    monkeypatch.setattr(writer, "RevisionRewriter", lambda p, blind=True: object())
    monkeypatch.setattr(writer, "ContentIntegrityChecker",
                        lambda p, blind=True: object())

    provider = DummyGenerationProvider(content="Original text.")
    result = writer.generate("austen", _req(), provider=provider,
                             data_root_=tmp_path, feedback_iterations=1,
                             evaluation_provider=_NoCallProvider())

    assert result["feedback"] is not None
    assert result["feedback"]["outcome"] == "accept"
    assert result["generated_text"] == "Revised text."   # accept → 采纳改写
    # 复用了既有闭环，且保守 max_iterations=1。
    assert seen["run_feedback_loop"]["max_iterations"] == 1
    assert seen["run_feedback_loop"]["original_text"] == "Original text."
    assert seen["measure"] == 1
    assert seen["compare"] == 1


# --------------------------------------------------------------------------- #
# 7. API 失败 → 用户可读错误
# --------------------------------------------------------------------------- #
def test_api_failure_raises_friendly_error(monkeypatch, tmp_path):
    _install_ready(monkeypatch)
    _install_plan_prompt_passage(monkeypatch)

    class _FailingProvider:
        provider_id = "deepseek"
        model = "deepseek-chat"
        base_url = "https://api.deepseek.com"

        def is_configured(self):
            return True

        def generate(self, prompt_text, parameters):
            raise LLMTransportError(
                "LLM 请求失败（deepseek-chat @ https://api.deepseek.com）: "
                "HTTP 401 Unauthorized")

    with pytest.raises(writer.WriterError, match="通信失败"):
        writer.generate("austen", _req(), provider=_FailingProvider(),
                        data_root_=tmp_path)


def test_missing_api_key_raises_friendly_error(monkeypatch, tmp_path):
    _install_ready(monkeypatch)
    _install_plan_prompt_passage(monkeypatch)

    class _UnconfiguredProvider:
        provider_id = "deepseek"
        model = "deepseek-chat"
        base_url = "https://api.deepseek.com"

        def is_configured(self):
            return False

        def generate(self, prompt_text, parameters):
            raise AssertionError("未配置 provider 不应被调用 generate")

    with pytest.raises(writer.WriterError, match="DEEPSEEK_API_KEY"):
        writer.generate("austen", _req(), provider=_UnconfiguredProvider(),
                        data_root_=tmp_path)


# --------------------------------------------------------------------------- #
# 8. UI/服务层绝不暴露 API key
# --------------------------------------------------------------------------- #
def test_service_never_exposes_api_key(monkeypatch, tmp_path):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-SECRET-12345")
    _install_ready(monkeypatch)
    _install_plan_prompt_passage(monkeypatch, text="Fine prose.")
    provider = DummyGenerationProvider(content="Fine prose.")

    result = writer.generate("austen", _req(), provider=provider,
                             data_root_=tmp_path)
    blob = json.dumps(result, ensure_ascii=False, sort_keys=True)
    assert "sk-SECRET-12345" not in blob
    assert "SECRET" not in blob
    # 错误消息亦不含密钥。
    assert "SECRET" not in writer._friendly_llm_error(LLMTransportError("HTTP 401"))


def test_webapp_never_exposes_api_key(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-SECRET-12345")
    from knowledge.service import webapp
    monkeypatch.setattr(webapp, "list_authors", lambda: [{
        "author_id": "austen", "display_name": "Jane Austen",
        "ready": True, "reason": "ready"}])

    form = webapp._form_page(webapp.list_authors())
    assert "sk-SECRET-12345" not in form
    assert "SECRET" not in form

    result = {"author_id": "austen", "display_name": "Jane Austen", "word_count": 3,
              "finish_reason": "stop", "provider": "deepseek",
              "model": "deepseek-chat", "generation_id": "gid",
              "generated_text": "Fine prose.", "feedback": None}
    page = webapp._result_page("austen", "Jane Austen", result)
    assert "sk-SECRET-12345" not in page
    assert "SECRET" not in page


# --------------------------------------------------------------------------- #
# 9. `_plan_and_prompt` 缺省仍用 NEUTRAL_REQUEST（Austen/Dickens 生成链不回归）
# --------------------------------------------------------------------------- #
def test_plan_and_prompt_defaults_to_neutral_request(monkeypatch):
    from knowledge.generation import run as gen_run
    from knowledge.planning.run import NEUTRAL_REQUEST

    seen = {}

    class _Planner:
        def __init__(self, band_thresholds=None):
            pass

        def plan(self, profile, request):
            seen["request"] = request
            return SimpleNamespace(
                style_plan_id="sp", source_profile_hash="sh", author_id="austen",
                writing_request=request.to_dict())

    class _Compiler:
        def compile(self, plan):
            return SimpleNamespace(text="PROMPT", sections=[])

    monkeypatch.setattr(gen_run, "StylePlanner", _Planner)
    monkeypatch.setattr(gen_run, "PromptCompiler", _Compiler)

    profile = SimpleNamespace(author_id="austen")
    gen_run._plan_and_prompt(profile, {}, author_names=[])
    assert seen["request"] is NEUTRAL_REQUEST            # 缺省：中性需求

    custom = WritingRequest(content="Custom brief.")
    gen_run._plan_and_prompt(profile, {}, author_names=[], request=custom)
    assert seen["request"] is custom                     # 自定义：透传


# --------------------------------------------------------------------------- #
# 10. 真实 bug 回归：空正文 GenerationError → 友好 WriterError（不裸抛）
# --------------------------------------------------------------------------- #
def test_empty_generation_raises_friendly_error(monkeypatch, tmp_path):
    """空正文经真实 `_build_passage`（GeneratedPassage 拒绝空生成）→ 服务层必须把
    GenerationError 映射为 WriterError，而非让 UI 显示 "生成失败：GenerationError"。
    """
    _install_ready(monkeypatch)
    plan = SimpleNamespace(style_plan_id="sp1", source_profile_hash="sh",
                           writing_request={"content": "x"})
    prompt = SimpleNamespace(text="PROMPT", sections=[])
    monkeypatch.setattr(writer, "_plan_and_prompt",
                        lambda *a, **k: (plan, prompt))

    provider = DummyGenerationProvider(content="")       # 空正文
    with pytest.raises(writer.WriterError, match="生成正文为空"):
        writer.generate("austen", _req(), provider=provider, data_root_=tmp_path)


# --------------------------------------------------------------------------- #
# 12. feedback=1 反馈闭环底层 LLM 错误 → 友好 WriterError（不裸抛 500）
# --------------------------------------------------------------------------- #
def test_feedback_eval_error_maps_to_friendly_writer_error(monkeypatch, tmp_path):
    """回归（真实验收第二次）：feedback=1 时改写器/评价器真实 LLM 返回无法解析的 JSON
    （LLMResponseError，如 402 余额不足 / 模型返回畸形输出）曾在 `_run_feedback` 中
    裸抛，UI 显示 "生成失败：LLMResponseError"（500）。服务层必须把它映射为友好
    WriterError（绝不向用户泄露内部堆栈/密钥）。零 LLM / 零 token。
    """
    _install_ready(monkeypatch)
    _install_plan_prompt_passage(monkeypatch, text="Original text.")

    def _boom(base, author_id, author_names, plan, profile, request, passage,
              band_thresholds, evaluation_provider):
        raise LLMResponseError("JSON 解析失败: Expecting ',' delimiter")

    monkeypatch.setattr(writer, "_run_feedback", _boom)

    provider = DummyGenerationProvider(content="Original text.")
    with pytest.raises(writer.WriterError, match="无法解析"):
        writer.generate("austen", _req(), provider=provider, data_root_=tmp_path,
                        feedback_iterations=1)


# --------------------------------------------------------------------------- #
# 11. 真实确定性链端到端（真实画像/planner/compiler/_build_passage，零 LLM）
# --------------------------------------------------------------------------- #
def test_generate_real_chain_end_to_end():
    """确定性回归：真实画像 → 真实 StylePlanner → 真实 PromptCompiler → 真实
    _build_passage（Dummy provider）。验证 Phase 9.5 service adapter 未错误假设既有
    对象的字段/API（Request C 的首要怀疑点）。data/ 缺失时跳过（已 gitignore）。"""
    from knowledge.config import data_root
    base = data_root()
    profile = base / "analysis" / "style_profiles" / "austen_style_profile.json"
    if not profile.exists():
        pytest.skip("真实画像数据缺失（data/ 已 gitignore）")

    content = "A quiet scene by the fire."
    provider = DummyGenerationProvider(content=content)
    req = WritingRequest(content=content)

    result = writer.generate("austen", req, provider=provider, data_root_=base)

    assert result["author_id"] == "austen"
    assert result["generated_text"] == content
    assert result["word_count"] == len(content.split())
    assert result["usage"]["total_tokens"] == 300
    assert result["feedback"] is None
