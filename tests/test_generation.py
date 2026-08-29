# tests/test_generation.py
"""Phase 7 / 7.1 Style-Conditioned Generation 测试（全部确定性，Dummy provider，零 token）。

覆盖（spec §十七 + Phase 7.1 §5）：
    - GenerationResult 序列化（含 request_id）、usage 解析、参数往返、GeneratedPassage
      往返（含 generation_condition_id / request_id）、空生成拒绝、prompt hash 正确。
    - Phase 7.1 身份模型：同条件不同正文 → 不同 generation_id；同 prompt/参数 → 同
      generation_condition_id；generation_id 含 request_id。
    - Phase 7.1 泄露守卫 A/B 分离：作者身份名单来自 author metadata（支持未来作者），
      用户 brief 正文里合法的 "imitate" 绝不误报为作者身份泄露。
    - Phase 7.1 plumbing gate：缺 / 失败 / 不匹配 plumbing 一律 fail-closed；合法
      plumbing 放行。
    - 渲染 Markdown 已解析元数据（无未解析的 `{p.` 占位符）。
    - 无自动评价 / 无自动改写；provider 未配置 fail-closed。

绝不调用真实模型；绝不读 DEEPSEEK_API_KEY；绝不写 `data/`（产物写入 tmp_path）。
"""
import hashlib
import json

import pytest

from knowledge.generation.provider import DummyGenerationProvider, GenerationProvider
from knowledge.generation.run import (
    EXPERIMENT_ID, EXPERIMENT_ID_82, GENERATION_PARAMETERS, _author_names_for,
    _band_thresholds, _build_passage, _plan_and_prompt, _render_passage_md,
    _require_valid_plumbing, run_generation,
)
from knowledge.generation.schema import (
    IMITATION_INSTRUCTION_TOKENS, GeneratedPassage, GenerationError,
    GenerationParameters, GenerationResult, GenerationUsage,
    assert_no_author_identity, assert_no_imitation_instruction,
    compiled_prompt_hash, make_generation_condition_id, make_generation_id,
    output_hash,
)
from knowledge.planning.run import NEUTRAL_REQUEST
from knowledge.profiles.style_profile import AuthorStyleProfileSynthesizer
from knowledge.providers.llm_provider import OpenAICompatibleProvider


# --------------------------------------------------------------------------- #
# 构造辅助（与 test_style_planner 同源，保证确定性且可编译）
# --------------------------------------------------------------------------- #
def _feature_summary(**over):
    base = {
        "n": 833, "mean": 0.5, "variance": 0.01, "std": 0.1,
        "n_expected": 833, "n_total": 833, "n_valid": 833,
        "n_missing": 0, "n_unobservable": 0, "n_insufficient": 0,
        "value_type": "continuous", "measurement_type": "statistical",
        "confidence": {"n": 833, "mean": 0.8},
    }
    base.update(over)
    return base


def _canonical(cid, name="N", status="discovered", works=1, chunks=1, conf=0.8):
    return {
        "canonical_strategy_id": cid,
        "canonical_name": name,
        "canonical_description": "d",
        "trigger_summary": "When a situation arises.",
        "operation_summary": "Apply the operation.",
        "effect_summary": "Achieve the effect.",
        "support_status": status,
        "confidence": conf,
        "source_strategy_ids": [f"{cid}_raw"],
        "supporting_work_ids": ["emma", "pride_and_prejudice"][:works],
        "supporting_chunk_ids": [f"c{i}" for i in range(chunks)],
        "number_of_distinct_works": works,
        "number_of_distinct_chunks": chunks,
        "number_of_raw_observations": 1,
    }


def _make_profile(author_id="austen", full_features=None, canonicals=None):
    synth = AuthorStyleProfileSynthesizer()
    return synth.synthesize(
        author_id=author_id,
        train_work_ids=["emma", "pride_and_prejudice"],
        held_out_work_ids=["persuasion"],
        profile_work_ids=["emma", "pride_and_prejudice"],
        full_corpus_features=full_features or {},
        sampled_features={},
        sampled_narrative={},
        canonical_strategies=canonicals or [],
        stylometry_author_target={"author_id": author_id, "n_samples": 833,
                                  "source_work_ids": ["emma", "pride_and_prejudice"],
                                  "centroid_norm": 0.123456},
        stylometry_validation_metadata={"n_features": 954},
    )


FEATURES = ["mean_sentence_length", "comma_density", "dialogue_ratio", "lexical_diversity"]


def _band_thresholds():
    return {
        "schema_version": "0.1.0",
        "train_only": True,
        "train_work_ids": ["emma", "pride_and_prejudice"],
        "features": {
            fid: {"q33": 0.3, "q67": 0.7, "n": 100, "min": 0.0,
                  "median": 0.5, "max": 1.0}
            for fid in FEATURES
        },
    }


def _profile(author_id="austen"):
    return _make_profile(
        author_id=author_id,
        full_features={fid: _feature_summary() for fid in FEATURES},
        canonicals=[_canonical("a::v", status="validated", works=2, chunks=5)],
    )


def _minimal_passage(text="A generated passage of original prose."):
    return GeneratedPassage(
        generation_id="gid", generation_condition_id="cid",
        schema_version="0.1.0", author_id="austen",
        style_plan_id="spid", source_profile_hash="h" * 64,
        writing_request=NEUTRAL_REQUEST.to_dict(),
        provider="dummy", model="dummy-model",
        generation_parameters={"temperature": 0.8, "top_p": 0.9, "max_tokens": 2048},
        compiled_prompt_hash="c" * 64, compiled_prompt="prompt",
        generated_text=text, finish_reason="stop",
        usage=GenerationUsage(100, 200, 300),
        generation_version="0.1.0", cache_hit=False, n_retries=0,
    )


def _two_passages():
    bt = _band_thresholds()
    prov = DummyGenerationProvider(
        content="A generated passage of original prose.", finish_reason="stop")
    out = {}
    for aid in ("austen", "dickens"):
        plan, prompt = _plan_and_prompt(_profile(aid), bt)
        result = prov.generate(prompt.text, GENERATION_PARAMETERS)
        out[aid] = _build_passage(
            aid, plan, prompt, result, prov, GENERATION_PARAMETERS, {},
            EXPERIMENT_ID, fresh_request=True)
    return out


def _valid_plumbing():
    """与 DummyGenerationProvider（dummy / dummy-model）匹配的合法 plumbing 记录。"""
    return {
        "experiment_id": EXPERIMENT_ID,
        "plumbing": True,
        "success": True,
        "author_id": "austen",
        "style_plan_id": "spid",
        "source_profile_hash": "h" * 64,
        "compiled_prompt_hash": "c" * 64,
        "provider": "dummy",
        "model": "dummy-model",
        "endpoint": "https://dummy.example/chat/completions",
        "generation_parameters": GENERATION_PARAMETERS.to_dict(),
        "finish_reason": "stop",
        "usage": {"prompt_tokens": 100, "completion_tokens": 200, "total_tokens": 300},
        "content_char_count": 40,
        "content_word_count": 8,
        "content_preview": "A generated passage",
        "n_retries": 0,
        "cache_hit": False,
        "fresh_request": True,
    }


def _patch_run_env(monkeypatch):
    profiles = {aid: _profile(aid) for aid in ("austen", "dickens")}
    monkeypatch.setattr("knowledge.generation.run._load_profile",
                        lambda base, aid: profiles[aid])
    monkeypatch.setattr("knowledge.generation.run._band_thresholds",
                        lambda base, twi: _band_thresholds())


# --------------------------------------------------------------------------- #
# 1–3：序列化 / usage 解析
# --------------------------------------------------------------------------- #
def test_generation_usage_round_trip_and_parse():
    u = GenerationUsage(prompt_tokens=10, completion_tokens=3, total_tokens=13)
    assert GenerationUsage.from_dict(u.to_dict()) == u
    # 非数值 → 0（绝不因 provider 返回垃圾而崩溃）
    assert GenerationUsage.from_dict(
        {"prompt_tokens": "x", "completion_tokens": True, "total_tokens": 5}).completion_tokens == 0
    assert GenerationUsage.from_dict({}).total_tokens == 0


def test_generation_result_serialization():
    r = GenerationResult(content="hi", finish_reason="stop",
                         usage=GenerationUsage(1, 2, 3), n_retries=1, cache_hit=False)
    assert r.to_dict() == {
        "content": "hi", "finish_reason": "stop",
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        "n_retries": 1, "cache_hit": False, "request_id": "",
    }


def test_generation_parameters_round_trip():
    p = GenerationParameters(temperature=0.8, top_p=0.9, max_tokens=2048)
    assert p.to_dict() == {"temperature": 0.8, "top_p": 0.9, "max_tokens": 2048}


# --------------------------------------------------------------------------- #
# 4 / 7 / 11：GeneratedPassage 往返、finish_reason、空生成拒绝
# --------------------------------------------------------------------------- #
def test_generated_passage_round_trip_and_finish_reason():
    p = _minimal_passage()
    d = p.to_dict()
    assert d["finish_reason"] == "stop"
    assert d["generation_condition_id"] == "cid"
    assert d["request_id"] == ""
    assert d["prompt_tokens"] == 100 and d["completion_tokens"] == 200 and d["total_tokens"] == 300
    rt = GeneratedPassage.from_dict(json.loads(json.dumps(d)))
    assert rt.to_dict() == d


def test_generated_passage_backward_compat_missing_condition_id():
    # Phase 7 旧产物只有 generation_id（当时即"条件 id"），from_dict 必须向后兼容，
    # 绝不要求重生成：缺 generation_condition_id 时回退到旧 generation_id。
    d = _minimal_passage().to_dict()
    del d["generation_condition_id"]
    rt = GeneratedPassage.from_dict(d)
    assert rt.generation_condition_id == rt.generation_id == "gid"


def test_generated_passage_empty_rejected():
    with pytest.raises(GenerationError):
        _minimal_passage(text="   ")


# --------------------------------------------------------------------------- #
# 5：prompt hash 正确
# --------------------------------------------------------------------------- #
def test_compiled_prompt_hash_matches_sha256():
    t = "prompt text"
    assert compiled_prompt_hash(t) == hashlib.sha256(t.encode("utf-8")).hexdigest()


def test_compiled_prompt_hash_sensitive_to_change():
    assert compiled_prompt_hash("a") != compiled_prompt_hash("b")


# --------------------------------------------------------------------------- #
# Phase 7.1 §1：身份模型（条件 vs 结果）
# --------------------------------------------------------------------------- #
def test_make_generation_condition_id_deterministic():
    a = make_generation_condition_id("austen", "sp", "h", "dummy", "dummy-model",
                                     {"temperature": 0.8, "top_p": 0.9, "max_tokens": 2048})
    b = make_generation_condition_id("austen", "sp", "h", "dummy", "dummy-model",
                                     {"temperature": 0.8, "top_p": 0.9, "max_tokens": 2048})
    assert a == b
    # 任一条件改变 → 条件 id 改变
    assert a != make_generation_condition_id("dickens", "sp", "h", "dummy",
                                              "dummy-model", {"temperature": 0.8, "top_p": 0.9, "max_tokens": 2048})


def test_same_prompt_params_same_condition_id():
    # 同 prompt/参数（作者/计划/provider/model 也相同）→ 同 condition_id
    bt = _band_thresholds()
    prov = DummyGenerationProvider()
    p1 = _build_passage("austen", *_plan_and_prompt(_profile(), bt),
                        prov.generate("x", GENERATION_PARAMETERS), prov,
                        GENERATION_PARAMETERS, {}, EXPERIMENT_ID, fresh_request=True)
    p2 = _build_passage("austen", *_plan_and_prompt(_profile(), bt),
                        prov.generate("x", GENERATION_PARAMETERS), prov,
                        GENERATION_PARAMETERS, {}, EXPERIMENT_ID, fresh_request=True)
    assert p1.generation_condition_id == p2.generation_condition_id


def test_make_generation_id_distinct_for_distinct_output():
    # 同条件、不同正文 → output hash 不同 → generation_id 不同（绝不依赖时间）
    cond = "condition1234"
    exp = EXPERIMENT_ID
    gid_a = make_generation_id(cond, exp, output_hash("The first passage."))
    gid_b = make_generation_id(cond, exp, output_hash("The second passage."))
    assert gid_a != gid_b
    # 同条件 + 同正文 → 同 generation_id（确定性）
    assert make_generation_id(cond, exp, output_hash("same")) == \
        make_generation_id(cond, exp, output_hash("same"))


def test_make_generation_id_sensitive_to_request_id():
    cond, exp, oh = "c", EXPERIMENT_ID, output_hash("x")
    assert make_generation_id(cond, exp, oh, "req-a") != make_generation_id(cond, exp, oh, "req-b")


# --------------------------------------------------------------------------- #
# Phase 7.1 §4：泄露守卫 A/B 分离（数据驱动，非硬编码）
# --------------------------------------------------------------------------- #
def test_assert_no_imitation_instruction_passes_clean():
    assert_no_imitation_instruction("Use moderate sentence length and light irony.")


def test_assert_no_imitation_instruction_detects_imitation():
    for bad in ("write like", "do not imitate any author", "write in the style of"):
        with pytest.raises(GenerationError):
            assert_no_imitation_instruction(bad)


def test_assert_no_author_identity_passes_clean():
    assert_no_author_identity("Write a scene about a letter arriving.",
                              ["Jane Austen", "Charles Dickens"])


def test_assert_no_author_identity_detects_author_names():
    with pytest.raises(GenerationError):
        assert_no_author_identity("In the manner of Jane Austen",
                                  ["Jane Austen", "Charles Dickens"])
    with pytest.raises(GenerationError):
        assert_no_author_identity("like Charles Dickens", ["Jane Austen", "Charles Dickens"])


def test_assert_no_author_identity_supports_future_authors():
    # 作者身份名单由调用方（来自 author metadata）传入，绝非硬编码 Austen/Dickens：
    # 传入一个"未来新增作者"的显示名，守卫同样生效。
    assert_no_author_identity("Write a scene.", ["Mark Twain", "Virginia Woolf"])
    with pytest.raises(GenerationError):
        assert_no_author_identity("in the style of Mark Twain",
                                  ["Mark Twain", "Virginia Woolf"])


def test_legitimate_user_content_imitate_not_author_leak():
    # "imitate" 是普通英语动词，可合法出现在用户 brief（CONTENT）正文里；作者身份
    # 守卫只查作者显示名，绝不因 "imitate" 误报为作者身份泄露。
    assert_no_author_identity(
        "The apprentice tries to imitate the master's brushwork.",
        ["Jane Austen", "Charles Dickens"])


def test_author_names_for_from_metadata():
    # 名单来自语料 metadata（数据驱动），新增作者时无需改守卫代码。
    assert _author_names_for(["austen"]) == ["Jane Austen"]
    assert _author_names_for(["dickens"]) == ["Charles Dickens"]


def test_imitation_instruction_tokens_are_the_required_set():
    assert set(IMITATION_INSTRUCTION_TOKENS) == {"write like", "imitate", "in the style of"}


# --------------------------------------------------------------------------- #
# 4（集成）：实际编译 prompt 无作者名 / 无模仿令牌
# --------------------------------------------------------------------------- #
def test_compiled_prompt_has_no_author_name_no_imitation():
    # _plan_and_prompt 内部已 fail-closed 校验（B 作者身份 + A 风格控制指令）。
    _, prompt = _plan_and_prompt(_profile(), _band_thresholds())
    text = prompt.text
    for banned in ("Austen", "Dickens", "Jane", "Charles",
                   "write like", "imitate", "in the style of"):
        assert banned.lower() not in text.lower()


# --------------------------------------------------------------------------- #
# 6：profile / style_plan provenance 保存
# --------------------------------------------------------------------------- #
def test_build_passage_provenance_saved():
    bt = _band_thresholds()
    plan, prompt = _plan_and_prompt(_profile(), bt)
    prov = DummyGenerationProvider(content="A generated passage of original prose.",
                                   finish_reason="stop")
    result = prov.generate(prompt.text, GENERATION_PARAMETERS)
    passage = _build_passage("austen", plan, prompt, result, prov,
                             GENERATION_PARAMETERS, {}, EXPERIMENT_ID, fresh_request=True)
    assert passage.source_profile_hash == plan.source_profile_hash
    assert passage.style_plan_id == plan.style_plan_id
    assert passage.author_id == "austen"
    assert passage.compiled_prompt_hash == compiled_prompt_hash(prompt.text)
    # 风格控制指令入 provenance，供渲染阶段复检（A/B 泄露守卫）
    assert "style_control_text" in passage.provenance


# --------------------------------------------------------------------------- #
# 2 / 3：同一 WritingRequest 共享 + provider/model/参数一致
# --------------------------------------------------------------------------- #
def test_same_writing_request_shared():
    out = _two_passages()
    assert out["austen"].writing_request == out["dickens"].writing_request
    assert out["austen"].writing_request == NEUTRAL_REQUEST.to_dict()


def test_provider_model_params_consistent():
    out = _two_passages()
    assert out["austen"].provider == out["dickens"].provider == "dummy"
    assert out["austen"].model == out["dickens"].model
    assert out["austen"].generation_parameters == out["dickens"].generation_parameters


# --------------------------------------------------------------------------- #
# Phase 7.1 §2：plumbing gate（fail-closed）
# --------------------------------------------------------------------------- #
def test_require_valid_plumbing_missing_blocks():
    prov = DummyGenerationProvider()
    with pytest.raises(GenerationError):
        _require_valid_plumbing(None, prov)


def test_require_valid_plumbing_failed_or_mismatched_blocks():
    prov = DummyGenerationProvider()
    good = _valid_plumbing()
    # 失败 / 空正文 / finish_reason 不合格 / provider 不匹配 / model 不匹配 /
    # 参数不一致 / 非 fresh_request / cache_hit —— 任一违反即 fail-closed。
    for field, bad_value in [
        ("success", False),
        ("content_char_count", 0),
        ("finish_reason", "length"),
        ("provider", "deepseek"),
        ("model", "other-model"),
        ("generation_parameters", {"temperature": 0.0, "top_p": 0.0, "max_tokens": 1}),
        ("fresh_request", False),
        ("cache_hit", True),
    ]:
        rec = dict(good)
        rec[field] = bad_value
        with pytest.raises(GenerationError):
            _require_valid_plumbing(rec, prov)


def test_require_valid_plumbing_valid_passes():
    prov = DummyGenerationProvider()
    _require_valid_plumbing(_valid_plumbing(), prov)  # 不抛即通过


def test_run_generation_missing_plumbing_blocks(tmp_path, monkeypatch):
    _patch_run_env(monkeypatch)
    # 无 generation_plumbing.json → 正式生成必须 fail-closed。
    with pytest.raises(GenerationError, match="plumbing"):
        run_generation(data_root_=tmp_path, provider=DummyGenerationProvider())


# --------------------------------------------------------------------------- #
# 13：provider 错误处理（未配置 fail-closed）
# --------------------------------------------------------------------------- #
def test_generation_provider_unconfigured_raises():
    prov = GenerationProvider(OpenAICompatibleProvider(api_key=""))
    assert prov.is_configured() is False
    with pytest.raises(GenerationError):
        prov.generate("hi", GENERATION_PARAMETERS)


# --------------------------------------------------------------------------- #
# 12 / 14 / 15：artifact 布局 + 无自动评价 / 无自动改写 + 合法 plumbing 放行
# --------------------------------------------------------------------------- #
def test_run_generation_writes_artifacts_no_auto_steps(tmp_path, monkeypatch):
    _patch_run_env(monkeypatch)
    # 合法 plumbing 放行：先写一份合法 plumbing 记录，再正式生成。
    out = tmp_path / "analysis" / "generation"
    out.mkdir(parents=True, exist_ok=True)
    (out / "generation_plumbing.json").write_text(
        json.dumps(_valid_plumbing(), ensure_ascii=False), encoding="utf-8")

    summary = run_generation(
        data_root_=tmp_path,
        provider=DummyGenerationProvider(
            content="A generated passage of original prose.", finish_reason="stop"))

    for name in ("generation_experiment.json", "austen_generation.json",
                 "dickens_generation.json", "austen_passage.md",
                 "dickens_passage.md", "generation_comparison_report.md",
                 "generation_summary.json"):
        assert (out / name).exists(), f"missing artifact: {name}"

    # 无自动评价 / 无自动改写（Phase 8 才做评价）
    assert summary["no_auto_evaluation"] is True
    assert summary["no_auto_revision"] is True
    assert summary["total_tokens"]["total_tokens"] == 600  # 2 × (100+200+300)

    austen = json.loads((out / "austen_generation.json").read_text(encoding="utf-8"))
    assert austen["finish_reason"] == "stop"
    assert austen["fresh_request"] is True
    assert austen["cache_hit"] is False
    assert "generation_condition_id" in austen
    assert "request_id" in austen
    for banned in ("Jane Austen", "Charles Dickens", "write like", "imitate",
                   "in the style of"):
        assert banned.lower() not in austen["compiled_prompt"].lower()
    # 评价 / 改写字段不得出现在生成产物（无自动评价 / 无自动改写）
    assert "evaluation" not in austen
    assert "revision" not in austen


# --------------------------------------------------------------------------- #
# Phase 7.1 §3：Markdown 已解析元数据（无未解析占位符）
# --------------------------------------------------------------------------- #
def test_passage_md_contains_resolved_metadata_no_placeholders():
    p = _two_passages()["austen"]
    md = _render_passage_md(p)
    # 渲染结果必须包含实际 ID（而非字面量占位符）。
    for actual in (p.experiment_id, p.generation_id, p.generation_condition_id,
                   p.schema_version, p.generation_version, p.style_plan_id,
                   p.source_profile_hash, p.compiled_prompt_hash):
        assert actual in md
    # 不得残留任何未解析的 `{p.` 占位符。
    assert "{p." not in md
    assert "{p.experiment_id}" not in md and "{p.generation_id}" not in md


# --------------------------------------------------------------------------- #
# 实验身份（Phase 8.2：独立 experiment_id → generation/{id}/ 子目录，绝不覆盖 Phase 7）
# --------------------------------------------------------------------------- #
def test_run_generation_custom_experiment_id_writes_subdir(tmp_path, monkeypatch):
    _patch_run_env(monkeypatch)
    # 合法 plumbing 在默认根目录（一次性传输验证，同 provider/model/params 复用，
    # 不随新实验重发）。
    default_out = tmp_path / "analysis" / "generation"
    default_out.mkdir(parents=True, exist_ok=True)
    (default_out / "generation_plumbing.json").write_text(
        json.dumps(_valid_plumbing(), ensure_ascii=False), encoding="utf-8")

    summary = run_generation(
        data_root_=tmp_path,
        provider=DummyGenerationProvider(
            content="A fresh passage for experiment two.", finish_reason="stop"),
        experiment_id=EXPERIMENT_ID_82)

    sub = default_out / EXPERIMENT_ID_82
    # 新产物写进子目录。
    for name in ("austen_generation.json", "dickens_generation.json",
                 "generation_experiment.json", "generation_summary.json",
                 "austen_passage.md", "dickens_passage.md",
                 "generation_comparison_report.md"):
        assert (sub / name).exists(), f"missing artifact: {name}"
    # 默认根目录只有 plumbing（Phase 7 产物未被触碰）。
    assert not (default_out / "generation_experiment.json").exists()

    # experiment_id 已写入 summary 与正文产物。
    assert summary["experiment_id"] == EXPERIMENT_ID_82
    austen = json.loads((sub / "austen_generation.json").read_text(encoding="utf-8"))
    assert austen["experiment_id"] == EXPERIMENT_ID_82
    assert austen["fresh_request"] is True
    assert austen["cache_hit"] is False
