# tests/test_generation.py
"""Phase 7 Style-Conditioned Generation 测试（全部确定性，Dummy provider，零 token）。

覆盖（spec §十七，16 项）：
    GenerationResult 序列化、同一 WritingRequest 共享、provider/model/参数一致、
    实际 prompt 无作者名、prompt hash 正确、profile/style_plan provenance 保存、
    空生成拒绝、prompt 泄露检出、作者名泄露检出、usage 解析、finish_reason 保存、
    artifact 布局、provider 错误处理、无自动评价、无自动改写。

绝不调用真实模型；绝不读 DEEPSEEK_API_KEY；绝不写 `data/`（产物写入 tmp_path）。
"""
import hashlib
import json

import pytest

from knowledge.generation.provider import DummyGenerationProvider, GenerationProvider
from knowledge.generation.run import (
    EXPERIMENT_ID, GENERATION_PARAMETERS, _band_thresholds, _build_passage,
    _plan_and_prompt, run_generation,
)
from knowledge.generation.schema import (
    BANNED_AUTHOR_LEAK_TOKENS, GeneratedPassage, GenerationError,
    GenerationParameters, GenerationResult, GenerationUsage,
    assert_no_author_leakage, compiled_prompt_hash, make_generation_id,
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
        generation_id="gid", schema_version="0.1.0", author_id="austen",
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
        "n_retries": 1, "cache_hit": False,
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
    assert d["prompt_tokens"] == 100 and d["completion_tokens"] == 200 and d["total_tokens"] == 300
    rt = GeneratedPassage.from_dict(json.loads(json.dumps(d)))
    assert rt.to_dict() == d


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


def test_make_generation_id_deterministic_and_sensitive():
    a = make_generation_id("austen", "sp", "h", {"temperature": 0.8})
    b = make_generation_id("austen", "sp", "h", {"temperature": 0.8})
    assert a == b
    assert a != make_generation_id("dickens", "sp", "h", {"temperature": 0.8})


# --------------------------------------------------------------------------- #
# 8 / 9：prompt 泄露检出 + 作者名泄露检出
# --------------------------------------------------------------------------- #
def test_assert_no_author_leakage_passes_clean():
    assert_no_author_leakage("Write a scene. Use moderate sentence length.")


def test_assert_no_author_leakage_detects_author_names():
    with pytest.raises(GenerationError):
        assert_no_author_leakage("Write like Jane Austen")
    with pytest.raises(GenerationError):
        assert_no_author_leakage("Write like Charles Dickens")


def test_assert_no_author_leakage_detects_imitation():
    with pytest.raises(GenerationError):
        assert_no_author_leakage("write like")
    with pytest.raises(GenerationError):
        assert_no_author_leakage("do not imitate any author")
    with pytest.raises(GenerationError):
        assert_no_author_leakage("write in the style of")


# --------------------------------------------------------------------------- #
# 4（集成）：实际编译 prompt 无作者名 / 无模仿令牌
# --------------------------------------------------------------------------- #
def test_compiled_prompt_has_no_author_name_no_imitation():
    _, prompt = _plan_and_prompt(_profile(), _band_thresholds())
    text = prompt.text
    assert_no_author_leakage(text)  # 不抛即通过
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
# 13：provider 错误处理（未配置 fail-closed）
# --------------------------------------------------------------------------- #
def test_generation_provider_unconfigured_raises():
    prov = GenerationProvider(OpenAICompatibleProvider(api_key=""))
    assert prov.is_configured() is False
    with pytest.raises(GenerationError):
        prov.generate("hi", GENERATION_PARAMETERS)


# --------------------------------------------------------------------------- #
# 12 / 14 / 15：artifact 布局 + 无自动评价 / 无自动改写
# --------------------------------------------------------------------------- #
def test_run_generation_writes_artifacts_no_auto_steps(tmp_path, monkeypatch):
    profiles = {aid: _profile(aid) for aid in ("austen", "dickens")}
    monkeypatch.setattr("knowledge.generation.run._load_profile",
                        lambda base, aid: profiles[aid])
    monkeypatch.setattr("knowledge.generation.run._band_thresholds",
                        lambda base, twi: _band_thresholds())

    summary = run_generation(
        data_root_=tmp_path,
        provider=DummyGenerationProvider(
            content="A generated passage of original prose.", finish_reason="stop"))

    out = tmp_path / "analysis" / "generation"
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
    for banned in ("Jane Austen", "Charles Dickens", "write like", "imitate",
                   "in the style of"):
        assert banned.lower() not in austen["compiled_prompt"].lower()
    # 评价 / 改写字段不得出现在生成产物（无自动评价 / 无自动改写）
    assert "evaluation" not in austen
    assert "revision" not in austen


def test_banned_tokens_are_the_required_set():
    # 铁律固定集合：作者名 + 三个模仿令牌（spec §8）。
    assert set(BANNED_AUTHOR_LEAK_TOKENS) == {
        "Jane Austen", "Charles Dickens", "write like", "imitate", "in the style of",
    }
