# tests/test_effect.py
"""Phase 8.2 改写有效性（Revision Effect）+ 测量有效性测试。

全部确定性、零 LLM、零 token、零真实 data/ 写入。覆盖（spec §六/§十四/§二十二）：

    - normalize_for_revision_comparison：只归一化排版标点 + 空白，绝不改词形/词序
      （"don't"→"do not" 与 "happy"→"glad" 仍被判为 SUBSTANTIVE）。
    - RevisionEffectAnalyzer：IDENTICAL / FORMATTING_ONLY / PUNCTUATION_ONLY /
      SUBSTANTIVE 分类，词/句改动计数、canonical hash。
    - decide_feedback_outcome 的 no_effect 短路语义（literary_guard_status /
      style_comparison_performed / content_integrity=None）。
    - run_evaluation 的 Gate 0 短路：no_effect 绝不调用 ContentIntegrityChecker /
      LiteraryEvaluator（after）/ measure_actual_profile（after）/ provider.complete。

绝不调用真实模型；绝不读 DEEPSEEK_API_KEY；绝不写真实 data/（产物写入 tmp_path）。
"""
import json
from types import SimpleNamespace

import pytest

from knowledge.evaluation.effect import (
    RevisionEffectAnalyzer, normalize_for_revision_comparison, normalize_whitespace,
)
from knowledge.evaluation import run as run_mod
from knowledge.evaluation.run import decide_feedback_outcome
from knowledge.evaluation.schema import (
    EFFECT_FORMATTING_ONLY, EFFECT_IDENTICAL, EFFECT_PUNCTUATION_ONLY,
    EFFECT_SUBSTANTIVE, FEEDBACK_NO_EFFECT, GUARD_NOT_APPLICABLE_NO_EFFECT,
    ActualStyleProfile, ComparisonResult, ContentIntegrityResult, EvaluationPolicy,
    FeedbackDecision, FeatureDeviation, RevisionEffectResult, RevisionItem,
    RevisionPlan, RevisionResult,
)

ANALYZER = RevisionEffectAnalyzer()

ORIGINAL_CURLY = "“She walked alone,” she said—at dusk."
REVISED_STRAIGHT = "\"She walked alone,\" she said-at dusk."


def _effect(original, revised):
    return ANALYZER.analyze(original, revised)


# --------------------------------------------------------------------------- #
# normalize_for_revision_comparison（spec §六）
# --------------------------------------------------------------------------- #
def test_normalize_is_idempotent_and_strips():
    t = "  “Hello”   —  world…  "
    once = normalize_for_revision_comparison(t)
    assert normalize_for_revision_comparison(once) == once
    assert not once.startswith(" ") and not once.endswith(" ")


def test_normalize_does_not_change_contractions_or_synonyms():
    # 铁律：绝不把 "don't" 合并成 "do not"，绝不把 "happy" 换成 "glad"。
    assert normalize_for_revision_comparison("don't") == "don't"
    assert normalize_for_revision_comparison("happy") == "happy"
    assert normalize_for_revision_comparison("do not") == "do not"


def test_normalize_whitespace_collapses_and_normalizes_nbsp():
    assert normalize_whitespace("a\r\nb\rc\td") == "a b c d"
    assert normalize_whitespace("a b  c") == "a b c"


# --------------------------------------------------------------------------- #
# RevisionEffectAnalyzer 分类（spec §二十二 1–8）
# --------------------------------------------------------------------------- #
def test_identical():
    r = _effect("same text", "same text")
    assert r.effect_status == EFFECT_IDENTICAL
    assert r.substantive_edit is False
    assert r.byte_identical is True
    assert r.word_change_count == 0


def test_curly_to_straight_is_punctuation_only():
    r = _effect(ORIGINAL_CURLY, REVISED_STRAIGHT)
    assert r.effect_status == EFFECT_PUNCTUATION_ONLY
    assert r.substantive_edit is False
    assert r.byte_identical is False
    assert r.word_change_count == 0
    assert r.canonical_original_hash == r.canonical_revised_hash


def test_crlf_to_lf_is_formatting_only():
    r = _effect("a b c", "a b c")
    r2 = _effect("line one\r\nline two", "line one\nline two")
    assert r.effect_status == EFFECT_IDENTICAL
    assert r2.effect_status == EFFECT_FORMATTING_ONLY
    assert r2.substantive_edit is False
    assert r2.word_change_count == 0


def test_multiple_spaces_is_formatting_only():
    r = _effect("a  b   c", "a b c")
    assert r.effect_status == EFFECT_FORMATTING_ONLY
    assert r.substantive_edit is False
    assert r.word_change_count == 0


def test_nbsp_to_space_is_formatting_only():
    r = _effect("a b", "a b")
    assert r.effect_status == EFFECT_FORMATTING_ONLY
    assert r.substantive_edit is False


def test_dash_variation_is_punctuation_only():
    r = _effect("word—word", "word-word")
    assert r.effect_status == EFFECT_PUNCTUATION_ONLY
    assert r.substantive_edit is False
    assert r.word_change_count == 0


def test_one_word_change_is_substantive():
    r = _effect("The cat sat on the mat.", "The dog sat on the mat.")
    assert r.effect_status == EFFECT_SUBSTANTIVE
    assert r.substantive_edit is True
    assert r.word_change_count == 1
    assert r.word_change_ratio == pytest.approx(1 / 6)


def test_word_order_change_is_substantive():
    r = _effect("a b c", "c b a")
    assert r.effect_status == EFFECT_SUBSTANTIVE
    assert r.substantive_edit is True


def test_contraction_expansion_is_substantive():
    r = _effect("don't stop", "do not stop")
    assert r.effect_status == EFFECT_SUBSTANTIVE
    assert r.substantive_edit is True


def test_synonym_change_is_substantive():
    r = _effect("happy days", "glad days")
    assert r.effect_status == EFFECT_SUBSTANTIVE
    assert r.substantive_edit is True


def test_word_and_sentence_change_counts():
    r = _effect("The cat sat. It was quiet.", "The dog sat. It was quiet.")
    assert r.word_change_count == 1
    assert r.sentence_change_count == 1

    r2 = _effect("The cat sat. It was quiet.", "The cat sat. It was loud.")
    assert r2.sentence_change_count == 1
    assert r2.word_change_count == 1


def test_effect_requires_no_llm():
    # RevisionEffectAnalyzer 不接受也不依赖 provider；纯文本 diff。
    import inspect
    params = list(inspect.signature(ANALYZER.analyze).parameters)
    assert params == ["original_text", "revised_text"]


# --------------------------------------------------------------------------- #
# RevisionEffectResult schema 往返 + 版本 guard
# --------------------------------------------------------------------------- #
def test_revision_effect_result_round_trip():
    r = _effect("a b", "a c")
    assert r.to_dict()["effect_status"] == EFFECT_SUBSTANTIVE
    rt = RevisionEffectResult.from_dict(r.to_dict())
    assert rt.effect_status == r.effect_status
    assert rt.word_change_count == r.word_change_count


def test_revision_effect_result_version_guard():
    from knowledge.evaluation.schema import EvalError
    d = _effect("a", "b").to_dict()
    d["schema_version"] = "999.0.0"
    with pytest.raises(EvalError):
        RevisionEffectResult.from_dict(d)


def test_revision_result_carries_revision_effect_round_trip():
    eff = _effect("a b", "a c")
    rr = RevisionResult(
        schema_version="0.2.0", author_id="austen", passage_id="p",
        original_passage_hash="h1", revised_passage_hash="h2", revised_text="a c",
        claimed_change_descriptions=["changed a word"],
        claimed_revision_items=["P3:f"], revision_effect=eff.to_dict())
    rt = RevisionResult.from_dict(rr.to_dict())
    assert rt.revision_effect is not None
    assert rt.revision_effect["effect_status"] == EFFECT_SUBSTANTIVE
    assert rt.claimed_change_descriptions == ["changed a word"]


# --------------------------------------------------------------------------- #
# decide_feedback_outcome：no_effect 短路语义（spec §十四/§二十二）
# --------------------------------------------------------------------------- #
def _cmp():
    return ComparisonResult(author_id="austen", passage_id="p")


def test_no_effect_decision_short_circuits():
    eff = _effect("a b c", "a b c")  # IDENTICAL → non-substantive
    d = decide_feedback_outcome(
        _cmp(), None, literary_before=8.5, literary_after=None,
        content_integrity=None, no_revision=False, revision_effect=eff,
        policy=EvaluationPolicy(), author_id="austen", passage_id="p")
    assert d.outcome == FEEDBACK_NO_EFFECT
    assert d.content_integrity_passed is None
    assert d.content_integrity is None
    assert d.style_comparison_performed is False
    assert d.literary_guard_status == GUARD_NOT_APPLICABLE_NO_EFFECT
    assert d.revision_effect["effect_status"] == EFFECT_IDENTICAL
    assert d.style_fidelity["high_priority_deviations_after"] is None
    assert d.style_fidelity["improved"] is False


def test_no_effect_distinct_from_no_action_and_rollback():
    eff = _effect("a b c", "a b c")
    d = decide_feedback_outcome(_cmp(), None, no_revision=True,
                                revision_effect=None, policy=EvaluationPolicy())
    assert d.outcome == "no_action"
    d2 = decide_feedback_outcome(_cmp(), None, no_revision=False,
                                 revision_effect=eff, policy=EvaluationPolicy())
    assert d2.outcome == FEEDBACK_NO_EFFECT
    # 实质改写被拒（无改善）→ roll_back，而非 no_effect。
    def _dev():
        return FeatureDeviation(
            feature_id="dialogue_ratio", target_band="high", actual_band="low",
            target_value=None, actual_value=None, status="below", measurable=True,
            reason="")
    before = ComparisonResult(author_id="austen", passage_id="p",
                              language_deviations=[_dev()])
    after = ComparisonResult(author_id="austen", passage_id="p",
                             language_deviations=[_dev()])
    d3 = decide_feedback_outcome(
        before, after, literary_before=8.0, literary_after=8.0,
        content_integrity=ContentIntegrityResult(
            schema_version="0.1.0", checker_version="0.1.0", passed=True,
            plot_facts_preserved=True, characters_preserved=True,
            relationships_preserved=True, constraints_preserved=True,
            new_major_events=False, removed_major_events=False, violations=[],
            reasoning_summary="", deterministic=True, blind=True),
        no_revision=False, revision_effect=None, policy=EvaluationPolicy(),
        author_id="austen", passage_id="p")
    assert d3.outcome == "roll_back"


def test_feedback_decision_round_trip_with_new_fields():
    eff = _effect("a b c", "a b c")
    d = decide_feedback_outcome(
        _cmp(), None, literary_before=8.5, revision_effect=eff,
        policy=EvaluationPolicy(), author_id="austen", passage_id="p")
    rt = FeedbackDecision.from_dict(d.to_dict())
    assert rt.outcome == FEEDBACK_NO_EFFECT
    assert rt.literary_guard_status == GUARD_NOT_APPLICABLE_NO_EFFECT
    assert rt.style_comparison_performed is False
    assert rt.revision_effect["effect_status"] == EFFECT_IDENTICAL


def test_substantive_effect_does_not_short_circuit():
    eff = _effect("a b c", "a b d")  # SUBSTANTIVE
    ci = ContentIntegrityResult(
        schema_version="0.1.0", checker_version="0.1.0", passed=True,
        plot_facts_preserved=True, characters_preserved=True,
        relationships_preserved=True, constraints_preserved=True,
        new_major_events=False, removed_major_events=False, violations=[],
        reasoning_summary="", deterministic=True, blind=True)
    before = _cmp()
    after = _cmp()
    d = decide_feedback_outcome(
        before, after, literary_before=8.0, literary_after=8.0,
        content_integrity=ci, no_revision=False, revision_effect=eff,
        policy=EvaluationPolicy(), author_id="austen", passage_id="p")
    assert d.outcome != FEEDBACK_NO_EFFECT
    assert d.style_comparison_performed is True


# --------------------------------------------------------------------------- #
# run_evaluation Gate 0 短路（spec §二十二：provider call count）
# --------------------------------------------------------------------------- #
class _CountingProvider:
    def __init__(self):
        self.calls = 0

    def is_configured(self):
        return True

    def complete(self, *a, **k):
        self.calls += 1
        raise AssertionError("no_effect path 绝不应调用 provider.complete")


class _FakeRewriter:
    def __init__(self, revised_text):
        self.revised_text = revised_text

    def rewrite(self, original, plan, author_names=()):
        return RevisionResult(
            schema_version="0.2.0", author_id=plan.author_id,
            passage_id=plan.passage_id, original_passage_hash="oh",
            revised_passage_hash="rh", revised_text=self.revised_text,
            claimed_change_descriptions=["normalized punctuation"],
            claimed_revision_items=["P3:f"])


class _FakeChecker:
    def __init__(self):
        self.calls = 0

    def check(self, *a, **k):
        self.calls += 1
        return ContentIntegrityResult(
            schema_version="0.1.0", checker_version="0.1.0", passed=True,
            plot_facts_preserved=True, characters_preserved=True,
            relationships_preserved=True, constraints_preserved=True,
            new_major_events=False, removed_major_events=False, violations=[],
            reasoning_summary="", deterministic=True, blind=True)


class _FakeEvaluator:
    def __init__(self):
        self.calls = 0

    def evaluate(self, *a, **k):
        self.calls += 1
        return None


def _fake_actual():
    return ActualStyleProfile(
        schema_version="0.1.0", author_id="austen", passage_id="g1",
        passage_hash="h", style_plan_id="sp1")


def _install_fakes(monkeypatch, tmp_path, *, revised_text):
    """装配 run_evaluation 的依赖，返回 (fake_checker, fake_evaluator, measure_calls, provider)。"""
    monkeypatch.setattr(run_mod, "AUTHOR_IDS", ("austen",))
    monkeypatch.setattr(run_mod, "_author_names", lambda ids: [])

    profile = SimpleNamespace(author_scope={"train_work_ids": []})
    passage = SimpleNamespace(generated_text=ORIGINAL_CURLY, generation_id="g1")
    plan = SimpleNamespace(
        style_plan_id="sp1", author_id="austen",
        writing_request={"content": "A quiet scene.", "desired_length": "short_scene",
                         "target_words": None, "language": "english", "pov": None,
                         "constraints": []})

    monkeypatch.setattr(run_mod, "_load_profile", lambda base, aid: profile)
    monkeypatch.setattr(run_mod, "_load_passage",
                        lambda base, aid, gen_exp=None: passage)
    monkeypatch.setattr(run_mod, "_load_plan", lambda base, aid: plan)
    monkeypatch.setattr(run_mod, "_band_thresholds", lambda base, ids: {})

    measure_calls = {"n": 0}
    def _measure(text, **kw):
        measure_calls["n"] += 1
        return _fake_actual()
    monkeypatch.setattr(run_mod, "measure_actual_profile", _measure)

    monkeypatch.setattr(
        run_mod, "compare_target_actual",
        lambda plan, profile, actual, thresholds: ComparisonResult(
            author_id="austen", passage_id="g1"))

    def _build_plan(comparison, plan, evaluation=None, weak_score_threshold=5.0):
        return RevisionPlan(
            schema_version="0.1.0", author_id="austen", passage_id="g1",
            style_plan_id="sp1",
            revision_items=[RevisionItem(
                priority="P3", category="language", target="f",
                instruction="Use dialogue.", reason="r")])
    monkeypatch.setattr(run_mod, "build_revision_plan", _build_plan)

    fake_checker = _FakeChecker()
    fake_evaluator = _FakeEvaluator()
    monkeypatch.setattr(run_mod, "ContentIntegrityChecker",
                        lambda provider, blind=True: fake_checker)
    monkeypatch.setattr(run_mod, "LiteraryEvaluator",
                        lambda provider, blind=True: fake_evaluator)
    monkeypatch.setattr(run_mod, "RevisionRewriter",
                        lambda provider, blind=True: _FakeRewriter(revised_text))

    provider = _CountingProvider()
    return fake_checker, fake_evaluator, measure_calls, provider


def test_run_no_effect_short_circuits_all_expensive_steps(monkeypatch, tmp_path):
    fake_checker, fake_evaluator, measure_calls, provider = _install_fakes(
        monkeypatch, tmp_path, revised_text=REVISED_STRAIGHT)

    summary = run_mod.run_evaluation(data_root_=tmp_path, provider=provider)

    decision = summary["authors"]["austen"]["decision"]
    assert decision["outcome"] == FEEDBACK_NO_EFFECT
    assert decision["style_comparison_performed"] is False
    assert decision["literary_guard_status"] == GUARD_NOT_APPLICABLE_NO_EFFECT

    # 短路：内容完整性检查器 0 次调用；文学评价器仅 "before" 1 次、无 "after"；
    # 重测仅 "before" 1 次、无 "after"；0 LLM 调用。
    assert fake_checker.calls == 0
    assert fake_evaluator.calls == 1        # 仅 "before"（改写前），无 "after"
    assert measure_calls["n"] == 1          # 仅 "before" 重测 1 次，无 "after"
    assert provider.calls == 0              # 0 LLM 调用

    # revision_effect 已附着并落盘。
    rr = summary["authors"]["austen"]["revision_effect"]
    assert rr["effect_status"] == EFFECT_PUNCTUATION_ONLY
    assert rr["substantive_edit"] is False


def test_run_substantive_edit_proceeds_through_integrity(monkeypatch, tmp_path):
    # 正向对照：实质改写不短路，完整性检查器被调用、after 重测发生。
    fake_checker, fake_evaluator, measure_calls, provider = _install_fakes(
        monkeypatch, tmp_path, revised_text="\"She walked alone,\" she said at noon.")

    summary = run_mod.run_evaluation(data_root_=tmp_path, provider=provider)
    decision = summary["authors"]["austen"]["decision"]

    assert decision["outcome"] != FEEDBACK_NO_EFFECT
    assert decision["style_comparison_performed"] is True
    assert fake_checker.calls == 1          # 完整性检查器被调用
    assert measure_calls["n"] == 2          # before + after 各 1 次


# --------------------------------------------------------------------------- #
# 实验身份 / 布局（spec §二十一：evaluation_v3/{author}_02/，绝不覆盖旧产物）
# --------------------------------------------------------------------------- #
def _write_generation_json(base, author_id, experiment_id=None):
    from knowledge.generation.schema import GeneratedPassage, GenerationUsage
    gen_root = base / "analysis" / "generation"
    if experiment_id:
        gen_root = gen_root / experiment_id
    gen_root.mkdir(parents=True, exist_ok=True)
    p = GeneratedPassage(
        generation_id="gid", generation_condition_id="cid",
        schema_version="0.1.0", author_id=author_id, style_plan_id="spid",
        source_profile_hash="h" * 64, writing_request={},
        provider="deepseek", model="deepseek-chat",
        generation_parameters={"temperature": 0.8, "top_p": 0.9, "max_tokens": 2048},
        compiled_prompt_hash="c" * 64, compiled_prompt="prompt",
        generated_text="A fresh passage.", finish_reason="stop",
        usage=GenerationUsage(10, 20, 30), generation_version="0.1.0",
        cache_hit=False, n_retries=0, experiment_id=experiment_id or "",
        fresh_request=True)
    (gen_root / f"{author_id}_generation.json").write_text(
        json.dumps(p.to_dict(), ensure_ascii=False), encoding="utf-8")


def test_load_passage_reads_custom_generation_subdir(tmp_path):
    _write_generation_json(tmp_path, "austen", experiment_id="phase8_2-generation-v0.1")
    passage = run_mod._load_passage(tmp_path, "austen", "phase8_2-generation-v0.1")
    assert passage.experiment_id == "phase8_2-generation-v0.1"
    assert passage.generated_text == "A fresh passage."
    # 默认（None）读扁平目录；自定义子目录不命中 → fail-closed。
    with pytest.raises(run_mod.EvalError):
        run_mod._load_passage(tmp_path, "austen")


def test_run_evaluation_writes_run_tag_subdir(monkeypatch, tmp_path):
    fake_checker, fake_evaluator, measure_calls, provider = _install_fakes(
        monkeypatch, tmp_path, revised_text=REVISED_STRAIGHT)
    summary = run_mod.run_evaluation(
        data_root_=tmp_path, provider=provider,
        generation_experiment_id="phase8_2-generation-v0.1",
        run_tag="02", summary_prefix="phase8_2_real_validation")

    root = tmp_path / "analysis" / "evaluation_v3"
    sub = root / "austen_02"
    assert (sub / "austen_actual_profile.json").exists()
    assert (sub / "austen_revision_plan.json").exists()
    assert (sub / "austen_revision_result.json").exists()
    assert (root / "phase8_2_real_validation_summary.json").exists()
    assert (root / "phase8_2_real_validation_report.md").exists()
    # 绝不写入扁平旧布局。
    assert not (root / "austen_actual_profile.json").exists()
    # summary 记录实验身份 + run_tag。
    assert summary["generation_experiment_id"] == "phase8_2-generation-v0.1"
    assert summary["run_tag"] == "02"


def test_run_evaluation_default_flat_layout_unchanged(monkeypatch, tmp_path):
    fake_checker, fake_evaluator, measure_calls, provider = _install_fakes(
        monkeypatch, tmp_path, revised_text=REVISED_STRAIGHT)
    run_mod.run_evaluation(data_root_=tmp_path, provider=provider)
    root = tmp_path / "analysis" / "evaluation_v3"
    assert (root / "austen_actual_profile.json").exists()
    assert (root / "evaluation_summary.json").exists()
    assert (root / "evaluation_report.md").exists()


def test_run_evaluation_threads_max_iterations(monkeypatch, tmp_path):
    # spec §二十：Phase 8.2 单轮改写必须用 max_iterations=1（改善 → accept，而非 continue）。
    fake_checker, fake_evaluator, measure_calls, provider = _install_fakes(
        monkeypatch, tmp_path, revised_text=REVISED_STRAIGHT)
    summary = run_mod.run_evaluation(
        data_root_=tmp_path, provider=provider, max_iterations=1)
    assert summary["max_iterations"] == 1
