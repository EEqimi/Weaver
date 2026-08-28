# tests/test_audit.py
"""Phase 8.1 Post-Run Audit 测试（全部确定性、零 LLM、零 token、零真实 data/ 写入）。

覆盖纯函数：
    - normalize_text / word_count / sentence_count：弯引号等 Unicode 排版标点归一化。
    - compute_text_diff：identical / punctuation_only / minimal / substantial 分类，
      词 token 改动计数、句子改动计数、opcode 统计。
    - _lang_classification / resolve_deviations：偏差 A/B/C/D/E 逐项对照。
    - verify_evidence：evidence 存在性 / 跨维复用 / 过短标记。

绝不调用真实模型；绝不读 DEEPSEEK_API_KEY；绝不写真实 data/。
"""
import pytest

from knowledge.evaluation import audit
from knowledge.evaluation.schema import (
    ComparisonResult, FeatureDeviation, NarrativeDeviation, StrategyCoverage,
)


# --------------------------------------------------------------------------- #
# 归一化 / 词数
# --------------------------------------------------------------------------- #
def test_normalize_text_curly_quotes():
    assert audit.normalize_text("“Hello” ‘world’ — dash –") == "\"Hello\" 'world' - dash -"


def test_word_count_curly_apostrophe_not_split():
    # 弯撇号的 "You’re" 不应被拆成 "You"/"re"。
    assert audit.word_count("You’re late") == 2
    assert audit.word_count("You're late") == 2


def test_sentence_count():
    assert audit.sentence_count("One. Two? Three!") == 3
    assert audit.sentence_count("") == 0


# --------------------------------------------------------------------------- #
# 文本 diff
# --------------------------------------------------------------------------- #
def test_diff_identical():
    d = audit.compute_text_diff("same text", "same text")
    assert d["classification"] == "identical"
    assert d["exact_equality"] is True
    assert d["normalized_equality"] is True
    assert d["changed_word_token_count"] == 0
    assert d["changed_sentence_count"] == 0


def test_diff_punctuation_only():
    # 只改弯引号 / 弯连字符，不改任何词。
    original = "“You’re late,” she said—softly."
    revised = "\"You're late,\" she said-softly."
    d = audit.compute_text_diff(original, revised)
    assert d["classification"] == "punctuation_only"
    assert d["exact_equality"] is False
    assert d["normalized_equality"] is True
    assert d["changed_word_token_count"] == 0
    assert d["changed_sentence_count"] == 0
    assert d["original_word_count"] == d["revised_word_count"]


def test_diff_minimal_word_change():
    original = "The cat sat on the mat and watched the rain."
    revised = "The cat sat on the mat and watched the storm."
    d = audit.compute_text_diff(original, revised)
    assert d["classification"] in ("minimal", "substantial")
    assert d["changed_word_token_count"] == 1


def test_diff_substantial():
    original = "hello brave world of prose"
    revised = "goodbye cowardly planet of verse"
    d = audit.compute_text_diff(original, revised)
    assert d["classification"] == "substantial"
    assert d["changed_word_token_count"] >= 2


# --------------------------------------------------------------------------- #
# 偏差分类 A/B/C/D/E
# --------------------------------------------------------------------------- #
def _fd(feature_id, target_band, actual_band, status):
    return FeatureDeviation(
        feature_id=feature_id, target_band=target_band, actual_band=actual_band,
        target_value=None, actual_value=None, status=status, measurable=True,
        reason="")


def test_lang_classification_disappeared():
    assert audit._lang_classification(
        _fd("x", "medium", "low", "below"),
        _fd("x", "medium", "medium", "on_target")) == "disappeared"


def test_lang_classification_new():
    assert audit._lang_classification(
        _fd("x", "medium", "medium", "on_target"),
        _fd("x", "medium", "low", "below")) == "new"


def test_lang_classification_improved_outside():
    # low → medium，向 target high 靠近但仍 outside。
    assert audit._lang_classification(
        _fd("x", "high", "low", "below"),
        _fd("x", "high", "medium", "below")) == "improved_outside"


def test_lang_classification_worsened():
    # medium → low，离 target high 更远。
    assert audit._lang_classification(
        _fd("x", "high", "medium", "below"),
        _fd("x", "high", "low", "below")) == "worsened"


def test_lang_classification_unchanged_same_band():
    assert audit._lang_classification(
        _fd("x", "medium", "low", "below"),
        _fd("x", "medium", "low", "below")) == "unchanged"


# --------------------------------------------------------------------------- #
# resolve_deviations 逐项对照 + 计数
# --------------------------------------------------------------------------- #
def _cmp(lang=None, narr=None, strat=None):
    return ComparisonResult(
        author_id="austen", passage_id="p",
        language_deviations=lang or [],
        narrative_deviations=narr or [],
        strategy_coverage=strat or [],
        summary={})


def test_resolve_deviations_strategy_flip_is_disappeared():
    before = _cmp(strat=[
        StrategyCoverage(strategy_id="a::s", active=True, matched=False, evidence_quotes=[]),
    ])
    after = _cmp(strat=[
        StrategyCoverage(strategy_id="a::s", active=True, matched=True, evidence_quotes=["q"]),
    ])
    res = audit.resolve_deviations(before, after)
    assert res["classification_counts"]["disappeared"] == 1
    assert res["strategy_coverage"][0]["classification"] == "disappeared"


def test_resolve_deviations_fully_unchanged():
    lang = [_fd("x", "medium", "low", "below")]
    before = _cmp(lang=lang)
    after = _cmp(lang=[_fd("x", "medium", "low", "below")])
    res = audit.resolve_deviations(before, after)
    assert res["classification_counts"]["unchanged"] == 1
    assert res["classification_counts"]["disappeared"] == 0


def test_resolve_deviations_counts_all_buckets():
    # 三语言：消失 / 不变 / 新增。
    before = _cmp(lang=[
        _fd("d", "medium", "low", "below"),
        _fd("u", "medium", "low", "below"),
        _fd("n", "medium", "medium", "on_target"),
    ])
    after = _cmp(lang=[
        _fd("d", "medium", "medium", "on_target"),
        _fd("u", "medium", "low", "below"),
        _fd("n", "medium", "low", "below"),
    ])
    res = audit.resolve_deviations(before, after)
    cc = res["classification_counts"]
    assert cc["disappeared"] == 1
    assert cc["unchanged"] == 1
    assert cc["new"] == 1


# --------------------------------------------------------------------------- #
# 证据契约审计
# --------------------------------------------------------------------------- #
class _Dim:
    def __init__(self, evidence):
        self.evidence = evidence


def test_verify_evidence_all_found_no_flags():
    text = "The cat sat on the mat and watched the rain fall quietly down."
    dims = {
        "pacing": _Dim(["The cat sat on the mat", "watched the rain fall quietly"]),
    }
    res = audit.verify_evidence(dims, text)
    assert res["per_dimension"]["pacing"]["flags"] == []


def test_verify_evidence_not_found():
    text = "The cat sat on the mat."
    dims = {"pacing": _Dim(["a completely absent phrase"])}
    res = audit.verify_evidence(dims, text)
    assert "not_found_in_passage" in res["per_dimension"]["pacing"]["flags"]


def test_verify_evidence_too_short():
    text = "The cat sat."
    dims = {"pacing": _Dim(["cat"])}
    res = audit.verify_evidence(dims, text)
    assert "too_short" in res["per_dimension"]["pacing"]["flags"]


def test_verify_evidence_reuse_across_dims():
    text = "The cat sat on the mat and it was quiet."
    dims = {
        "pacing": _Dim(["The cat sat on the mat"]),
        "theme": _Dim(["The cat sat on the mat"]),
    }
    res = audit.verify_evidence(dims, text)
    assert res["n_reused_quotes"] == 1
    assert set(res["reused_quotes"].keys()) == {"The cat sat on the mat"}


# --------------------------------------------------------------------------- #
# 决策重构（纯函数，复用 run.decide_feedback_outcome）
# --------------------------------------------------------------------------- #
def test_reconstruct_decision_rollback_on_no_improvement():
    from knowledge.evaluation.schema import (
        ContentIntegrityResult, EvaluationPolicy,
    )
    ci = ContentIntegrityResult(
        schema_version="0.1.0", checker_version="0.1.0", passed=True,
        plot_facts_preserved=True, characters_preserved=True,
        relationships_preserved=True, constraints_preserved=True,
        new_major_events=False, removed_major_events=False,
        violations=[], reasoning_summary="", deterministic=True, blind=True)
    before = _cmp(lang=[_fd("x", "medium", "low", "below")])
    after = _cmp(lang=[_fd("x", "medium", "low", "below")])
    d = audit.reconstruct_decision(
        before, after, literary_before=8.5, literary_after=8.5,
        content_integrity=ci, no_revision=False,
        policy=EvaluationPolicy(), author_id="austen", passage_id="p")
    assert d["outcome"] == "roll_back"


def test_reconstruct_decision_continue_on_improvement():
    from knowledge.evaluation.schema import (
        ContentIntegrityResult, EvaluationPolicy,
    )
    ci = ContentIntegrityResult(
        schema_version="0.1.0", checker_version="0.1.0", passed=True,
        plot_facts_preserved=True, characters_preserved=True,
        relationships_preserved=True, constraints_preserved=True,
        new_major_events=False, removed_major_events=False,
        violations=[], reasoning_summary="", deterministic=True, blind=True)
    before = _cmp(lang=[
        _fd("a", "medium", "low", "below"),
        _fd("b", "medium", "low", "below"),
    ])
    after = _cmp(lang=[
        _fd("a", "medium", "medium", "on_target"),
        _fd("b", "medium", "low", "below"),
    ])
    d = audit.reconstruct_decision(
        before, after, literary_before=8.5, literary_after=8.7,
        content_integrity=ci, no_revision=False,
        policy=EvaluationPolicy(), author_id="austen", passage_id="p")
    assert d["outcome"] == "continue"
