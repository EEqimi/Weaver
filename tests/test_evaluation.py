# tests/test_evaluation.py
"""Phase 8 Style Feedback Loop + 文学评价测试（全部确定性，Dummy provider，零 token）。

覆盖（spec §15 / §19.5 / §21）：
    - schema 往返 + schema-version guard（ActualStyleProfile / LiteraryEvaluation /
      RevisionPlan / RevisionResult / 各偏差与改写项）。
    - LiteraryEvaluator：盲测 prompt、6 维解析与权重总分、score 越界 / 缺维度报错、
      证据逐字校验、未配置返回 AnalysisUnavailable。
    - compare_target_actual：band 分类（on/above/below/not_measurable）、叙事
      on/off/not_measurable、策略覆盖。
    - build_revision_plan：P0>P1>P2>P3 优先级、指令不含作者名 / 原始数值 / 微观
      stylometric 指纹、P4 恒无改写项（stylometric 仅诊断）。
    - RevisionRewriter：P0 保护约束入 prompt、盲测（无作者名 / 无模仿令牌）、空计划
      确定性短路、合法 JSON 解析、作者名泄露 fail-closed。
    - decide_feedback_outcome：Accept / Continue / Roll Back 的确定性规则。
    - measure_actual_profile：Layer A 统计（22 特征）+ Layer D stylometric 重拟合诊断
      （合成 fixture），LLM 层在未配置 provider 下显式 unavailable，绝不伪造。

绝不调用真实模型；绝不读 DEEPSEEK_API_KEY；绝不写真实 `data/`（产物写入 tmp_path）。
"""
import json

import pytest

from knowledge.analysis.base import AnalysisUnavailable, LLMResponseError
from knowledge.evaluation.analyze import _layer_d_diagnostic, measure_actual_profile
from knowledge.evaluation.compare import compare_target_actual
from knowledge.evaluation.integrity import ContentIntegrityChecker
from knowledge.evaluation.literary import LiteraryEvaluator
from knowledge.evaluation.revision import (
    RevisionRewriter, _build_system_prompt, build_revision_plan,
)
from knowledge.evaluation.run import (
    MAX_ITERATIONS, _count_high_priority_deviations, decide_feedback_outcome,
)
from knowledge.evaluation.schema import (
    ASSESSMENT_INSUFFICIENT, ASSESSMENT_OBSERVED, FEEDBACK_ACCEPT,
    FEEDBACK_CONTINUE, FEEDBACK_NO_ACTION, FEEDBACK_ROLL_BACK, INTEGRITY_CRITICAL,
    INTEGRITY_WARNING, ActualStyleProfile, ComparisonResult,
    ContentIntegrityResult, ContentIntegrityViolation, DimensionScore, EvalError,
    EvaluationPolicy, FeatureDeviation, FeedbackDecision, LiteraryEvaluation,
    NarrativeDeviation, RevisionItem, RevisionPlan, RevisionResult,
    StrategyCoverage,
)
from knowledge.generation.schema import (
    IMITATION_INSTRUCTION_TOKENS, assert_no_author_identity,
    assert_no_imitation_instruction,
)
from knowledge.planning.schema import (
    PlannedControl, PlannedNarrativeControl, PlannedStrategy, StylePlan,
    WritingRequest,
)
from knowledge.profiles.style_profile import AuthorStyleProfileSynthesizer
from knowledge.providers.llm_provider import DummyLLMProvider, UnconfiguredLLMProvider
from knowledge.schema.versions import (
    CONTENT_INTEGRITY_VERSION, EVALUATION_SCHEMA_VERSION,
    FEEDBACK_DECISION_SCHEMA_VERSION, LITERARY_EVALUATION_SCHEMA_VERSION,
    STYLE_PLAN_SCHEMA_VERSION,
)
from knowledge.stylometry.extract import StylometricVectorizer

TEXT = "She walked alone at dusk. The garden was quiet. She remembered him clearly."


# --------------------------------------------------------------------------- #
# 构造辅助
# --------------------------------------------------------------------------- #
class RecordingProvider:
    """记录每次调用的 messages，并返回预设响应（供盲测/保护约束断言）。"""

    provider_id = "dummy"
    model = "dummy-model"

    def __init__(self, response: str = ""):
        self._response = response
        self.messages: list[list[dict]] = []

    def is_configured(self) -> bool:
        return True

    def complete(self, messages: list[dict], **kwargs) -> str:
        self.messages.append(messages)
        return self._response


def _feature_summary(mean=0.5, **over):
    base = {
        "n": 833, "mean": mean, "variance": 0.01, "std": 0.1,
        "n_expected": 833, "n_total": 833, "n_valid": 833,
        "n_missing": 0, "n_unobservable": 0, "n_insufficient": 0,
        "value_type": "continuous", "measurement_type": "statistical",
        "confidence": {"n": 833, "mean": 0.8},
    }
    base.update(over)
    return base


def _make_profile(author_id="austen", features=None, canonicals=None):
    synth = AuthorStyleProfileSynthesizer()
    return synth.synthesize(
        author_id=author_id,
        train_work_ids=["emma", "pride_and_prejudice"],
        held_out_work_ids=["persuasion"],
        profile_work_ids=["emma", "pride_and_prejudice"],
        full_corpus_features=features or {},
        sampled_features={},
        sampled_narrative={},
        canonical_strategies=canonicals or [],
        stylometry_author_target={"author_id": author_id, "n_samples": 1},
        stylometry_validation_metadata={"n_features": 954},
    )


_REQUEST = WritingRequest(
    content="A short test brief.", desired_length="short_scene",
    target_words=300, language="english", pov=None, constraints=["No new characters"])


def _plan(author_id="austen"):
    return StylePlan(
        style_plan_id="sp1", schema_version=STYLE_PLAN_SCHEMA_VERSION,
        author_id=author_id, source_profile_hash="h",
        writing_request=_REQUEST.to_dict(),
        language_controls=[
            PlannedControl(
                feature_id="dialogue_ratio", registry_control_role="candidate_core",
                activation="strong", bucket="primary", source_scope="full_train_corpus",
                support={}, reason="r", guidance="Use dialogue relatively often.",
                source="s"),
        ],
        narrative_controls=[
            PlannedNarrativeControl(
                field="pov", activation="medium", value_type="categorical",
                summary={"mode": "third"}, reason="r",
                guidance="Use a third-person point of view."),
        ],
        strategy_controls=[
            PlannedStrategy(
                canonical_strategy_id="austen::s1", canonical_name="S1",
                support_status="validated", confidence=0.8, control_priority=1,
                n_supporting_works=2, n_supporting_chunks=5, activation="active",
                trigger="When a character reflects.", operation="Blend the voice.",
                effect="Interiority.", reason="r"),
        ],
    )


def _thresholds(q33=0.3, q67=0.7):
    return {
        "features": {
            "dialogue_ratio": {"q33": q33, "q67": q67, "n": 100, "min": 0.0,
                               "median": 0.5, "max": 1.0},
        },
    }


def _actual(stat_values=None, narrative=None, matched_strategy_ids=()):
    stat = {fid: {"value": v} for fid, v in (stat_values or {}).items()}
    strategies = [{"strategy_id": sid,
                   "evidence": {"quotes": ["quote"], "quote": "quote"}}
                  for sid in matched_strategy_ids]
    return ActualStyleProfile(
        schema_version=EVALUATION_SCHEMA_VERSION, author_id="austen",
        passage_id="p1", passage_hash="h", style_plan_id="sp1",
        layer_a_statistical=stat, layer_a_judgment={},
        layer_b_narrative=narrative, layer_c_strategies=strategies,
        layer_d_stylometric=None, unavailable={}, provenance={},
    )


def _eval_response(scores=None):
    dims = {
        "plot_logic": {"score": scores.get("plot_logic", 8) if scores else 8,
                       "summary": "Coherent.", "strength": "Clear premise.",
                       "weakness": "Ending is rushed.",
                       "evidence": ["She walked alone at dusk."]},
        "characterization": {"score": scores.get("characterization", 7) if scores else 7,
                             "summary": "Vivid.", "strength": "Interiority.",
                             "weakness": "Thin secondary.",
                             "evidence": ["She remembered him clearly."]},
        "language_texture": {"score": scores.get("language_texture", 6) if scores else 6,
                             "summary": "Plain.", "strength": "Clean prose.",
                             "weakness": "Repetitive.",
                             "evidence": ["The garden was quiet."]},
        "theme_expression": {"score": scores.get("theme_expression", 7) if scores else 7,
                             "summary": "Clear.", "strength": "Memory theme.",
                             "weakness": "Underdeveloped.",
                             "evidence": ["She remembered him clearly."]},
        "pacing": {"score": scores.get("pacing", 7) if scores else 7,
                   "summary": "Even.", "strength": "Steady.",
                   "weakness": "Abrupt close.",
                   "evidence": ["She walked alone at dusk."]},
        "emotional_resonance": {"score": scores.get("emotional_resonance", 7) if scores else 7,
                                "summary": "Moving.", "strength": "Melancholy.",
                                "weakness": "Restrained.",
                                "evidence": ["She remembered him clearly."]},
    }
    return json.dumps({"dimensions": dims, "summary": "A quiet, coherent passage."})


# --------------------------------------------------------------------------- #
# Schema 往返 + 版本守卫
# --------------------------------------------------------------------------- #
def test_schema_round_trips():
    d = FeatureDeviation(feature_id="f", target_band="medium", actual_band="low",
                         target_value=0.5, actual_value=0.2, status="below",
                         measurable=True, reason="r").to_dict()
    assert FeatureDeviation.from_dict(d).to_dict() == d

    d = NarrativeDeviation(field="pov", target_value="third", actual_value="first",
                           status="off_target", reason="r").to_dict()
    assert NarrativeDeviation.from_dict(d).to_dict() == d

    d = StrategyCoverage(strategy_id="s", active=True, matched=False,
                         evidence_quotes=["q"]).to_dict()
    assert StrategyCoverage.from_dict(d).to_dict() == d

    cmp_ = ComparisonResult(
        author_id="austen", passage_id="p1",
        language_deviations=[FeatureDeviation(
            feature_id="f", target_band="medium", actual_band="low",
            target_value=0.5, actual_value=0.2, status="below", measurable=True,
            reason="r")],
    ).to_dict()
    assert ComparisonResult.from_dict(cmp_).to_dict() == cmp_

    ev = LiteraryEvaluation(
        schema_version=LITERARY_EVALUATION_SCHEMA_VERSION, author_id="austen",
        passage_id="p1",
        dimensions={"plot_logic": DimensionScore(
            dimension="plot_logic", label="Plot Logic", score=7.5, summary="s",
            strength="st", weakness="w", evidence=["e"])},
        weights={"plot_logic": 1.0}, total_score=7.5, summary="ok", blind=True,
        evaluator_version="0.2.0")
    assert LiteraryEvaluation.from_dict(ev.to_dict()).total_score == 7.5

    item = RevisionItem(priority="P3", category="language", target="f",
                        instruction="Use dialogue relatively often.", reason="r")
    plan = RevisionPlan(schema_version=EVALUATION_SCHEMA_VERSION, author_id="austen",
                        passage_id="p1", style_plan_id="sp1",
                        revision_items=[item], metadata={})
    assert RevisionPlan.from_dict(plan.to_dict()).revision_items[0].target == "f"

    res = RevisionResult(
        schema_version=EVALUATION_SCHEMA_VERSION, author_id="austen", passage_id="p1",
        original_passage_hash="a", revised_passage_hash="b", revised_text="t",
        change_descriptions=["d"], revision_items_applied=["P3:f"])
    assert RevisionResult.from_dict(res.to_dict()).revised_text == "t"


def test_phase_81_schema_round_trips():
    pol = EvaluationPolicy(max_literary_drop=0.5, weak_score_threshold=5.0)
    assert EvaluationPolicy.from_dict(pol.to_dict()).max_literary_drop == 0.5

    v = ContentIntegrityViolation("plot_facts", INTEGRITY_CRITICAL, "changed a fact")
    assert ContentIntegrityViolation.from_dict(v.to_dict()).kind == "plot_facts"

    ci = ContentIntegrityResult(
        schema_version=EVALUATION_SCHEMA_VERSION,
        checker_version=CONTENT_INTEGRITY_VERSION, passed=True,
        plot_facts_preserved=True, characters_preserved=True,
        relationships_preserved=True, constraints_preserved=True,
        new_major_events=False, removed_major_events=False,
        violations=[v], reasoning_summary="ok", deterministic=False, blind=True)
    assert ContentIntegrityResult.from_dict(ci.to_dict()).passed is True

    d = FeedbackDecision(
        schema_version=FEEDBACK_DECISION_SCHEMA_VERSION, outcome=FEEDBACK_ACCEPT,
        reason="resolved", content_integrity_passed=True,
        content_integrity=ci.to_dict(),
        style_fidelity={"high_priority_deviations_before": 2,
                        "high_priority_deviations_after": 0, "improved": True},
        literary_quality={"before": 8.0, "after": 8.0, "drop": 0.0,
                          "max_literary_drop": 0.5, "drop_exceeded": False,
                          "evaluated": True},
        iteration=1, max_iterations=2, author_id="austen", passage_id="p1")
    assert FeedbackDecision.from_dict(d.to_dict()).outcome == FEEDBACK_ACCEPT


def test_dimension_score_backward_compatible_defaults():
    # 旧字段缺失时（无 assessment_status / verified_evidence_count）→ 向后兼容默认。
    d = {"dimension": "plot_logic", "label": "Plot Logic", "score": 7.0,
         "summary": "s", "strength": "st", "weakness": "w", "evidence": ["a", "b"]}
    ds = DimensionScore.from_dict(d)
    assert ds.assessment_status == ASSESSMENT_OBSERVED
    assert ds.verified_evidence_count == 2


def test_schema_version_guard_rejects_wrong_version():
    cases = {
        ActualStyleProfile: {
            "schema_version": "999.0.0", "author_id": "a", "passage_id": "p",
            "passage_hash": "h", "style_plan_id": "sp",
            "layer_a_statistical": {}, "layer_a_judgment": {},
            "layer_b_narrative": None, "layer_c_strategies": [],
            "layer_d_stylometric": None, "unavailable": {}, "provenance": {},
        },
        LiteraryEvaluation: {
            "schema_version": "999.0.0", "author_id": "a", "passage_id": "p",
            "dimensions": {}, "weights": {}, "total_score": 0.0, "summary": "s",
            "blind": True, "evaluator_version": "v",
        },
        RevisionPlan: {
            "schema_version": "999.0.0", "author_id": "a", "passage_id": "p",
            "style_plan_id": "sp", "revision_items": [], "metadata": {},
        },
        RevisionResult: {
            "schema_version": "999.0.0", "author_id": "a", "passage_id": "p",
            "original_passage_hash": "a", "revised_passage_hash": "b",
            "revised_text": "t", "change_descriptions": [],
            "revision_items_applied": [], "blind": True, "rewriter_version": "v",
        },
        ContentIntegrityResult: {
            "schema_version": "999.0.0", "checker_version": "v", "passed": True,
            "plot_facts_preserved": True, "characters_preserved": True,
            "relationships_preserved": True, "constraints_preserved": True,
            "new_major_events": False, "removed_major_events": False,
            "violations": [], "reasoning_summary": "s", "deterministic": False,
            "blind": True,
        },
        FeedbackDecision: {
            "schema_version": "999.0.0", "outcome": "accept", "reason": "r",
            "content_integrity_passed": None, "content_integrity": None,
            "style_fidelity": {}, "literary_quality": {}, "iteration": 1,
            "max_iterations": 2,
        },
    }
    for cls, d in cases.items():
        with pytest.raises(EvalError):
            cls.from_dict(d)


# --------------------------------------------------------------------------- #
# LiteraryEvaluator
# --------------------------------------------------------------------------- #
def test_literary_evaluator_blind_prompt_and_total():
    prov = RecordingProvider(_eval_response())
    ev = LiteraryEvaluator(prov, blind=True)
    out = ev.evaluate(TEXT, author_id="austen", passage_id="p1")
    assert isinstance(out, LiteraryEvaluation)
    assert out.total_score == pytest.approx(7.0, abs=0.01)  # equal scores 8,7,6,7,7,7 -> 7.0
    # 盲测：system+user 不含作者名或模仿令牌
    system = prov.messages[0][0]["content"]
    for tok in IMITATION_INSTRUCTION_TOKENS:
        assert tok not in system.lower()
    assert "austen" not in system.lower()


def test_literary_evaluator_score_out_of_range_rejected():
    prov = RecordingProvider(_eval_response({"plot_logic": 12}))
    with pytest.raises(LLMResponseError):
        LiteraryEvaluator(prov).evaluate(TEXT)


def test_literary_evaluator_missing_dimension_rejected():
    raw = {"dimensions": {"plot_logic": {"score": 8, "summary": "s", "strength": "st",
                                         "weakness": "w", "evidence": ["e"]}},
           "summary": "ok"}
    prov = RecordingProvider(json.dumps(raw))
    with pytest.raises(LLMResponseError):
        LiteraryEvaluator(prov).evaluate(TEXT)


def test_literary_evaluator_evidence_verified():
    raw = json.loads(_eval_response())
    # 注入一条无法逐字验证的引文，应被丢弃（不保留未验证引文）
    raw["dimensions"]["plot_logic"]["evidence"].append("a fabricated quote")
    prov = RecordingProvider(json.dumps(raw))
    out = LiteraryEvaluator(prov).evaluate(TEXT)
    assert out.dimensions["plot_logic"].evidence == ["She walked alone at dusk."]


def test_literary_evaluator_unconfigured_returns_unavailable():
    out = LiteraryEvaluator(UnconfiguredLLMProvider()).evaluate(TEXT)
    assert isinstance(out, AnalysisUnavailable)


def test_literary_insufficient_dim_excluded_from_total():
    # plot_logic 的 evidence 全部无法逐字验证 → insufficient_evidence，不进加权总分。
    raw = json.loads(_eval_response())
    raw["dimensions"]["plot_logic"]["evidence"] = ["a fabricated quote"]
    prov = RecordingProvider(json.dumps(raw))
    out = LiteraryEvaluator(prov).evaluate(TEXT)
    assert isinstance(out, LiteraryEvaluation)
    dim = out.dimensions["plot_logic"]
    assert dim.assessment_status == ASSESSMENT_INSUFFICIENT
    assert dim.verified_evidence_count == 0
    # 其余 5 维 observed，总分只在这些维度上加权。
    observed = [d for d in out.dimensions.values()
                if d.assessment_status == ASSESSMENT_OBSERVED]
    assert len(observed) == 5


def test_literary_all_insufficient_returns_unavailable():
    # 6 维证据全部验证失败 → 整体 unavailable，绝不返回伪总分。
    raw = json.loads(_eval_response())
    for dim in raw["dimensions"].values():
        dim["evidence"] = ["a fabricated quote"]
    prov = RecordingProvider(json.dumps(raw))
    out = LiteraryEvaluator(prov).evaluate(TEXT)
    assert isinstance(out, AnalysisUnavailable)


def test_literary_extra_dimension_rejected():
    # 严格 exactly-six：多余未知维度 → reject（绝不 silent ignore）。
    raw = json.loads(_eval_response())
    raw["dimensions"]["mystery_dim"] = {"score": 8, "summary": "s", "strength": "st",
                                        "weakness": "w", "evidence": ["e"]}
    prov = RecordingProvider(json.dumps(raw))
    with pytest.raises(LLMResponseError):
        LiteraryEvaluator(prov).evaluate(TEXT)


# --------------------------------------------------------------------------- #
# compare_target_actual
# --------------------------------------------------------------------------- #
def test_compare_language_band_classification():
    profile = _make_profile(features={"dialogue_ratio": _feature_summary(mean=0.5)})
    plan = _plan()
    thresholds = _thresholds()
    # 实际 0.2 < q33=0.3 → low，目标 medium → below
    below = compare_target_actual(plan, profile, _actual({"dialogue_ratio": 0.2}),
                                  thresholds)
    assert below.language_deviations[0].status == "below"
    # 实际 0.8 > q67=0.7 → high → above
    above = compare_target_actual(plan, profile, _actual({"dialogue_ratio": 0.8}),
                                  thresholds)
    assert above.language_deviations[0].status == "above"
    # 实际 0.5 → medium → on_target
    on = compare_target_actual(plan, profile, _actual({"dialogue_ratio": 0.5}),
                               thresholds)
    assert on.language_deviations[0].status == "on_target"
    # 无阈值 → not_measurable
    none = compare_target_actual(plan, profile, _actual({"dialogue_ratio": 0.5}), {})
    assert none.language_deviations[0].status == "not_measurable"


def test_compare_narrative_and_strategy():
    profile = _make_profile(features={"dialogue_ratio": _feature_summary(mean=0.5)})
    plan = _plan()
    thresholds = _thresholds()
    # 叙事 on_target / off_target / not_measurable
    on = compare_target_actual(
        plan, profile, _actual({"dialogue_ratio": 0.5}, narrative={"pov": "third"}),
        thresholds)
    assert on.narrative_deviations[0].status == "on_target"
    off = compare_target_actual(
        plan, profile, _actual({"dialogue_ratio": 0.5}, narrative={"pov": "first"}),
        thresholds)
    assert off.narrative_deviations[0].status == "off_target"
    # 策略覆盖：active 但未命中
    assert off.strategy_coverage[0].active is True
    assert off.strategy_coverage[0].matched is False
    hit = compare_target_actual(
        plan, profile, _actual({"dialogue_ratio": 0.5}, narrative={"pov": "third"},
                               matched_strategy_ids=["austen::s1"]),
        thresholds)
    assert hit.strategy_coverage[0].matched is True


# --------------------------------------------------------------------------- #
# build_revision_plan
# --------------------------------------------------------------------------- #
def test_revision_plan_priority_order_and_interpretable():
    profile = _make_profile(features={"dialogue_ratio": _feature_summary(mean=0.5)})
    plan = _plan()
    # 制造：语言偏离（P3）+ 叙事偏离（P1）+ 策略未命中（P2）+ 文学 plot_logic 弱（P0）
    cmp_ = compare_target_actual(
        plan, profile,
        _actual({"dialogue_ratio": 0.2}, narrative={"pov": "first"}),
        _thresholds())
    eval_ = LiteraryEvaluation(
        schema_version=EVALUATION_SCHEMA_VERSION, author_id="austen", passage_id="p1",
        dimensions={"plot_logic": DimensionScore(
            dimension="plot_logic", label="Plot Logic", score=3.0, summary="s",
            strength="st", weakness="the ending is rushed", evidence=["e"])},
        weights={"plot_logic": 1.0}, total_score=3.0, summary="ok", blind=True,
        evaluator_version="0.1.0")
    rev = build_revision_plan(cmp_, plan, evaluation=eval_)
    priorities = [i.priority for i in rev.revision_items]
    assert priorities == sorted(priorities, key=lambda p: ("P0", "P1", "P2", "P3", "P4").index(p))
    assert "P0" in priorities and "P1" in priorities and "P2" in priorities and "P3" in priorities
    # 可解释：指令不含作者名 / 原始数值 / 微观 stylometric 指纹
    for item in rev.revision_items:
        for name in ("austen", "dickens", "Austen", "Dickens"):
            assert name not in item.instruction
        assert not any(ch.isdigit() for ch in item.instruction)
        for token in ("char 3-gram", "trigram", "char_trigram", "function word",
                      "centroid", "PCA"):
            assert token.lower() not in item.instruction.lower()


def test_revision_plan_p4_never_emitted():
    profile = _make_profile(features={"dialogue_ratio": _feature_summary(mean=0.5)})
    plan = _plan()
    cmp_ = compare_target_actual(
        plan, profile, _actual({"dialogue_ratio": 0.5}, narrative={"pov": "third"},
                               matched_strategy_ids=["austen::s1"]),
        _thresholds())
    rev = build_revision_plan(cmp_, plan, evaluation=None)
    assert all(i.priority != "P4" for i in rev.revision_items)
    assert "stylometric 距离仅诊断" in rev.metadata["stylometric_note"]


def test_revision_plan_empty_when_on_target():
    profile = _make_profile(features={"dialogue_ratio": _feature_summary(mean=0.5)})
    plan = _plan()
    cmp_ = compare_target_actual(
        plan, profile, _actual({"dialogue_ratio": 0.5}, narrative={"pov": "third"},
                               matched_strategy_ids=["austen::s1"]),
        _thresholds())
    rev = build_revision_plan(cmp_, plan, evaluation=None)
    assert rev.revision_items == []


# --------------------------------------------------------------------------- #
# RevisionRewriter
# --------------------------------------------------------------------------- #
def test_rewriter_prompt_has_p0_constraint_and_is_blind():
    system = _build_system_prompt([RevisionItem(
        priority="P3", category="language", target="f",
        instruction="Use dialogue relatively often.", reason="r")])
    assert "Do NOT change the plot" in system
    assert "characters" in system
    for tok in IMITATION_INSTRUCTION_TOKENS:
        assert tok not in system.lower()


def test_rewriter_returns_result():
    prov = RecordingProvider(json.dumps({
        "revised_text": "She walked alone at dusk, in the quiet garden.",
        "change_descriptions": ["combined two sentences"],
    }))
    item = RevisionItem(priority="P3", category="language", target="f",
                        instruction="Use dialogue relatively often.", reason="r")
    plan = RevisionPlan(schema_version=EVALUATION_SCHEMA_VERSION, author_id="austen",
                        passage_id="p1", style_plan_id="sp1", revision_items=[item],
                        metadata={})
    out = RevisionRewriter(prov).rewrite(TEXT, plan)
    assert isinstance(out, RevisionResult)
    assert out.revised_text.startswith("She walked alone")
    assert out.change_descriptions == ["combined two sentences"]
    assert out.original_passage_hash != out.revised_passage_hash


def test_rewriter_empty_plan_short_circuits():
    prov = RecordingProvider("")
    plan = RevisionPlan(schema_version=EVALUATION_SCHEMA_VERSION, author_id="austen",
                        passage_id="p1", style_plan_id="sp1", revision_items=[],
                        metadata={})
    out = RevisionRewriter(prov).rewrite(TEXT, plan)
    assert isinstance(out, RevisionResult)
    assert out.revised_text == TEXT
    assert prov.messages == []  # 绝不烧 token


def test_rewriter_author_name_leak_fails_closed():
    prov = RecordingProvider("{}")
    item = RevisionItem(priority="P3", category="language", target="f",
                        instruction="Use dialogue relatively often.", reason="r")
    plan = RevisionPlan(schema_version=EVALUATION_SCHEMA_VERSION, author_id="austen",
                        passage_id="p1", style_plan_id="sp1", revision_items=[item],
                        metadata={})
    with pytest.raises(EvalError):
        RevisionRewriter(prov).rewrite("Jane Austen wrote this passage.", plan,
                                       author_names=["Jane Austen"])


# --------------------------------------------------------------------------- #
# ContentIntegrityChecker（Phase 8.1）
# --------------------------------------------------------------------------- #
_CI_OK = json.dumps({
    "plot_facts_preserved": True, "characters_preserved": True,
    "relationships_preserved": True, "constraints_preserved": True,
    "new_major_events": False, "removed_major_events": False,
    "violations": [], "summary": "no content changed",
})


def test_integrity_identical_text_passes_deterministically():
    prov = RecordingProvider("")
    out = ContentIntegrityChecker(prov).check(TEXT, TEXT, _REQUEST)
    assert isinstance(out, ContentIntegrityResult)
    assert out.passed is True and out.deterministic is True
    assert prov.messages == []  # 零 token 短路


def test_integrity_empty_revised_fails_deterministically():
    prov = RecordingProvider("")
    out = ContentIntegrityChecker(prov).check(TEXT, "   ", _REQUEST)
    assert isinstance(out, ContentIntegrityResult)
    assert out.passed is False and out.deterministic is True
    assert out.removed_major_events is True
    assert any(v.kind == "removed_event" and v.severity == INTEGRITY_CRITICAL
               for v in out.violations)
    assert prov.messages == []


def test_integrity_llm_passed():
    prov = RecordingProvider(_CI_OK)
    out = ContentIntegrityChecker(prov).check(TEXT, "She walked alone. The garden quieted.", _REQUEST)
    assert isinstance(out, ContentIntegrityResult)
    assert out.passed is True and out.deterministic is False
    assert out.new_major_events is False and out.removed_major_events is False


def test_integrity_llm_fail_on_removed_event():
    raw = json.loads(_CI_OK)
    raw["removed_major_events"] = True
    raw["violations"] = [{"kind": "removed_event", "severity": INTEGRITY_CRITICAL,
                          "description": "removed the garden scene"}]
    prov = RecordingProvider(json.dumps(raw))
    out = ContentIntegrityChecker(prov).check(TEXT, "She walked alone.", _REQUEST)
    assert isinstance(out, ContentIntegrityResult)
    assert out.passed is False
    assert out.removed_major_events is True


def test_integrity_non_bool_rejected():
    raw = json.loads(_CI_OK)
    raw["characters_preserved"] = "yes"  # 非法：必须是 boolean
    prov = RecordingProvider(json.dumps(raw))
    with pytest.raises(LLMResponseError):
        ContentIntegrityChecker(prov).check(TEXT, "changed", _REQUEST)


def test_integrity_bad_violation_kind_rejected():
    raw = json.loads(_CI_OK)
    raw["violations"] = [{"kind": "style_issue", "severity": INTEGRITY_CRITICAL,
                          "description": "x"}]
    prov = RecordingProvider(json.dumps(raw))
    with pytest.raises(LLMResponseError):
        ContentIntegrityChecker(prov).check(TEXT, "changed", _REQUEST)


def test_integrity_unconfigured_returns_unavailable():
    out = ContentIntegrityChecker(UnconfiguredLLMProvider()).check(TEXT, "changed", _REQUEST)
    assert isinstance(out, AnalysisUnavailable)


def test_integrity_prompt_is_blind():
    prov = RecordingProvider(_CI_OK)
    ContentIntegrityChecker(prov, blind=True).check(
        "She walked alone at dusk.", "She walked alone at dusk, quietly.", _REQUEST,
        author_names=["Jane Austen"])
    system = prov.messages[0][0]["content"]
    user = prov.messages[0][1]["content"]
    for tok in IMITATION_INSTRUCTION_TOKENS:
        assert tok not in system.lower()
    assert "austen" not in (system + user).lower()


def test_integrity_author_name_in_text_fails_closed():
    # 若正文本身含作者身份名，泄露守卫 fail-closed（绝不发送）。
    prov = RecordingProvider(_CI_OK)
    with pytest.raises(EvalError):
        ContentIntegrityChecker(prov).check(
            "Jane Austen wrote this.", "Jane Austen wrote this, too.", _REQUEST,
            author_names=["Jane Austen"])


# --------------------------------------------------------------------------- #
# decide_feedback_outcome
# --------------------------------------------------------------------------- #
def _cmp(n_lang=0, n_narr=0, n_strat=0):
    return ComparisonResult(
        author_id="austen", passage_id="p1",
        language_deviations=[FeatureDeviation(
            feature_id=f"f{i}", target_band="medium", actual_band="low",
            target_value=0.5, actual_value=0.2, status="below", measurable=True,
            reason="r") for i in range(n_lang)],
        narrative_deviations=[NarrativeDeviation(
            field=f"n{i}", target_value="third", actual_value="first",
            status="off_target", reason="r") for i in range(n_narr)],
        strategy_coverage=[StrategyCoverage(
            strategy_id=f"s{i}", active=True, matched=False, evidence_quotes=[])
            for i in range(n_strat)],
    )


def _ci(passed=True, *, removed=False, new=False, violations=None):
    return ContentIntegrityResult(
        schema_version=EVALUATION_SCHEMA_VERSION,
        checker_version=CONTENT_INTEGRITY_VERSION, passed=passed,
        plot_facts_preserved=passed, characters_preserved=passed,
        relationships_preserved=passed, constraints_preserved=passed,
        new_major_events=new, removed_major_events=removed,
        violations=violations or [], reasoning_summary="ok",
        deterministic=False, blind=True)


def test_decide_feedback_outcome_rules():
    # after None → roll_back
    assert decide_feedback_outcome(_cmp(1), None).outcome == FEEDBACK_ROLL_BACK
    # 零高优先级偏差 → accept
    assert decide_feedback_outcome(_cmp(1), _cmp(0)).outcome == FEEDBACK_ACCEPT
    # 改善且未达上限 → continue
    assert decide_feedback_outcome(_cmp(3), _cmp(1), iteration=1,
                                   max_iterations=2).outcome == FEEDBACK_CONTINUE
    # 改善但已达上限 → accept
    assert decide_feedback_outcome(_cmp(3), _cmp(1), iteration=2,
                                   max_iterations=2).outcome == FEEDBACK_ACCEPT
    # 未改善 → roll_back
    assert decide_feedback_outcome(_cmp(1), _cmp(2)).outcome == FEEDBACK_ROLL_BACK


def test_decide_no_action_when_no_revision():
    # 改写计划为空 → no_action，独立于 roll_back（不是"回滚"）。
    d = decide_feedback_outcome(_cmp(2), None, no_revision=True)
    assert d.outcome == FEEDBACK_NO_ACTION
    assert d.content_integrity_passed is None
    assert d.content_integrity is None
    assert d.literary_quality["evaluated"] is False


def test_decide_integrity_violation_rolls_back_even_if_style_improved():
    # STEP 1 最高 gate：内容破坏 → roll_back，即便风格偏差减少（P1/P2/P3 全改善）。
    ci = _ci(passed=False, removed=True)
    d = decide_feedback_outcome(
        _cmp(3), _cmp(1), content_integrity=ci,
        literary_before=8.0, literary_after=8.0)
    assert d.outcome == FEEDBACK_ROLL_BACK
    assert d.content_integrity_passed is False
    assert "integrity" in d.reason.lower()


def test_decide_literary_drop_rolls_back():
    # STEP 2：文学总分下降超过容忍度 → roll_back（即便风格改善）。
    d = decide_feedback_outcome(
        _cmp(3), _cmp(1), content_integrity=_ci(passed=True),
        literary_before=8.0, literary_after=5.0)
    assert d.outcome == FEEDBACK_ROLL_BACK
    assert d.literary_quality["drop_exceeded"] is True
    assert d.literary_quality["drop"] == pytest.approx(3.0)


def test_decide_literary_drop_within_tolerance_continues():
    # 文学下降在容忍度内 + 风格改善 → continue（不阻断）。
    d = decide_feedback_outcome(
        _cmp(3), _cmp(1), content_integrity=_ci(passed=True),
        literary_before=8.0, literary_after=7.7)
    assert d.outcome == FEEDBACK_CONTINUE
    assert d.literary_quality["drop_exceeded"] is False


def test_decide_literary_guard_skipped_when_unavailable():
    # literary_before/after 为 None → 文学 guard 不阻断，进入风格 gate。
    d = decide_feedback_outcome(_cmp(1), _cmp(0))
    assert d.outcome == FEEDBACK_ACCEPT
    assert d.literary_quality["evaluated"] is False


def test_decide_custom_tolerance_configurable():
    # max_literary_drop 可配置（非科学真值）：调大容忍度后不再 roll_back。
    policy = EvaluationPolicy(max_literary_drop=5.0)
    d = decide_feedback_outcome(
        _cmp(3), _cmp(1), content_integrity=_ci(passed=True),
        literary_before=8.0, literary_after=5.0, policy=policy)
    assert d.outcome == FEEDBACK_CONTINUE


def test_decide_reports_style_and_literary_separately():
    # 绝不合并成单一加权总分：style_fidelity 与 literary_quality 分别报告。
    d = decide_feedback_outcome(
        _cmp(3), _cmp(1), content_integrity=_ci(passed=True),
        literary_before=8.0, literary_after=7.7)
    assert "high_priority_deviations_before" in d.style_fidelity
    assert "high_priority_deviations_after" in d.style_fidelity
    assert "before" in d.literary_quality and "after" in d.literary_quality
    assert "weighted_total" not in d.literary_quality


def test_high_priority_count_excludes_p0_p4():
    assert _count_high_priority_deviations(_cmp(1, 1, 1)) == 3
    assert _count_high_priority_deviations(_cmp(0, 0, 0)) == 0


def test_decide_literary_after_unavailable_rolls_back_even_if_style_improved():
    # 基线有效 + 改写后文学评价 unavailable + 风格改善 → 仍 roll_back（无法验证文学保留）。
    d = decide_feedback_outcome(
        _cmp(3), _cmp(1), content_integrity=_ci(passed=True),
        literary_before=8.0, literary_after=None)
    assert d.outcome == FEEDBACK_ROLL_BACK
    assert "post-revision literary evaluation unavailable" in d.reason
    assert d.literary_quality["guard"] == "unavailable"
    assert d.literary_quality["evaluated"] is False
    assert d.literary_quality["after"] is None   # 绝不把 unavailable 转成 0 分


def test_decide_literary_after_unavailable_rolls_back_even_if_style_perfect():
    # 即便风格偏差归零（perfect），只要文学无法验证 → roll_back。
    d = decide_feedback_outcome(
        _cmp(3), _cmp(0), content_integrity=_ci(passed=True),
        literary_before=8.0, literary_after=None)
    assert d.outcome == FEEDBACK_ROLL_BACK


def test_decide_literary_before_unavailable_marks_guard_unavailable():
    # 基线本身 unavailable → 不伪造基线，允许走 Style Fidelity，但显式标记 guard 不可用。
    d = decide_feedback_outcome(_cmp(2), _cmp(0),
                                literary_before=None, literary_after=None)
    assert d.outcome == FEEDBACK_ACCEPT
    assert d.literary_quality["guard"] == "unavailable"
    assert d.literary_quality["evaluated"] is False
    assert d.literary_quality["before"] is None and d.literary_quality["after"] is None


# --------------------------------------------------------------------------- #
# measure_actual_profile（Layer A 统计 + Layer D 诊断，合成 fixture）
# --------------------------------------------------------------------------- #
def _build_stylo_fixture(tmp_path, author_id="austen"):
    base = tmp_path
    (base / "chunks").mkdir(parents=True)
    (base / "analysis" / "stylometry").mkdir(parents=True)
    (base / "analysis" / "style_profiles").mkdir(parents=True)
    work_id = "emma"
    texts = [
        "It is a truth universally acknowledged. The garden was quiet in the evening.",
        "She walked alone at dusk and thought of her home and of the years gone by.",
        "He spoke little, but his meaning was clear enough to those who listened well.",
    ]
    lines = [json.dumps({"chunk_id": f"c{i}", "text": t, "work_id": work_id})
             for i, t in enumerate(texts)]
    (base / "chunks" / f"{work_id}__2000.jsonl").write_text("\n".join(lines))
    vec = StylometricVectorizer().fit(texts)
    (base / "analysis" / "stylometry" / "index.json").write_text(
        json.dumps({"feature_names": vec.feature_names_, "train_work_ids": [work_id]}))
    centroid = vec.transform(texts).mean(axis=0).tolist()
    (base / "analysis" / "style_profiles" / "stylometric_author_targets.json").write_text(
        json.dumps({"authors": {author_id: {"author_id": author_id,
                                            "centroid": centroid}}}))
    return base


def test_measure_actual_profile_deterministic_layers(tmp_path):
    base = _build_stylo_fixture(tmp_path)
    profile = _make_profile()
    actual = measure_actual_profile(
        TEXT, author_id="austen", passage_id="p1", style_plan_id="sp1",
        profile=profile, provider=UnconfiguredLLMProvider(), data_root_=base)
    assert actual.schema_version == EVALUATION_SCHEMA_VERSION
    assert len(actual.layer_a_statistical) == 22
    assert actual.layer_a_judgment == {}
    assert actual.layer_b_narrative is None
    assert actual.layer_c_strategies == []
    assert actual.layer_d_stylometric is not None
    assert actual.layer_d_stylometric["feature_names_match"] is True
    assert isinstance(actual.layer_d_stylometric["cosine_distance"], float)
    # 未配置的 LLM 层显式记录 unavailable，绝不伪造
    assert "layer_a_judgment" in actual.unavailable
    assert "layer_b_narrative" in actual.unavailable
    assert "layer_c_strategies" in actual.unavailable


def test_layer_d_diagnostic_fails_closed_on_mismatch(tmp_path):
    base = _build_stylo_fixture(tmp_path)
    (base / "analysis" / "stylometry" / "index.json").write_text(
        json.dumps({"feature_names": ["bogus:1", "bogus:2"], "train_work_ids": ["emma"]}))
    with pytest.raises(EvalError):
        _layer_d_diagnostic(TEXT, "austen", base)
