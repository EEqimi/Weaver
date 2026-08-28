# tests/test_style_planner.py
"""Phase 6 Style Planner & Prompt Compiler 测试（全部确定性，无 LLM，无生成正文）。

覆盖：三层 schema 往返、激活政策（candidate_core 门槛 / descriptive / experimental /
diagnostic）、控制预算（语言 / 叙事 / 策略，绝不静默丢弃）、POV 覆盖、确定性复现、
提示词铁律（不提作者名 / 不写微观 stylometric / 不改写用户 brief）、画像完整性 fail-closed。
"""
import json
from pathlib import Path

import pytest

from knowledge.planning.bands import (
    band_label, compute_band_thresholds, describe_feature,
)
from knowledge.planning.compiler import CompiledPrompt, PromptCompiler
from knowledge.planning.planner import StylePlanner
from knowledge.planning.policy import language_activation
from knowledge.planning.schema import (
    PlannedControl, PlannedNarrativeControl, PlannedStrategy, PlannerPolicy,
    PlanningError, PromptBudgetError, StylePlan, WritingRequest,
    make_style_plan_id,
)
from knowledge.profiles.style_profile import (
    AuthorStyleProfile, AuthorStyleProfileSynthesizer, _reproducibility_hash,
)
from knowledge.schema.versions import STYLE_PLAN_SCHEMA_VERSION


# --------------------------------------------------------------------------- #
# 构造辅助
# --------------------------------------------------------------------------- #
def _feature_summary(value_type="continuous", measurement_type="statistical", **over):
    base = {
        "n": 833, "mean": 0.5, "variance": 0.01, "std": 0.1,
        "n_expected": 833, "n_total": 833, "n_valid": 833,
        "n_missing": 0, "n_unobservable": 0, "n_insufficient": 0,
        "value_type": value_type, "measurement_type": measurement_type,
        "confidence": {"n": 833, "mean": 0.8},
    }
    base.update(over)
    return base


def _canonical(cid, name="N", status="discovered", works=1, chunks=1, conf=0.8, raw=1):
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
        "number_of_raw_observations": raw,
    }


def _make_profile(author_id="austen", full_features=None, sampled_features=None,
                  narrative=None, canonicals=None, profile_work_ids=None):
    synth = AuthorStyleProfileSynthesizer()
    return synth.synthesize(
        author_id=author_id,
        train_work_ids=["emma", "pride_and_prejudice"],
        held_out_work_ids=["persuasion"],
        profile_work_ids=profile_work_ids or ["emma", "pride_and_prejudice"],
        full_corpus_features=full_features or {},
        sampled_features=sampled_features or {},
        sampled_narrative=narrative or {},
        canonical_strategies=canonicals or [],
        stylometry_author_target={"author_id": author_id, "n_samples": 833,
                                  "source_work_ids": ["emma", "pride_and_prejudice"],
                                  "centroid_norm": 0.123456},
        stylometry_validation_metadata={"n_features": 954},
    )


def _pov_narrative(mode="third"):
    return {"n": 20, "n_total": 20, "n_valid": 20, "n_missing": 0,
            "pov": {"n": 20, "counts": {mode: 19, "other": 1},
                    "proportions": {mode: 0.95, "other": 0.05}, "mode": mode}}


REQUEST = WritingRequest(
    content="A short test brief.", desired_length="short_scene",
    target_words=300, language="english", pov=None, constraints=["No new characters"])


def _band_thresholds(feature_ids, q33=0.3, q67=0.7):
    """合成一份最小经验 band 阈值 artifact（供测试：使特征可编译）。"""
    return {
        "schema_version": "0.1.0",
        "train_only": True,
        "features": {
            fid: {"q33": q33, "q67": q67, "n": 100, "min": 0.0,
                  "median": 0.5, "max": 1.0}
            for fid in feature_ids
        },
    }


def _planner(feature_ids=None, q33=0.3, q67=0.7):
    return StylePlanner(band_thresholds=_band_thresholds(feature_ids or [], q33, q67))


# --------------------------------------------------------------------------- #
# Schema 往返
# --------------------------------------------------------------------------- #
def test_writing_request_round_trip():
    d = REQUEST.to_dict()
    assert WritingRequest.from_dict(json.loads(json.dumps(d))).to_dict() == d


def test_writing_request_empty_content_rejected():
    with pytest.raises(ValueError):
        WritingRequest(content="   ")


def test_planner_policy_round_trip():
    p = PlannerPolicy(max_primary_controls=3, max_strategies=2)
    assert PlannerPolicy.from_dict(p.to_dict()).to_dict() == p.to_dict()


def test_planned_control_missing_field_rejected():
    d = {"feature_id": "x", "registry_control_role": "descriptive",
         "activation": "weak", "bucket": "primary"}
    with pytest.raises(ValueError):
        PlannedControl.from_dict(d)


def test_planned_strategy_round_trip():
    s = PlannedStrategy(
        canonical_strategy_id="a::x", canonical_name="N", support_status="validated",
        confidence=0.9, control_priority=1, n_supporting_works=2,
        n_supporting_chunks=5, activation="active", trigger="t", operation="o",
        effect="e", reason="r")
    assert PlannedStrategy.from_dict(s.to_dict()).to_dict() == s.to_dict()


def test_style_plan_from_dict_round_trip():
    plan = StylePlanner().plan(_make_profile(), REQUEST)
    d = plan.to_dict()
    assert StylePlan.from_dict(json.loads(json.dumps(d))).to_dict() == d


def test_style_plan_wrong_schema_version_rejected():
    plan = StylePlanner().plan(_make_profile(), REQUEST)
    d = plan.to_dict()
    d["schema_version"] = "9.9.9"
    with pytest.raises(ValueError):
        StylePlan.from_dict(d)


def test_make_style_plan_id_deterministic_and_sensitive():
    a = make_style_plan_id("austen", "h", REQUEST, PlannerPolicy())
    b = make_style_plan_id("austen", "h", REQUEST, PlannerPolicy())
    assert a == b
    c = make_style_plan_id("dickens", "h", REQUEST, PlannerPolicy())
    assert a != c


# --------------------------------------------------------------------------- #
# 激活政策
# --------------------------------------------------------------------------- #
def test_candidate_core_full_corpus_strong():
    act, _ = language_activation("mean_sentence_length", "candidate_core",
                                 "full_train_corpus", _feature_summary(mean=17.3, variance=32.0, std=5.6),
                                 PlannerPolicy())
    assert act == "strong"


def test_candidate_core_insufficient_evidence_suppressed():
    act, reason = language_activation(
        "mean_sentence_length", "candidate_core", "full_train_corpus",
        _feature_summary(n_insufficient=3), PlannerPolicy())
    assert act == "suppressed"
    assert "insufficient" in reason


def test_candidate_core_sampled_scope_medium():
    act, _ = language_activation("mean_sentence_length", "candidate_core",
                                 "calibration_sample", _feature_summary(),
                                 PlannerPolicy())
    assert act == "medium"


def test_descriptive_weak():
    act, _ = language_activation("comma_density", "descriptive", "full_train_corpus",
                                 _feature_summary(), PlannerPolicy())
    assert act == "weak"


def test_experimental_reference():
    act, _ = language_activation("simile_frequency", "experimental",
                                 "calibration_sample", _feature_summary(),
                                 PlannerPolicy())
    assert act == "reference"


def test_diagnostic_never_controls_generation():
    act, reason = language_activation("mfw_frequency", "diagnostic",
                                      "full_train_corpus", _feature_summary(),
                                      PlannerPolicy())
    assert act == "suppressed"
    assert "diagnostic" in reason


# --------------------------------------------------------------------------- #
# 控制预算（绝不静默丢弃）
# --------------------------------------------------------------------------- #
def test_language_budget_records_suppression():
    full = {fid: _feature_summary(mean=50.0) for fid in
            ["comma_density", "semicolon_density", "dash_density", "quotation_density",
             "exclamation_frequency", "question_frequency", "period_density",
             "long_sentence_ratio", "short_sentence_ratio", "sentence_length_cv",
             "type_token_ratio", "hapax_ratio", "word_repetition_rate",
             "mean_word_length", "connective_density"]}
    plan = _planner(full.keys()).plan(_make_profile(full_features=full), REQUEST)
    assert len(plan.language_controls) <= 6 + 4  # primary + secondary
    budget_dropped = [c for c in plan.suppressed_controls
                      if "suppressed_due_to_budget" in c.reason]
    assert budget_dropped  # 超出预算被显式记录，未静默


def test_experimental_lands_in_reference_controls():
    plan = StylePlanner().plan(
        _make_profile(sampled_features={"simile_frequency": _feature_summary(measurement_type="hybrid")}),
        REQUEST)
    assert all(c.bucket == "reference" for c in plan.reference_controls)
    assert any(c.feature_id == "simile_frequency" for c in plan.reference_controls)


def test_narrative_budget_max_4():
    narrative = _pov_narrative()
    narrative.update({
        "focalization": {"n": 20, "counts": {"internal": 19, "zero": 1},
                         "proportions": {}, "mode": "internal"},
        "narrator_presence": {"n": 20, "counts": {"low": 15, "medium": 5},
                              "proportions": {}, "mode": "low"},
        "narrative_distance": {"n": 20, "counts": {"close": 15, "medium": 5},
                               "proportions": {}, "mode": "close"},
        "perspective_stability": {"n": 20, "counts": {"stable": 16, "mostly_stable": 4},
                                  "proportions": {}, "mode": "stable"},
        "information_access": {"n": 20, "counts": {"limited": 19, "omniscient": 1},
                               "proportions": {}, "mode": "limited"},
    })
    plan = StylePlanner().plan(_make_profile(narrative=narrative), REQUEST)
    active = [n for n in plan.narrative_controls if n.activation == "medium"]
    assert len(active) <= 4


def test_strategy_selection_validated_candidate_active_discovered_reference():
    canonicals = [
        _canonical("a::v", status="validated", works=2, chunks=5),
        _canonical("a::c", status="candidate", works=1, chunks=3),
        _canonical("a::d", status="discovered", works=1, chunks=1),
    ]
    plan = StylePlanner().plan(_make_profile(canonicals=canonicals), REQUEST)
    active_ids = {s.canonical_strategy_id for s in plan.strategy_controls}
    ref_ids = {s.canonical_strategy_id for s in plan.reference_strategy_controls}
    assert active_ids == {"a::v", "a::c"}
    assert ref_ids == {"a::d"}


def test_strategy_budget_active_le_max():
    canonicals = [_canonical(f"a::v{i}", status="validated", works=2, chunks=5)
                  for i in range(10)]
    plan = StylePlanner().plan(_make_profile(canonicals=canonicals), REQUEST)
    assert len(plan.strategy_controls) <= 6
    assert len(plan.reference_strategy_controls) >= 4  # 溢出进 reference，未丢弃


# --------------------------------------------------------------------------- #
# POV 覆盖
# --------------------------------------------------------------------------- #
def test_pov_override_overrides_with_warning():
    req = WritingRequest(content="brief", desired_length="short_scene",
                         target_words=300, language="english", pov="first")
    plan = StylePlanner().plan(_make_profile(narrative=_pov_narrative("third")), req)
    pov_ctrl = next(n for n in plan.narrative_controls if n.field == "pov")
    assert pov_ctrl.overridden is True
    assert pov_ctrl.activation == "suppressed"
    assert any("conflicts with explicit user constraint" in w for w in plan.warnings)


def test_pov_override_no_warning_when_same():
    req = WritingRequest(content="brief", desired_length="short_scene",
                         target_words=300, language="english", pov="third")
    plan = StylePlanner().plan(_make_profile(narrative=_pov_narrative("third")), req)
    assert not any("conflicts" in w for w in plan.warnings)


# --------------------------------------------------------------------------- #
# 画像完整性 fail-closed
# --------------------------------------------------------------------------- #
def test_planner_rejects_hash_tamper():
    profile = _make_profile()
    profile.reproducibility_hash = "0" * 64
    with pytest.raises(PlanningError):
        StylePlanner().plan(profile, REQUEST)


def test_planner_rejects_held_out_contamination():
    profile = _make_profile()
    profile.author_scope["held_out_isolation"]["clean"] = False
    body = profile.to_dict()
    body.pop("reproducibility_hash", None)
    profile.reproducibility_hash = _reproducibility_hash(body)
    with pytest.raises(PlanningError):
        StylePlanner().plan(profile, REQUEST)


# --------------------------------------------------------------------------- #
# 确定性
# --------------------------------------------------------------------------- #
def test_plan_deterministic():
    profile = _make_profile(
        full_features={"mean_sentence_length": _feature_summary(mean=17.3)},
        canonicals=[_canonical("a::v", status="validated", works=2, chunks=5)])
    p1 = StylePlanner().plan(profile, REQUEST)
    p2 = StylePlanner().plan(profile, REQUEST)
    assert p1.to_dict() == p2.to_dict()
    assert p1.style_plan_id == p2.style_plan_id


def test_prompt_deterministic():
    profile = _make_profile(canonicals=[_canonical("a::v", status="validated", works=2, chunks=5)])
    plan = StylePlanner().plan(profile, REQUEST)
    t1 = PromptCompiler().compile(plan)
    t2 = PromptCompiler().compile(plan)
    assert t1.text == t2.text


# --------------------------------------------------------------------------- #
# 提示词铁律
# --------------------------------------------------------------------------- #
def test_compiled_prompt_sections_present():
    plan = StylePlanner().plan(_make_profile(), REQUEST)
    text = PromptCompiler().compile(plan).text
    for heading in ("ROLE", "CONTENT", "STYLE CONTROL", "NARRATIVE",
                    "CONDITIONAL STRATEGIES", "IMPORTANT"):
        assert f"## {heading}" in text


def test_compiled_prompt_no_author_name_no_stylometric():
    plan = StylePlanner().plan(
        _make_profile(canonicals=[_canonical("a::v", status="validated", works=2, chunks=5)]),
        REQUEST)
    text = PromptCompiler().compile(plan).text
    for banned in ("Austen", "Dickens", "Jane", "Charles", "centroid", "PCA",
                   "char_trigram", "mfw_frequency", "n-gram", "trigram"):
        assert banned not in text


def test_compiled_prompt_preserves_user_content():
    plan = StylePlanner().plan(_make_profile(), REQUEST)
    text = PromptCompiler().compile(plan).text
    assert "A short test brief." in text
    assert "No new characters" in text


def test_compiled_prompt_pov_override_uses_user_pov():
    req = WritingRequest(content="brief", desired_length="short_scene",
                         target_words=300, language="english", pov="first")
    plan = StylePlanner().plan(_make_profile(narrative=_pov_narrative("third")), req)
    text = PromptCompiler().compile(plan).text
    # POV 只出现在 CONTENT，不再作为 NARRATIVE 覆盖行重复出现。
    assert "first-person point of view" in text
    assert "explicit user requirement" not in text
    # 编译提示词里 POV 只出现一次（CONTENT-only，无 NARRATIVE 重复）。
    assert text.count("first-person point of view") == 1


def test_compiled_prompt_budget_degradation_drops_strategies_first():
    plan = StylePlanner().plan(
        _make_profile(canonicals=[_canonical(f"a::v{i}", status="validated", works=2, chunks=5)
                                  for i in range(6)]),
        REQUEST)
    full = PromptCompiler().compile(plan)
    assert not full.degraded
    budget = PlannerPolicy(max_prompt_chars=max(120, full.char_count - 100))
    degraded = PromptCompiler(budget).compile(plan)
    assert degraded.degraded is True
    assert degraded.char_count <= budget.max_prompt_chars
    assert any(r.startswith("strategy:") for r in degraded.removed_controls)
    assert "dropped" in degraded.degradation_note
    # 用户内容永不硬截断：CONTENT 里的核心句子仍在。
    assert "A short test brief." in degraded.text


def test_compiled_prompt_mandatory_overflow_raises():
    plan = StylePlanner().plan(_make_profile(), REQUEST)
    tiny = PlannerPolicy(max_prompt_chars=10)
    with pytest.raises(PromptBudgetError):
        PromptCompiler(tiny).compile(plan)


def test_compiled_prompt_sections_reconstruct_text():
    plan = StylePlanner().plan(
        _make_profile(canonicals=[_canonical("a::v", status="validated", works=2, chunks=5)]),
        REQUEST)
    prompt = PromptCompiler().compile(plan)
    assert len(prompt.sections) == 6
    assert PromptCompiler._assemble(prompt.sections) == prompt.text


# --------------------------------------------------------------------------- #
# 真实产物集成（gitignored data/）
# --------------------------------------------------------------------------- #
def _load_real_profile(author_id):
    path = Path("data") / "analysis" / "style_profiles" / f"{author_id}_style_profile.json"
    if not path.exists():
        pytest.skip("data/ artifacts 不存在（gitignored）")
    profile = AuthorStyleProfile.from_dict(json.loads(path.read_text(encoding="utf-8")))
    assert profile.verify_reproducibility_hash()
    return profile


def _load_real_band_thresholds():
    path = Path("data") / "analysis" / "planning" / "band_thresholds.json"
    if not path.exists():
        pytest.skip("data/analysis/planning/band_thresholds.json 不存在（gitignored）")
    return json.loads(path.read_text(encoding="utf-8"))


def test_real_artifacts_austen_dickens_plan_and_differ():
    bt = _load_real_band_thresholds()
    plans = {}
    for aid in ("austen", "dickens"):
        plans[aid] = StylePlanner(band_thresholds=bt).plan(_load_real_profile(aid), REQUEST)
    assert plans["austen"].style_plan_id != plans["dickens"].style_plan_id
    g = {aid: next(c.guidance for c in plans[aid].language_controls
                   if c.feature_id == "dialogue_ratio") for aid in ("austen", "dickens")}
    assert g["austen"] != g["dickens"]  # 对话占比方向相反


def test_real_artifacts_prompt_no_author_name():
    bt = _load_real_band_thresholds()
    for aid in ("austen", "dickens"):
        plan = StylePlanner(band_thresholds=bt).plan(_load_real_profile(aid), REQUEST)
        text = PromptCompiler().compile(plan).text
        for banned in ("Austen", "Dickens", "Jane", "Charles"):
            assert banned not in text
        assert text.count("## ") >= 6


# --------------------------------------------------------------------------- #
# Phase 6.1：经验 band 阈值 + 预算完整性
# --------------------------------------------------------------------------- #
def _chunk(work_id, **feature_values):
    return {"chunk_id": f"{work_id}_c", "work_id": work_id,
            "features": {fid: {"value": v} for fid, v in feature_values.items()}}


def test_band_thresholds_train_only():
    chunks = [
        _chunk("emma", comma_density=10.0),
        _chunk("pride_and_prejudice", comma_density=20.0),
        _chunk("persuasion", comma_density=999.0),  # held-out，必须被排除
    ]
    bt = compute_band_thresholds(chunks, train_work_ids=["emma", "pride_and_prejudice"])
    assert set(bt["train_work_ids"]) == {"emma", "pride_and_prejudice"}
    assert bt["train_only"] is True
    f = bt["features"]["comma_density"]
    assert f["n"] == 2
    assert f["max"] == 20.0  # 999 被排除


def test_band_thresholds_held_out_does_not_affect():
    base = [_chunk("emma", comma_density=10.0), _chunk("emma", comma_density=20.0)]
    with_held = base + [_chunk("persuasion", comma_density=999.0)]
    a = compute_band_thresholds(base, train_work_ids=["emma"])
    b = compute_band_thresholds(with_held, train_work_ids=["emma"])
    assert a["features"] == b["features"]  # held-out 不改变阈值
    assert b["n_train_chunks"] == 2


def test_band_thresholds_deterministic():
    chunks = [_chunk("emma", comma_density=10.0),
              _chunk("pride_and_prejudice", comma_density=20.0)]
    assert compute_band_thresholds(chunks) == compute_band_thresholds(chunks)


def test_band_thresholds_shared_across_authors():
    chunks = [
        _chunk("emma", comma_density=10.0),
        _chunk("pride_and_prejudice", comma_density=20.0),
        _chunk("great_expectations", comma_density=30.0),
        _chunk("david_copperfield", comma_density=40.0),
    ]
    bt = compute_band_thresholds(chunks)
    # 一份跨作者合并阈值（非 per-author），两位作者共用同一 band 系统。
    assert "q33" in bt["features"]["comma_density"]
    assert "q67" in bt["features"]["comma_density"]
    assert "austen" not in bt["features"] and "dickens" not in bt["features"]


def test_band_label_tertiles():
    bt = _band_thresholds(["f"], q33=0.3, q67=0.7)
    assert band_label("f", 0.1, bt) == "low"
    assert band_label("f", 0.3, bt) == "medium"  # 边界 ∈ medium
    assert band_label("f", 0.5, bt) == "medium"
    assert band_label("f", 0.7, bt) == "medium"
    assert band_label("f", 0.9, bt) == "high"
    assert band_label("f", 0.5, None) is None


def test_describe_feature_literal_no_overclaim():
    bt = _band_thresholds(["comma_density", "semicolon_density", "mean_sentence_length"])
    comma_high = describe_feature("comma_density", {"mean": 0.9}, bt)
    assert comma_high == "Use commas relatively frequently."
    for banned in ("subordinate", "parenthetical", "clause", "insertion"):
        assert banned not in comma_high
    semi_high = describe_feature("semicolon_density", {"mean": 0.9}, bt)
    assert semi_high == "Use semicolons relatively frequently."
    for banned in ("antithetical", "paired", "construction"):
        assert banned not in semi_high
    # 长句：字面指令，不附会"从句/插入语"等未测机制
    long_sent = describe_feature("mean_sentence_length", {"mean": 0.9}, bt)
    assert long_sent == "Favor relatively long sentences."


def test_describe_feature_no_band_returns_none():
    assert describe_feature("comma_density", {"mean": 0.5}, _band_thresholds([])) is None
    assert describe_feature("unknown_feature", {"mean": 0.5},
                            _band_thresholds(["unknown_feature"])) is None
    assert describe_feature("comma_density", {"mean": None},
                            _band_thresholds(["comma_density"])) is None


def test_not_compilable_downgraded_to_reference():
    profile = _make_profile(
        full_features={"mean_sentence_length": _feature_summary(
            mean=17.3, variance=32.0, std=5.6)})
    plan = StylePlanner(band_thresholds=_band_thresholds([])).plan(profile, REQUEST)
    refs = {c.feature_id: c for c in plan.reference_controls}
    assert "mean_sentence_length" in refs
    assert refs["mean_sentence_length"].activation == "reference"
    assert "not_compilable" in refs["mean_sentence_length"].reason
    assert plan.language_controls == []


def test_long_content_never_hard_truncated():
    long_content = ("A young woman returns to her family's country house. " * 40).strip()
    req = WritingRequest(content=long_content, desired_length="short_scene",
                         target_words=400, language="english", pov=None,
                         constraints=["No new characters"])
    profile = _make_profile(canonicals=[_canonical(f"a::v{i}", status="validated",
                                                   works=2, chunks=5) for i in range(6)])
    plan = _planner([]).plan(profile, req)
    for budget_chars in (300, 1000, 3000, 10000):
        policy = PlannerPolicy(max_prompt_chars=budget_chars)
        try:
            prompt = PromptCompiler(policy).compile(plan)
        except PromptBudgetError:
            continue  # 强制内容放不下 → 显式失败，绝不截断
        assert long_content in prompt.text  # 内容永不丢失/切片


def test_low_priority_removed_before_mandatory():
    full = {fid: _feature_summary(mean=50.0) for fid in
            ["comma_density", "semicolon_density", "dash_density", "quotation_density",
             "exclamation_frequency", "question_frequency", "period_density",
             "long_sentence_ratio", "short_sentence_ratio", "sentence_length_cv",
             "type_token_ratio", "hapax_ratio", "word_repetition_rate",
             "mean_word_length", "connective_density"]}
    profile = _make_profile(full_features=full,
                            canonicals=[_canonical(f"a::v{i}", status="validated",
                                                   works=2, chunks=5) for i in range(6)])
    plan = _planner(full.keys()).plan(profile, REQUEST)
    full_prompt = PromptCompiler().compile(plan)
    degraded = PromptCompiler(
        PlannerPolicy(max_prompt_chars=full_prompt.char_count - 600)).compile(plan)
    assert degraded.degraded
    # 只移除策略/语言控制/措辞，绝不动用户内容
    assert "A short test brief." in degraded.text
    assert all(r.startswith(("strategy:", "language:", "wording:"))
               for r in degraded.removed_controls)
    # 降级顺序：策略先于语言控制
    seen_lang = False
    for r in degraded.removed_controls:
        if r.startswith("language:"):
            seen_lang = True
        elif r.startswith("strategy:"):
            assert not seen_lang, "strategy 在 language control 之后被移除（顺序错误）"


def test_real_band_thresholds_train_only():
    bt = _load_real_band_thresholds()
    assert bt["train_only"] is True
    train = set(bt["train_work_ids"])
    assert train == {"emma", "pride_and_prejudice",
                     "great_expectations", "david_copperfield"}
    assert "persuasion" not in train and "tale_of_two_cities" not in train
    assert bt["n_train_chunks"] == 2328
    assert len(bt["features"]) == 22
