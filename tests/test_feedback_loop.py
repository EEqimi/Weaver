# tests/test_feedback_loop.py
"""Phase 9.1 多轮反馈闭环测试（全部确定性，Dummy/fake，零 LLM，零 token）。

覆盖（spec §15.5 停止条件 / §二十 max_iterations）：
    - continue → accept：真正迭代，逐轮 artifacts 加 _iter{N} 后缀。
    - 回归 → roll_back 保留 best-so-far（第 N 轮 before = 第 N−1 轮改写，绝不回退原文）。
    - max_iterations 界：第 max_iterations 轮改善 → accept，绝不无限循环。
    - 中途 no_effect：第 N 轮改写相对当前最佳正文无实质变化 → 停。
    - 完整性每轮对照不可变原文（recording checker 捕获 args）。
    - 单轮（max_iterations=1）向后兼容：无后缀 artifacts + iterations 恰好一条。
    - {author}_iterations.json 写入且带 loop_version / final_* / iterations。

绝不调用真实模型；绝不读 DEEPSEEK_API_KEY；绝不写真实 data/（产物写入 tmp_path）。
"""
import json
from types import SimpleNamespace

import pytest

from knowledge.evaluation import run as run_mod
from knowledge.evaluation.schema import (
    FEEDBACK_ACCEPT, FEEDBACK_CONTINUE, FEEDBACK_NO_EFFECT, FEEDBACK_ROLL_BACK,
    ActualStyleProfile, ComparisonResult, ContentIntegrityResult, FeatureDeviation,
    RevisionItem, RevisionPlan, RevisionResult,
)
from knowledge.generation.schema import output_hash
from knowledge.schema.versions import FEEDBACK_LOOP_VERSION


def _fake_actual():
    return ActualStyleProfile(
        schema_version="0.1.0", author_id="austen", passage_id="g1",
        passage_hash="h", style_plan_id="sp1")


def _cmp_with_n(n):
    return ComparisonResult(
        author_id="austen", passage_id="g1",
        language_deviations=[FeatureDeviation(
            feature_id=f"f{i}", target_band="high", actual_band="low",
            target_value=None, actual_value=None, status="below",
            measurable=True, reason="") for i in range(n)])


class _RecordingChecker:
    """记录每次 check 的 (original, revised) 参数，返回 passed。"""

    def __init__(self):
        self.calls = 0
        self.args = []

    def check(self, original, revised, request, author_names=()):
        self.calls += 1
        self.args.append((original, revised))
        return ContentIntegrityResult(
            schema_version="0.1.0", checker_version="0.1.0", passed=True,
            plot_facts_preserved=True, characters_preserved=True,
            relationships_preserved=True, constraints_preserved=True,
            new_major_events=False, removed_major_events=False, violations=[],
            reasoning_summary="", deterministic=True, blind=True)


class _RecordingEvaluator:
    """文学评价器：返回 None（文学 guard 不参与），计数调用。"""

    def __init__(self):
        self.calls = 0

    def evaluate(self, *a, **k):
        self.calls += 1
        return None


class _ScriptedRewriter:
    """按调用序返回预置 revised_text（每轮不同，确保实质改写）。"""

    def __init__(self, revised_texts):
        self.revised_texts = list(revised_texts)
        self.calls = 0
        self.originals = []

    def rewrite(self, original, plan, author_names=()):
        self.originals.append(original)
        idx = self.calls
        self.calls += 1
        text = self.revised_texts[idx] if idx < len(self.revised_texts) \
            else self.revised_texts[-1]
        return RevisionResult(
            schema_version="0.2.0", author_id=plan.author_id,
            passage_id=plan.passage_id, original_passage_hash="oh",
            revised_passage_hash="rh", revised_text=text,
            claimed_change_descriptions=[f"round {self.calls} change"],
            claimed_revision_items=["P3:f"])


class _NoCallProvider:
    def __init__(self):
        self.calls = 0

    def is_configured(self):
        return True

    def complete(self, *a, **k):
        self.calls += 1
        raise AssertionError("多轮闭环测试绝不应调用 provider.complete")


def _install_multi_round_fakes(monkeypatch, tmp_path, *, before_devs, after_devs_seq,
                               revised_texts):
    """装配 run_evaluation 依赖，返回各 fake/计数器。

    compare_target_actual 脚本化：第 1 次调用 = 初始 before（before_devs），
    其后 = 各轮 after（after_devs_seq）。rewriter 按序返回 revised_texts。
    """
    monkeypatch.setattr(run_mod, "AUTHOR_IDS", ("austen",))
    monkeypatch.setattr(run_mod, "_author_names", lambda ids: [])

    profile = SimpleNamespace(author_scope={"train_work_ids": []})
    passage = SimpleNamespace(generated_text="Original passage text.", generation_id="g1")
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

    compare_calls = {"n": 0}
    def _compare(plan, profile, actual, thresholds):
        idx = compare_calls["n"]
        compare_calls["n"] += 1
        if idx == 0:
            return _cmp_with_n(before_devs)
        after_idx = idx - 1
        if after_idx < len(after_devs_seq):
            return _cmp_with_n(after_devs_seq[after_idx])
        return _cmp_with_n(after_devs_seq[-1])
    monkeypatch.setattr(run_mod, "compare_target_actual", _compare)

    def _build_plan(comparison, plan, evaluation=None, weak_score_threshold=5.0):
        return RevisionPlan(
            schema_version="0.1.0", author_id="austen", passage_id="g1",
            style_plan_id="sp1",
            revision_items=[RevisionItem(
                priority="P3", category="language", target="f",
                instruction="Use dialogue.", reason="r")])
    monkeypatch.setattr(run_mod, "build_revision_plan", _build_plan)

    checker = _RecordingChecker()
    evaluator = _RecordingEvaluator()
    rewriter = _ScriptedRewriter(revised_texts)
    monkeypatch.setattr(run_mod, "ContentIntegrityChecker",
                        lambda provider, blind=True: checker)
    monkeypatch.setattr(run_mod, "LiteraryEvaluator",
                        lambda provider, blind=True: evaluator)
    monkeypatch.setattr(run_mod, "RevisionRewriter",
                        lambda provider, blind=True: rewriter)

    provider = _NoCallProvider()
    return checker, evaluator, measure_calls, compare_calls, rewriter, provider


# --------------------------------------------------------------------------- #
# 多轮迭代
# --------------------------------------------------------------------------- #
def test_continue_then_accept(monkeypatch, tmp_path):
    checker, evaluator, measure_calls, compare_calls, rewriter, provider = \
        _install_multi_round_fakes(
            monkeypatch, tmp_path, before_devs=3, after_devs_seq=[1, 0],
            revised_texts=["Revised passage text one.", "Revised passage text two."])

    summary = run_mod.run_evaluation(data_root_=tmp_path, provider=provider,
                                     max_iterations=3)
    a = summary["authors"]["austen"]

    assert a["n_iterations"] == 2
    assert a["final_outcome"] == FEEDBACK_ACCEPT
    assert a["final_iteration"] == 2
    assert [it["iteration"] for it in a["iterations"]] == [1, 2]
    assert [it["decision"]["outcome"] for it in a["iterations"]] == \
        [FEEDBACK_CONTINUE, FEEDBACK_ACCEPT]
    assert checker.calls == 2
    assert measure_calls["n"] == 3            # before + 2 轮 after
    # 每轮 delta：第 2 轮 before = 第 1 轮改写（当前最佳正文）。
    assert rewriter.originals == ["Original passage text.", "Revised passage text one."]
    assert a["final_text_hash"] == output_hash("Revised passage text two.")

    root = tmp_path / "analysis" / "evaluation_v3"
    assert (root / "austen_revision_result.json").exists()
    assert (root / "austen_revision_result_iter2.json").exists()
    assert not (root / "austen_revision_result_iter3.json").exists()


def test_regression_rolls_back_to_best_so_far(monkeypatch, tmp_path):
    checker, evaluator, measure_calls, compare_calls, rewriter, provider = \
        _install_multi_round_fakes(
            monkeypatch, tmp_path, before_devs=3, after_devs_seq=[1, 2],
            revised_texts=["Revised one.", "Revised two (worse)."])

    summary = run_mod.run_evaluation(data_root_=tmp_path, provider=provider,
                                     max_iterations=3)
    a = summary["authors"]["austen"]

    assert a["n_iterations"] == 2
    assert a["final_outcome"] == FEEDBACK_ROLL_BACK
    assert a["final_iteration"] == 2
    # 保留 best-so-far = 第 1 轮改写，绝不回退原文，也绝不接受第 2 轮劣化文本。
    assert a["final_text_hash"] == output_hash("Revised one.")
    assert "best-so-far" in a["decision"]["reason"]


def test_stops_at_max_iterations(monkeypatch, tmp_path):
    checker, evaluator, measure_calls, compare_calls, rewriter, provider = \
        _install_multi_round_fakes(
            monkeypatch, tmp_path, before_devs=3, after_devs_seq=[2, 1],
            revised_texts=["Revised one.", "Revised two."])

    summary = run_mod.run_evaluation(data_root_=tmp_path, provider=provider,
                                     max_iterations=2)
    a = summary["authors"]["austen"]

    assert a["n_iterations"] == 2
    assert a["final_outcome"] == FEEDBACK_ACCEPT   # 第 2 轮改善但 iteration>=max → accept
    assert a["final_iteration"] == 2
    assert measure_calls["n"] == 3
    root = tmp_path / "analysis" / "evaluation_v3"
    assert not (root / "austen_revision_result_iter3.json").exists()  # 绝不无限循环


def test_no_effect_mid_loop_terminates(monkeypatch, tmp_path):
    checker, evaluator, measure_calls, compare_calls, rewriter, provider = \
        _install_multi_round_fakes(
            monkeypatch, tmp_path, before_devs=3, after_devs_seq=[1],
            revised_texts=["Revised one.", "Revised one."])  # 第 2 轮返回相同文本

    summary = run_mod.run_evaluation(data_root_=tmp_path, provider=provider,
                                     max_iterations=3)
    a = summary["authors"]["austen"]

    assert a["n_iterations"] == 2
    assert a["final_outcome"] == FEEDBACK_NO_EFFECT
    assert a["final_iteration"] == 2
    assert a["final_text_hash"] == output_hash("Revised one.")
    # no_effect 轮之后绝不再调 checker（短路）。
    assert checker.calls == 1


def test_integrity_checked_against_original_every_round(monkeypatch, tmp_path):
    checker, evaluator, measure_calls, compare_calls, rewriter, provider = \
        _install_multi_round_fakes(
            monkeypatch, tmp_path, before_devs=3, after_devs_seq=[1, 0],
            revised_texts=["Revised passage text one.", "Revised passage text two."])

    run_mod.run_evaluation(data_root_=tmp_path, provider=provider, max_iterations=3)

    # 完整性每轮对照不可变原文，绝不对照漂移的中间态。
    assert [orig for orig, _ in checker.args] == \
        ["Original passage text.", "Original passage text."]
    assert [rev for _, rev in checker.args] == \
        ["Revised passage text one.", "Revised passage text two."]


# --------------------------------------------------------------------------- #
# 单轮向后兼容 + iterations.json
# --------------------------------------------------------------------------- #
def test_single_round_backward_compatible(monkeypatch, tmp_path):
    checker, evaluator, measure_calls, compare_calls, rewriter, provider = \
        _install_multi_round_fakes(
            monkeypatch, tmp_path, before_devs=3, after_devs_seq=[0],
            revised_texts=["Revised one."])

    summary = run_mod.run_evaluation(data_root_=tmp_path, provider=provider,
                                     max_iterations=1)
    a = summary["authors"]["austen"]

    assert a["n_iterations"] == 1
    assert a["final_outcome"] == FEEDBACK_ACCEPT
    assert a["final_iteration"] == 1
    assert len(a["iterations"]) == 1
    root = tmp_path / "analysis" / "evaluation_v3"
    assert (root / "austen_revision_result.json").exists()
    assert not (root / "austen_revision_result_iter2.json").exists()


def test_iterations_json_written_with_loop_version(monkeypatch, tmp_path):
    checker, evaluator, measure_calls, compare_calls, rewriter, provider = \
        _install_multi_round_fakes(
            monkeypatch, tmp_path, before_devs=3, after_devs_seq=[1, 0],
            revised_texts=["Revised one.", "Revised two."])

    summary = run_mod.run_evaluation(data_root_=tmp_path, provider=provider,
                                     max_iterations=3)
    assert summary["loop_version"] == FEEDBACK_LOOP_VERSION

    root = tmp_path / "analysis" / "evaluation_v3"
    data = json.loads((root / "austen_iterations.json").read_text(encoding="utf-8"))
    assert data["loop_version"] == FEEDBACK_LOOP_VERSION
    assert data["n_iterations"] == 2
    assert data["final_outcome"] == FEEDBACK_ACCEPT
    assert data["final_iteration"] == 2
    assert [it["iteration"] for it in data["iterations"]] == [1, 2]
    # 紧凑摘要：不内嵌全文 profile 大对象。
    assert "layer_a_statistical" not in data["iterations"][0]
