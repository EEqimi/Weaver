# tests/test_controllability.py
"""§19.5 生成可控性实验测试（全部确定性，Dummy/fake，零 LLM，零 token）。

覆盖：
    - apply_intensity：三档重标语言控制 activation、互异确定性 plan id、guidance 保留、
      reference/suppressed 不动、原 plan 不被改动、reason 追加溯源。
    - 强度措辞：编译后 STYLE CONTROL 段前缀随强度变化，且不泄露作者名/模仿令牌。
    - check_monotonic：递减 / 非单调 / 平坦 / 容差内近等。
    - run_controllability 编排（monkeypatch 重协作者 + 记录 provider）：3 passages/作者、
      产物落盘、summary 携带距离与单调判定、provider 零真实调用。

绝不调用真实模型；绝不读 DEEPSEEK_API_KEY；绝不写真实 data/（产物写入 tmp_path）。
"""
import json
from types import SimpleNamespace

import pytest

from knowledge.generation.controllability import (
    EXPERIMENT_ID_95, INTENSITY_TO_ACTIVATION, INTENSITY_LEVELS,
    _effect_direction, _summarize_samples, apply_intensity, check_monotonic,
    run_controllability, run_controllability_repeated,
)
from knowledge.generation import controllability as cmod
from knowledge.generation.run import generation_layout
from knowledge.generation.schema import (
    assert_no_author_identity, assert_no_imitation_instruction,
)
from knowledge.planning.compiler import PromptCompiler
from knowledge.planning.schema import PlannedControl, StylePlan
from knowledge.schema.versions import (
    CONTROLLABILITY_REPEATED_VERSION, CONTROLLABILITY_VERSION,
    STYLE_PLAN_SCHEMA_VERSION,
)


def _pc(feature_id, activation, bucket, guidance="Use dialogue relatively often."):
    return PlannedControl(
        feature_id=feature_id, registry_control_role="descriptive",
        activation=activation, bucket=bucket, source_scope="full_train_corpus",
        support={}, reason="descriptive: auxiliary/weak control",
        guidance=guidance, source="artifact")


def _make_plan() -> StylePlan:
    return StylePlan(
        style_plan_id="sp-base", schema_version=STYLE_PLAN_SCHEMA_VERSION,
        author_id="austen", source_profile_hash="ph",
        writing_request={
            "content": "A quiet scene in a garden at dusk.",
            "desired_length": "short_scene", "target_words": None,
            "language": "english", "pov": None, "constraints": []},
        language_controls=[
            _pc("dialogue_ratio", "strong", "primary"),
            _pc("comma_density", "medium", "secondary"),
        ],
        reference_controls=[_pc("semicolon_density", "reference", "reference")],
        suppressed_controls=[_pc("dash_density", "suppressed", "suppressed")],
        warnings=["w0"],
    )


# --------------------------------------------------------------------------- #
# apply_intensity（纯函数）
# --------------------------------------------------------------------------- #
def test_apply_intensity_relabels_activation_and_distinct_ids():
    plan = _make_plan()
    seen_ids = set()
    for intensity, target in INTENSITY_TO_ACTIVATION.items():
        out = apply_intensity(plan, intensity)
        assert [c.activation for c in out.language_controls] == [target, target]
        # 互异确定性 id（区别于 base，也区别于其它强度）。
        assert out.style_plan_id != plan.style_plan_id
        assert out.style_plan_id not in seen_ids
        seen_ids.add(out.style_plan_id)
        # guidance 保留；reference/suppressed 原样不动。
        assert [c.guidance for c in out.language_controls] == \
            [c.guidance for c in plan.language_controls]
        assert [c.activation for c in out.reference_controls] == ["reference"]
        assert [c.activation for c in out.suppressed_controls] == ["suppressed"]
        # reason 追加溯源；warnings 追加强度覆写标记。
        assert all(c.reason.endswith(f"; intensity override -> {target}")
                   for c in out.language_controls)
        assert len(out.warnings) == len(plan.warnings) + 1
        # 绝不改动调用方 plan。
    assert [c.activation for c in plan.language_controls] == ["strong", "medium"]


def test_apply_intensity_rejects_unknown_intensity():
    with pytest.raises(ValueError):
        apply_intensity(_make_plan(), "extreme")


# --------------------------------------------------------------------------- #
# 强度措辞 + 泄露守卫
# --------------------------------------------------------------------------- #
def test_intensity_wording_and_no_leak():
    compiler = PromptCompiler()
    prefixes = {"low": "As a general tendency: ", "medium": "Tend toward: ",
                "high": "Strongly prefer: "}
    for intensity in INTENSITY_LEVELS:
        out = apply_intensity(_make_plan(), intensity)
        prompt = compiler.compile(out)
        # STYLE CONTROL 段前缀随强度变化（强度旋钮生效）。
        assert prefixes[intensity] in prompt.text
        # 泄露守卫：不注入作者名、不写模仿令牌（fail-closed）。
        assert_no_author_identity(prompt.text, ["Jane Austen"])
        assert_no_imitation_instruction(prompt.text)


# --------------------------------------------------------------------------- #
# check_monotonic（纯函数）
# --------------------------------------------------------------------------- #
def test_check_monotonic_decreasing():
    v = check_monotonic({"low": 0.3, "medium": 0.2, "high": 0.1})
    assert v["monotonic"] is True
    assert v["direction"] == "decreasing"


def test_check_monotonic_non_monotonic():
    v = check_monotonic({"low": 0.1, "medium": 0.3, "high": 0.2})
    assert v["monotonic"] is False
    assert v["direction"] == "non_monotonic"


def test_check_monotonic_flat_and_tolerance():
    assert check_monotonic({"low": 0.2, "medium": 0.2, "high": 0.2})["direction"] == "flat"
    # 容差内近等 → 视为非劣化（monotonic）。
    assert check_monotonic(
        {"low": 0.2000001, "medium": 0.2, "high": 0.2})["monotonic"] is True


# --------------------------------------------------------------------------- #
# run_controllability 编排（monkeypatch 重协作者，零 LLM）
# --------------------------------------------------------------------------- #
class _Usage:
    def __init__(self):
        self.prompt_tokens = 100
        self.completion_tokens = 200
        self.total_tokens = 300

    def to_dict(self):
        return {"prompt_tokens": 100, "completion_tokens": 200, "total_tokens": 300}


class _Passage:
    def __init__(self, author_id, intensity):
        self.author_id = author_id
        self.experiment_id = EXPERIMENT_ID_95
        self.generation_id = f"g-{author_id}-{intensity}"
        self.style_plan_id = intensity
        self.compiled_prompt_hash = "ph"
        self.finish_reason = "stop"
        self.cache_hit = False
        self.fresh_request = True
        self.generated_text = f"{author_id}:{intensity}"
        self.usage = _Usage()

    def to_dict(self):
        return {"generation_id": self.generation_id}


class _RecordingProvider:
    provider_id = "dummy"
    model = "dummy-model"
    base_url = "https://dummy.example"

    def __init__(self):
        self.calls = 0

    def is_configured(self):
        return True

    def generate(self, prompt_text, parameters):
        self.calls += 1
        return SimpleNamespace(content="generated", finish_reason="stop", usage=None)


_DIST = {
    "austen:low": 0.30, "austen:medium": 0.20, "austen:high": 0.10,   # 单调递减
    "dickens:low": 0.10, "dickens:medium": 0.30, "dickens:high": 0.20,  # 非单调
}


def _install_controllability_fakes(monkeypatch, tmp_path):
    monkeypatch.setattr(
        cmod, "_load_profile",
        lambda base, aid: SimpleNamespace(author_id=aid,
                                          author_scope={"train_work_ids": []}))
    monkeypatch.setattr(cmod, "_band_thresholds", lambda base, ids: {})
    monkeypatch.setattr(cmod, "_read_plumbing", lambda out_dir: {})
    plumbing_calls = {"n": 0}

    def _require(plumbing, provider):
        plumbing_calls["n"] += 1
    monkeypatch.setattr(cmod, "_require_valid_plumbing", _require)

    def _plan_and_prompt(profile, band_thresholds, intensity):
        plan = SimpleNamespace(style_plan_id=intensity, author_id=profile.author_id)
        prompt = SimpleNamespace(text=f"prompt:{profile.author_id}:{intensity}")
        return plan, prompt
    monkeypatch.setattr(cmod, "_plan_and_prompt_at_intensity", _plan_and_prompt)
    monkeypatch.setattr(cmod, "_provenance", lambda plan, bt, profile: {})

    def _build_passage(author_id, plan, prompt, result, provider, parameters,
                       provenance, experiment_id, fresh_request=True):
        return _Passage(author_id, plan.style_plan_id)
    monkeypatch.setattr(cmod, "_build_passage", _build_passage)
    monkeypatch.setattr(cmod, "stylometric_distance",
                        lambda text, author_id, base: _DIST[text])

    provider = _RecordingProvider()
    return provider, plumbing_calls


def test_run_controllability_writes_artifacts_and_monotonic_verdict(monkeypatch, tmp_path):
    provider, plumbing_calls = _install_controllability_fakes(monkeypatch, tmp_path)

    summary = run_controllability(data_root_=tmp_path, provider=provider)

    assert summary["controllability_version"] == CONTROLLABILITY_VERSION
    assert summary["intensity_levels"] == list(INTENSITY_LEVELS)
    # 每作者三档距离 + 单调判定（austen 单调递减，dickens 非单调）。
    a = summary["authors"]["austen"]
    assert a["distances"] == {"low": 0.30, "medium": 0.20, "high": 0.10}
    assert a["monotonic"] is True
    assert a["direction"] == "decreasing"
    d = summary["authors"]["dickens"]
    assert d["monotonic"] is False
    assert d["direction"] == "non_monotonic"
    # plumbing gate 被调用；provider 零真实调用（仅计数 fake）。
    assert plumbing_calls["n"] == 1
    assert provider.calls == 2 * len(INTENSITY_LEVELS)

    root = tmp_path / "analysis" / "generation" / EXPERIMENT_ID_95
    for aid in ("austen", "dickens"):
        for intensity in INTENSITY_LEVELS:
            assert (root / f"{aid}_{intensity}_generation.json").exists()
            assert (root / f"{aid}_{intensity}_passage.md").exists()
    assert (root / "controllability_summary.json").exists()
    assert (root / "controllability_report.md").exists()


# --------------------------------------------------------------------------- #
# Phase 9.3 repeated-sampling（确定性，零 LLM）
# --------------------------------------------------------------------------- #
def test_summarize_samples_mean_median_std():
    s = _summarize_samples([0.1, 0.2, 0.3])
    assert s["n"] == 3
    assert s["mean"] == pytest.approx(0.2)
    assert s["median"] == pytest.approx(0.2)
    assert s["min"] == pytest.approx(0.1)
    assert s["max"] == pytest.approx(0.3)
    assert s["std"] > 0  # 总体 std，非零
    assert s["samples"] == [0.1, 0.2, 0.3]


def test_effect_direction():
    assert _effect_direction({"low": 0.2, "high": 0.1})[0] == "decreasing"
    assert _effect_direction({"low": 0.1, "high": 0.2})[0] == "increasing"
    assert _effect_direction({"low": 0.1, "high": 0.1})[0] == "flat"
    # effect_size = low - high（正值 = 符合假设方向）。
    assert _effect_direction({"low": 0.2, "high": 0.1})[1] == pytest.approx(0.1)


def _write_existing_summary(tmp_path, first_distances):
    out_dir = generation_layout(tmp_path, EXPERIMENT_ID_95)["root"]
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "authors": {aid: {"distances": first_distances[aid]}
                    for aid in ("austen", "dickens")},
    }
    (out_dir / "controllability_summary.json").write_text(
        json.dumps(summary), encoding="utf-8")


def _install_repeated_fakes(monkeypatch, new_distances):
    monkeypatch.setattr(
        cmod, "_load_profile",
        lambda base, aid: SimpleNamespace(author_id=aid,
                                          author_scope={"train_work_ids": []}))
    monkeypatch.setattr(cmod, "_band_thresholds", lambda base, ids: {})
    monkeypatch.setattr(cmod, "_read_plumbing", lambda out_dir: {})
    plumbing_calls = {"n": 0}

    def _require(plumbing, provider):
        plumbing_calls["n"] += 1
    monkeypatch.setattr(cmod, "_require_valid_plumbing", _require)
    monkeypatch.setattr(
        cmod, "_plan_and_prompt_at_intensity",
        lambda profile, bt, intensity: (
            SimpleNamespace(style_plan_id=intensity, author_id=profile.author_id,
                            source_profile_hash="ph"),
            SimpleNamespace(text=f"prompt:{profile.author_id}:{intensity}")))
    monkeypatch.setattr(cmod, "_provenance", lambda plan, bt, profile: {})
    monkeypatch.setattr(
        cmod, "_build_passage",
        lambda author_id, plan, prompt, result, provider, parameters, provenance,
               experiment_id, fresh_request=True: _Passage(author_id,
                                                           plan.style_plan_id))

    state: dict[tuple[str, str], int] = {}

    def _stylo(text, author_id, base):
        intensity = text.split(":")[1]
        key = (author_id, intensity)
        idx = state.get(key, 0)
        state[key] = idx + 1
        return new_distances[key][idx]
    monkeypatch.setattr(cmod, "stylometric_distance", _stylo)

    return _RecordingProvider(), plumbing_calls


def test_run_controllability_repeated_n3_and_monotonic(monkeypatch, tmp_path):
    first = {
        "austen": {"low": 0.15, "medium": 0.27, "high": 0.19},
        "dickens": {"low": 0.16, "medium": 0.13, "high": 0.11},
    }
    # 新样本：austen medium 仍偏高（均值非单调）；dickens 保持递减。
    new = {
        ("austen", "low"): [0.14, 0.16],
        ("austen", "medium"): [0.26, 0.28],
        ("austen", "high"): [0.18, 0.20],
        ("dickens", "low"): [0.17, 0.15],
        ("dickens", "medium"): [0.14, 0.12],
        ("dickens", "high"): [0.10, 0.12],
    }
    _write_existing_summary(tmp_path, first)
    provider, plumbing_calls = _install_repeated_fakes(monkeypatch, new)

    summary = run_controllability_repeated(data_root_=tmp_path, provider=provider)

    assert summary["controllability_repeated_version"] == CONTROLLABILITY_REPEATED_VERSION
    assert summary["n_total_samples_per_cell"] == 3
    assert summary["new_request_count"] == 2 * len(INTENSITY_LEVELS) * 2  # 12

    # Austen：n=3，medium 偏高未被平均掉 → 均值仍 non-monotonic。
    a = summary["authors"]["austen"]
    assert a["per_intensity"]["low"]["n"] == 3
    assert a["per_intensity"]["low"]["samples"] == [0.15, 0.14, 0.16]
    assert a["per_intensity"]["low"]["mean"] == pytest.approx(0.15)
    assert a["monotonic_on_mean"]["monotonic"] is False
    assert a["monotonic_on_mean"]["direction"] == "non_monotonic"

    # Dickens：n=3 后递减趋势仍成立。
    d = summary["authors"]["dickens"]
    assert d["monotonic_on_mean"]["monotonic"] is True
    assert d["monotonic_on_mean"]["direction"] == "decreasing"
    assert d["effect_direction"] == "decreasing"
    assert d["effect_size_mean"] == pytest.approx(0.05)

    assert provider.calls == 12
    assert plumbing_calls["n"] == 1

    out_dir = generation_layout(tmp_path, EXPERIMENT_ID_95)["root"]
    assert (out_dir / "controllability_repeated_summary.json").exists()
    assert (out_dir / "controllability_repeated_report.md").exists()
    for aid in ("austen", "dickens"):
        for intensity in INTENSITY_LEVELS:
            assert (out_dir / f"{aid}_{intensity}_rep1_generation.json").exists()
            assert (out_dir / f"{aid}_{intensity}_rep2_passage.md").exists()
    # 原单次 summary 未被覆盖（首样本距离原样保留）。
    original = json.loads(
        (out_dir / "controllability_summary.json").read_text(encoding="utf-8"))
    assert original["authors"]["austen"]["distances"]["low"] == 0.15


def test_run_controllability_repeated_missing_summary_fails_closed(monkeypatch, tmp_path):
    _install_repeated_fakes(monkeypatch, {})
    with pytest.raises(cmod.GenerationError):
        run_controllability_repeated(data_root_=tmp_path, provider=_RecordingProvider())
