# knowledge/generation/controllability.py
"""§19.5 生成可控性实验：同一中性 brief 以 low/medium/high 三档风格强度重生成。

目标：验证测量到的 Layer D 余弦距离（到作者质心）随强度**单调递减**——强度越高 →
生成正文越贴近作者文体学指纹。这是**报告观测**（LLM 抽样随机、单次 3 点采样证据弱），
不是硬门。

强度旋钮 = 语言控制的 `activation`（唯一带强度措辞的层）：low→weak（"As a general
tendency"）/ medium→medium（"Tend toward"）/ high→strong（"Strongly prefer"），由
`PromptCompiler._ACTIVATION_PREFIX` 翻译为 prompt 措辞差异；叙事/策略无强度措辞，不动。

铁律（spec）：
    - 同一 WritingRequest、同一模型、同一生成参数；唯一变量是强度覆写后的风格控制。
    - 实际 prompt 绝不含作者名 / "write like" / "imitate" / "in the style of"：
      复用 `assert_no_author_identity` / `assert_no_imitation_instruction`，fail-closed。
    - 复用 DeepSeekProvider（OpenAI 兼容）的 HTTP 传输，绝不另写第二套 client。
    - 复用 `run_generation` 的 plumbing gate（`_require_valid_plumbing`）：正式生成前
      必须有合法 plumbing 记录，否则 fail-closed。
    - 绝不自动评价（Phase 8）、绝不自动改写正文；密钥只读；独立 experiment_id /
      无缓存（每次 generate 都是 fresh request）。
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

from ..config import data_root as default_data_root
from ..evaluation.analyze import stylometric_distance
from ..planning.compiler import PromptCompiler
from ..planning.planner import StylePlanner
from ..planning.run import (
    AUTHOR_IDS, NEUTRAL_REQUEST, _band_thresholds, _load_profile,
)
from ..planning.schema import StylePlan, make_intensity_plan_id
from ..schema.versions import CONTROLLABILITY_REPEATED_VERSION, CONTROLLABILITY_VERSION
from .provider import GenerationProvider
from .run import (
    GENERATION_PARAMETERS, MAX_PROMPT_CHARS_GUARD, _assert_prompt_safe,
    _author_names_for, _build_passage, _provenance, _read_plumbing,
    _require_valid_plumbing, build_generation_provider, generation_layout,
)
from .schema import GenerationError

EXPERIMENT_ID_95 = "phase9_5-controllability-v0.1"

# 强度 → 语言控制激活级别（三档，确定性；唯一变量是风格控制的强度措辞）。
INTENSITY_LEVELS: tuple[str, ...] = ("low", "medium", "high")
INTENSITY_TO_ACTIVATION: dict[str, str] = {
    "low": "weak", "medium": "medium", "high": "strong",
}


def apply_intensity(plan: StylePlan, intensity: str) -> StylePlan:
    """把已激活语言控制统一重标为目标激活级别，返回互异确定性 id 的新 plan（纯函数）。

    只改 `language_controls`（已激活的 strong/medium/weak）；`reference_controls` /
    `suppressed_controls` / narrative / strategy 原样保留（无强度措辞）。深拷贝，绝不
    改动调用方 plan。
    """
    if intensity not in INTENSITY_TO_ACTIVATION:
        raise ValueError(
            f"unknown intensity {intensity!r} (expected one of {list(INTENSITY_LEVELS)})")
    target = INTENSITY_TO_ACTIVATION[intensity]
    plan_copy = StylePlan.from_dict(plan.to_dict())
    for c in plan_copy.language_controls:
        c.activation = target
        c.reason = f"{c.reason}; intensity override -> {target}"
    plan_copy.style_plan_id = make_intensity_plan_id(plan.style_plan_id, intensity)
    plan_copy.warnings = list(plan_copy.warnings) + [
        f"§19.5 intensity override: activated language controls re-labelled to {target}"]
    return plan_copy


def check_monotonic(distances: dict[str, float], eps: float = 1e-6) -> dict[str, Any]:
    """判定 low/medium/high 三档距离是否随强度单调非增（强度↑ → 距离↓）。

    容差 eps 吸收浮点噪声（近等视为非劣化）。这是**报告观测**，非硬门——LLM 抽样
    随机、单次 3 点采样证据弱，non-monotonic 不视为失败，可重跑确认。
    """
    low = float(distances["low"])
    med = float(distances["medium"])
    high = float(distances["high"])
    monotonic = (low + eps >= med) and (med + eps >= high)
    if monotonic:
        direction = "flat" if abs(low - high) <= eps else "decreasing"
    else:
        direction = "non_monotonic"
    return {
        "monotonic": monotonic,
        "direction": direction,
        "distances": {"low": round(low, 8), "medium": round(med, 8),
                      "high": round(high, 8)},
        "eps": eps,
        "note": ("单次 3 点采样证据弱、LLM 抽样随机；non-monotonic 不视为失败，"
                 "可重跑确认"),
    }


def _write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _plan_and_prompt_at_intensity(profile: Any, band_thresholds: dict[str, Any],
                                  intensity: str) -> tuple[StylePlan, Any]:
    """画像 → StylePlan → apply_intensity → CompiledPrompt（确定性），fail-closed 无泄露。"""
    planner = StylePlanner(band_thresholds=band_thresholds)
    compiler = PromptCompiler()
    plan = apply_intensity(planner.plan(profile, NEUTRAL_REQUEST), intensity)
    prompt = compiler.compile(plan)
    _assert_prompt_safe(prompt, _author_names_for([profile.author_id]))
    if len(prompt.text) > MAX_PROMPT_CHARS_GUARD:
        raise GenerationError(
            f"{profile.author_id}: prompt {len(prompt.text)} chars 超保护上限 "
            f"{MAX_PROMPT_CHARS_GUARD}，拒绝发送")
    return plan, prompt


def run_controllability(data_root_: Path | None = None,
                        provider: GenerationProvider | None = None,
                        experiment_id: str = EXPERIMENT_ID_95) -> dict[str, Any]:
    """每作者 × 每强度生成一遍，测整段 Layer D 距离，判定单调性，落盘产物。

    复用 run_generation 的 plumbing gate / 泄露守卫 / 预算守卫；正式生成前必须有合法
    plumbing 记录（fail-closed）。真实生成需真实 LLM——调用前须通过 §十六 成本预检 +
    显式批准。
    """
    base = Path(data_root_) if data_root_ is not None else default_data_root()
    out_dir = generation_layout(base, experiment_id)["root"]
    out_dir.mkdir(parents=True, exist_ok=True)

    provider = provider or build_generation_provider()
    if not provider.is_configured():
        raise GenerationError("未配置 LLM provider（缺 DEEPSEEK_API_KEY）")

    profiles = {aid: _load_profile(base, aid) for aid in AUTHOR_IDS}
    train_work_ids = sorted(
        {w for p in profiles.values() for w in p.author_scope.get("train_work_ids", [])})
    band_thresholds = _band_thresholds(base, train_work_ids)

    plumbing = _read_plumbing(generation_layout(base)["root"])
    _require_valid_plumbing(plumbing, provider)

    per_author: dict[str, dict[str, Any]] = {}
    for author_id in AUTHOR_IDS:
        profile = profiles[author_id]
        by_intensity: dict[str, dict[str, Any]] = {}
        for intensity in INTENSITY_LEVELS:
            plan, prompt = _plan_and_prompt_at_intensity(
                profile, band_thresholds, intensity)
            provenance = _provenance(plan, band_thresholds, profile)
            result = provider.generate(prompt.text, GENERATION_PARAMETERS)
            passage = _build_passage(
                author_id, plan, prompt, result, provider, GENERATION_PARAMETERS,
                provenance, experiment_id, fresh_request=True)
            distance = stylometric_distance(passage.generated_text, author_id, base)
            by_intensity[intensity] = {"passage": passage, "distance": distance}
            _write_json(out_dir / f"{author_id}_{intensity}_generation.json",
                        passage.to_dict())
            (out_dir / f"{author_id}_{intensity}_passage.md").write_text(
                _render_passage_md(passage, intensity), encoding="utf-8")
        per_author[author_id] = by_intensity

    summary = _build_summary(per_author, provider, plumbing, experiment_id)
    _write_json(out_dir / "controllability_summary.json", summary)
    (out_dir / "controllability_report.md").write_text(
        _render_report(per_author, provider, plumbing, experiment_id),
        encoding="utf-8")
    return summary


def _build_summary(per_author: dict[str, dict[str, Any]], provider: GenerationProvider,
                   plumbing: dict[str, Any] | None,
                   experiment_id: str) -> dict[str, Any]:
    authors: dict[str, Any] = {}
    total_prompt = total_completion = total = 0
    for author_id, by_intensity in per_author.items():
        distances = {k: by_intensity[k]["distance"] for k in INTENSITY_LEVELS}
        verdict = check_monotonic(distances)
        authors[author_id] = {
            "distances": verdict["distances"],
            "monotonic": verdict["monotonic"],
            "direction": verdict["direction"],
            "by_intensity": {
                k: _author_intensity_summary(by_intensity[k]["passage"])
                for k in INTENSITY_LEVELS
            },
        }
        for k in INTENSITY_LEVELS:
            p = by_intensity[k]["passage"]
            total_prompt += p.usage.prompt_tokens
            total_completion += p.usage.completion_tokens
            total += p.usage.total_tokens

    return {
        "stage": "generation_controllability",
        "experiment_id": experiment_id,
        "controllability_version": CONTROLLABILITY_VERSION,
        "deterministic_plan_and_prompt": True,
        "real_generation": True,
        "no_auto_evaluation": True,
        "no_auto_revision": True,
        "provider": provider.provider_id,
        "model": provider.model,
        "endpoint": f"{provider.base_url}/chat/completions",
        "generation_parameters": GENERATION_PARAMETERS.to_dict(),
        "writing_request": NEUTRAL_REQUEST.to_dict(),
        "intensity_levels": list(INTENSITY_LEVELS),
        "intensity_to_activation": dict(INTENSITY_TO_ACTIVATION),
        "plumbing": plumbing,
        "authors": authors,
        "total_tokens": {
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "total_tokens": total,
        },
        "note": ("monotonic 判定是报告观测（单次 3 点采样、LLM 随机），非硬门"),
    }


def _author_intensity_summary(p: Any) -> dict[str, Any]:
    return {
        "generation_id": p.generation_id,
        "style_plan_id": p.style_plan_id,
        "compiled_prompt_hash": p.compiled_prompt_hash,
        "finish_reason": p.finish_reason,
        "word_count": len(p.generated_text.split()),
        "total_tokens": p.usage.total_tokens,
        "cache_hit": p.cache_hit,
        "fresh_request": p.fresh_request,
    }


def _render_passage_md(p: Any, intensity: str) -> str:
    return "\n".join([
        f"# {p.author_id.capitalize()} — Controllability ({intensity}) Passage",
        "",
        f"- **experiment_id**: `{p.experiment_id}`",
        f"- **intensity**: `{intensity}`",
        f"- **generation_id**: `{p.generation_id}`",
        f"- **style_plan_id**: `{p.style_plan_id}`",
        f"- **finish_reason**: `{p.finish_reason}`  **usage**: {p.usage.to_dict()}",
        "",
        "---",
        "",
        p.generated_text.strip(),
        "",
    ])


def _render_report(per_author: dict[str, dict[str, Any]], provider: GenerationProvider,
                   plumbing: dict[str, Any] | None,
                   experiment_id: str) -> str:
    lines = [
        "# Weaver Style Engine — 生成可控性报告（§19.5）",
        "",
        "同一中性写作需求、同一模型、同一生成参数；**唯一变量**是风格控制强度"
        "（low/medium/high → weak/medium/strong 语言控制措辞）。",
        "Layer D 余弦距离越**小** → 越贴近作者文体学指纹。预期：强度越高，距离越小。",
        "",
        f"- **experiment_id**: `{experiment_id}`",
        f"- **provider / model**: `{provider.provider_id}` / `{provider.model}`",
        f"- **生成参数**: {GENERATION_PARAMETERS.to_dict()}",
        f"- **强度→激活级别**: {INTENSITY_TO_ACTIVATION}",
        "",
        "## 每作者三档距离与单调性",
        "",
        "| 作者 | low | medium | high | 单调递减? | 方向 |",
        "|---|---|---|---|---|---|",
    ]
    for author_id in AUTHOR_IDS:
        by = per_author[author_id]
        distances = {k: by[k]["distance"] for k in INTENSITY_LEVELS}
        verdict = check_monotonic(distances)
        lines.append(
            f"| {author_id} | {verdict['distances']['low']} | "
            f"{verdict['distances']['medium']} | {verdict['distances']['high']} | "
            f"{verdict['monotonic']} | {verdict['direction']} |")

    lines += [
        "",
        "> 注：单调性判定是单次 3 点采样观测（LLM 随机），非硬门；non-monotonic 可重跑确认。",
        "> 正文全文见 `{author}_{intensity}_passage.md`；机器可读产物见 "
        "`{author}_{intensity}_generation.json` + `controllability_summary.json`。",
        "",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Phase 9.3 repeated-sampling：追加重复样本，n=3/档，判断趋势在均值/中位数是否稳定
# --------------------------------------------------------------------------- #
def _summarize_samples(samples: list[float]) -> dict[str, Any]:
    """n 个距离样本的汇总统计（均值/中位数/总体 std/min/max）。n 小、纯观测。"""
    n = len(samples)
    mean = sum(samples) / n
    s = sorted(samples)
    median = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    var = sum((x - mean) ** 2 for x in samples) / n  # 总体方差（n=3，观测自身散布）
    return {
        "n": n,
        "mean": round(mean, 8),
        "median": round(median, 8),
        "std": round(math.sqrt(var), 8),
        "min": round(min(samples), 8),
        "max": round(max(samples), 8),
        "samples": [round(x, 8) for x in samples],
    }


def _effect_direction(means: dict[str, float], eps: float = 1e-6) -> tuple[str, float]:
    """low→high 均值方向：decreasing（强度↑→距离↓，符合假设）/ increasing / flat。

    返回 (direction, effect_size_mean) where effect_size_mean = mean_low - mean_high
    （正值 = 符合假设方向）。"""
    effect = means["low"] - means["high"]
    if abs(effect) <= eps:
        return "flat", round(effect, 8)
    return ("decreasing" if effect > 0 else "increasing"), round(effect, 8)


def _ranges_overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def run_controllability_repeated(data_root_: Path | None = None,
                                 provider: GenerationProvider | None = None,
                                 experiment_id: str = EXPERIMENT_ID_95,
                                 n_new: int = 2) -> dict[str, Any]:
    """对 §19.5 单次实验追加重复样本：每作者 × 每强度保留首样本 + n_new 个 fresh 样本。

    只复用现有 harness（planner/compiler/Layer D 都不改），唯一变量仍是强度覆写；同一
    WritingRequest / 模型 / 生成参数。每 cell 达 n = 1 + n_new，报告均值/中位数层面是否
    单调、effect 方向、以及样本间散布/重叠。**绝不覆盖**原单次实验产物（新样本以
    `{author}_{intensity}_rep{i}_*` 落盘；汇总写 controllability_repeated_*）。
    """
    base = Path(data_root_) if data_root_ is not None else default_data_root()
    out_dir = generation_layout(base, experiment_id)["root"]

    # 保留首样本：读既有单次实验 summary 的三档距离（fail-closed 缺失即拒跑）。
    existing_path = out_dir / "controllability_summary.json"
    if not existing_path.exists():
        raise GenerationError(
            f"缺单次实验 summary（{existing_path}）：repeated-sampling 必须先有 "
            "§19.5 单次结果，绝不凭空重采样")
    existing = _load_json(existing_path)
    first_distances = {
        aid: dict(existing["authors"][aid]["distances"]) for aid in AUTHOR_IDS
    }

    provider = provider or build_generation_provider()
    if not provider.is_configured():
        raise GenerationError("未配置 LLM provider（缺 DEEPSEEK_API_KEY）")

    profiles = {aid: _load_profile(base, aid) for aid in AUTHOR_IDS}
    train_work_ids = sorted(
        {w for p in profiles.values() for w in p.author_scope.get("train_work_ids", [])})
    band_thresholds = _band_thresholds(base, train_work_ids)

    plumbing = _read_plumbing(generation_layout(base)["root"])
    _require_valid_plumbing(plumbing, provider)

    per_author: dict[str, dict[str, list[float]]] = {}
    tot_prompt = tot_completion = tot_total = 0
    new_requests = 0
    for author_id in AUTHOR_IDS:
        profile = profiles[author_id]
        by_intensity: dict[str, list[float]] = {}
        for intensity in INTENSITY_LEVELS:
            plan, prompt = _plan_and_prompt_at_intensity(
                profile, band_thresholds, intensity)
            provenance = _provenance(plan, band_thresholds, profile)
            distances = [float(first_distances[author_id][intensity])]
            for rep in range(1, n_new + 1):
                result = provider.generate(prompt.text, GENERATION_PARAMETERS)
                passage = _build_passage(
                    author_id, plan, prompt, result, provider, GENERATION_PARAMETERS,
                    provenance, experiment_id, fresh_request=True)
                distances.append(
                    stylometric_distance(passage.generated_text, author_id, base))
                tot_prompt += passage.usage.prompt_tokens
                tot_completion += passage.usage.completion_tokens
                tot_total += passage.usage.total_tokens
                new_requests += 1
                _write_json(
                    out_dir / f"{author_id}_{intensity}_rep{rep}_generation.json",
                    passage.to_dict())
                (out_dir / f"{author_id}_{intensity}_rep{rep}_passage.md").write_text(
                    _render_passage_md(passage, f"{intensity} (rep {rep})"),
                    encoding="utf-8")
            by_intensity[intensity] = distances
        per_author[author_id] = by_intensity

    summary = _build_repeated_summary(
        per_author, provider, plumbing, experiment_id, n_new,
        {"prompt_tokens": tot_prompt, "completion_tokens": tot_completion,
         "total_tokens": tot_total}, new_requests)
    _write_json(out_dir / "controllability_repeated_summary.json", summary)
    (out_dir / "controllability_repeated_report.md").write_text(
        _render_repeated_report(per_author, provider, plumbing, experiment_id, n_new),
        encoding="utf-8")
    return summary


def _build_repeated_summary(per_author: dict[str, dict[str, list[float]]],
                            provider: GenerationProvider,
                            plumbing: dict[str, Any] | None,
                            experiment_id: str, n_new: int,
                            total_tokens: dict[str, int],
                            new_requests: int) -> dict[str, Any]:
    authors: dict[str, Any] = {}
    for author_id, by_intensity in per_author.items():
        stats = {k: _summarize_samples(by_intensity[k]) for k in INTENSITY_LEVELS}
        means = {k: stats[k]["mean"] for k in INTENSITY_LEVELS}
        medians = {k: stats[k]["median"] for k in INTENSITY_LEVELS}
        direction, effect_size = _effect_direction(means)
        low_range = (stats["low"]["min"], stats["low"]["max"])
        med_range = (stats["medium"]["min"], stats["medium"]["max"])
        high_range = (stats["high"]["min"], stats["high"]["max"])
        authors[author_id] = {
            "per_intensity": stats,
            "monotonic_on_mean": check_monotonic(means),
            "monotonic_on_median": check_monotonic(medians),
            "effect_direction": direction,
            "effect_size_mean": effect_size,
            "range_overlap": {
                "low_medium": _ranges_overlap(low_range, med_range),
                "medium_high": _ranges_overlap(med_range, high_range),
            },
            "max_std": round(max(stats[k]["std"] for k in INTENSITY_LEVELS), 8),
        }

    return {
        "stage": "generation_controllability_repeated",
        "experiment_id": experiment_id,
        "controllability_repeated_version": CONTROLLABILITY_REPEATED_VERSION,
        "controllability_version": CONTROLLABILITY_VERSION,
        "deterministic_plan_and_prompt": True,
        "real_generation": True,
        "no_auto_evaluation": True,
        "no_auto_revision": True,
        "n_existing_samples": 1,
        "n_new_samples_per_cell": n_new,
        "n_total_samples_per_cell": 1 + n_new,
        "new_request_count": new_requests,
        "provider": provider.provider_id,
        "model": provider.model,
        "endpoint": f"{provider.base_url}/chat/completions",
        "generation_parameters": GENERATION_PARAMETERS.to_dict(),
        "writing_request": NEUTRAL_REQUEST.to_dict(),
        "intensity_levels": list(INTENSITY_LEVELS),
        "intensity_to_activation": dict(INTENSITY_TO_ACTIVATION),
        "plumbing": plumbing,
        "authors": authors,
        "total_tokens": total_tokens,
        "note": ("monotonic_on_mean/median 是 n=3 小样本观测，非硬门；首样本复用既有单次"
                 "实验（不重复计费），本轮 token 仅为新增 12 fresh 请求"),
    }


def _render_repeated_report(per_author: dict[str, dict[str, list[float]]],
                            provider: GenerationProvider,
                            plumbing: dict[str, Any] | None,
                            experiment_id: str, n_new: int) -> str:
    lines = [
        "# Weaver Style Engine — 生成可控性重复采样报告（§19.5 repeated-sampling）",
        "",
        "对既有单次实验追加重复样本（每 cell n=3 = 首样本 + 2 个 fresh），判断 low/medium/high",
        "的 Layer D 距离趋势在**均值/中位数层面**是否稳定。同一 WritingRequest / 模型 / 生成",
        "参数；唯一变量仍是风格控制强度。距离越**小** → 越贴近作者指纹。",
        "",
        f"- **experiment_id**: `{experiment_id}`",
        f"- **provider / model**: `{provider.provider_id}` / `{provider.model}`",
        f"- **生成参数**: {GENERATION_PARAMETERS.to_dict()}",
        f"- **每档样本数**: 1 首样本 + {n_new} 新样本 = {1 + n_new}",
        "",
        "## 每作者三档距离（n=3）与单调性",
        "",
        "| 作者 | 档 | n | mean | median | std | min | max |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for author_id in AUTHOR_IDS:
        by = per_author[author_id]
        for intensity in INTENSITY_LEVELS:
            s = _summarize_samples(by[intensity])
            lines.append(
                f"| {author_id} | {intensity} | {s['n']} | {s['mean']} | "
                f"{s['median']} | {s['std']} | {s['min']} | {s['max']} |")

    lines += [
        "",
        "## 单调性判定（观测，非硬门）",
        "",
        "| 作者 | mean 单调? | mean 方向 | median 单调? | median 方向 | effect 方向 | "
        "effect_size(low−high) | 区间重叠 low↔med / med↔high |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for author_id in AUTHOR_IDS:
        by = per_author[author_id]
        stats = {k: _summarize_samples(by[k]) for k in INTENSITY_LEVELS}
        means = {k: stats[k]["mean"] for k in INTENSITY_LEVELS}
        medians = {k: stats[k]["median"] for k in INTENSITY_LEVELS}
        m_mean = check_monotonic(means)
        m_med = check_monotonic(medians)
        direction, effect_size = _effect_direction(means)
        low_range = (stats["low"]["min"], stats["low"]["max"])
        med_range = (stats["medium"]["min"], stats["medium"]["max"])
        high_range = (stats["high"]["min"], stats["high"]["max"])
        overlap = f"{_ranges_overlap(low_range, med_range)} / " \
                  f"{_ranges_overlap(med_range, high_range)}"
        lines.append(
            f"| {author_id} | {m_mean['monotonic']} | {m_mean['direction']} | "
            f"{m_med['monotonic']} | {m_med['direction']} | {direction} | "
            f"{effect_size} | {overlap} |")

    lines += [
        "",
        "> 注：n=3 小样本，均值/中位数观测仍弱、LLM 抽样随机；区间重叠反映档间散布是否"
        "大于档间差异。non-monotonic 不视为失败。原始单次实验产物（无 `_rep` 后缀）未被"
        "覆盖；新样本见 `{author}_{intensity}_rep{{1..2}}_*`。",
        "",
    ]
    return "\n".join(lines) + "\n"


def main_repeated() -> None:
    summary = run_controllability_repeated()
    print(f"experiment_id: {summary['experiment_id']}")
    print(f"provider/model: {summary['provider']} / {summary['model']}")
    print(f"new_request_count: {summary['new_request_count']}")
    print(f"total_tokens(new): {summary['total_tokens']}")
    for aid in AUTHOR_IDS:
        a = summary["authors"][aid]
        print(f"{aid}: mean={ {k: a['per_intensity'][k]['mean'] for k in INTENSITY_LEVELS} } "
              f"median={ {k: a['per_intensity'][k]['median'] for k in INTENSITY_LEVELS} } "
              f"std={ {k: a['per_intensity'][k]['std'] for k in INTENSITY_LEVELS} }")
        print(f"  monotonic_on_mean={a['monotonic_on_mean']['monotonic']} "
              f"({a['monotonic_on_mean']['direction']})  "
              f"monotonic_on_median={a['monotonic_on_median']['monotonic']} "
              f"({a['monotonic_on_median']['direction']})  "
              f"effect_direction={a['effect_direction']} "
              f"effect_size={a['effect_size_mean']}")
    print("artifacts: data/analysis/generation/"
          f"{EXPERIMENT_ID_95}/controllability_repeated_summary.json + "
          "controllability_repeated_report.md + {author}_{intensity}_rep{1..2}_*")


def main() -> None:
    summary = run_controllability()
    print(f"experiment_id: {summary['experiment_id']}")
    print(f"provider/model: {summary['provider']} / {summary['model']}")
    for aid in AUTHOR_IDS:
        a = summary["authors"][aid]
        print(f"{aid}: distances={a['distances']} monotonic={a['monotonic']} "
              f"direction={a['direction']}")
    print(f"total_tokens: {summary['total_tokens']}")
    print("artifacts: data/analysis/generation/"
          f"{EXPERIMENT_ID_95}/{{author}}_{{intensity}}_generation.json + "
          "{author}_{intensity}_passage.md + controllability_summary.json + "
          "controllability_report.md")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "repeated":
        main_repeated()
    else:
        main()
