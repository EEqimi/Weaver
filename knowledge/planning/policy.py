# knowledge/planning/policy.py
"""Phase 6 确定性策略：激活政策、控制预算、数值→自然语言描述。

全部纯函数：无 I/O、无 LLM、无随机。这是"哪个控制该以多强激活 + 为什么 + 怎么表达"
的唯一权威实现（planner / compiler 复用，避免两处口径不一致）。

Phase 6.1 起，连续统计特征的"怎么表达"（数值→字面 guidance）由 `bands.py` 提供
（TRAIN-only 经验分位数阈值）；本模块只负责激活政策、预算与叙事字段的表述。

自然语言描述一律为 **English**（目标生成文本为英文，且 spec §18 的示例即英文
"tends toward longer complete sentences"）；代码注释保留中文以与库内风格一致。
"""
from __future__ import annotations

from typing import Any

from .schema import ActivationLevel, PlannerPolicy

# --------------------------------------------------------------------------- #
# 激活排序 / 角色排序（确定性）
# --------------------------------------------------------------------------- #
_ACTIVATION_RANK = {
    ActivationLevel.STRONG.value: 0,
    ActivationLevel.MEDIUM.value: 1,
    ActivationLevel.WEAK.value: 2,
    ActivationLevel.REFERENCE.value: 3,
    ActivationLevel.SUPPRESSED.value: 4,
}
_ROLE_RANK = {"candidate_core": 0, "descriptive": 1, "experimental": 2, "core": 0}
_STRATEGY_TIER_RANK = {"validated": 0, "candidate": 1, "discovered": 2}

# descriptive 辅助控制的确定性优先序（区分度高的标点/节奏/词汇维度先于一般维度）。
# 这是文档化策略，不是随机抽样；未列出的 descriptive 按 feature_id 升序兜底。
_DESCRIPTIVE_PRIORITY = [
    "comma_density", "semicolon_density", "dash_density", "quotation_density",
    "exclamation_frequency", "question_frequency", "period_density",
    "long_sentence_ratio", "short_sentence_ratio", "sentence_length_cv",
    "type_token_ratio", "hapax_ratio", "word_repetition_rate",
    "mean_word_length", "connective_density",
]


# --------------------------------------------------------------------------- #
# 证据门槛
# --------------------------------------------------------------------------- #
def _int(summary: dict[str, Any], key: str) -> int:
    v = summary.get(key)
    return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0


def _evidence_blocking(summary: dict[str, Any]) -> str | None:
    """not_observable / insufficient / missing 绝不成为积极写作指令。"""
    if _int(summary, "n_unobservable") > 0:
        return "not_observable"
    if _int(summary, "n_insufficient") > 0:
        return "insufficient_evidence"
    if _int(summary, "n_expected") <= 0:
        return "missing"
    if _int(summary, "n_valid") <= 0:
        return "missing"
    return None


def _support_snapshot(summary: dict[str, Any]) -> dict[str, Any]:
    """抽取可用于解释的 support/uncertainty 摘要（不重算，不丢字段语义）。"""
    conf = summary.get("confidence") or {}
    return {
        "n_expected": _int(summary, "n_expected"),
        "n_valid": _int(summary, "n_valid"),
        "n_missing": _int(summary, "n_missing"),
        "n_unobservable": _int(summary, "n_unobservable"),
        "n_insufficient": _int(summary, "n_insufficient"),
        "variance": summary.get("variance"),
        "confidence_mean": conf.get("mean"),
    }


# --------------------------------------------------------------------------- #
# 语言控制激活政策
# --------------------------------------------------------------------------- #
def _gate_candidate_core(summary: dict[str, Any], source_scope: str,
                         policy: PlannerPolicy) -> tuple[str, str]:
    block = _evidence_blocking(summary)
    if block:
        return ActivationLevel.SUPPRESSED.value, f"candidate_core gated: {block}"
    n_expected = _int(summary, "n_expected")
    n_valid = _int(summary, "n_valid")
    completeness = n_valid / n_expected if n_expected else 0.0
    if completeness < policy.candidate_core_min_completeness:
        return (ActivationLevel.WEAK.value,
                f"candidate_core gated: completeness {completeness:.2f} "
                f"< {policy.candidate_core_min_completeness}")
    variance = summary.get("variance")
    if isinstance(variance, (int, float)) and not isinstance(variance, bool) and variance <= 0:
        return ActivationLevel.WEAK.value, "candidate_core gated: zero variance (constant)"
    if source_scope != "full_train_corpus":
        return ActivationLevel.MEDIUM.value, "candidate_core gated: sampled scope (not full corpus)"
    mean = summary.get("mean")
    std = summary.get("std")
    if isinstance(mean, (int, float)) and isinstance(std, (int, float)) and abs(mean) > 1e-9:
        if std / abs(mean) > 3.0:
            return ActivationLevel.MEDIUM.value, "candidate_core gated: high relative dispersion"
    return (ActivationLevel.STRONG.value,
            "candidate_core: full-corpus, complete, stable (still NOT a verified core)")


def language_activation(feature_id: str, registry_role: str, source_scope: str,
                        summary: dict[str, Any],
                        policy: PlannerPolicy) -> tuple[str, str]:
    """确定性决定一个语言控制的激活级别 + 原因。绝不晋升 candidate_core 为 core。"""
    if registry_role == "diagnostic":
        return ActivationLevel.SUPPRESSED.value, "diagnostic never controls generation"
    if registry_role == "core":
        # 未来验证通过的正式核心才可能走到这里；当前 V0.1 无 core。
        return ActivationLevel.STRONG.value, "core (verified strong control)"
    if registry_role == "experimental":
        block = _evidence_blocking(summary)
        if block:
            return ActivationLevel.SUPPRESSED.value, f"experimental gated: {block}"
        return ActivationLevel.REFERENCE.value, "experimental (sampled LLM) — reference_only"
    if registry_role == "candidate_core":
        return _gate_candidate_core(summary, source_scope, policy)
    if registry_role == "descriptive":
        block = _evidence_blocking(summary)
        if block:
            return ActivationLevel.SUPPRESSED.value, f"descriptive gated: {block}"
        return ActivationLevel.WEAK.value, "descriptive: auxiliary/weak control"
    return ActivationLevel.SUPPRESSED.value, f"unknown registry role {registry_role!r}"


# --------------------------------------------------------------------------- #
# 语言控制预算：primary / secondary / reference / suppressed
# --------------------------------------------------------------------------- #
def _sort_key_language(c: dict[str, Any]) -> tuple:
    role = c.get("registry_control_role", "descriptive")
    prio = (_DESCRIPTIVE_PRIORITY.index(c["feature_id"])
            if c["feature_id"] in _DESCRIPTIVE_PRIORITY else len(_DESCRIPTIVE_PRIORITY))
    return (
        _ACTIVATION_RANK.get(c["activation"], 9),
        _ROLE_RANK.get(role, 9),
        prio,
        c["feature_id"],
    )


def assign_language_buckets(controls: list[dict[str, Any]],
                            policy: PlannerPolicy) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """把已定激活级别的语言控制分成 (activated, reference, suppressed)。

    activated 内按 _sort_key_language 排序，再切 primary/secondary；
    超出预算的进 suppressed（bucket 标 suppressed，reason 记 budget）。
    """
    activated: list[dict[str, Any]] = []
    reference: list[dict[str, Any]] = []
    suppressed: list[dict[str, Any]] = []
    for c in controls:
        act = c["activation"]
        if act == ActivationLevel.REFERENCE.value:
            c["bucket"] = "reference"
            reference.append(c)
        elif act == ActivationLevel.SUPPRESSED.value:
            c["bucket"] = "suppressed"
            suppressed.append(c)
        else:
            activated.append(c)

    activated.sort(key=_sort_key_language)
    primary_n = min(policy.max_primary_controls, len(activated))
    secondary_n = min(policy.max_secondary_controls, max(0, len(activated) - primary_n))
    for i, c in enumerate(activated):
        if i < primary_n:
            c["bucket"] = "primary"
        elif i < primary_n + secondary_n:
            c["bucket"] = "secondary"
        else:
            c["bucket"] = "suppressed"
            c["activation"] = ActivationLevel.SUPPRESSED.value
            c["reason"] = "suppressed_due_to_budget: 超出 language control budget"
            suppressed.append(c)
    activated = [c for c in activated if c["bucket"] != "suppressed"]
    return activated, reference, suppressed


# --------------------------------------------------------------------------- #
# 叙事控制激活政策
# --------------------------------------------------------------------------- #
def _narrative_mode(summary: dict[str, Any]) -> str | None:
    mode = summary.get("mode")
    return mode if isinstance(mode, str) else None


def narrative_activation(field: str, summary: dict[str, Any], value_type: str,
                         user_pov: str | None) -> tuple[str, str, bool]:
    """叙事维度激活：sampled 证据 → 不超 medium；用户显式 POV 覆盖作者倾向。

    注意：narrative summary 只有 `n`（40-chunk 样本量），没有 n_valid/n_expected；
    这里用 `n` 判缺失，用 mode / mean_distribution 判可观测性。
    """
    if field == "pov" and user_pov is not None:
        return (ActivationLevel.SUPPRESSED.value,
                "user explicit POV overrides author tendency", True)

    n = _int(summary, "n")
    if value_type == "categorical":
        mode = _narrative_mode(summary)
        if mode is None:
            return ActivationLevel.SUPPRESSED.value, "no mode (missing)", False
        if mode in ("not_observable", "insufficient_evidence", "unknown"):
            return ActivationLevel.SUPPRESSED.value, f"mode={mode}", False
        if n == 0:
            return ActivationLevel.SUPPRESSED.value, "n=0 (missing)", False
        # sampled（40-chunk）→ 不超 medium
        return ActivationLevel.MEDIUM.value, "sampled narrative tendency (calibration sample)", False
    # distribution
    dist = summary.get("mean_distribution")
    if not dist or n == 0:
        return ActivationLevel.SUPPRESSED.value, "no distribution / missing", False
    return ActivationLevel.MEDIUM.value, "sampled narrative distribution (calibration sample)", False


def _dominant_key(dist: dict[str, Any]) -> str:
    if not dist:
        return ""
    return max(dist, key=lambda k: float(dist[k]))


# 叙事字段的确定性优先序（视角/叙述者存在感等"框架性"维度先于节奏/细节维度）。
_NARRATIVE_PRIORITY = [
    "pov", "narrator_presence", "focalization", "narrative_distance",
    "perspective_stability", "information_access", "temporal_order",
    "narrator_evaluative_intervention", "temporal_pace", "scene_detail",
]


def _narrative_sort_key(c: dict[str, Any]) -> tuple:
    prio = (_NARRATIVE_PRIORITY.index(c["field"])
            if c["field"] in _NARRATIVE_PRIORITY else len(_NARRATIVE_PRIORITY))
    return (_ACTIVATION_RANK.get(c.get("activation"), 9), prio, c["field"])


def apply_narrative_budget(controls: list[dict[str, Any]],
                           policy: PlannerPolicy) -> list[dict[str, Any]]:
    """预算内 medium 保持；超出 → suppressed + reason（记录，绝不静默丢弃）。

    overridden 字段（用户显式约束）不受预算影响；返回按字段优先序稳定排序的列表。
    """
    ordered = sorted(controls, key=_narrative_sort_key)
    kept = 0
    for c in ordered:
        if c["activation"] == ActivationLevel.MEDIUM.value:
            if kept < policy.max_narrative_controls:
                kept += 1
            else:
                c["activation"] = ActivationLevel.SUPPRESSED.value
                c["reason"] = "suppressed_due_to_budget: 超出 narrative control budget"
    return sorted(controls, key=lambda c: (
        _NARRATIVE_PRIORITY.index(c["field"]) if c["field"] in _NARRATIVE_PRIORITY
        else len(_NARRATIVE_PRIORITY),
        c["field"],
    ))


# --------------------------------------------------------------------------- #
# 策略选择政策
# --------------------------------------------------------------------------- #
def select_strategies(canonicals: list[dict[str, Any]],
                      policy: PlannerPolicy) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """按 tier + control_priority 选择：validated > candidate > discovered。

    discovered 默认 reference（不主动激活），除非 policy.allow_discovered_strategies_as_active。
    返回 (active, reference)。
    """
    ordered = sorted(canonicals, key=lambda c: (
        _STRATEGY_TIER_RANK.get(c.get("support_status"), 9),
        int(c.get("control_priority") or 0),
    ))
    active: list[dict[str, Any]] = []
    reference: list[dict[str, Any]] = []
    for c in ordered:
        status = c.get("support_status", "discovered")
        if status == "discovered" and not policy.allow_discovered_strategies_as_active:
            reference.append(c)
        else:
            if len(active) < policy.max_strategies:
                active.append(c)
            else:
                reference.append(c)  # 超出策略预算 → 保留为 reference，不删除
    return active, reference


# --------------------------------------------------------------------------- #
# 叙事 → 自然语言描述（English）
# --------------------------------------------------------------------------- #
# 注：语言特征（连续统计量）的数值→自然语言 banding 已移入 bands.py（Phase 6.1），
# 改用 TRAIN-only 经验分位数阈值，不再用人工绝对阈值，也不自造未测量的文学机制。
_NARRATIVE_VALUE_LABELS: dict[str, dict[str, str]] = {
    "pov": {"first": "first-person point of view", "third": "third-person point of view",
            "second": "second-person point of view"},
    "focalization": {
        "internal": "internal focalization (limited to a single character's perspective and knowledge)",
        "zero": "zero focalization (an omniscient narrator)",
        "external": "external focalization (only external behavior is shown)"},
    "perspective_stability": {"stable": "a stable viewpoint",
                              "mostly_stable": "a mostly stable viewpoint",
                              "shifting": "a viewpoint that shifts across characters"},
    "narrative_distance": {"close": "close narrative distance (near the character's inner life)",
                           "medium": "medium narrative distance",
                           "distant": "distant narrative distance"},
    "narrator_presence": {"low": "low narrator presence (the narrator recedes behind the characters)",
                          "medium": "medium narrator presence",
                          "high": "high narrator presence (overt commentary)"},
    "narrator_evaluative_intervention": {"low": "little evaluative narrator intervention",
                                         "medium": "moderate narrator intervention",
                                         "high": "frequent evaluative narrator intervention"},
    "information_access": {"limited": "limited information access (close to what characters know)",
                           "omniscient": "omniscient information access",
                           "objective": "objective presentation (does not enter minds)"},
    "temporal_order": {"chronological": "chronological order",
                       "analepsis": "analepsis (flashback)",
                       "prolepsis": "prolepsis (flash-forward)"},
}
_DIST_VALUE_LABELS: dict[str, dict[str, str]] = {
    "temporal_pace": {"scene": "scene", "summary": "summary", "ellipsis": "ellipsis",
                      "pause": "pause"},
    "scene_detail": {"dialogue": "dialogue", "action": "action", "psychology": "psychology",
                     "social_relations": "social relations", "environment": "environment",
                     "objects": "objects"},
}


def describe_narrative(field: str, summary: dict[str, Any], value_type: str) -> str:
    """叙事字段 → 自然语言（mode / 主导成分）。"""
    if value_type == "categorical":
        mode = _narrative_mode(summary)
        labels = _NARRATIVE_VALUE_LABELS.get(field, {})
        return labels.get(mode, f"{field}={mode}")
    dist = summary.get("mean_distribution") or {}
    key = _dominant_key(dist)
    labels = _DIST_VALUE_LABELS.get(field, {})
    label = labels.get(key, key)
    return f"dominated by {label}"
