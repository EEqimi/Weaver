# knowledge/evaluation/compare.py
"""Phase 8 目标 vs 实际比较（纯函数，确定性，无 LLM、无随机、无时间）。

把 ActualStyleProfile（再测量）与目标（StylePlan / AuthorStyleProfile + band 阈值）
对齐，产出三类偏差：语言特征 band 偏差、叙事字段偏差、策略覆盖。输出只含
low/medium/high 这类可解释 band 与 on_target/above/below/off_target 状态，绝不含
微观 stylometric 指纹，也不生成任何改写指令（改写计划由 revision.py 负责）。
"""
from __future__ import annotations

from typing import Any

from ..planning.bands import band_label
from ..planning.schema import StylePlan
from ..profiles.style_profile import AuthorStyleProfile
from ..schema.narrative_schema import UNKNOWN_VALUES
from .schema import (
    ComparisonResult, FeatureDeviation, NarrativeDeviation, StrategyCoverage,
)

_BAND_RANK: dict[str, int] = {"low": 0, "medium": 1, "high": 2}

_STATUS_ON = "on_target"
_STATUS_OFF = "off_target"
_STATUS_ABOVE = "above"
_STATUS_BELOW = "below"
_STATUS_NOT_MEASURABLE = "not_measurable"


def _num(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _band_status(target_band: str | None, actual_band: str | None) -> str:
    if target_band is None or actual_band is None:
        return _STATUS_NOT_MEASURABLE
    if target_band == actual_band:
        return _STATUS_ON
    return _STATUS_ABOVE if _BAND_RANK[actual_band] > _BAND_RANK[target_band] else _STATUS_BELOW


def _actual_feature_value(actual, feature_id: str) -> float | None:
    """从 ActualStyleProfile 取某语言特征的实测数值（统计优先，其次判断层）。"""
    for layer in (actual.layer_a_statistical, actual.layer_a_judgment):
        fv = layer.get(feature_id)
        v = _num(fv.get("value")) if isinstance(fv, dict) else None
        if v is not None:
            return v
    return None


def _language_deviations(plan: StylePlan, profile: AuthorStyleProfile,
                         actual, thresholds: dict) -> list[FeatureDeviation]:
    out: list[FeatureDeviation] = []
    for c in plan.language_controls:
        gc = profile.generation_controls.get(c.feature_id)
        target_value = _num(gc.summary.get("mean")) if gc else None
        target_band = (band_label(c.feature_id, target_value, thresholds)
                       if target_value is not None else None)

        actual_value = _actual_feature_value(actual, c.feature_id)
        actual_band = (band_label(c.feature_id, actual_value, thresholds)
                       if actual_value is not None else None)

        measurable = (actual_value is not None and target_band is not None
                      and actual_band is not None)
        if actual_value is None:
            status = _STATUS_NOT_MEASURABLE
        else:
            status = _band_status(target_band, actual_band)

        out.append(FeatureDeviation(
            feature_id=c.feature_id,
            target_band=target_band,
            actual_band=actual_band,
            target_value=target_value,
            actual_value=actual_value,
            status=status,
            measurable=measurable,
            reason=(f"target={target_band}, actual={actual_band}"
                    if measurable else "value 或 band 阈值缺失，无法分类"),
        ))
    return out


def _narrative_deviations(plan: StylePlan, actual) -> list[NarrativeDeviation]:
    out: list[NarrativeDeviation] = []
    narrative = actual.layer_b_narrative or {}
    for nc in plan.narrative_controls:
        if nc.activation != "medium":
            continue
        target = nc.summary.get("mode")
        if nc.value_type == "categorical" and target is not None:
            actual_value = narrative.get(nc.field)
            if actual_value is None or actual_value in UNKNOWN_VALUES:
                out.append(NarrativeDeviation(
                    field=nc.field, target_value=str(target),
                    actual_value=actual_value, status=_STATUS_NOT_MEASURABLE,
                    reason="实测叙事字段缺失或不可判定"))
            elif actual_value == target:
                out.append(NarrativeDeviation(
                    field=nc.field, target_value=str(target),
                    actual_value=actual_value, status=_STATUS_ON))
            else:
                out.append(NarrativeDeviation(
                    field=nc.field, target_value=str(target),
                    actual_value=actual_value, status=_STATUS_OFF,
                    reason=f"measured {actual_value!r}, target {target!r}"))
        else:
            out.append(NarrativeDeviation(
                field=nc.field, target_value=(str(target) if target is not None else None),
                actual_value=None, status=_STATUS_NOT_MEASURABLE,
                reason="distribution 字段或目标 mode 缺失，V0.1 不比较"))
    return out


def _strategy_coverage(plan: StylePlan, actual) -> list[StrategyCoverage]:
    matched: dict[str, list[str]] = {}
    for entry in actual.layer_c_strategies:
        sid = entry.get("strategy_id")
        ev = entry.get("evidence") or {}
        quotes = [q for q in (ev.get("quotes") or []) if isinstance(q, str)]
        if not quotes:
            q0 = ev.get("quote")
            if isinstance(q0, str) and q0:
                quotes = [q0]
        matched[sid] = quotes

    out: list[StrategyCoverage] = []
    for s in plan.strategy_controls:
        active = s.activation == "active"
        if active:
            out.append(StrategyCoverage(
                strategy_id=s.canonical_strategy_id, active=True,
                matched=s.canonical_strategy_id in matched,
                evidence_quotes=list(matched.get(s.canonical_strategy_id, [])),
            ))
    return out


def compare_target_actual(plan: StylePlan, profile: AuthorStyleProfile,
                          actual, thresholds: dict) -> ComparisonResult:
    """目标（plan/profile）vs 实际（actual）→ ComparisonResult（纯函数）。"""
    lang = _language_deviations(plan, profile, actual, thresholds)
    narr = _narrative_deviations(plan, actual)
    strat = _strategy_coverage(plan, actual)

    def _counts(items, key):
        from collections import Counter
        return dict(sorted(Counter(getattr(i, key) for i in items).items()))

    summary = {
        "n_language_controls": len(lang),
        "language_status": _counts(lang, "status"),
        "n_narrative_controls_active": len(narr),
        "narrative_status": _counts(narr, "status"),
        "n_strategies_active": sum(1 for s in strat if s.active),
        "n_strategies_matched": sum(1 for s in strat if s.matched),
        "stylometric": actual.layer_d_stylometric,
    }
    return ComparisonResult(
        author_id=plan.author_id,
        passage_id=actual.passage_id,
        language_deviations=lang,
        narrative_deviations=narr,
        strategy_coverage=strat,
        summary=summary,
    )
