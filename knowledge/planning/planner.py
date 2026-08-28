# knowledge/planning/planner.py
"""Phase 6 StylePlanner：AuthorStyleProfile → StylePlan（本次写作激活哪些控制）。

纯函数、确定性、无 LLM、无随机、无时间戳。绝不改写用户 core story facts / 人物关系 /
约束；绝不把画像 JSON 直接塞进 plan（plan 只保留"本次要激活的控制 + 理由 + 自然语言
指导"，数值与 stylometric 指纹一律留在画像层，由 compiler 再翻译）。
"""
from __future__ import annotations

from typing import Any

from ..profiles.style_profile import AuthorStyleProfile
from ..schema.versions import STYLE_PLAN_SCHEMA_VERSION, STYLE_PLANNER_VERSION
from .policy import (
    apply_narrative_budget, assign_language_buckets, describe_feature,
    describe_narrative, language_activation, narrative_activation,
    select_strategies, _support_snapshot,
)
from .schema import (
    PlannedControl, PlannedNarrativeControl, PlannedStrategy, PlannerPolicy,
    PlanningError, StylePlan, WritingRequest, make_style_plan_id,
)


class StylePlanner:
    """把画像（观察）翻译成计划（本次激活哪些控制）。"""

    def __init__(self, policy: PlannerPolicy | None = None) -> None:
        self.policy = policy or PlannerPolicy()

    # ------------------------------------------------------------------ #
    # 入口
    # ------------------------------------------------------------------ #
    def plan(self, profile: AuthorStyleProfile, request: WritingRequest) -> StylePlan:
        self._validate_profile(profile)

        language, reference, suppressed = self._plan_language(profile)
        narrative = self._plan_narrative(profile, request)
        active_strategies, reference_strategies = self._plan_strategies(profile)
        warnings = self._build_warnings(profile, request, language)

        style_plan_id = make_style_plan_id(
            profile.author_id, profile.reproducibility_hash, request, self.policy)

        return StylePlan(
            style_plan_id=style_plan_id,
            schema_version=STYLE_PLAN_SCHEMA_VERSION,
            author_id=profile.author_id,
            source_profile_hash=profile.reproducibility_hash,
            writing_request=request.to_dict(),
            language_controls=language,
            narrative_controls=narrative,
            strategy_controls=active_strategies,
            reference_controls=reference,
            reference_strategy_controls=reference_strategies,
            suppressed_controls=suppressed,
            warnings=warnings,
            planner_metadata=self._build_metadata(
                profile, language, reference, suppressed,
                active_strategies, reference_strategies, self.policy),
        )

    # ------------------------------------------------------------------ #
    # 画像完整性校验（fail-closed）
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_profile(profile: AuthorStyleProfile) -> None:
        if not profile.verify_reproducibility_hash():
            raise PlanningError(
                f"{profile.author_id}: reproducibility hash 不匹配，拒绝规划（画像可能被篡改）。")
        iso = profile.author_scope.get("held_out_isolation", {})
        if not iso.get("clean", False):
            raise PlanningError(
                f"{profile.author_id}: held-out 隔离不干净，拒绝规划（越过实验边界）。")

    # ------------------------------------------------------------------ #
    # Layer A：语言控制
    # ------------------------------------------------------------------ #
    def _plan_language(self, profile: AuthorStyleProfile) -> tuple[
            list[PlannedControl], list[PlannedControl], list[PlannedControl]]:
        raw: list[dict[str, Any]] = []
        for feature_id, gc in profile.generation_controls.items():
            summary = gc.summary or {}
            activation, reason = language_activation(
                feature_id, gc.registry_control_role, gc.source_scope,
                summary, self.policy)
            raw.append({
                "feature_id": feature_id,
                "registry_control_role": gc.registry_control_role,
                "activation": activation,
                "bucket": "",
                "source_scope": gc.source_scope,
                "support": _support_snapshot(summary),
                "reason": reason,
                "guidance": describe_feature(feature_id, summary),
                "source": gc.source_artifact,
            })
        activated, reference, suppressed = assign_language_buckets(raw, self.policy)
        return (
            [PlannedControl.from_dict(c) for c in activated],
            [PlannedControl.from_dict(c) for c in reference],
            [PlannedControl.from_dict(c) for c in suppressed],
        )

    # ------------------------------------------------------------------ #
    # Layer B：叙事控制
    # ------------------------------------------------------------------ #
    def _plan_narrative(self, profile: AuthorStyleProfile,
                        request: WritingRequest) -> list[PlannedNarrativeControl]:
        raw: list[dict[str, Any]] = []
        for field, nc in profile.narrative_controls.items():
            summary = nc.summary or {}
            activation, reason, overridden = narrative_activation(
                field, summary, nc.value_type, request.pov)
            raw.append({
                "field": field,
                "activation": activation,
                "value_type": nc.value_type,
                "summary": summary,
                "reason": reason,
                "guidance": describe_narrative(field, summary, nc.value_type),
                "overridden": overridden,
            })
        ordered = apply_narrative_budget(raw, self.policy)
        return [PlannedNarrativeControl.from_dict(c) for c in ordered]

    # ------------------------------------------------------------------ #
    # Layer C：canonical 策略
    # ------------------------------------------------------------------ #
    def _plan_strategies(self, profile: AuthorStyleProfile) -> tuple[
            list[PlannedStrategy], list[PlannedStrategy]]:
        canonical_dicts: list[dict[str, Any]] = []
        for s in profile.strategy_controls:
            canonical_dicts.append({
                "canonical_strategy_id": s.canonical_strategy_id,
                "canonical_name": s.canonical_name,
                "canonical_description": s.canonical_description,
                "trigger_summary": s.trigger_summary,
                "operation_summary": s.operation_summary,
                "effect_summary": s.effect_summary,
                "support_status": s.support_status,
                "confidence": s.confidence,
                "control_priority": s.control_priority,
                "source_strategy_ids": s.source_strategy_ids,
                "supporting_work_ids": s.supporting_work_ids,
                "supporting_chunk_ids": s.supporting_chunk_ids,
                "source_artifact": s.source_artifact,
            })
        active, reference = select_strategies(canonical_dicts, self.policy)
        return (
            [self._make_planned_strategy(c, "active") for c in active],
            [self._make_planned_strategy(c, "reference") for c in reference],
        )

    @staticmethod
    def _make_planned_strategy(c: dict[str, Any], activation: str) -> PlannedStrategy:
        if activation == "active":
            reason = f"{c['support_status']} canonical strategy (priority {c['control_priority']})"
        elif c["support_status"] == "discovered":
            reason = "discovered strategy kept as reference (not forced active)"
        else:
            reason = "strategy budget exceeded → kept as reference (never silently dropped)"
        return PlannedStrategy(
            canonical_strategy_id=c["canonical_strategy_id"],
            canonical_name=c["canonical_name"],
            support_status=c["support_status"],
            confidence=c["confidence"],
            control_priority=c["control_priority"],
            n_supporting_works=len(c["supporting_work_ids"]),
            n_supporting_chunks=len(c["supporting_chunk_ids"]),
            activation=activation,
            trigger=c["trigger_summary"],
            operation=c["operation_summary"],
            effect=c["effect_summary"],
            reason=reason,
        )

    # ------------------------------------------------------------------ #
    # warnings / metadata
    # ------------------------------------------------------------------ #
    def _build_warnings(self, profile: AuthorStyleProfile, request: WritingRequest,
                        language: list[PlannedControl]) -> list[str]:
        warnings: list[str] = []
        if any(c.registry_control_role == "candidate_core" and c.activation == "strong"
               for c in language):
            warnings.append(
                "candidate_core controls activated as strong are CANDIDATE — not a "
                "verified core (calibration sample insufficient for core promotion).")
        if request.pov is not None:
            pov_nc = profile.narrative_controls.get("pov")
            author_pov = pov_nc.summary.get("mode") if pov_nc else None
            if author_pov not in (None, request.pov):
                warnings.append(
                    "author tendency conflicts with explicit user constraint: "
                    f"pov={request.pov} (author tendency={author_pov}). User wins.")
        return warnings

    @staticmethod
    def _build_metadata(profile: AuthorStyleProfile,
                        language: list[PlannedControl],
                        reference: list[PlannedControl],
                        suppressed: list[PlannedControl],
                        active_strategies: list[PlannedStrategy],
                        reference_strategies: list[PlannedStrategy],
                        policy: PlannerPolicy) -> dict[str, Any]:
        return {
            "planner_version": STYLE_PLANNER_VERSION,
            "policy": policy.to_dict(),
            "author_scope": profile.author_scope,
            "generation_control_count": len(profile.generation_controls),
            "narrative_control_count": len(profile.narrative_controls),
            "strategy_control_count": len(profile.strategy_controls),
            "activated_language_controls": len(language),
            "reference_language_controls": len(reference),
            "suppressed_language_controls": len(suppressed),
            "active_strategies": len(active_strategies),
            "reference_strategies": len(reference_strategies),
            # 单作者规划通常无冲突；结构预留给未来的多作者风格混合。
            "conflicts": [],
            "resolution_required": False,
        }
