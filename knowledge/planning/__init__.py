# knowledge/planning/__init__.py
"""Phase 6：Style Planner & Prompt Compiler（作者画像 → 写作计划 → 生成提示词）。

三层严格分离（绝不混成一个结构）：
    AuthorStyleProfile = "我们观察到作者是什么样"（Phase 5，只读）
    StylePlan          = "这次写作应该激活哪些风格控制"（planner 输出）
    CompiledPrompt     = "如何把 StylePlan 翻译成生成模型可执行的指令"（compiler 输出）

本包全部确定性：无 LLM、无随机数、无时间戳内容、无生成正文。
"""
from .schema import (
    ActivationLevel, PlannedControl, PlannedNarrativeControl, PlannedStrategy,
    PlannerPolicy, PlanningError, PromptBudgetError, StylePlan, WritingRequest,
)
from .planner import StylePlanner
from .compiler import CompiledPrompt, PromptCompiler
from .bands import (
    band_label, compute_band_thresholds, describe_feature,
    build_band_thresholds_artifact, load_band_thresholds,
)

__all__ = [
    "ActivationLevel", "PlannedControl", "PlannedNarrativeControl",
    "PlannedStrategy", "PlannerPolicy", "PlanningError", "PromptBudgetError",
    "StylePlan", "WritingRequest", "StylePlanner", "CompiledPrompt",
    "PromptCompiler", "band_label", "compute_band_thresholds", "describe_feature",
    "build_band_thresholds_artifact", "load_band_thresholds",
]
