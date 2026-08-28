# knowledge/evaluation/__init__.py
"""Phase 8：Style Feedback Loop + 独立 LLM 文学评价（spec §15 / §19.5 / §21）。

把 Phase 7 的生成正文重新送入测量管线，得到 Actual Style Profile（Layer A 统计 +
Layer A 判断 LLM + Layer B 叙事 + Layer C 策略 + Layer D stylometric），与目标画像
（StylePlan / AuthorStyleProfile）比较出偏差，产出优先化的 Revision Plan（P0–P4），
做一次最小编辑改写，再分析，跑 stylometric 诊断，最终确定性决定 Accept / Continue /
Roll Back。另加**独立**的 LLM 文学评价（6 维 1–10 + 证据引文）。

铁律：
    - 文学评价与改写器同为盲测：prompt 绝不含作者名或 "write like"/"imitate"/
      "in the style of"（复用 generation/schema.py 的 A/B 泄露守卫，fail-closed）。
    - P0（故事情节 / 语义连贯）绝不因低优先级风格编辑而被破坏：改写指令显式禁止
      改动情节 / 事实 / 人物 / 中性 brief 约束。
    - 改写指令只含可解释的自然语言（字面 guidance / 有限词汇），绝不含原始数值或
      微观 stylometric 指纹（"增加 char 3-gram"等）；stylometric 距离只作诊断。
    - 密钥只读（DEEPSEEK_API_KEY），绝不打印 / 保存 / 提交；复用 DeepSeekProvider。
"""
from .schema import (
    ActualStyleProfile, ComparisonResult, DimensionScore, EvalError,
    FeatureDeviation, LiteraryEvaluation, NarrativeDeviation, RevisionItem,
    RevisionPlan, RevisionResult, StrategyCoverage,
    CATEGORY_TO_PRIORITY, DEFAULT_DIMENSION_WEIGHTS, LITERARY_DIMENSIONS,
    REVISION_PRIORITIES, priority_rank,
)

__all__ = [
    "ActualStyleProfile", "ComparisonResult", "DimensionScore", "EvalError",
    "FeatureDeviation", "LiteraryEvaluation", "NarrativeDeviation", "RevisionItem",
    "RevisionPlan", "RevisionResult", "StrategyCoverage",
    "CATEGORY_TO_PRIORITY", "DEFAULT_DIMENSION_WEIGHTS", "LITERARY_DIMENSIONS",
    "REVISION_PRIORITIES", "priority_rank",
]
