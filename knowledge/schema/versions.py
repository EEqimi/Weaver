# knowledge/schema/versions.py
"""统一版本号。每个产物必须记录其 schema/analyzer 版本，保证可复现与可追溯。"""

# 语料处理
CLEANER_VERSION = "0.1.0"
CHUNKER_VERSION = "0.1.0"

# Schema
SCHEMA_VERSION = "0.1.0"
FEATURE_SCHEMA_VERSION = "0.1.0"
NARRATIVE_SCHEMA_VERSION = "0.2.0"
STRATEGY_SCHEMA_VERSION = "0.1.0"
# Phase 4.5：作者级 canonical 策略 schema 独立版本（不 bump STRATEGY_SCHEMA_VERSION，
# 以免让 Phase 4.4 的 strategy_match/discover 缓存键失效，从而破坏"复用标定结果"）。
CANONICAL_STRATEGY_SCHEMA_VERSION = "0.1.0"
STYLOMETRY_SCHEMA_VERSION = "0.1.0"

# Analyzer（与 schema_version 分离，见 spec §17.6）
STATISTICAL_ANALYZER_VERSION = "0.1.0"
LLM_ANALYZER_VERSION = "0.2.0"
NARRATIVE_ANALYZER_VERSION = "0.2.0"
STRATEGY_MINER_VERSION = "0.2.0"
STRATEGY_CONSOLIDATOR_VERSION = "0.1.0"
STYLOMETRY_VERSION = "0.1.0"
AGGREGATION_VERSION = "0.2.0"
SAMPLING_VERSION = "0.1.0"
# Phase 5：作者风格画像（AuthorStyleProfile）schema 版本。这是对既有 AuthorProfile +
# canonical strategy set + stylometry 的**确定性合成**（不重新分析、不调用 LLM），
# 因此独立版本，避免影响既有 aggregation/consolidation 缓存与产物。
AUTHOR_STYLE_PROFILE_SCHEMA_VERSION = "0.1.0"

# Phase 6：Style Planner & Prompt Compiler（确定性，无 LLM，无生成正文）。
# 独立版本，避免影响既有画像/聚合/consolidation 的缓存与产物。
WRITING_REQUEST_SCHEMA_VERSION = "0.1.0"
STYLE_PLAN_SCHEMA_VERSION = "0.1.0"
STYLE_PLANNER_VERSION = "0.1.0"
PROMPT_COMPILER_VERSION = "0.1.0"

# Phase 6.1：经验 band 阈值（TRAIN-only 分位数）schema 版本。独立版本，避免影响
# 既有画像/计划缓存；band 阈值由 TRAIN chunk 分布派生，held-out 绝不参与。
BAND_SCHEMA_VERSION = "0.1.0"

# Phase 7：style-conditioned generation（真实模型生成正文）。独立版本，避免影响
# 既有画像/计划/阈值缓存；生成响应（GeneratedPassage）与 analysis 测量严格分离。
GENERATION_SCHEMA_VERSION = "0.1.0"
GENERATION_VERSION = "0.1.0"

# Phase 8：style feedback loop + 文学评价（对生成正文再测量 → 目标 vs 实际 → 改写）。
# 独立版本，避免影响既有 analysis/generation/planning 的缓存与产物。LLM 文学评价器与
# 改写器的版本进一步与 evaluation schema 分离（同 spec §17.6 的版本分离原则）。
EVALUATION_SCHEMA_VERSION = "0.1.0"
LITERARY_EVALUATOR_VERSION = "0.1.0"
REVISION_REWRITER_VERSION = "0.1.0"

# Phase 8.1：评价决策完整性 + 改写安全性。只改文学评价 evidence contract、决策 gate、
# 内容完整性检查、no_action 语义；不改既有画像/计划/生成。文学评价 schema 独立版本
# （DimensionScore 增 assessment_status / verified_evidence_count），bump 文学评价器
# 版本使旧 Phase 8 文学 cache 失效（绝不错误复用旧语义）。其它 evaluation schema
# （ActualStyleProfile / RevisionPlan / RevisionResult）保持不变，仍用 EVALUATION_SCHEMA_VERSION。
LITERARY_EVALUATION_SCHEMA_VERSION = "0.2.0"
LITERARY_EVALUATOR_VERSION = "0.2.0"
CONTENT_INTEGRITY_VERSION = "0.1.0"
FEEDBACK_DECISION_SCHEMA_VERSION = "0.1.0"
