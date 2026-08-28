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
