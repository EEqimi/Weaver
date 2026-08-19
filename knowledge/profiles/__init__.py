# knowledge/profiles/__init__.py
"""Profile 聚合（Phase 4）：ChunkProfile → WorkProfile → AuthorProfile。

按 feature 的 value_type 做类型感知聚合：
    continuous/discrete → mean + variance + distribution（min/max/quartiles）
    categorical         → 类别分布（proportions）
    distribution        → 逐类均值分布
    narrative（Layer B）→ 枚举类别分布 + 比例字段逐类均值
    strategies（Layer C）→ 证据数 / 生命周期状态

严禁把作者风格从单部作品/单 chunk 推断出来：所有聚合都显式保留 sample_count
与分布，不产出"只有一个数据点却看似确凿"的结论。
"""
