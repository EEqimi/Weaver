# knowledge/sampling/__init__.py
"""受控标定采样（Phase 3 §9）：从 TRAIN 作品选出代表性 chunk，供 LLM 标定。

关键约束：
    - 只从 TRAIN 作品采样（P&P / Emma / GE / DC），绝不用 held-out（Persuasion / TOTC）；
    - 每部作品 8–12 个 chunk，2000 字符档优先；
    - 分层：position（早/中/晚）× dialogue（对话/混合/叙述）；
    - 确定性：无随机数，按稳定键排序 + 均匀间隔选取，可复现；
    - 只产出 sample manifest，绝不自动触发 LLM 调用。
"""
