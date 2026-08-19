# knowledge/__init__.py
"""Weaver Style Engine 知识系统（V0.1 原型）。

独立的 knowledge/ 包，与 AI_coding/agents 解耦；原型验证后再与现有 agents 集成。

分层（对应 STYLE_ENGINE_SPEC_V0.1.md §2）：
    A. Interpretable Language Style  —— analysis/
    B. Narrative Profile              —— analysis/ + schema/
    C. Creative Strategies            —— strategies/
    D. Stylometric Fingerprint        —— stylometry/
    E. Evidence / Confidence          —— schema/ + profiles/
"""
