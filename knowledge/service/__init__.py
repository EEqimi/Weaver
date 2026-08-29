# knowledge/service/__init__.py
"""Writer 服务层（UI / CLI / Web API 共享的单一业务入口）。

只做编排：把 UI 输入转成 WritingRequest，再复用既有
StylePlanner → PromptCompiler → Generation → Evaluation → Revision → Feedback Loop，
绝不重实现任何核心分析/生成逻辑。当前提供一个最小 stdlib Web UI（`webapp`）。
"""
from __future__ import annotations

from .writer import (
    WRITER_EXPERIMENT_ID,
    WriterError,
    build_request,
    generate,
    list_authors,
)

__all__ = [
    "WRITER_EXPERIMENT_ID",
    "WriterError",
    "build_request",
    "generate",
    "list_authors",
]
