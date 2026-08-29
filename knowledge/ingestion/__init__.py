# knowledge/ingestion/__init__.py
"""Generic Author Onboarding / Corpus Registry 入口。

V0.1 验收：新增第三位作者**无需修改任何 Style Engine 核心分析代码**，只需
一份 author manifest + 对应语料。本包提供：

    - `validate_author`  — 校验 manifest schema + 语料存在 + 无 id 冲突。
    - `register_author`  — 把作者 manifest 写入 corpus registry（committed）。
    - `build_author`     — 确定性处理（discover→clean→chunk→QC→metadata），零 LLM。
    - `onboard_author`   — 编排上述步骤；需要 LLM 的后续步骤（特征分析/聚合/
                          画像合成）绝不自动执行，返回 REQUIRES_LLM_APPROVAL。

CLI：`python -m knowledge.ingestion.add_author path/to/author_manifest.{json,yaml}`
"""
from .onboarding import (
    STATUS_INVALID,
    STATUS_READY_FOR_NEXT_STEP,
    STATUS_REQUIRES_LLM_APPROVAL,
    build_author,
    onboard_author,
    register_author,
    validate_author,
)

__all__ = [
    "STATUS_INVALID",
    "STATUS_READY_FOR_NEXT_STEP",
    "STATUS_REQUIRES_LLM_APPROVAL",
    "validate_author",
    "register_author",
    "build_author",
    "onboard_author",
]
