# knowledge/generation/__init__.py
"""Phase 7：Style-Conditioned Generation（第一次真实作者风格生成实验）。

把 Phase 6 的 CompiledPrompt（不含作者名 / 模仿指令 / 微观 stylometric 指令）交给
真实生成模型，产出 GeneratedPassage（正文 + 完成状态 + token 用量 + 完整 provenance）。

与 analysis 严格分离：analysis 测量文本，generation 产生正文。生成响应（正文 + finish
reason + usage）绝不混进 analysis schema（spec §5）。

Phase 7.1 provenance/integrity hardening：`generation_condition_id`（确定性条件身份）与
`generation_id`（具体结果身份，含 output hash）分离；泄露守卫 A/B 分离，作者身份名单
来自 author metadata（`assert_no_author_identity`），模仿指令只查风格控制指令
（`assert_no_imitation_instruction`）；`run_generation` 强制 plumbing gate（fail-closed）。

铁律：
    - 实际 prompt 绝不含当前作者显示名（名单来自 metadata；`assert_no_author_identity`）
      或模仿指令 "write like" / "imitate" / "in the style of"（`assert_no_imitation_instruction`
      只查风格控制指令，绝不查用户 brief 正文）——均 fail-closed。
    - 复用 OpenAICompatibleProvider / DeepSeekProvider 的 HTTP 传输，绝不另写第二套 client。
    - 同一 WritingRequest、同一模型、同一生成参数；唯一变量是画像导出的风格控制。
    - 不自动评价（Phase 8 才做）；不自动改写正文。
"""
from .schema import (
    GeneratedPassage, GenerationError, GenerationParameters, GenerationResult,
    GenerationUsage, assert_no_author_identity, assert_no_imitation_instruction,
    compiled_prompt_hash, make_generation_condition_id, make_generation_id,
    output_hash,
)
from .provider import DummyGenerationProvider, GenerationProvider

__all__ = [
    "GeneratedPassage", "GenerationError", "GenerationParameters",
    "GenerationResult", "GenerationUsage", "assert_no_author_identity",
    "assert_no_imitation_instruction", "compiled_prompt_hash",
    "make_generation_condition_id", "make_generation_id", "output_hash",
    "DummyGenerationProvider", "GenerationProvider",
]
