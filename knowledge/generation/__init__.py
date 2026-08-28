# knowledge/generation/__init__.py
"""Phase 7：Style-Conditioned Generation（第一次真实作者风格生成实验）。

把 Phase 6 的 CompiledPrompt（不含作者名 / 模仿指令 / 微观 stylometric 指令）交给
真实生成模型，产出 GeneratedPassage（正文 + 完成状态 + token 用量 + 完整 provenance）。

与 analysis 严格分离：analysis 测量文本，generation 产生正文。生成响应（正文 + finish
reason + usage）绝不混进 analysis schema（spec §5）。

铁律：
    - 实际 prompt 绝不含 "Jane Austen" / "Charles Dickens" / "write like" / "imitate" /
      "in the style of"（作者 ID 只在 metadata；`assert_no_author_leakage` fail-closed）。
    - 复用 OpenAICompatibleProvider / DeepSeekProvider 的 HTTP 传输，绝不另写第二套 client。
    - 同一 WritingRequest、同一模型、同一生成参数；唯一变量是画像导出的风格控制。
    - 不自动评价（Phase 8 才做）；不自动改写正文。
"""
from .schema import (
    BANNED_AUTHOR_LEAK_TOKENS, GeneratedPassage, GenerationError,
    GenerationParameters, GenerationResult, GenerationUsage,
    assert_no_author_leakage, compiled_prompt_hash, make_generation_id,
)
from .provider import DummyGenerationProvider, GenerationProvider

__all__ = [
    "BANNED_AUTHOR_LEAK_TOKENS", "GeneratedPassage", "GenerationError",
    "GenerationParameters", "GenerationResult", "GenerationUsage",
    "assert_no_author_leakage", "compiled_prompt_hash", "make_generation_id",
    "DummyGenerationProvider", "GenerationProvider",
]
