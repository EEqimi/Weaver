# knowledge/generation/provider.py
"""Phase 7 GenerationProvider：把编译提示词交给真实模型，返回 GenerationResult。

复用 `OpenAICompatibleProvider` 的 HTTP 传输（绝不另写第二套 client），只在其上做
"生成语义"的最小扩展：`generate()` 返回 GenerationResult（content + finish_reason +
per-call usage），而 analysis 的 `complete()` 只返回文本、语义不同，故不复用。

`DummyGenerationProvider` 供测试，绝不调用模型、绝不产生 token。
"""
from __future__ import annotations

from ..providers.llm_provider import OpenAICompatibleProvider
from .schema import GenerationError, GenerationParameters, GenerationResult, GenerationUsage


class GenerationProvider:
    """把单条 prompt 文本交给 OpenAI 兼容后端，返回结构化生成结果。"""

    def __init__(self, inner: OpenAICompatibleProvider):
        self._inner = inner

    @property
    def provider_id(self) -> str:
        return self._inner.provider_id

    @property
    def model(self) -> str:
        return self._inner.model

    @property
    def base_url(self) -> str:
        return getattr(self._inner, "base_url", "")

    def is_configured(self) -> bool:
        return self._inner.is_configured()

    def generate(self, prompt_text: str, parameters: GenerationParameters) -> GenerationResult:
        if not self.is_configured():
            raise GenerationError("未配置 LLM provider（缺 DEEPSEEK_API_KEY）")
        meta = self._inner.complete_with_metadata(
            [{"role": "user", "content": prompt_text}],
            temperature=parameters.temperature,
            top_p=parameters.top_p,
            max_tokens=parameters.max_tokens,
        )
        return GenerationResult(
            content=meta["content"],
            finish_reason=meta["finish_reason"],
            usage=GenerationUsage(
                prompt_tokens=int(meta["usage"].get("prompt_tokens") or 0),
                completion_tokens=int(meta["usage"].get("completion_tokens") or 0),
                total_tokens=int(meta["usage"].get("total_tokens") or 0),
            ),
            n_retries=meta["n_retries"],
            cache_hit=False,
        )


class DummyGenerationProvider:
    """测试用：返回预设 GenerationResult，绝不调用模型 / 产生 token。"""

    def __init__(self, content: str = "A generated passage of original prose.",
                 finish_reason: str = "stop",
                 usage: GenerationUsage | None = None,
                 provider_id: str = "dummy", model: str = "dummy-model"):
        self._content = content
        self._finish_reason = finish_reason
        self._usage = usage or GenerationUsage(
            prompt_tokens=100, completion_tokens=200, total_tokens=300)
        self.provider_id = provider_id
        self.model = model
        self.base_url = "https://dummy.example"

    def is_configured(self) -> bool:
        return True

    def generate(self, prompt_text: str, parameters: GenerationParameters) -> GenerationResult:
        return GenerationResult(
            content=self._content,
            finish_reason=self._finish_reason,
            usage=self._usage,
            n_retries=0,
            cache_hit=False,
        )
