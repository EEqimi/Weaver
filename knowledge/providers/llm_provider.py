# knowledge/providers/llm_provider.py
"""LLM Provider 抽象（spec §17.4 + Phase 3 §7）。

- 分析管线在无 LLM 配置时仍可用（返回 AnalysisUnavailable，绝不伪造结果）。
- 结果可缓存：缓存键基于 text hash + analyzer 与 schema 版本 + model/provider，
  保证重复实验不重复烧 token（spec §10.1）。
- 不绑定现有 Streamlit secrets：provider 由调用方注入。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol

from ..analysis.base import LLMNotConfiguredError


class LLMProvider(Protocol):
    """LLM 后端的最小接口。"""

    provider_id: str
    model: str

    def is_configured(self) -> bool: ...

    def complete(self, messages: list[dict], **kwargs) -> str:
        """把 messages（OpenAI 风格 [{"role","content"}]）交给模型，返回文本。

        未配置时抛 LLMNotConfiguredError。
        """
        ...


class UnconfiguredLLMProvider:
    """未配置任何后端的占位 provider：分析将显式不可用。"""

    provider_id = "unconfigured"
    model = "unconfigured"

    def is_configured(self) -> bool:
        return False

    def complete(self, messages: list[dict], **kwargs) -> str:
        raise LLMNotConfiguredError("未配置 LLM provider")


class DummyLLMProvider:
    """测试用 provider：返回调用方预设的固定文本，永不真正调用模型。"""

    def __init__(self, response: str = "", provider_id: str = "dummy",
                 model: str = "dummy-model"):
        self._response = response
        self.provider_id = provider_id
        self.model = model

    def is_configured(self) -> bool:
        return True

    def complete(self, messages: list[dict], **kwargs) -> str:
        return self._response


def cache_key(*, text: str, analyzer_id: str, analyzer_version: str,
              schema_version: str, model: str, provider_id: str,
              prompt_name: str, extra: dict | None = None) -> str:
    """基于文本与版本/模型的稳定缓存键（确定性，可复现）。

    text 不直接入键，而是其 SHA256（避免长文本做键）；其它身份字段原样入键。
    """
    payload: dict = {
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "analyzer_id": analyzer_id,
        "analyzer_version": analyzer_version,
        "schema_version": schema_version,
        "model": model,
        "provider_id": provider_id,
        "prompt_name": prompt_name,
    }
    if extra is not None:
        payload["extra"] = extra
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class LLMCache:
    """基于 JSON 文件的磁盘缓存：key → 原始文本响应。"""

    def __init__(self, cache_dir: str | Path):
        self._dir = Path(cache_dir)

    def _path(self, key: str) -> Path:
        return self._dir / f"{key}.json"

    def get(self, key: str) -> str | None:
        p = self._path(key)
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))["response"]

    def put(self, key: str, response: str) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path(key).write_text(
            json.dumps({"response": response}, ensure_ascii=False), encoding="utf-8")


class CacheBackedLLMProvider:
    """包装一个真实 provider，先查缓存再调用；命中则不再发请求。"""

    def __init__(self, inner: LLMProvider, cache: LLMCache):
        self._inner = inner
        self._cache = cache

    @property
    def provider_id(self) -> str:
        return self._inner.provider_id

    @property
    def model(self) -> str:
        return self._inner.model

    def is_configured(self) -> bool:
        return self._inner.is_configured()

    def complete(self, messages: list[dict], *, cache_hint: str | None = None,
                 **kwargs) -> str:
        if cache_hint is not None:
            cached = self._cache.get(cache_hint)
            if cached is not None:
                return cached
        result = self._inner.complete(messages, **kwargs)
        if cache_hint is not None:
            self._cache.put(cache_hint, result)
        return result
