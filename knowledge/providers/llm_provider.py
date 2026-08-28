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
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Protocol

from ..analysis.base import LLMNotConfiguredError


class LLMTransportError(RuntimeError):
    """真实 provider 的网络/HTTP 错误（重试耗尽后抛出）。

    与 LLMResponseError（响应已拿到但无法通过 schema 校验）语义分离，便于调用方
    区分"传输失败"与"模型输出不合格"。
    """


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


class OpenAICompatibleProvider:
    """OpenAI 兼容 HTTP 后端（通用传输层，可接 DeepSeek / DashScope / 智谱GLM 等）。

    - 用标准库 urllib 实现，无第三方依赖（`.venv` 未装 `openai`/`requests`）；
    - 密钥从环境变量读取，绝不落盘、绝不打印（也不进入 cache key）；
    - 记录每次调用的 token 用量与重试次数，供冒烟/标定报表使用。

    子类通过类属性提供默认值；显式参数 > 环境变量 > 类默认：
        api_key    显式密钥，缺省读 `{ENV_PREFIX}_API_KEY`
        base_url   显式地址，缺省读 `{ENV_PREFIX}_BASE_URL`，再缺省用类默认
        model      显式模型，缺省读 `{ENV_PREFIX}_MODEL`，再缺省用类默认
    """

    provider_id: str = "openai-compatible"
    env_prefix: str = "OPENAI"
    default_base_url: str = "https://api.openai.com/v1"
    default_model: str = "gpt-4o-mini"

    def __init__(self, *, api_key: str | None = None, base_url: str | None = None,
                 model: str | None = None, provider_id: str | None = None,
                 timeout: float = 120.0, max_retries: int = 2,
                 temperature: float = 0.0, max_tokens: int = 2048):
        self._env_prefix = self.env_prefix.rstrip("_").upper()
        self._api_key = (api_key if api_key is not None
                         else os.environ.get(f"{self._env_prefix}_API_KEY", ""))
        self._base_url = (base_url or os.environ.get(f"{self._env_prefix}_BASE_URL")
                          or self.default_base_url).rstrip("/")
        self.model = (model or os.environ.get(f"{self._env_prefix}_MODEL")
                      or self.default_model)
        self.provider_id = provider_id or self.provider_id
        self._timeout = timeout
        self._max_retries = max_retries
        self._temperature = temperature
        self._max_tokens = max_tokens
        # 运行期计量（供冒烟报表；不参与缓存键）
        self.n_calls = 0                 # 实际发出的 HTTP 请求数（= 缓存未命中数）
        self.n_success = 0               # 成功返回文本的次数
        self.n_retries = 0               # 因瞬态错误重试的次数
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def is_configured(self) -> bool:
        return bool(self._api_key)

    @property
    def base_url(self) -> str:
        """后端 base URL（供 provenance/报表记录 endpoint，不含密钥）。"""
        return self._base_url

    def complete(self, messages: list[dict], **kwargs) -> str:
        """返回模型生成的纯文本（analysis 语义）。"""
        return self._request(messages, **kwargs)["content"]

    def complete_with_metadata(self, messages: list[dict], **kwargs) -> dict:
        """返回完整响应（generation 语义）：content + finish_reason + per-call usage。

        与 complete 共用同一 HTTP 传输（绝不另写第二套 client），只是把单次请求的
        finish_reason / token 用量一并暴露，供 Phase 7 生成记录。
        """
        return self._request(messages, **kwargs)

    def _request(self, messages: list[dict], **kwargs) -> dict:
        if not self.is_configured():
            raise LLMNotConfiguredError(
                f"未配置 LLM provider（缺 {self._env_prefix}_API_KEY）")
        self.n_calls += 1
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self._temperature),
            "max_tokens": kwargs.get("max_tokens", self._max_tokens),
        }
        top_p = kwargs.get("top_p")
        if top_p is not None:
            payload["top_p"] = top_p
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        url = f"{self._base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

        last_err: Exception | None = None
        retries_this_call = 0
        for attempt in range(self._max_retries + 1):
            if attempt > 0:
                retries_this_call += 1
                self.n_retries += 1
                time.sleep(float(attempt))  # 确定性退避：1s, 2s, ...
            try:
                req = urllib.request.Request(url, data=body, method="POST",
                                             headers=headers)
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    raw = resp.read().decode("utf-8")
                data = json.loads(raw)
                choices = data.get("choices") or []
                content = (choices[0].get("message", {}).get("content", "")
                           if choices else "")
                finish_reason = (choices[0].get("finish_reason", "")
                                 if choices else "")
                usage = data.get("usage") or {}
                self._accumulate_usage(usage)
                self.n_success += 1
                return {"content": content or "",
                        "finish_reason": finish_reason,
                        "usage": dict(usage),
                        "n_retries": retries_this_call}
            except (urllib.error.HTTPError, urllib.error.URLError,
                    TimeoutError, OSError, json.JSONDecodeError, KeyError) as e:
                last_err = e
                status = getattr(e, "code", None)
                # 4xx（除 429）为永久错误，不重试；其余瞬态错误重试
                if isinstance(status, int) and 400 <= status < 500 and status != 429:
                    break
                continue
        raise LLMTransportError(f"LLM 请求失败（{self.model} @ {self._base_url}）: "
                                f"{self._error_detail(last_err)}")

    @staticmethod
    def _error_detail(e: Exception) -> str:
        """提取错误详情（含 HTTP 响应体），供冒烟/标定报表定位真实失败原因。

        400 等客户端错误常携带服务端给出的具体原因（内容审核、参数越界等）；这里
        只读响应体（服务端错误描述），不读请求体，故不含密钥、不落盘、不打印密钥。
        """
        if isinstance(e, urllib.error.HTTPError):
            try:
                body = e.read().decode("utf-8", errors="replace").strip()
            except Exception:
                body = ""
            reason = getattr(e, "reason", "") or getattr(e, "msg", "")
            return f"HTTP {e.code} {reason}" + (f" | {body[:300]}" if body else "")
        return repr(e)

    def _accumulate_usage(self, usage: dict) -> None:
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            v = usage.get(key)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                self.usage[key] = self.usage.get(key, 0) + int(v)


class DashScopeProvider(OpenAICompatibleProvider):
    """Aliyun DashScope（百炼）compatible-mode 后端。"""

    provider_id = "dashscope"
    env_prefix = "DASHSCOPE"
    default_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    default_model = "qwen-plus"


class DeepSeekProvider(OpenAICompatibleProvider):
    """DeepSeek 后端（OpenAI 兼容；默认 `deepseek-chat`）。"""

    provider_id = "deepseek"
    env_prefix = "DEEPSEEK"
    default_base_url = "https://api.deepseek.com"
    default_model = "deepseek-chat"


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
        self.cache_hits = 0
        self.cache_misses = 0

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
                self.cache_hits += 1
                return cached
            self.cache_misses += 1
        result = self._inner.complete(messages, **kwargs)
        if cache_hint is not None:
            self._cache.put(cache_hint, result)
        return result
