# knowledge/generation/schema.py
"""Phase 7 GeneratedPassage 正式 schema（真实模型生成正文）。

与 analysis 的 `FeatureValue` / `NarrativeObservation` / `CreativeStrategy` **严格分离**：
生成响应是"一段正文 + 完成状态 + token 用量"，不是"对文本的测量"。绝不把 generation
response 混进 analysis schema（spec §五）。

铁律（spec §八）：真实发给模型的主 prompt 绝不含作者名 / "write like" / "imitate" /
"in the style of"。作者 ID 只存在于程序 metadata，不进 generation instruction。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from ..schema.versions import GENERATION_SCHEMA_VERSION

# 主 prompt 中禁止出现的泄露令牌（Phase 6 原则延续到真实生成）。
BANNED_AUTHOR_LEAK_TOKENS: tuple[str, ...] = (
    "Jane Austen", "Charles Dickens",
    "write like", "imitate", "in the style of",
)


class GenerationError(Exception):
    """generation 失败（provider 未配置 / 传输失败 / 空生成 / prompt 泄露等）。"""


def compiled_prompt_hash(prompt_text: str) -> str:
    """编译提示词的 sha256（`如果 prompt artifact 被修改，hash 应该不同`）。"""
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()


def assert_no_author_leakage(prompt_text: str) -> None:
    """主 prompt 绝不含作者名 / 模仿指令；否则拒绝发送（fail-closed）。"""
    lowered = prompt_text.lower()
    for tok in BANNED_AUTHOR_LEAK_TOKENS:
        if tok.lower() in lowered:
            raise GenerationError(f"prompt 含作者名/模仿指令，拒绝发送: {tok!r}")


def make_generation_id(author_id: str, style_plan_id: str, prompt_hash: str,
                       parameters: dict[str, Any]) -> str:
    """确定性 generation_id：同一 (author, plan, prompt, 参数) 恒得同一 id。"""
    blob = json.dumps([author_id, style_plan_id, prompt_hash, parameters],
                      sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


@dataclass
class GenerationUsage:
    """单次生成的 token 用量（provider 返回的 usage 直录）。"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GenerationUsage":
        def _i(v: Any) -> int:
            return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0
        return cls(prompt_tokens=_i(d.get("prompt_tokens")),
                   completion_tokens=_i(d.get("completion_tokens")),
                   total_tokens=_i(d.get("total_tokens")))


@dataclass
class GenerationParameters:
    """本次生成实验的采样参数（两位作者严格一致，唯一变量是画像导出的风格控制）。"""
    temperature: float
    top_p: float
    max_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return {"temperature": self.temperature, "top_p": self.top_p,
                "max_tokens": self.max_tokens}


@dataclass
class GenerationResult:
    """provider 单次生成调用返回的原始结果（尚未绑定 author/plan，语义中性）。"""
    content: str
    finish_reason: str
    usage: GenerationUsage
    n_retries: int = 0
    cache_hit: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "finish_reason": self.finish_reason,
            "usage": self.usage.to_dict(),
            "n_retries": self.n_retries,
            "cache_hit": self.cache_hit,
        }


@dataclass
class GeneratedPassage:
    """正式生成产物：一段正文 + 完整 provenance（机器可读）。"""
    generation_id: str
    schema_version: str
    author_id: str
    style_plan_id: str
    source_profile_hash: str
    writing_request: dict[str, Any]
    provider: str
    model: str
    generation_parameters: dict[str, Any]
    compiled_prompt_hash: str
    compiled_prompt: str
    generated_text: str
    finish_reason: str
    usage: GenerationUsage
    generation_version: str
    cache_hit: bool
    n_retries: int
    provenance: dict[str, Any] = field(default_factory=dict)
    experiment_id: str = ""
    fresh_request: bool = True

    def __post_init__(self) -> None:
        # 空生成绝不落盘（spec §十七 测试 7）。
        if not isinstance(self.generated_text, str) or not self.generated_text.strip():
            raise GenerationError("生成正文为空，拒绝保存 GeneratedPassage")

    def to_dict(self) -> dict[str, Any]:
        return {
            "generation_id": self.generation_id,
            "schema_version": self.schema_version,
            "author_id": self.author_id,
            "style_plan_id": self.style_plan_id,
            "source_profile_hash": self.source_profile_hash,
            "writing_request": self.writing_request,
            "provider": self.provider,
            "model": self.model,
            "generation_parameters": self.generation_parameters,
            "compiled_prompt_hash": self.compiled_prompt_hash,
            "compiled_prompt": self.compiled_prompt,
            "generated_text": self.generated_text,
            "finish_reason": self.finish_reason,
            "usage": self.usage.to_dict(),
            "prompt_tokens": self.usage.prompt_tokens,
            "completion_tokens": self.usage.completion_tokens,
            "total_tokens": self.usage.total_tokens,
            "generation_version": self.generation_version,
            "cache_hit": self.cache_hit,
            "n_retries": self.n_retries,
            "experiment_id": self.experiment_id,
            "fresh_request": self.fresh_request,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GeneratedPassage":
        return cls(
            generation_id=d["generation_id"],
            schema_version=d["schema_version"],
            author_id=d["author_id"],
            style_plan_id=d["style_plan_id"],
            source_profile_hash=d["source_profile_hash"],
            writing_request=d["writing_request"],
            provider=d["provider"],
            model=d["model"],
            generation_parameters=d["generation_parameters"],
            compiled_prompt_hash=d["compiled_prompt_hash"],
            compiled_prompt=d["compiled_prompt"],
            generated_text=d["generated_text"],
            finish_reason=d["finish_reason"],
            usage=GenerationUsage.from_dict(d["usage"]),
            generation_version=d["generation_version"],
            cache_hit=d["cache_hit"],
            n_retries=d["n_retries"],
            provenance=d.get("provenance", {}),
            experiment_id=d.get("experiment_id", ""),
            fresh_request=d.get("fresh_request", True),
        )
