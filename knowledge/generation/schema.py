# knowledge/generation/schema.py
"""Phase 7.1 GeneratedPassage 正式 schema（真实模型生成正文）。

与 analysis 的测量 schema 严格分离：生成响应 = 一段正文 + 完成状态 + token 用量 +
完整 provenance，绝不混进 analysis 的 `FeatureValue` / `NarrativeObservation`。

身份模型（Phase 7.1 §1）——把"生成条件"与"具体生成结果"分离：
    - `generation_condition_id`：作者 / 计划 / prompt / provider / model / 参数的
      确定性 hash，标识"这次生成的条件"。同一条件可产生多次随机抽样，条件 id 不变。
    - `generation_id`：具体生成结果的 identity。同一条件下两次 fresh 随机抽样若正文
      不同，其 `generation_id` 必须不同。由 `condition_id + experiment_id + output
      hash`（+ provider request id，若有）派生，**绝不依赖当前时间**。

泄露守卫（Phase 7.1 §4）——A/B 分离，作者身份名单来自 author metadata（绝不硬编码）：
    - A. 通用风格模仿指令守卫（"write like"/"imitate"/"in the style of"）只查**我们
      生成的风格控制指令**，绝不查用户 brief 正文（"imitate" 是普通英语动词，可能
      合法出现在故事情节里）。
    - B. 当前作者身份守卫：作者显示名由调用方从 author metadata 传入。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable

from ..schema.versions import GENERATION_SCHEMA_VERSION

# A. 通用"风格模仿"指令令牌——只用于检查编译器生成的风格控制指令（ROLE / STYLE
# CONTROL / NARRATIVE / CONDITIONAL STRATEGIES / IMPORTANT），绝不用于用户 brief
# （CONTENT）。"imitate" 是普通英语动词，故不可对 CONTENT 全文一刀切。
IMITATION_INSTRUCTION_TOKENS: tuple[str, ...] = (
    "write like", "imitate", "in the style of",
)


class GenerationError(Exception):
    """generation 失败（provider 未配置 / 传输失败 / 空生成 / prompt 泄露 / 缺 plumbing 等）。"""


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compiled_prompt_hash(prompt_text: str) -> str:
    """编译提示词的 sha256（`如果 prompt artifact 被修改，hash 应该不同`）。"""
    return _sha256(prompt_text)


def output_hash(text: str) -> str:
    """生成正文的 sha256（具体结果 identity 的组成，与 prompt hash 语义分离）。"""
    return _sha256(text)


def assert_no_imitation_instruction(text: str) -> None:
    """A. 风格控制指令绝不含"模仿"指令；否则拒绝发送（fail-closed）。"""
    lowered = text.lower()
    for tok in IMITATION_INSTRUCTION_TOKENS:
        if tok in lowered:
            raise GenerationError(f"风格控制指令含模仿令牌，拒绝发送: {tok!r}")


def assert_no_author_identity(text: str, author_names: Iterable[str]) -> None:
    """B. prompt 绝不含当前作者显示名（名单来自 author metadata，非硬编码）。"""
    lowered = text.lower()
    for name in author_names:
        n = str(name).strip().lower()
        if n and n in lowered:
            raise GenerationError(f"prompt 含作者身份名，拒绝发送: {name!r}")


def make_generation_condition_id(author_id: str, style_plan_id: str,
                                 prompt_hash: str, provider: str, model: str,
                                 parameters: dict[str, Any]) -> str:
    """生成条件的确定性 id：同一 (作者/计划/prompt/provider/model/参数) 恒得同一 id。"""
    blob = json.dumps([author_id, style_plan_id, prompt_hash, provider, model,
                       parameters], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def make_generation_id(condition_id: str, experiment_id: str, output_sha256: str,
                       request_id: str = "") -> str:
    """具体生成结果的 identity：条件 + 实验 + 正文 hash（+ provider request id）。

    同一条件下两次 fresh 抽样若正文不同 → output_sha256 不同 → generation_id 不同。
    绝不依赖当前时间。
    """
    blob = json.dumps([condition_id, experiment_id, output_sha256, request_id],
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
    request_id: str = ""        # provider 响应里的 request id（若有）

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "finish_reason": self.finish_reason,
            "usage": self.usage.to_dict(),
            "n_retries": self.n_retries,
            "cache_hit": self.cache_hit,
            "request_id": self.request_id,
        }


@dataclass
class GeneratedPassage:
    """正式生成产物：一段正文 + 完整 provenance（机器可读）。

    Phase 7.1 新增 `generation_condition_id`（条件身份）与 `request_id`（provider
    request id）。`generation_id` 变为具体结果身份（含 output hash）。
    """
    generation_id: str
    generation_condition_id: str
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
    request_id: str = ""

    def __post_init__(self) -> None:
        # 空生成绝不落盘（spec §十七 测试 7）。
        if not isinstance(self.generated_text, str) or not self.generated_text.strip():
            raise GenerationError("生成正文为空，拒绝保存 GeneratedPassage")

    def to_dict(self) -> dict[str, Any]:
        return {
            "generation_id": self.generation_id,
            "generation_condition_id": self.generation_condition_id,
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
            "request_id": self.request_id,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GeneratedPassage":
        return cls(
            generation_id=d["generation_id"],
            # 向后兼容：Phase 7 旧产物只有 generation_id（当时它其实就是"条件 id"）。
            # 缺 generation_condition_id 时回退到旧的 generation_id，绝不要求重生成。
            generation_condition_id=d.get("generation_condition_id", d["generation_id"]),
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
            request_id=d.get("request_id", ""),
        )
