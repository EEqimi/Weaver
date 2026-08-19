# knowledge/analysis/base.py
"""分析器公共约定：显式的"不可用"结果（区别于静默跳过 / 伪造结果）。

当 LLM/NLP 未配置时，依赖这些能力的 analyzer 返回 AnalysisUnavailable，
而不是编造一个结果。调用方据此记录明确状态。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass


@dataclass
class AnalysisUnavailable:
    """某个分析因缺少依赖能力而不可用（显式状态，非错误）。"""

    kind: str                 # 被分析的对象标识（feature_id / "narrative" / "strategy" 等）
    analyzer_id: str
    analyzer_version: str
    reason: str
    status: str = "unavailable"

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "analyzer_id": self.analyzer_id,
            "analyzer_version": self.analyzer_version,
            "reason": self.reason,
            "status": self.status,
        }


class LLMNotConfiguredError(RuntimeError):
    """在未配置 provider 的情况下强制调用 LLM 时抛出。"""


class LLMResponseError(ValueError):
    """LLM 返回了无法解析/无法通过 schema 校验的响应。"""


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_json_response(text: str) -> dict:
    """从 LLM 文本中稳健地提取一个 JSON 对象（容忍 markdown 代码围栏）。

    失败抛 LLMResponseError（malformed 响应必须显式报错，而非静默吞掉）。
    """
    t = text.strip()
    m = _FENCE_RE.search(t)
    if m:
        t = m.group(1).strip()
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end <= start:
        raise LLMResponseError(f"响应中未找到 JSON 对象: {t[:120]!r}")
    try:
        obj = json.loads(t[start:end + 1])
    except json.JSONDecodeError as e:
        raise LLMResponseError(f"JSON 解析失败: {e}") from e
    if not isinstance(obj, dict):
        raise LLMResponseError("LLM 返回的 JSON 顶层必须是对象")
    return obj
