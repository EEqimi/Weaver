# knowledge/evaluation/integrity.py
"""Phase 8.1 内容完整性检查器（改写后，风格重测之前）。

职责：比较 Original Passage / Revised Passage / WritingRequest，判断改写是否破坏了
用户内容（情节事实 / 角色 / 人物关系 / 明确约束 / 主要事件增删）。这是决策的**最高
优先级 gate**：若改写破坏了内容，直接 roll_back，绝不因风格改善而接受，也不必再浪费
昂贵的 Layer B/C/Literary 重测 token（spec §十一）。

实现（spec §九/§十）：
    - 先做**确定性检查**（零 token）：原改文本一致 → pass；改写为空 → fail；
    - 再做**盲测 LLM 语义层**：prompt 不含作者名、不讨论"风格像不像"，只比较内容
      保真；JSON schema 严格验证（boolean 字段必须是 bool，违规项 kind/severity 枚举）；
    - 只保存简短 `reasoning_summary`，绝不保存 hidden chain-of-thought。

铁律：
    - blind；prompt 绝不含作者名或 "write like"/"imitate"/"in the style of"
      （复用 generation/schema.py A/B 守卫，fail-closed）；
    - passed = 四个 preserved 全真 且 无新增/删除主要事件 且 无 critical 违规；
    - 密钥只读；provider 由调用方注入（测试用 DummyLLMProvider）。
"""
from __future__ import annotations

import json
from typing import Any, Iterable

from ..analysis.base import AnalysisUnavailable, LLMResponseError, parse_json_response
from ..generation.schema import (
    GenerationError, assert_no_author_identity, assert_no_imitation_instruction,
)
from ..planning.schema import WritingRequest
from ..providers.llm_provider import LLMProvider, cache_key
from ..schema.versions import CONTENT_INTEGRITY_VERSION, EVALUATION_SCHEMA_VERSION
from .schema import (
    ContentIntegrityResult, ContentIntegrityViolation, EvalError,
    INTEGRITY_CRITICAL,
)

ANALYZER_ID = "ContentIntegrityChecker"
ANALYZER_VERSION = CONTENT_INTEGRITY_VERSION

_KINDS = frozenset({
    "plot_facts", "characters", "relationships", "constraints",
    "new_event", "removed_event",
})
_SEVERITIES = frozenset({INTEGRITY_CRITICAL, "warning"})


def _build_system_prompt() -> str:
    return (
        "You are a strict content-integrity reviewer. Compare an ORIGINAL passage and "
        "its REVISED version against the user's writing request. Judge ONLY whether the "
        "revision preserved the user's content — the plot facts, named characters, "
        "relationships between characters, and every stated constraint. Ignore style, "
        "tone, and phrasing differences entirely.\n"
        "Return ONLY a JSON object (no prose, no markdown fences) with exactly these "
        "keys:\n"
        '  "plot_facts_preserved": boolean,\n'
        '  "characters_preserved": boolean,\n'
        '  "relationships_preserved": boolean,\n'
        '  "constraints_preserved": boolean,\n'
        '  "new_major_events": boolean (true = the revision ADDED a major story event),\n'
        '  "removed_major_events": boolean (true = the revision REMOVED a major story '
        "event),\n"
        '  "violations": an array of { "kind", "severity", "description" } where kind is '
        "one of plot_facts/characters/relationships/constraints/new_event/removed_event "
        "and severity is one of critical/warning,\n"
        '  "summary": one short sentence. Do NOT include step-by-step reasoning.\n'
    )


def _request_str(wr: WritingRequest) -> str:
    parts = [f"content: {wr.content}"]
    if wr.pov:
        parts.append(f"pov: {wr.pov}")
    if wr.constraints:
        parts.append("constraints:\n" + "\n".join(f"- {c}" for c in wr.constraints))
    return "\n".join(parts)


def _user_prompt(original: str, revised: str, wr: WritingRequest) -> str:
    return (
        f"ORIGINAL PASSAGE:\n\"\"\"{original}\"\"\"\n\n"
        f"REVISED PASSAGE:\n\"\"\"{revised}\"\"\"\n\n"
        f"WRITING REQUEST:\n{_request_str(wr)}"
    )


def _deterministic_result(passed: bool, *, summary: str,
                          violations: list[ContentIntegrityViolation] | None = None
                          ) -> ContentIntegrityResult:
    """确定性短路结果（零 token）。passed=False 时情节/角色/关系/约束全判未保留。"""
    violations = list(violations or [])
    return ContentIntegrityResult(
        schema_version=EVALUATION_SCHEMA_VERSION,
        checker_version=ANALYZER_VERSION,
        passed=passed,
        plot_facts_preserved=passed,
        characters_preserved=passed,
        relationships_preserved=passed,
        constraints_preserved=passed,
        new_major_events=False,
        removed_major_events=not passed,
        violations=violations,
        reasoning_summary=summary,
        deterministic=True,
        blind=True,
    )


class ContentIntegrityChecker:
    """改写后内容完整性检查（确定性短路 + 盲测 LLM 语义层）。"""

    def __init__(self, provider: LLMProvider, blind: bool = True):
        self._provider = provider
        self.blind = blind

    def check(self, original_text: str, revised_text: str,
              writing_request: WritingRequest,
              author_names: Iterable[str] = ()
              ) -> ContentIntegrityResult | AnalysisUnavailable:
        if not self._provider.is_configured():
            return AnalysisUnavailable("content_integrity", ANALYZER_ID,
                                       ANALYZER_VERSION, "未配置 LLM provider")

        # 确定性短路（零 token）：一致 → pass。
        if original_text.strip() == revised_text.strip():
            return _deterministic_result(
                True, summary="revised_text == original_text; no content changed")
        # 确定性短路（零 token）：改写为空 → 删除全部内容，fail。
        if not revised_text.strip():
            return _deterministic_result(
                False, summary="revised_text is empty",
                violations=[ContentIntegrityViolation(
                    "removed_event", INTEGRITY_CRITICAL, "revised text is empty")])

        system = _build_system_prompt()
        user = _user_prompt(original_text, revised_text, writing_request)

        # A/B 泄露守卫（fail-closed）：指令不含模仿令牌；全文不含作者身份名。
        try:
            assert_no_imitation_instruction(system)
            assert_no_author_identity(system + "\n" + user, author_names)
        except GenerationError as e:
            raise EvalError(f"content_integrity prompt 泄露守卫触发: {e}") from e

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        combined = (original_text + "\n---REVISED---\n" + revised_text
                    + "\n---REQUEST---\n" + _request_str(writing_request))
        key = cache_key(
            text=combined, analyzer_id=ANALYZER_ID, analyzer_version=ANALYZER_VERSION,
            schema_version=EVALUATION_SCHEMA_VERSION, model=self._provider.model,
            provider_id=self._provider.provider_id,
            prompt_name=f"content_integrity:blind={self.blind}",
        )
        raw = self._provider.complete(messages, cache_hint=key)
        data = parse_json_response(raw)
        return self._to_result(data)

    # ------------------------------------------------------------------ #
    def _to_result(self, data: dict) -> ContentIntegrityResult:
        def _bool(key: str) -> bool:
            v = data.get(key)
            if not isinstance(v, bool):
                raise LLMResponseError(f"content_integrity 的 {key} 必须是 boolean")
            return v

        plot = _bool("plot_facts_preserved")
        chars = _bool("characters_preserved")
        rels = _bool("relationships_preserved")
        cons = _bool("constraints_preserved")
        new_ev = _bool("new_major_events")
        rem_ev = _bool("removed_major_events")

        raw_v = data.get("violations", [])
        if not isinstance(raw_v, list):
            raise LLMResponseError("content_integrity 的 violations 必须是数组")
        violations: list[ContentIntegrityViolation] = []
        for v in raw_v:
            if not isinstance(v, dict):
                raise LLMResponseError("content_integrity 的 violations 每项必须是对象")
            kind = v.get("kind")
            severity = v.get("severity")
            desc = v.get("description")
            if (kind not in _KINDS or severity not in _SEVERITIES
                    or not isinstance(desc, str) or not desc.strip()):
                raise LLMResponseError(f"content_integrity 违规项字段非法: {v!r}")
            violations.append(ContentIntegrityViolation(
                kind=kind, severity=severity, description=desc.strip()))

        summary = data.get("summary", "")
        summary = summary.strip() if isinstance(summary, str) else ""

        passed = (plot and chars and rels and cons and not new_ev and not rem_ev
                  and not any(v.severity == INTEGRITY_CRITICAL for v in violations))
        return ContentIntegrityResult(
            schema_version=EVALUATION_SCHEMA_VERSION,
            checker_version=ANALYZER_VERSION,
            passed=passed,
            plot_facts_preserved=plot, characters_preserved=chars,
            relationships_preserved=rels, constraints_preserved=cons,
            new_major_events=new_ev, removed_major_events=rem_ev,
            violations=violations, reasoning_summary=summary,
            deterministic=False, blind=self.blind,
        )
