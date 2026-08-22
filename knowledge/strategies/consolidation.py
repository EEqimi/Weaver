# knowledge/strategies/consolidation.py
"""Phase 4.5：作者级策略合并（author-scoped Strategy Consolidation）。

把某位作者在 40-chunk 标定中发现的 raw strategies 合并为 canonical strategies。
关键约束（spec Phase 4.5）：
    - 严格 author-scoped：一次只处理一位作者；混入其他作者或缺失 author 一律拒绝。
    - 绝不删除/覆盖原始 raw strategies：LLM 只返回 structured 映射
      （canonical group → source_strategy_ids），不直接改数据。
    - LLM 输出必须是结构化映射；确定性校验保证每个输入 raw id 恰好出现一次、
      无幻觉 id、无重复赋值、无丢失。
    - 不存 chain-of-thought：只存简短 reasoning_summary。
    - 确定性预处理只做低风险归一（Unicode / name / whitespace + 精确结构重复折叠），
      绝不拿任意相似度阈值做文学语义合并（最终语义合并归 LLM）。
"""
from __future__ import annotations

import unicodedata
from collections import Counter
from typing import Any

from ..analysis.base import LLMNotConfiguredError, LLMResponseError, parse_json_response
from ..providers.llm_provider import LLMProvider, cache_key
from ..schema.strategy_schema import (
    CanonicalStrategy, ConsolidationGroup, RawStrategy, StrategyStatus,
    canonical_strategy_id,
)
from ..schema.versions import (
    CANONICAL_STRATEGY_SCHEMA_VERSION, STRATEGY_CONSOLIDATOR_VERSION,
)

ANALYZER_ID = "StrategyConsolidator"
ANALYZER_VERSION = STRATEGY_CONSOLIDATOR_VERSION


class ConsolidationError(ValueError):
    """consolidation 校验失败（作者越界 / 覆盖缺失 / 幻觉 / 重复赋值 / 缺 provider）。"""


def _normalize(text: str) -> str:
    """低风险归一：NFC + 空白折叠（用于 name/description/trigger/operation/effect）。"""
    return " ".join(unicodedata.normalize("NFC", text or "").split())


class StrategyConsolidator:
    """作者级策略合并器：确定性预处理 + 结构化 LLM 映射 + 校验 + canonical 构建。"""

    def __init__(self, provider: LLMProvider | None = None, blind: bool = True):
        self._provider = provider
        self.blind = blind

    # ------------------------------------------------------------------ #
    # 确定性预处理（只做低风险归一，不替代 LLM 语义合并）
    # ------------------------------------------------------------------ #
    @staticmethod
    def _content_key(raw: RawStrategy) -> tuple:
        return (
            _normalize(raw.name),
            _normalize(raw.description),
            tuple(_normalize(t) for t in raw.triggers),
            tuple(_normalize(o) for o in raw.operations),
            tuple(_normalize(e) for e in raw.intended_effects),
        )

    def prepare(self, raw_strategies: list[RawStrategy]) -> list[RawStrategy]:
        """归一化 + 精确结构重复折叠（内容完全一致才折叠，绝不语义近似合并）。

        折叠后幸存者的 source_strategy_ids 会并入被折叠者的原始 id，证据合并；
        被折叠者从输入中移除（其 id 不再出现在 LLM 输入，而是经 source_strategy_ids
        追溯），因此不会丢失任何原始 raw strategy 的可追溯性。
        """
        out: list[RawStrategy] = []
        by_key: dict[tuple, RawStrategy] = {}
        for r in raw_strategies:
            norm = RawStrategy(
                strategy_id=r.strategy_id, author_id=r.author_id,
                name=_normalize(r.name), description=_normalize(r.description),
                triggers=[_normalize(t) for t in r.triggers],
                operations=[_normalize(o) for o in r.operations],
                intended_effects=[_normalize(e) for e in r.intended_effects],
                status=r.status, confidence=r.confidence,
                evidence=list(r.evidence), source_work=r.source_work,
                source_strategy_ids=list(r.source_strategy_ids),
            )
            key = self._content_key(norm)
            if key in by_key:
                survivor = by_key[key]
                survivor.source_strategy_ids.extend(norm.source_strategy_ids)
                survivor.evidence.extend(norm.evidence)
            else:
                by_key[key] = norm
                out.append(norm)
        return out

    # ------------------------------------------------------------------ #
    # 作者越界校验
    # ------------------------------------------------------------------ #
    @staticmethod
    def validate_author_scope(raw_strategies: list[RawStrategy], author_id: str) -> None:
        if not author_id:
            raise ConsolidationError("author_id 为空，拒绝 consolidation")
        for r in raw_strategies:
            if not r.author_id:
                raise ConsolidationError(
                    f"raw strategy `{r.strategy_id}` 缺 author_id，拒绝 consolidation")
            if r.author_id != author_id:
                raise ConsolidationError(
                    f"author scope 越界：`{r.strategy_id}` 属于 `{r.author_id}`，"
                    f"但本次 consolidation 为 `{author_id}`")

    # ------------------------------------------------------------------ #
    # 结构化映射校验（spec §七：恰好一次、无幻觉、无重复、无丢失）
    # ------------------------------------------------------------------ #
    @staticmethod
    def validate_mapping(input_ids: list[str], groups: list[ConsolidationGroup]) -> None:
        expected = set(input_ids)
        if len(expected) != len(input_ids):
            raise ConsolidationError("输入 raw strategy id 重复")
        seen: Counter[str] = Counter()
        for g in groups:
            if not g.source_strategy_ids:
                raise ConsolidationError(f"group `{g.canonical_name}` 缺 source_strategy_ids")
            for sid in g.source_strategy_ids:
                seen[sid] += 1
                if sid not in expected:
                    raise ConsolidationError(f"幻觉 source id：`{sid}` 不在输入中")
        for sid, n in seen.items():
            if n > 1:
                raise ConsolidationError(f"source id `{sid}` 被重复赋值 {n} 次")
        missing = sorted(expected - set(seen))
        if missing:
            raise ConsolidationError(f"输入 raw strategy 未出现在输出：{missing}")

    # ------------------------------------------------------------------ #
    # canonical 构建（纯函数，完全可追溯）
    # ------------------------------------------------------------------ #
    @staticmethod
    def build_canonicals(raw_by_id: dict[str, RawStrategy],
                         groups: list[ConsolidationGroup],
                         author_id: str) -> list[CanonicalStrategy]:
        canonicals: list[CanonicalStrategy] = []
        for g in groups:
            source_ids: list[str] = []
            evidence: list = []
            for node_id in g.source_strategy_ids:
                node = raw_by_id[node_id]
                source_ids.extend(node.source_strategy_ids)
                evidence.extend(node.evidence)
            chunk_ids = sorted({e.chunk_id for e in evidence if e.chunk_id})
            work_ids = sorted({e.work_id for e in evidence if e.work_id})
            support_status = StrategyConsolidator._support_status(len(work_ids), len(chunk_ids))
            canonicals.append(CanonicalStrategy(
                canonical_strategy_id=canonical_strategy_id(author_id, g.canonical_name),
                author_id=author_id,
                canonical_name=g.canonical_name,
                canonical_description=g.canonical_description,
                trigger_summary=g.trigger_summary,
                operation_summary=g.operation_summary,
                effect_summary=g.effect_summary,
                source_strategy_ids=source_ids,
                supporting_chunk_ids=chunk_ids,
                supporting_work_ids=work_ids,
                reasoning_summary=g.reasoning_summary,
                confidence=g.confidence,
                number_of_raw_observations=len(source_ids),
                number_of_distinct_chunks=len(chunk_ids),
                number_of_distinct_works=len(work_ids),
                support_status=support_status,
                evidence=evidence,
            ))
        return canonicals

    @staticmethod
    def _support_status(n_works: int, n_chunks: int) -> str:
        """作者范围内的支持层级（长期语义 = 该策略在作者内部获得多少支持）。"""
        if n_works >= 2:
            return StrategyStatus.VALIDATED.value
        if n_chunks >= 2:
            return StrategyStatus.CANDIDATE.value
        return StrategyStatus.DISCOVERED.value

    @staticmethod
    def _verified_quotes(raw: RawStrategy, max_quotes: int = 2,
                         max_chars: int = 80) -> list[str]:
        """抽取至多 max_quotes 条**已验证**短引文（绝不发送全部证据，控制 token）。

        优先取 `quotes`（已验证引文全集），退化为 `quote`（首条已验证）；每条截断到
        max_chars 并压缩空白，去重。未验证引文（unverified_quotes）绝不进入提示词。
        """
        out: list[str] = []
        for e in raw.evidence:
            for q in (e.quotes or []) or ([e.quote] if e.quote else []):
                t = " ".join(str(q).split())
                if not t or t in out:
                    continue
                if len(t) > max_chars:
                    t = t[: max_chars - 1].rstrip() + "…"
                out.append(t)
                if len(out) >= max_quotes:
                    return out
        return out

    # ------------------------------------------------------------------ #
    # 提示词
    # ------------------------------------------------------------------ #
    def build_prompt(self, raw_strategies: list[RawStrategy],
                     author_id: str) -> tuple[str, str]:
        system = (
            "You are a literary style analyst consolidating writing strategies "
            "discovered across chunks of a SINGLE author's work. Group the raw "
            "strategies below into canonical strategies, each representing one "
            "repeatable literary mechanism.\n"
            "Merge two or more raw strategies only when they share the same "
            "underlying mechanism: same trigger conditions, same writing operation, "
            "and same intended effect. Do NOT merge on name similarity alone; do NOT "
            "merge when trigger/operation/effect differ even if the names look alike. "
            "Conversely, merge even when the names differ if the mechanism is identical.\n"
            "Every input raw strategy must appear in EXACTLY one group — do not drop "
            "any, and give a single-source group to a strategy that matches nothing "
            "else.\n"
            "Do not assume the author's identity beyond the given author id.\n"
            "Return ONLY a JSON object:\n"
            '{"groups": [{"canonical_name": "...", "canonical_description": "...", '
            '"source_strategy_ids": ["...", "..."], "trigger_summary": "...", '
            '"operation_summary": "...", "effect_summary": "...", '
            '"reasoning_summary": "...", "confidence": 0.0}]}\n'
            "reasoning_summary is a concise explanation of why the grouped raw "
            "strategies are the same literary mechanism (do not produce hidden "
            "chain-of-thought)."
        )
        lines = [f"AUTHOR: {author_id}", "", "RAW STRATEGIES:"]
        for r in raw_strategies:
            n_chunks = len({e.chunk_id for e in r.evidence if e.chunk_id})
            n_works = len({e.work_id for e in r.evidence if e.work_id})
            status = self._support_status(n_works, n_chunks)
            conf = f" confidence={r.confidence}" if r.confidence is not None else ""
            quotes = self._verified_quotes(r)
            block = (
                f"- [{r.strategy_id}] {r.name}\n"
                f"  description: {r.description}\n"
                f"  trigger: {', '.join(r.triggers) or '(none)'}\n"
                f"  operation: {', '.join(r.operations) or '(none)'}\n"
                f"  effect: {', '.join(r.intended_effects) or '(none)'}\n"
                f"  support: chunks={n_chunks} works={n_works} status={status}{conf}"
            )
            if quotes:
                block += "\n  evidence: " + " | ".join(f'"{q}"' for q in quotes)
            lines.append(block)
        user = "\n".join(lines)
        return system, user

    # ------------------------------------------------------------------ #
    # 全流程（真正调用 LLM 之前，请先 review 输入产物与估算）
    # ------------------------------------------------------------------ #
    def consolidate(self, raw_strategies: list[RawStrategy],
                    author_id: str) -> list[CanonicalStrategy]:
        if self._provider is None:
            raise ConsolidationError("未提供 provider，无法执行 LLM consolidation")
        if not self._provider.is_configured():
            raise LLMNotConfiguredError("未配置 LLM provider，无法执行 consolidation")
        self.validate_author_scope(raw_strategies, author_id)
        prepared = self.prepare(raw_strategies)
        system, user = self.build_prompt(prepared, author_id)
        key = cache_key(
            text=user, analyzer_id=ANALYZER_ID, analyzer_version=ANALYZER_VERSION,
            schema_version=CANONICAL_STRATEGY_SCHEMA_VERSION,
            model=self._provider.model, provider_id=self._provider.provider_id,
            prompt_name=f"strategy_consolidation:blind={self.blind}:author={author_id}",
        )
        raw_text = self._provider.complete(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}], cache_hint=key)
        data = parse_json_response(raw_text)
        groups_raw = data.get("groups", [])
        if not isinstance(groups_raw, list):
            raise LLMResponseError("consolidation 的 groups 必须是列表")
        groups: list[ConsolidationGroup] = []
        for g in groups_raw:
            if not isinstance(g, dict):
                continue
            try:
                groups.append(ConsolidationGroup.from_dict(g))
            except ValueError as e:
                raise LLMResponseError(f"consolidation 分组字段非法: {e}") from e
        self.validate_mapping([p.strategy_id for p in prepared], groups)
        raw_by_id = {p.strategy_id: p for p in prepared}
        return self.build_canonicals(raw_by_id, groups, author_id)
