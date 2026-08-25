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
from dataclasses import dataclass
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

# repair 阶段专用契约版本：首轮合并（build_prompt）契约不变，但 repair 从「按
# canonical_name 匹配」升级为「按 canonical_strategy_id 匹配 + merge_existing /
# create_new 显式动作」。该版本只进入 repair 的 cache key，绝不污染首次合并缓存。
REPAIR_CONTRACT_VERSION = "2.0"


class ConsolidationError(ValueError):
    """consolidation 校验失败（作者越界 / 覆盖缺失 / 幻觉 / 重复赋值 / 缺 provider）。"""


def _normalize(text: str) -> str:
    """低风险归一：NFC + 空白折叠（用于 name/description/trigger/operation/effect）。"""
    return " ".join(unicodedata.normalize("NFC", text or "").split())


@dataclass
class RepairAssignment:
    """repair 阶段对遗漏 raw 策略的一次处理指令（显式区分 merge / create）。

    - `merge_existing`：只带 `target_canonical_id`（指向某已有 canonical 的稳定 id）。
      绝不依赖返回的 name 做匹配——name 只是给人看的标签，paraphrase 不会新建 canonical。
    - `create_new`：带完整 canonical 定义（等价于一个 ConsolidationGroup），用于构建新组。
    """
    action: str                       # "merge_existing" | "create_new"
    source_strategy_ids: list[str]
    target_canonical_id: str = ""             # merge_existing 时
    canonical_name: str = ""                   # create_new 时
    canonical_description: str = ""
    trigger_summary: str = ""
    operation_summary: str = ""
    effect_summary: str = ""
    reasoning_summary: str = ""
    confidence: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RepairAssignment":
        action = str(data.get("action", "")).strip().lower()
        if action not in ("merge_existing", "create_new"):
            raise ValueError("action 必须是 merge_existing 或 create_new")
        src = data.get("source_strategy_ids")
        if not isinstance(src, list):
            raise ValueError("source_strategy_ids 必须是列表")
        src_ids = [x for x in src if isinstance(x, str) and x.strip()]
        if not src_ids:
            raise ValueError("source_strategy_ids 必须是非空字符串列表")
        if action == "merge_existing":
            target = data.get("target_canonical_id")
            if not isinstance(target, str) or not target.strip():
                raise ValueError("merge_existing 必须提供 target_canonical_id")
            return cls(action=action, source_strategy_ids=src_ids,
                       target_canonical_id=target.strip())
        # create_new
        name = data.get("canonical_name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("create_new 必须提供 canonical_name")
        desc = data.get("canonical_description")
        if not isinstance(desc, str) or not desc.strip():
            raise ValueError("create_new 必须提供 canonical_description")
        conf = data.get("confidence")
        if isinstance(conf, bool) or not isinstance(conf, (int, float)):
            raise ValueError("confidence 必须是 [0,1] 内的数值（int/float）")
        conf = float(conf)
        if not 0.0 <= conf <= 1.0:
            raise ValueError("confidence 必须在 [0,1] 内")
        return cls(
            action=action, source_strategy_ids=src_ids,
            canonical_name=name.strip(), canonical_description=desc.strip(),
            trigger_summary=str(data.get("trigger_summary") or ""),
            operation_summary=str(data.get("operation_summary") or ""),
            effect_summary=str(data.get("effect_summary") or ""),
            reasoning_summary=str(data.get("reasoning_summary") or ""),
            confidence=conf,
        )


class StrategyConsolidator:
    """作者级策略合并器：确定性预处理 + 结构化 LLM 映射 + 校验 + canonical 构建。"""

    def __init__(self, provider: LLMProvider | None = None, blind: bool = True,
                 max_tokens: int = 8192):
        self._provider = provider
        self.blind = blind
        # consolidation 输出比单次 analyzer 长（一次产出多组 canonical 描述）；
        # provider 默认 2048 会把 JSON 截断，这里显式放宽到 deepseek-chat 输出上限。
        self.max_tokens = max_tokens

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
    def _strategy_block(self, r: RawStrategy) -> str:
        """单个 raw strategy 的紧凑定义 + 支持/证据块（build_prompt 与 repair 共用）。"""
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
        return block

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
            lines.append(self._strategy_block(r))
        user = "\n".join(lines)
        return system, user

    # ------------------------------------------------------------------ #
    # 覆盖修复：首轮遗漏的 raw 策略（merge 进已有 canonical 或新建）
    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_groups(groups_raw: list[Any]) -> list[ConsolidationGroup]:
        groups: list[ConsolidationGroup] = []
        for g in groups_raw:
            if not isinstance(g, dict):
                continue
            try:
                groups.append(ConsolidationGroup.from_dict(g))
            except ValueError as e:
                raise LLMResponseError(f"consolidation 分组字段非法: {e}") from e
        return groups

    @staticmethod
    def _missing_ids(input_ids: list[str],
                     groups: list[ConsolidationGroup]) -> list[str]:
        placed = {sid for g in groups for sid in g.source_strategy_ids}
        return [sid for sid in input_ids if sid not in placed]

    @staticmethod
    def _parse_assignments(assignments_raw: list[Any]) -> list[RepairAssignment]:
        out: list[RepairAssignment] = []
        for a in assignments_raw:
            if not isinstance(a, dict):
                continue
            try:
                out.append(RepairAssignment.from_dict(a))
            except ValueError as e:
                raise LLMResponseError(f"repair assignment 字段非法: {e}") from e
        return out

    def _existing_group_block(self, cid: str, g: ConsolidationGroup) -> str:
        """已有 canonical 的紧凑描述，暴露稳定 id（merge 目标由 id 引用，绝不按 name）。"""
        return (
            f"- canonical_strategy_id: {cid}\n"
            f"  name: {g.canonical_name}\n"
            f"  description: {g.canonical_description}\n"
            f"  trigger: {g.trigger_summary or '(none)'}\n"
            f"  operation: {g.operation_summary or '(none)'}\n"
            f"  effect: {g.effect_summary or '(none)'}"
        )

    def repair(self, existing_groups: list[ConsolidationGroup],
               raw_by_id: dict[str, RawStrategy], missing_ids: list[str],
               author_id: str) -> list[RepairAssignment]:
        """对首轮遗漏的 raw 策略做一次针对性处理（显式 merge_existing / create_new）。

        只处理 missing_ids。merge 目标用**稳定 canonical_strategy_id** 引用（绝不用
        name 匹配）；create_new 才带完整 canonical 定义。绝不改数据、绝不重复首轮已
        正确分组的 id。
        """
        missing_raws = [raw_by_id[i] for i in missing_ids]
        existing = [(canonical_strategy_id(author_id, g.canonical_name), g)
                    for g in existing_groups]
        system = (
            "You are completing a literary strategy consolidation. A first pass "
            "grouped most of a SINGLE author's raw writing strategies into "
            "canonical strategies, but a few were omitted. For each omitted raw "
            "strategy, decide whether it belongs to an EXISTING canonical strategy "
            "(same mechanism: trigger/operation/effect) or needs a NEW canonical "
            "strategy.\n"
            "Return ONLY a JSON object with an 'assignments' array:\n"
            '{"assignments": [\n'
            '  {"source_strategy_ids": ["..."], "action": "merge_existing", '
            '"target_canonical_id": "..."},\n'
            '  {"source_strategy_ids": ["..."], "action": "create_new", '
            '"canonical_name": "...", "canonical_description": "...", '
            '"trigger_summary": "...", "operation_summary": "...", '
            '"effect_summary": "...", "reasoning_summary": "...", "confidence": 0.0}\n'
            "]}\n"
            "- action is exactly 'merge_existing' or 'create_new'.\n"
            "- merge_existing: set target_canonical_id to the EXACT "
            "canonical_strategy_id of the existing canonical strategy to merge into "
            "(copy the id verbatim; do NOT paraphrase it; do NOT set canonical_name).\n"
            "- create_new: provide canonical_name, canonical_description, "
            "trigger_summary, operation_summary, effect_summary, reasoning_summary, "
            "confidence.\n"
            "Every omitted raw strategy id must appear in EXACTLY one assignment."
        )
        lines = [f"AUTHOR: {author_id}", "",
                 "EXISTING CANONICAL STRATEGIES (reference by canonical_strategy_id):"]
        for cid, g in existing:
            lines.append(self._existing_group_block(cid, g))
        lines += ["", "OMITTED RAW STRATEGIES:"]
        for r in missing_raws:
            lines.append(self._strategy_block(r))
        user = "\n".join(lines)

        key = cache_key(
            text=user, analyzer_id=ANALYZER_ID, analyzer_version=ANALYZER_VERSION,
            schema_version=CANONICAL_STRATEGY_SCHEMA_VERSION,
            model=self._provider.model, provider_id=self._provider.provider_id,
            prompt_name=f"strategy_consolidation_repair:blind={self.blind}:author={author_id}",
            extra={"max_tokens": self.max_tokens,
                   "repair_contract_version": REPAIR_CONTRACT_VERSION},
        )
        raw_text = self._provider.complete(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            cache_hint=key, max_tokens=self.max_tokens)
        data = parse_json_response(raw_text)
        assignments_raw = data.get("assignments", [])
        if not isinstance(assignments_raw, list):
            raise LLMResponseError("repair 的 assignments 必须是列表")
        return self._parse_assignments(assignments_raw)

    @staticmethod
    def _apply_repair(existing_groups: list[ConsolidationGroup],
                      assignments: list[RepairAssignment], author_id: str,
                      missing_ids: list[str]) -> list[ConsolidationGroup]:
        """按 canonical_strategy_id 应用修复（先校验再改动，失败即抛，不改已通过结果）。

        校验规则（全部确定性）：
            - merge 目标 id 必须存在；跨作者（前缀 != author_id）或幻觉 id 一律拒绝；
            - 每个遗漏 id 恰好被分配一次（重复分配 / 遗漏未覆盖 / 幻觉 raw id 都拒绝）。
        """
        by_id: dict[str, ConsolidationGroup] = {
            canonical_strategy_id(author_id, g.canonical_name): g for g in existing_groups}
        merged = list(existing_groups)
        missing_set = set(missing_ids)
        seen: Counter[str] = Counter()
        for a in assignments:
            for sid in a.source_strategy_ids:
                if sid not in missing_set:
                    raise ConsolidationError(
                        f"repair 引用了未遗漏的 raw id：`{sid}`（可能幻觉或重复处理已分组 id）")
                seen[sid] += 1
                if seen[sid] > 1:
                    raise ConsolidationError(f"repair 重复分配 raw id：`{sid}`")
            if a.action == "merge_existing":
                tid = a.target_canonical_id
                if "::" in tid and tid.split("::", 1)[0] != author_id:
                    raise ConsolidationError(
                        f"repair 跨作者 target canonical id：`{tid}` 不属于 `{author_id}`")
                target = by_id.get(tid)
                if target is None:
                    raise ConsolidationError(f"repair target canonical id 不存在：`{tid}`")
                target.source_strategy_ids.extend(a.source_strategy_ids)
            else:  # create_new
                merged.append(ConsolidationGroup(
                    canonical_name=a.canonical_name,
                    canonical_description=a.canonical_description,
                    source_strategy_ids=a.source_strategy_ids,
                    trigger_summary=a.trigger_summary,
                    operation_summary=a.operation_summary,
                    effect_summary=a.effect_summary,
                    reasoning_summary=a.reasoning_summary,
                    confidence=a.confidence,
                ))
        uncovered = sorted(missing_set - set(seen))
        if uncovered:
            raise ConsolidationError(f"repair 未覆盖的遗漏 raw id：{uncovered}")
        return merged

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
            extra={"max_tokens": self.max_tokens},
        )
        raw_text = self._provider.complete(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            cache_hint=key, max_tokens=self.max_tokens)
        data = parse_json_response(raw_text)
        groups_raw = data.get("groups", [])
        if not isinstance(groups_raw, list):
            raise LLMResponseError("consolidation 的 groups 必须是列表")
        groups = self._parse_groups(groups_raw)
        prepared_ids = [p.strategy_id for p in prepared]
        raw_by_id = {p.strategy_id: p for p in prepared}
        try:
            self.validate_mapping(prepared_ids, groups)
        except ConsolidationError:
            missing = self._missing_ids(prepared_ids, groups)
            if not missing:
                # 幻觉 id / 重复赋值等非"覆盖缺失"问题：修复无意义，直接上抛
                raise
            assignments = self.repair(groups, raw_by_id, missing, author_id)
            groups = self._apply_repair(groups, assignments, author_id, missing)
            self.validate_mapping(prepared_ids, groups)  # 修复后必须完整
        return self.build_canonicals(raw_by_id, groups, author_id)
