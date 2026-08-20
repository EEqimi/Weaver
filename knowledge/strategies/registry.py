# knowledge/strategies/registry.py
"""Creative Strategy 注册表与验证生命周期（spec §6 / Phase 3 §5）。

生命周期（单调，只升不降）：
    discovered → candidate → validated
    - >= 2 个不同 chunk 证据        → candidate
    - >= 2 个不同 work（同一作者）  → validated

新发现的策略绝不立即成为 Author Strategy；已注册的"文献候选"策略也不会因
单一证据而降级。
"""
from __future__ import annotations

from ..schema.strategy_schema import CreativeStrategy, StrategyEvidence, StrategyStatus

_STATUS_RANK = {
    StrategyStatus.DISCOVERED.value: 0,
    StrategyStatus.CANDIDATE.value: 1,
    StrategyStatus.VALIDATED.value: 2,
}


class StrategyRegistry:
    def __init__(self) -> None:
        self._strategies: dict[str, CreativeStrategy] = {}

    def register(self, strategy: CreativeStrategy) -> None:
        if strategy.strategy_id in self._strategies:
            raise ValueError(f"重复的 strategy id: {strategy.strategy_id}")
        self._strategies[strategy.strategy_id] = strategy

    def get(self, strategy_id: str) -> CreativeStrategy:
        return self._strategies[strategy_id]

    def has(self, strategy_id: str) -> bool:
        return strategy_id in self._strategies

    def all(self) -> list[CreativeStrategy]:
        return list(self._strategies.values())

    def by_status(self, status: str) -> list[CreativeStrategy]:
        return [s for s in self._strategies.values() if s.status == status]

    def __len__(self) -> int:
        return len(self._strategies)

    def __iter__(self):
        return iter(self._strategies.values())

    # ---- 生命周期 ----
    def record_evidence(self, strategy_id: str, evidence: StrategyEvidence) -> str:
        """追加一条证据并单调推进状态，返回新状态。"""
        s = self._strategies[strategy_id]
        s.evidence.append(evidence)
        s.status = max(s.status, self._evidence_status(s), key=_STATUS_RANK.get)
        return s.status

    @staticmethod
    def _evidence_status(s: CreativeStrategy) -> str:
        # 严格作者一致性（task item 5）：仅统计 author_id 与 work_id 均非空的证据；
        # 缺失 author/work 元数据的证据绝不参与作者级验证。
        counted = [e for e in s.evidence if e.author_id and e.work_id]
        works = {e.work_id for e in counted}
        authors = {e.author_id for e in counted}
        chunks = {e.chunk_id for e in s.evidence if e.chunk_id}
        # VALIDATED：>= 2 个不同 work，且所有可计证据来自**同一位**作者
        if len(works) >= 2 and len(authors) == 1:
            return StrategyStatus.VALIDATED.value
        if len(chunks) >= 2:
            return StrategyStatus.CANDIDATE.value
        return StrategyStatus.DISCOVERED.value


def seed_default_registry() -> StrategyRegistry:
    """预置少量文献中公认的高阶写作策略（均为候选，待语料证据验证）。

    这些不是"风格形容词"，而是 TRIGGER → OPERATION → EFFECT 的可重复操作。
    """
    reg = StrategyRegistry()
    seeds = [
        CreativeStrategy(
            strategy_id="free_indirect_discourse",
            name="自由间接引语",
            description="第三人称叙述中混入人物内心声音，模糊叙述者与人物边界。",
            triggers=["character_internal_state_focus", "third_person_narration"],
            operations=["merge_narrator_and_character_voice",
                        "report_thought_without_quotation"],
            intended_effects=["interiority", "narrator_character_blend", "ironic_distance"],
            constraints=["keep_grammatical_third_person"],
            source_author=None, source_work=None, status=StrategyStatus.CANDIDATE.value,
        ),
        CreativeStrategy(
            strategy_id="dramatic_irony",
            name="戏剧性反讽",
            description="让读者先于人物知晓信息，制造悬置与悲剧预感。",
            triggers=["reader_knowledge_exceeds_character"],
            operations=["reveal_information_to_reader_only", "withhold_information_from_character"],
            intended_effects=["suspense", "tragic_anticipation", "empathy"],
            source_author=None, source_work=None, status=StrategyStatus.CANDIDATE.value,
        ),
        CreativeStrategy(
            strategy_id="narrative_irony",
            name="叙述反讽",
            description="在表面表述与真实评价之间制造张力（褒义措辞承载贬义评价）。",
            triggers=["gap_between_stated_and_meant"],
            operations=["praise_with_negative_undertone", "juxtapose_ideal_and_reality"],
            intended_effects=["satire", "humor", "critical_distance"],
            constraints=["use_sparingly"],
            source_author=None, source_work=None, status=StrategyStatus.CANDIDATE.value,
        ),
        CreativeStrategy(
            strategy_id="character_revelation_through_dialogue",
            name="以对话显人物",
            description="通过人物言说方式与内容（而非叙述者直接陈述）揭示性格与关系。",
            triggers=["character_introduction", "relationship_definition"],
            operations=["differentiate_idiolect", "reveal_character_through_speech"],
            intended_effects=["speech_individuality", "implicit_characterization"],
            source_author=None, source_work=None, status=StrategyStatus.CANDIDATE.value,
        ),
    ]
    for s in seeds:
        reg.register(s)
    return reg
