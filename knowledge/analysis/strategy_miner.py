# knowledge/analysis/strategy_miner.py
"""Layer C 策略挖掘器（Phase 3 §5 + Phase 3–4.1 标定就绪），支持两种模式：

    Mode 1 — 已知策略匹配（match）：判断已注册策略是否出现在文本中，带证据；
    Mode 2 — 候选发现（discover）：当强证据支持时提出新的候选策略。

新发现策略绝不立即成为 Author Strategy：以 status="discovered" 进入注册表，
经多 chunk / 跨作品证据逐步晋升（见 strategies/registry.py 生命周期）。
默认盲测、无 provider 时返回 AnalysisUnavailable。

标定就绪（task item 5/6/7）：
    - 所有 evidence 经共享校验逐字比对 passage，未验证引文显式标记；
    - 保留 match confidence 与全部**有效**引文（不丢弃置信度、不只留第一条）；
    - 零验证证据的正向匹配/发现绝不构成生命周期证据（无论置信度高低），
      不静默接受编造引文（task item 6）；
    - StrategyEvidence 携带 analyzer/schema 溯源。
"""
from __future__ import annotations

import hashlib
import re

from ..providers.llm_provider import LLMProvider, cache_key
from ..schema.strategy_schema import (
    CreativeStrategy, StrategyEvidence, StrategyStatus,
)
from ..schema.versions import STRATEGY_MINER_VERSION, STRATEGY_SCHEMA_VERSION
from ..strategies.registry import StrategyRegistry
from .base import AnalysisUnavailable, LLMResponseError, parse_json_response
from .evidence import verify_evidence_quotes

ANALYZER_ID = "StrategyMiner"
ANALYZER_VERSION = STRATEGY_MINER_VERSION

# 正向判定所需的最小已验证证据数（task item 6：零验证证据绝不构成生命周期证据）
_MIN_EVIDENCE = 1


class StrategyMiner:
    def __init__(self, provider: LLMProvider, registry: StrategyRegistry,
                 blind: bool = True, rejections: list[dict] | None = None):
        self._provider = provider
        self._registry = registry
        self.blind = blind
        # 可选拒绝收集器（供冒烟/标定报表统计"因零验证证据而被拒"的输出数）。
        # 默认 None 时行为不变；传入 list 则每次拒绝追加一条 {stage, reason, ...}。
        self._rejections = rejections

    def _record_rejection(self, stage: str, reason: str, **fields) -> None:
        if self._rejections is not None:
            self._rejections.append({"stage": stage, "reason": reason, **fields})

    # ---- Mode 1：已知策略匹配 ----
    def match(self, text: str, chunk_id: str = "", work_id: str = "",
              author_id: str = "") -> list[tuple[str, StrategyEvidence]] | AnalysisUnavailable:
        if not self._provider.is_configured():
            return AnalysisUnavailable("strategy", ANALYZER_ID, ANALYZER_VERSION,
                                       "未配置 LLM provider")
        strategies = sorted(self._registry.all(), key=lambda s: s.strategy_id)
        catalog = "\n".join(f"- {s.strategy_id}: {s.description}" for s in strategies)
        system = (
            "You are a literary strategy analyst. You will be given a list of known "
            "writing strategies and a text passage. For each strategy, decide whether "
            "it is CLEARLY present in the passage, based on explicit evidence only.\n"
            "Do not assume the author's identity.\n"
            "Return ONLY a JSON object:\n"
            '{"matches": [{"strategy_id": "...", "evidence": ["verbatim quote", ...], '
            '"confidence": 0.0}]}\n'
            "Include an entry only for strategies actually present; empty list if none."
        )
        user = f"STRATEGIES:\n{catalog}\n\nPASSAGE:\n\"\"\"{text}\"\"\""
        key = cache_key(
            text=text, analyzer_id=ANALYZER_ID, analyzer_version=ANALYZER_VERSION,
            schema_version=STRATEGY_SCHEMA_VERSION, model=self._provider.model,
            provider_id=self._provider.provider_id,
            prompt_name=f"strategy_match:blind={self.blind}",
        )
        raw = self._provider.complete(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}], cache_hint=key)
        data = parse_json_response(raw)
        matches = data.get("matches", [])
        if not isinstance(matches, list):
            raise LLMResponseError("strategy_match 的 matches 必须是列表")
        out: list[tuple[str, StrategyEvidence]] = []
        for m in matches:
            if not isinstance(m, dict):
                continue
            sid = m.get("strategy_id")
            if not self._registry.has(sid):
                self._record_rejection("match", "unknown_strategy", strategy_id=sid)
                continue  # 忽略模型杜撰的未知策略
            raw_quotes = [q for q in (m.get("evidence") or []) if isinstance(q, str)]
            check = verify_evidence_quotes(raw_quotes, text)
            confidence = self._confidence(m)
            # 零验证证据的正向匹配不构成生命周期证据（task item 6），无论置信度高低
            if check.n_verified < _MIN_EVIDENCE:
                self._record_rejection("match", "zero_verified_evidence",
                                       strategy_id=sid, confidence=confidence)
                continue
            verified = [q for q in check.verified if isinstance(q, str)]
            unverified = [q for q in check.unverified if isinstance(q, str)]
            ev = StrategyEvidence(
                chunk_id=chunk_id, work_id=work_id, author_id=author_id,
                strategy_id=sid,
                quote=verified[0] if verified else "",
                quotes=verified, unverified_quotes=unverified,
                confidence=confidence,
                analyzer_id=ANALYZER_ID, analyzer_version=ANALYZER_VERSION,
                schema_version=STRATEGY_SCHEMA_VERSION,
            )
            out.append((sid, ev))
        return out

    # ---- Mode 2：候选发现 ----
    def discover(self, text: str, chunk_id: str = "", work_id: str = "",
                 author_id: str = "") -> list[CreativeStrategy] | AnalysisUnavailable:
        if not self._provider.is_configured():
            return AnalysisUnavailable("strategy", ANALYZER_ID, ANALYZER_VERSION,
                                       "未配置 LLM provider")
        system = (
            "You are a literary strategy analyst. Identify up to 2 repeatable "
            "high-level writing strategies in the passage.\n"
            "A strategy has the form TRIGGER -> OPERATION -> EFFECT: under what "
            "condition the author does WHAT writing operation to produce WHICH "
            "literary effect. It must be a repeatable craft operation, NOT a vague "
            "style adjective (do NOT return things like 'uses vivid language' or "
            "'writes complex sentences').\n"
            "Return an empty list if nothing is strongly supported.\n"
            "Do not assume the author's identity.\n"
            "Return ONLY a JSON object:\n"
            '{"strategies": [{"name": "...", "description": "...", "triggers": [...], '
            '"operations": [...], "intended_effects": [...], '
            '"evidence": ["verbatim quote"], "confidence": 0.0}]}'
        )
        user = f'PASSAGE:\n"""{text}"""'
        key = cache_key(
            text=text, analyzer_id=ANALYZER_ID, analyzer_version=ANALYZER_VERSION,
            schema_version=STRATEGY_SCHEMA_VERSION, model=self._provider.model,
            provider_id=self._provider.provider_id,
            prompt_name=f"strategy_discover:blind={self.blind}",
        )
        raw = self._provider.complete(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}], cache_hint=key)
        data = parse_json_response(raw)
        items = data.get("strategies", [])
        if not isinstance(items, list):
            raise LLMResponseError("strategy_discover 的 strategies 必须是列表")
        out: list[CreativeStrategy] = []
        for it in items:
            if not isinstance(it, dict) or not it.get("name"):
                continue
            s = self._to_strategy(it, chunk_id, work_id, author_id, text)
            if s is not None:
                out.append(s)
        return out

    def _to_strategy(self, it: dict, chunk_id: str, work_id: str,
                     author_id: str, text: str) -> CreativeStrategy | None:
        name = str(it["name"])
        sid = self._slugify(name)
        if self._registry.has(sid):
            sid = f"{sid}_{hashlib.sha256(it.get('description', '').encode()).hexdigest()[:6]}"
        evidence = it.get("evidence") or []
        if isinstance(evidence, str):
            evidence = [evidence]
        evidence = [e for e in evidence if isinstance(e, str)]
        check = verify_evidence_quotes(evidence, text)
        confidence = self._confidence(it)
        # 零验证证据的正向发现不构成生命周期证据（task item 6），无论置信度高低
        if check.n_verified < _MIN_EVIDENCE:
            self._record_rejection("discover", "zero_verified_evidence",
                                   name=name, confidence=confidence)
            return None
        verified = [e for e in check.verified if isinstance(e, str)]
        return CreativeStrategy(
            strategy_id=sid,
            name=name,
            description=it.get("description", ""),
            triggers=[str(t) for t in (it.get("triggers") or [])],
            operations=[str(o) for o in (it.get("operations") or [])],
            intended_effects=[str(e) for e in (it.get("intended_effects") or [])],
            confidence=confidence,
            evidence=[StrategyEvidence(chunk_id=chunk_id, work_id=work_id,
                                       author_id=author_id, strategy_id=sid,
                                       quote=q, quotes=[q],
                                       confidence=confidence,
                                       analyzer_id=ANALYZER_ID,
                                       analyzer_version=ANALYZER_VERSION,
                                       schema_version=STRATEGY_SCHEMA_VERSION)
                      for q in verified],
            source_author=author_id or None,
            source_work=work_id or None,
            status=StrategyStatus.DISCOVERED.value,
        )

    # ---- 共享 ----
    @staticmethod
    def _confidence(data: dict) -> float | None:
        c = data.get("confidence")
        if c is None or isinstance(c, bool) or not isinstance(c, (int, float)):
            return None
        c = float(c)
        if not 0.0 <= c <= 1.0:
            return None
        return c

    @staticmethod
    def _slugify(name: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        return slug or "strategy"
