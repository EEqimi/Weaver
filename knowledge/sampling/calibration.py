# knowledge/sampling/calibration.py
"""确定性分层采样：从 TRAIN 作品选代表性 chunk 生成标定样本清单（Phase 3 §9）。

分层维度（Phase 3 §9.4）：
    position_band ∈ {early, middle, late}   —— 作品内位置
    dialogue_band ∈ {dialogue, mixed, narration} —— 对话/叙述（描述性经叙事档近似）

确定性保证：无随机数；按 (chapter, seq) 排序后均匀间隔选取，结果可复现。
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

from ..schema.versions import SAMPLING_VERSION
from ..schema.feature_registry import build_default_registry
from ..analysis.statistical_analyzer import StatisticalAnalyzer
from ..corpus.metadata import CORPUS, TRAIN

# 对话档阈值（dialogue_ratio 为引号内字符占比）
_DIALOGUE_HIGH = 0.25
_DIALOGUE_LOW = 0.05

DIALOGUE_BANDS = ("dialogue", "mixed", "narration")
POSITION_BANDS = ("early", "middle", "late")


def dialogue_band(dialogue_ratio: float) -> str:
    if dialogue_ratio >= _DIALOGUE_HIGH:
        return "dialogue"
    if dialogue_ratio >= _DIALOGUE_LOW:
        return "mixed"
    return "narration"


def position_band(seq: int, n: int) -> str:
    """按作品内顺序分三档（n==0 时退化为 early）。"""
    if n <= 1:
        return "early"
    r = (seq - 1) / (n - 1) if n > 1 else 0.0
    if r < 1 / 3:
        return "early"
    if r < 2 / 3:
        return "middle"
    return "late"


@dataclass
class SampleChunk:
    chunk_id: str
    work_id: str
    chapter: str
    seq: int
    char_count: int
    dialogue_ratio: float
    mean_sentence_length: float
    position: str
    dialogue: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "work_id": self.work_id,
            "chapter": self.chapter,
            "seq": self.seq,
            "char_count": self.char_count,
            "dialogue_ratio": self.dialogue_ratio,
            "mean_sentence_length": self.mean_sentence_length,
            "position": self.position,
            "dialogue": self.dialogue,
        }


def enrich_chunks(chunks: list[dict], work_id: str) -> list[SampleChunk]:
    """为每个 chunk 计算采样所需统计量（对话占比 + 平均句长）并分层。"""
    analyzer = StatisticalAnalyzer()
    registry = build_default_registry()
    dial_f = registry.get("dialogue_ratio")
    ms_f = registry.get("mean_sentence_length")
    # 先按 (chapter, seq) 稳定排序，使 position 定义明确
    ordered = sorted(chunks, key=lambda c: (str(c.get("chapter", "")), int(c.get("seq", 0))))
    n = len(ordered)
    out: list[SampleChunk] = []
    for i, c in enumerate(ordered):
        text = c.get("text", "")
        dr = analyzer.analyze(text, dial_f)
        ms = analyzer.analyze(text, ms_f)
        dr_v = float(dr.value) if dr else 0.0
        ms_v = float(ms.value) if ms else 0.0
        seq = int(c.get("seq", i + 1))
        out.append(SampleChunk(
            chunk_id=c["chunk_id"], work_id=work_id,
            chapter=str(c.get("chapter", "")), seq=seq,
            char_count=int(c.get("char_count", 0)),
            dialogue_ratio=dr_v, mean_sentence_length=ms_v,
            position=position_band(i + 1, n), dialogue=dialogue_band(dr_v),
        ))
    return out


def _allocate(group_sizes: dict[str, int], target: int) -> dict[str, int]:
    """把 target 个名额分配到各层：先每层 1 个，再按层大小贪心补齐（确定性）。"""
    groups = sorted(group_sizes)
    alloc = {g: 0 for g in groups}
    remaining = target
    for g in groups:
        if remaining > 0:
            alloc[g] += 1
            remaining -= 1
    while remaining > 0:
        candidates = [g for g in groups if alloc[g] < group_sizes[g]]
        if not candidates:
            break
        g = sorted(candidates, key=lambda x: (-group_sizes[x], x))[0]
        alloc[g] += 1
        remaining -= 1
    return alloc


def _evenly_space(items: list[SampleChunk], k: int) -> list[SampleChunk]:
    """在已按 (chapter, seq) 排序的列表中均匀间隔取 k 个（确定性）。"""
    items = sorted(items, key=lambda c: (c.chapter, c.seq))
    n = len(items)
    if k <= 0:
        return []
    if k >= n:
        return list(items)
    if k == 1:
        return [items[n // 2]]  # 单名额取中位，保证位置均衡
    idxs: list[int] = []
    for i in range(k):
        idxs.append(round(i * (n - 1) / (k - 1)))
    seen: set[int] = set()
    return [items[i] for i in idxs if not (i in seen or seen.add(i))]


def select_stratified(chunks: list[SampleChunk], target: int) -> list[SampleChunk]:
    """按 position × dialogue 分层、均匀间隔的确定性选择。"""
    if target <= 0:
        return []
    strata: dict[str, list[SampleChunk]] = {}
    for c in chunks:
        strata.setdefault(f"{c.position}:{c.dialogue}", []).append(c)
    sizes = {k: len(v) for k, v in strata.items()}
    alloc = _allocate(sizes, min(target, len(chunks)))
    selected: list[SampleChunk] = []
    for k in sorted(alloc):
        selected.extend(_evenly_space(strata[k], alloc[k]))
    return sorted(selected, key=lambda c: (c.chapter, c.seq))


def _coverage(selected: list[SampleChunk]) -> dict[str, Any]:
    """采样覆盖度 QC：章节数、位置/对话档分布。"""
    pos = Counter(c.position for c in selected)
    dial = Counter(c.dialogue for c in selected)
    return {
        "n_selected": len(selected),
        "n_chapters": len({c.chapter for c in selected}),
        "position_band_counts": {b: pos.get(b, 0) for b in POSITION_BANDS},
        "dialogue_band_counts": {b: dial.get(b, 0) for b in DIALOGUE_BANDS},
        "mean_sentence_length_range": _range([c.mean_sentence_length for c in selected]),
    }


def _range(xs: list[float]) -> tuple[float, float] | None:
    if not xs:
        return None
    return (min(xs), max(xs))


def build_calibration_manifest(
    chunks_by_work: dict[str, list[dict]],
    target_per_work: int = 10,
    author_by_work: dict[str, str] | None = None,
) -> dict[str, Any]:
    """从各 TRAIN 作品的分块记录生成分层采样清单。

    chunks_by_work: work_id -> chunk 记录列表（含 chunk_id/chapter/seq/char_count/text）。
    只允许 TRAIN 作品；held-out 作品传入即报错（spec §9.3）。
    """
    train_ids = {m.work_id for m in CORPUS if m.role == TRAIN}
    author_by_work = author_by_work or {m.work_id: m.author_id for m in CORPUS}
    works: dict[str, Any] = {}
    totals = {"n_works": 0, "n_chunks": 0}
    for work_id in sorted(chunks_by_work):
        if work_id not in train_ids:
            raise ValueError(f"held-out / 未知作品禁止进入标定采样: {work_id}")
        enriched = enrich_chunks(chunks_by_work[work_id], work_id)
        if len(enriched) < 8:
            raise ValueError(f"作品 {work_id} 可用 chunk 不足 8（{len(enriched)}）")
        target = max(8, min(12, target_per_work, len(enriched)))
        selected = select_stratified(enriched, target)
        works[work_id] = {
            "work_id": work_id,
            "author_id": author_by_work.get(work_id, ""),
            "n_available": len(enriched),
            "n_selected": len(selected),
            "target": target,
            "coverage": _coverage(selected),
            "selected": [c.to_dict() for c in selected],
        }
        totals["n_works"] += 1
        totals["n_chunks"] += len(selected)

    return {
        "schema_version": SAMPLING_VERSION,
        "strategy": "position_band × dialogue_band 分层，均匀间隔，确定性",
        "target_per_work": target_per_work,
        "held_out_excluded": sorted({m.work_id for m in CORPUS if m.role != TRAIN}),
        "totals": totals,
        "works": works,
    }
