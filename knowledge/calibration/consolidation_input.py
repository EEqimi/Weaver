# knowledge/calibration/consolidation_input.py
"""Phase 4.5：生成作者级 consolidation 输入产物（不调用 LLM）。

从 Phase 4.4 的 strategy_registry.json 读回 raw strategies，按作者分区，写出每位
作者的 consolidation_input.json（含准备发送给 LLM 的 prompt 与请求/token 估算），
并写汇总。**复用 Phase 4.4 结果，绝不重新调用 40-chunk analyzers。**

调用 `build_consolidation_inputs()` 后，请在真正执行付费 LLM consolidation 之前
先 review 这些产物与估算。
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import data_root as default_data_root
from ..providers.llm_provider import DeepSeekProvider
from ..schema.strategy_schema import RawStrategy, StrategyEvidence, StrategyStatus
from ..schema.versions import (
    CANONICAL_STRATEGY_SCHEMA_VERSION, STRATEGY_CONSOLIDATOR_VERSION,
)
from ..strategies.consolidation import StrategyConsolidator
from .smoke import calibration_layout

CONSOLIDATION_DIRNAME = "consolidation"


def consolidation_layout(data_root_: Path | None = None) -> dict[str, Path]:
    base = Path(data_root_) if data_root_ is not None else default_data_root()
    return {"root": base / "analysis" / CONSOLIDATION_DIRNAME}


def _estimate_tokens(text: str) -> int:
    """粗略 token 估算（英文约 4 字符/token；无第三方 tokenizer，只作量级参考）。"""
    return max(1, len(text) // 4)


def _author_scoped_status(evidence: list[StrategyEvidence]) -> str:
    """按某作者自身的证据重算支持层级（区别于全局注册表的 legacy status）。

    全局注册表的生命周期是单调、跨作者的：一个策略先在 Dickens 内 validated，之后
    又混入 Austen 证据，其 status 仍残留 validated（单调不降）。这里重算作者范围内
    的真实支持，避免把它误当作该作者的最终画像。
    """
    works = {e.work_id for e in evidence if e.work_id}
    chunks = {e.chunk_id for e in evidence if e.chunk_id}
    if len(works) >= 2:
        return StrategyStatus.VALIDATED.value
    if len(chunks) >= 2:
        return StrategyStatus.CANDIDATE.value
    return StrategyStatus.DISCOVERED.value


def _partition_raw_strategies(registry_state: dict[str, Any]) -> dict[str, list[RawStrategy]]:
    """把注册表快照按作者分区，生成作者范围内的 raw strategies。

    一个 strategy（含 seed）只要在某作者名下有证据，就为该作者贡献一个 RawStrategy；
    其证据只取该作者的（严格 author scope）。seed 跨作者时分别进入两位作者的清单，
    各自独立 consolidation，互不干扰。
    """
    by_author: dict[str, list[RawStrategy]] = {}
    for s in registry_state["strategies"]:
        ev_by_author: dict[str, list[dict]] = {}
        for e in s.get("evidence", []):
            aid = e.get("author_id")
            if aid:
                ev_by_author.setdefault(aid, []).append(e)
        for aid, author_evs in ev_by_author.items():
            evidence = [
                StrategyEvidence(
                    chunk_id=e.get("chunk_id", ""),
                    work_id=e.get("work_id", ""),
                    author_id=e.get("author_id", ""),
                    strategy_id=e.get("strategy_id", s["strategy_id"]),
                    quote=e.get("quote", ""),
                    quotes=e.get("quotes", []),
                    unverified_quotes=e.get("unverified_quotes", []),
                    confidence=e.get("confidence"),
                    analyzer_id=e.get("analyzer_id", "StrategyMiner"),
                    analyzer_version=e.get("analyzer_version", ""),
                    schema_version=e.get("schema_version", ""),
                ) for e in author_evs
            ]
            raw = RawStrategy(
                strategy_id=s["strategy_id"],
                author_id=aid,
                name=s.get("name", s["strategy_id"]),
                description=s.get("description", ""),
                triggers=s.get("triggers", []),
                operations=s.get("operations", []),
                intended_effects=s.get("intended_effects", []),
                status=s.get("status", "discovered"),
                confidence=s.get("confidence"),
                evidence=evidence,
                source_work=s.get("source_work"),
                source_strategy_ids=[s["strategy_id"]],
            )
            by_author.setdefault(aid, []).append(raw)
    return by_author


def build_consolidation_inputs(data_root_: Path | None = None) -> dict[str, Any]:
    """生成每位作者的 consolidation 输入产物 + 汇总（无任何 LLM 调用）。"""
    out_dir = consolidation_layout(data_root_)
    out_dir["root"].mkdir(parents=True, exist_ok=True)
    reg_path = calibration_layout(data_root_)["root"] / "strategy_registry.json"
    registry_state = json.loads(reg_path.read_text(encoding="utf-8"))

    by_author = _partition_raw_strategies(registry_state)
    # 只构建 prompt、不调用 LLM；用真实 provider 预设仅为了拿 provider_id/model。
    consolidator = StrategyConsolidator(provider=None)
    provider = DeepSeekProvider()

    summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stage": "consolidation_input",
        "source_artifact": str(reg_path),
        "provider_id": provider.provider_id,
        "model": provider.model,
        "schema_version": CANONICAL_STRATEGY_SCHEMA_VERSION,
        "consolidator_version": STRATEGY_CONSOLIDATOR_VERSION,
        "n_authors": len(by_author),
        "authors": {},
    }

    for author_id in sorted(by_author):
        raw_list = by_author[author_id]
        prepared = consolidator.prepare(raw_list)          # 归一化 + 精确去重
        system, user = consolidator.build_prompt(prepared, author_id)
        est_input = _estimate_tokens(system) + _estimate_tokens(user)
        est_output = len(prepared) * 60                     # 每组 name/desc/3 summary/reasoning ≈ 60 token
        raw_records = []
        for r in prepared:
            d = r.to_dict()
            # 作者范围内重算支持层级（legacy `status` 是全局单调生命周期，可能含跨作者证据）
            d["author_scoped_support_status"] = _author_scoped_status(r.evidence)
            raw_records.append(d)
        record: dict[str, Any] = {
            "author_id": author_id,
            "n_raw_strategies": len(raw_list),
            "n_after_exact_dedup": len(prepared),
            "estimate": {
                "provider_id": provider.provider_id,
                "model": provider.model,
                "n_requests": 1,
                "est_input_tokens": est_input,
                "est_output_tokens": est_output,
                "est_total_tokens": est_input + est_output,
                "cache_status": "no existing consolidation cache",
            },
            "prompt_system": system,
            "prompt_user": user,
            "raw_strategies": raw_records,
        }
        summary["authors"][author_id] = {
            "n_raw_strategies": len(raw_list),
            "n_after_exact_dedup": len(prepared),
            "n_requests": 1,
            "est_input_tokens": est_input,
            "est_output_tokens": est_output,
        }
        (out_dir["root"] / f"{author_id}_consolidation_input.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 数据异常标注：全局注册表的跨作者残留 ----
    status_by_id = {s["strategy_id"]: s.get("status") for s in registry_state["strategies"]}
    appearing_authors: dict[str, set[str]] = {}
    for aid, raw_list in by_author.items():
        for r in raw_list:
            appearing_authors.setdefault(r.strategy_id, set()).add(aid)
    cross_author_ids = sorted(sid for sid, authors in appearing_authors.items()
                              if len(authors) > 1)
    cross_author_validated = sorted(
        sid for sid in cross_author_ids if status_by_id.get(sid) == StrategyStatus.VALIDATED.value)
    summary["anomalies"] = {
        "note": ("全局注册表生命周期单调且跨作者：策略先在单一作者内 validated 后，"
                 "再混入另一作者证据时 status 不降，导致 'validated' 残留跨作者证据。"
                 "author_scoped_support_status 为按作者重算后的真实支持层级。"),
        "n_cross_author_strategies": len(cross_author_ids),
        "cross_author_strategy_ids": cross_author_ids,
        "cross_author_validated_ids": cross_author_validated,
    }

    (out_dir["root"] / "consolidation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    summary = build_consolidation_inputs()
    for aid, a in summary["authors"].items():
        print(f"{aid}: raw={a['n_raw_strategies']} "
              f"after_exact_dedup={a['n_after_exact_dedup']} "
              f"requests={a['n_requests']} "
              f"est_tokens_in={a['est_input_tokens']} out={a['est_output_tokens']}")
    print("artifacts: data/analysis/consolidation/{author_id}_consolidation_input.json "
          "+ consolidation_summary.json")


if __name__ == "__main__":
    main()
