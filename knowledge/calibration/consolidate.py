# knowledge/calibration/consolidate.py
"""Phase 4.5：执行作者级付费 consolidation（每作者一次真实 LLM 请求）。

复用 Phase 4.4 的 `strategy_registry.json`，按作者分区后调用 `StrategyConsolidator`
做结构化映射合并，产出每作者 canonical strategy 集合 + 汇总报告。

关键约束（spec Phase 4.5）：
    - **绝不重跑 chunk-level analyzer** —— 只读注册表快照，不碰任何 chunk。
    - 一次只处理一位作者（严格 author scope），越界由 consolidator 拒绝。
    - 缓存后端：内容寻址缓存键保证重复运行不重复烧 token。
    - 未配置 provider / 传输失败 / 非法输出：显式记录并继续，绝不伪造结果。

执行 `run_consolidation()` 即触发真实付费 LLM 请求（DeepSeek `deepseek-chat`，
每作者 1 次）。请先 review `consolidation_input.py` 产物与估算。
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..analysis.base import LLMNotConfiguredError, LLMResponseError
from ..config import data_root as default_data_root
from ..providers.llm_provider import (
    CacheBackedLLMProvider, DeepSeekProvider, LLMCache, LLMTransportError,
)
from ..schema.strategy_schema import CanonicalStrategy
from ..schema.versions import (
    CANONICAL_STRATEGY_SCHEMA_VERSION, STRATEGY_CONSOLIDATOR_VERSION,
)
from ..strategies.consolidation import ConsolidationError, StrategyConsolidator
from .consolidation_input import _partition_raw_strategies, consolidation_layout
from .smoke import calibration_layout


def _count_support_status(canonicals: list[CanonicalStrategy]) -> dict[str, int]:
    return dict(sorted(Counter(c.support_status for c in canonicals).items()))


def run_consolidation(data_root_: Path | None = None) -> dict[str, Any]:
    """按作者执行付费 consolidation 并落盘 canonical 集合 + 报告。"""
    out_dir = consolidation_layout(data_root_)
    out_dir["root"].mkdir(parents=True, exist_ok=True)
    # 复用标定阶段同一 llm_cache（内容寻址，跨阶段不冲突、重复运行免费）
    cache_dir = calibration_layout(data_root_)["cache"]
    cache_dir.mkdir(parents=True, exist_ok=True)

    reg_path = calibration_layout(data_root_)["root"] / "strategy_registry.json"
    registry_state = json.loads(reg_path.read_text(encoding="utf-8"))

    provider = CacheBackedLLMProvider(DeepSeekProvider(), LLMCache(cache_dir))
    consolidator = StrategyConsolidator(provider, blind=True)
    by_author = _partition_raw_strategies(registry_state)

    authors: dict[str, Any] = {}
    for author_id in sorted(by_author):
        raw_list = by_author[author_id]
        rec: dict[str, Any] = {"author_id": author_id,
                               "n_raw_strategies": len(raw_list)}
        try:
            canonicals = consolidator.consolidate(raw_list, author_id)
        except LLMNotConfiguredError as e:
            rec.update(status="unconfigured", error=str(e))
        except LLMTransportError as e:
            rec.update(status="transport_error", error=str(e))
        except (LLMResponseError, ConsolidationError) as e:
            rec.update(status="invalid_response", error=str(e))
        else:
            canonical_dicts = [c.to_dict() for c in canonicals]
            rec.update(
                status="ok",
                n_canonical=len(canonicals),
                by_support_status=_count_support_status(canonicals),
                canonical_strategies=canonical_dicts,
            )
            (out_dir["root"] / f"{author_id}_canonical_strategies.json").write_text(
                json.dumps({
                    "author_id": author_id,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "schema_version": CANONICAL_STRATEGY_SCHEMA_VERSION,
                    "n_canonical": len(canonicals),
                    "canonical_strategies": canonical_dicts,
                }, ensure_ascii=False, indent=2), encoding="utf-8")
        authors[author_id] = rec

    inner = provider._inner  # DeepSeekProvider（暴露运行期计量）
    metrics = {
        "requests": {"total": inner.n_calls, "successful": inner.n_success,
                     "failed": inner.n_calls - inner.n_success,
                     "retries": inner.n_retries},
        "cache": {"hits": provider.cache_hits, "misses": provider.cache_misses},
        "token_usage": dict(inner.usage),
    }

    report = {
        "run_meta": {
            "stage": "author_consolidation",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "canonical_schema_version": CANONICAL_STRATEGY_SCHEMA_VERSION,
            "consolidator_version": STRATEGY_CONSOLIDATOR_VERSION,
            "blind": True,
            "n_authors": len(by_author),
            "source_artifact": str(reg_path),
        },
        "provider": {
            "provider_id": inner.provider_id,
            "model": inner.model,
            "configured": inner.is_configured(),
            "cache_enabled": True,
            "cache_dir": str(cache_dir),
        },
        "metrics": metrics,
        "authors": authors,
    }

    (out_dir["root"] / "consolidation_results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir["root"] / "consolidation_report.md").write_text(
        _render_markdown(report), encoding="utf-8")
    return report


def _render_markdown(report: dict[str, Any]) -> str:
    meta, prov, mx = report["run_meta"], report["provider"], report["metrics"]
    lines: list[str] = [
        "# Weaver Style Engine — 作者级策略合并（Consolidation）报告",
        "",
        f"- 生成时间（UTC）：`{meta['generated_at']}`",
        f"- canonical schema：`{meta['canonical_schema_version']}`  consolidator：`{meta['consolidator_version']}`",
        f"- blind：`{meta['blind']}`  作者数：`{meta['n_authors']}`",
        f"- 来源：`{meta['source_artifact']}`",
        "",
        "## Provider",
        "",
        f"- provider：`{prov['provider_id']}`  model：`{prov['model']}`  configured：`{prov['configured']}`",
        f"- cache：enabled，目录 `{prov['cache_dir']}`",
        "",
        "## 运行期计量",
        "",
        "| 指标 | 值 |",
        "|---|---|",
        f"| LLM 请求总数 | {mx['requests']['total']} |",
        f"| 成功请求 | {mx['requests']['successful']} |",
        f"| 失败请求 | {mx['requests']['failed']} |",
        f"| 重试次数 | {mx['requests']['retries']} |",
        f"| 缓存命中 / 未命中 | {mx['cache']['hits']} / {mx['cache']['misses']} |",
        f"| input tokens | {mx['token_usage']['prompt_tokens']} |",
        f"| output tokens | {mx['token_usage']['completion_tokens']} |",
        f"| total tokens | {mx['token_usage']['total_tokens']} |",
        "",
    ]
    for aid, a in report["authors"].items():
        lines += [f"## 作者：{aid}", ""]
        if a["status"] != "ok":
            lines.append(f"- **{a['status']}**：{a.get('error', '')}")
            lines.append("")
            continue
        lines += [
            f"- raw 策略：{a['n_raw_strategies']} → canonical：{a['n_canonical']}",
            f"- 支持层级分布：{a['by_support_status']}",
            "",
            "| canonical_strategy_id | name | 支持作品 | 支持 chunk | support_status | confidence |",
            "|---|---|---|---|---|---|",
        ]
        for cs in a["canonical_strategies"]:
            lines.append(
                f"| `{cs['canonical_strategy_id']}` | {cs['canonical_name']} | "
                f"{len(cs['supporting_work_ids'])} | {len(cs['supporting_chunk_ids'])} | "
                f"{cs['support_status']} | {cs['confidence']} |")
        lines.append("")
    lines.append("> 本报告为作者级 consolidation；原始 raw strategies 仍保留在 "
                 "`data/analysis/calibration/strategy_registry.json`，绝不覆盖。")
    return "\n".join(lines)


def main() -> None:
    report = run_consolidation()
    mx = report["metrics"]
    print(f"consolidation done: model={report['provider']['model']} "
          f"requests={mx['requests']['total']} success={mx['requests']['successful']} "
          f"retries={mx['requests']['retries']} cache_hit={mx['cache']['hits']}")
    print(f"  tokens: in={mx['token_usage']['prompt_tokens']} "
          f"out={mx['token_usage']['completion_tokens']} "
          f"total={mx['token_usage']['total_tokens']}")
    for aid, a in report["authors"].items():
        if a["status"] == "ok":
            print(f"  {aid}: raw={a['n_raw_strategies']} -> canonical={a['n_canonical']} "
                  f"support={a['by_support_status']}")
        else:
            print(f"  {aid}: {a['status']} — {a.get('error', '')}")
    print("  artifacts: data/analysis/consolidation/{author_id}_canonical_strategies.json "
          "+ consolidation_results.json + consolidation_report.md")


if __name__ == "__main__":
    main()
