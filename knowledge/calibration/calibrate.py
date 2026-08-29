# knowledge/calibration/calibrate.py
"""Phase 4.4：40-chunk 采样 LLM 标定（Layer A judgment/hybrid + B 叙事 + C 策略）。

与 smoke.py（4 chunk，测量系统验证，smoke-only 注册表、不做聚合）不同，这是第一次
真实采样标定：
    - 从 40-chunk 采样清单（data/analysis/calibration_sample.json）取 chunk；
    - 对每个 chunk 跑 Layer A（8 个 LLM 特征）、B（叙事）、C（策略 match+discover）；
    - 用**规范**策略注册表并写回生命周期证据（discovered → candidate → validated）；
    - 聚合 ChunkProfile → WorkProfile → AuthorProfile（类型感知）；
    - 产出 JSON + Markdown 报告、LLM 画像、策略注册表快照。

仍然盲测默认、缓存后端、无 provider 显式不可用、绝不伪造。
"""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..analysis.base import AnalysisUnavailable, LLMResponseError
from ..analysis.narrative_analyzer import NarrativeAnalyzer
from ..analysis.strategy_miner import StrategyMiner
from ..analysis.style_analyzer import LLMFeatureAnalyzer
from ..config import data_layout, data_root as default_data_root
from ..corpus.metadata import CORPUS, author_display_names, author_ids
from ..profiles.aggregation import Aggregator, ChunkProfile
from ..providers.llm_provider import (
    CacheBackedLLMProvider, DeepSeekProvider, LLMCache, LLMTransportError,
)
from ..schema.feature_registry import build_default_registry
from ..schema.narrative_schema import NarrativeObservation
from ..schema.rubrics import (
    ASSESSMENT_INSUFFICIENT, ASSESSMENT_NOT_OBSERVABLE, ASSESSMENT_OBSERVED,
)
from ..schema.strategy_schema import CreativeStrategy, StrategyEvidence
from ..schema.style_schema import FeatureValue
from ..schema.versions import (
    AGGREGATION_VERSION, LLM_ANALYZER_VERSION, NARRATIVE_ANALYZER_VERSION,
    SAMPLING_VERSION, SCHEMA_VERSION, STRATEGY_MINER_VERSION,
)
from ..strategies.registry import seed_default_registry
from .smoke import (
    _author_by_work, _bump, _load_chunk, _run, _serialize, calibration_layout,
)

LLM_FEATURE_ANALYZER = "LlmFeatureAnalyzer"
TARGET_CHARS = 2000


def _sample_chunks(data_root_: Path | None) -> list[dict]:
    """从采样清单读取 40 个 chunk（按 work 稳定顺序展开），返回 selected 记录列表。

    每条 selected 记录含 chunk_id/work_id/chapter/seq 等；chunk 文本另行从 chunk
    文件按 chunk_id 加载（见 `_load_chunk`）。
    """
    base = Path(data_root_) if data_root_ is not None else default_data_root()
    manifest_path = base / "analysis" / "calibration_sample.json"
    sample = json.loads(manifest_path.read_text(encoding="utf-8"))
    chunks: list[dict] = []
    for work_id in sorted(sample["works"]):
        chunks.extend(sample["works"][work_id]["selected"])
    return chunks


def _registry_by_status(registry) -> dict[str, int]:
    from collections import Counter
    return dict(sorted(Counter(s.status for s in registry.all()).items()))


def run_sampled_calibration(data_root_: Path | None = None) -> dict[str, Any]:
    layout = data_layout(data_root_)
    out_dir = calibration_layout(data_root_)
    out_dir["root"].mkdir(parents=True, exist_ok=True)
    out_dir["cache"].mkdir(parents=True, exist_ok=True)

    # ---- provider（真实后端 + 磁盘缓存；与冒烟共用 llm_cache 目录）----
    provider = CacheBackedLLMProvider(
        DeepSeekProvider(),
        LLMCache(out_dir["cache"]),
    )

    feat_registry = build_default_registry()
    llm_features = sorted(
        (f for f in feat_registry.all() if f.analyzer == LLM_FEATURE_ANALYZER),
        key=lambda f: f.id,
    )
    # 规范策略注册表：标定就是要向它写回生命周期证据（与 smoke-only 相对）。
    strategy_registry = seed_default_registry()

    feature_analyzer = LLMFeatureAnalyzer(provider, blind=True)
    narrative_analyzer = NarrativeAnalyzer(provider, blind=True)
    author_by_work = _author_by_work()

    sample_chunks = _sample_chunks(data_root_)

    # 计量累加器（结构同冒烟，便于对照）
    m = {
        "layer_a": {"calls": 0, "ok": 0, "unavailable": 0, "schema_json": 0,
                    "transport": 0},
        "layer_b": {"calls": 0, "ok": 0, "unavailable": 0, "schema_json": 0,
                    "transport": 0, "downgraded": 0},
        "layer_c_match": {"calls": 0, "ok": 0, "unavailable": 0, "schema_json": 0,
                          "transport": 0},
        "layer_c_discover": {"calls": 0, "ok": 0, "unavailable": 0, "schema_json": 0,
                             "transport": 0},
        "assessment": {ASSESSMENT_OBSERVED: 0, ASSESSMENT_INSUFFICIENT: 0,
                       ASSESSMENT_NOT_OBSERVABLE: 0},
        "evidence": {"verified": 0, "unverified": 0},
        "strategy": {"matches": 0, "discoveries": 0,
                     "rejected_zero_evidence": 0, "rejected_unknown_strategy": 0},
    }

    profiles: list[ChunkProfile] = []
    chunks_out: list[dict[str, Any]] = []

    for sel in sample_chunks:
        work_id = sel["work_id"]
        chunk_id = sel["chunk_id"]
        rec = _load_chunk(layout, work_id, chunk_id)
        text = rec.get("text", "")
        author_id = author_by_work.get(work_id, "")
        chunk_rec = {
            "chunk_id": chunk_id, "work_id": work_id, "author_id": author_id,
            "chapter": rec.get("chapter", ""), "char_count": rec.get("char_count", 0),
            "layer_a": {}, "layer_b": {}, "layer_c": {},
        }

        feature_values: dict[str, FeatureValue] = {}

        # ---- Layer A：全部 LLM 派生特征（8 个）----
        for feat in llm_features:
            m["layer_a"]["calls"] += 1
            out = _run(lambda f=feat: feature_analyzer.analyze(
                text, f, chunk_id=chunk_id))
            if out["status"] != "ok":
                _bump(m["layer_a"], out["error_type"])
                chunk_rec["layer_a"][feat.id] = out
                continue
            res = out["result"]
            if isinstance(res, AnalysisUnavailable):
                m["layer_a"]["unavailable"] += 1
                chunk_rec["layer_a"][feat.id] = _serialize(res)
                continue
            m["layer_a"]["ok"] += 1
            feature_values[feat.id] = res
            chunk_rec["layer_a"][feat.id] = res.to_dict()
            prov = res.provenance if isinstance(res, FeatureValue) else {}
            if isinstance(prov, dict):
                st = prov.get("assessment_status")
                if st in m["assessment"]:
                    m["assessment"][st] += 1
                if feat.measurement_protocol == "frequency":
                    m["evidence"]["verified"] += int(prov.get("n_instances_verified", 0))
                    m["evidence"]["unverified"] += int(prov.get("n_instances_unverified", 0))
                else:  # ordinal：verified/unverified 来自 evidence / unverified_evidence
                    m["evidence"]["verified"] += len(res.evidence or [])
                    m["evidence"]["unverified"] += len(prov.get("unverified_evidence", []))

        # ---- Layer B：叙事观察 ----
        narrative: NarrativeObservation | None = None
        m["layer_b"]["calls"] += 1
        out = _run(lambda: narrative_analyzer.analyze(text, chunk_id=chunk_id))
        if out["status"] != "ok":
            _bump(m["layer_b"], out["error_type"])
            chunk_rec["layer_b"] = out
        else:
            res = out["result"]
            if isinstance(res, AnalysisUnavailable):
                m["layer_b"]["unavailable"] += 1
                chunk_rec["layer_b"] = _serialize(res)
            else:
                m["layer_b"]["ok"] += 1
                narrative = res
                chunk_rec["layer_b"] = res.to_dict()
                m["evidence"]["verified"] += len(res.observed_evidence or [])
                m["evidence"]["unverified"] += len(res.unverified_evidence or [])
                if "high_confidence_substantive_without_verified_evidence" in (
                        res.evidence_issues or []):
                    m["layer_b"]["downgraded"] += 1

        # ---- Layer C：策略匹配 + 发现（规范注册表，写回生命周期）----
        rejections: list[dict] = []
        miner = StrategyMiner(provider, strategy_registry, blind=True,
                              rejections=rejections)
        strategy_evidence: list[StrategyEvidence] = []

        m["layer_c_match"]["calls"] += 1
        out = _run(lambda: miner.match(text, chunk_id=chunk_id, work_id=work_id,
                                       author_id=author_id))
        match_list: list[dict] = []
        if out["status"] != "ok":
            _bump(m["layer_c_match"], out["error_type"])
        else:
            res = out["result"]
            if isinstance(res, AnalysisUnavailable):
                m["layer_c_match"]["unavailable"] += 1
            else:
                m["layer_c_match"]["ok"] += 1
                for sid, ev in res:
                    strategy_evidence.append(ev)
                    strategy_registry.record_evidence(sid, ev)  # 生命周期写回
                    match_list.append({"strategy_id": sid, "evidence": ev.to_dict()})
                m["strategy"]["matches"] += len(match_list)
                for _, ev in res:
                    m["evidence"]["verified"] += len(ev.quotes or [])
                    m["evidence"]["unverified"] += len(ev.unverified_quotes or [])

        m["layer_c_discover"]["calls"] += 1
        out = _run(lambda: miner.discover(text, chunk_id=chunk_id, work_id=work_id,
                                          author_id=author_id))
        disc_list: list[dict] = []
        if out["status"] != "ok":
            _bump(m["layer_c_discover"], out["error_type"])
        else:
            res = out["result"]
            if isinstance(res, AnalysisUnavailable):
                m["layer_c_discover"]["unavailable"] += 1
            else:
                m["layer_c_discover"]["ok"] += 1
                for s in res:
                    if not strategy_registry.has(s.strategy_id):
                        # 以空证据注册（discovered 状态），再逐条 record_evidence 推进
                        # 生命周期，避免证据重复入列。
                        strategy_registry.register(replace(s, evidence=[]))
                    for ev in s.evidence:
                        strategy_evidence.append(ev)
                        strategy_registry.record_evidence(s.strategy_id, ev)
                    disc_list.append(s.to_dict())
                m["strategy"]["discoveries"] += len(disc_list)
                for s in res:
                    m["evidence"]["verified"] += sum(len(e.quotes or [])
                                                    for e in s.evidence)

        for r in rejections:
            if r.get("reason") == "zero_verified_evidence":
                m["strategy"]["rejected_zero_evidence"] += 1
            elif r.get("reason") == "unknown_strategy":
                m["strategy"]["rejected_unknown_strategy"] += 1

        chunk_rec["layer_c"] = {
            "match": match_list,
            "discover": disc_list,
            "rejections": rejections,
        }
        chunks_out.append(chunk_rec)
        profiles.append(ChunkProfile(
            chunk_id=chunk_id, work_id=work_id, author_id=author_id,
            feature_values=feature_values, narrative=narrative,
            strategy_evidence=strategy_evidence,
        ))

    # ---- 聚合 Chunk → Work → Author ----
    agg = Aggregator()
    by_work: dict[str, list[ChunkProfile]] = {}
    by_author: dict[str, list[ChunkProfile]] = {}
    for p in profiles:
        by_work.setdefault(p.work_id, []).append(p)
        by_author.setdefault(p.author_id, []).append(p)
    work_profiles = {wid: agg.aggregate_work(by_work[wid]) for wid in sorted(by_work)}
    author_profiles = {aid: agg.aggregate_author(by_author[aid])
                       for aid in sorted(by_author)}

    # ---- 策略注册表快照 ----
    registry_state = {
        "n_strategies": len(strategy_registry),
        "by_status": _registry_by_status(strategy_registry),
        "strategies": [s.to_dict() for s in sorted(strategy_registry.all(),
                                                   key=lambda s: s.strategy_id)],
    }

    # ---- 汇总 ----
    inner = provider._inner  # DeepSeekProvider（暴露计量）
    metrics = {
        "requests": {
            "total": inner.n_calls,
            "successful": inner.n_success,
            "failed": inner.n_calls - inner.n_success,
            "retries": inner.n_retries,
        },
        "cache": {"hits": provider.cache_hits, "misses": provider.cache_misses},
        "token_usage": dict(inner.usage),
        "analysis": m,
    }

    report = {
        "run_meta": {
            "stage": "sampled_calibration",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "schema_version": SCHEMA_VERSION,
            "aggregation_version": AGGREGATION_VERSION,
            "sampling_version": SAMPLING_VERSION,
            "llm_analyzer_version": LLM_ANALYZER_VERSION,
            "narrative_analyzer_version": NARRATIVE_ANALYZER_VERSION,
            "strategy_miner_version": STRATEGY_MINER_VERSION,
            "blind": True,
            "n_chunks": len(profiles),
        },
        "provider": {
            "provider_id": inner.provider_id,
            "model": inner.model,
            "configured": inner.is_configured(),
            "cache_enabled": True,
            "cache_dir": str(out_dir["cache"]),
        },
        "metrics": metrics,
        "strategy_registry": registry_state,
        "profiles": {
            "n_chunk_profiles": len(profiles),
            "n_work_profiles": len(work_profiles),
            "n_author_profiles": len(author_profiles),
            "work_profiles": {k: v.to_dict() for k, v in work_profiles.items()},
            "author_profiles": {k: v.to_dict() for k, v in author_profiles.items()},
        },
        "chunks": chunks_out,
    }

    # ---- 写产物 ----
    prof_dir = out_dir["root"] / "profiles"
    prof_dir.mkdir(parents=True, exist_ok=True)
    with (prof_dir / "chunk_profiles.jsonl").open("w", encoding="utf-8") as fh:
        for p in profiles:
            fh.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")
    (prof_dir / "work_profiles.json").write_text(json.dumps(
        {k: v.to_dict() for k, v in work_profiles.items()},
        ensure_ascii=False, indent=2), encoding="utf-8")
    (prof_dir / "author_profiles.json").write_text(json.dumps(
        {k: v.to_dict() for k, v in author_profiles.items()},
        ensure_ascii=False, indent=2), encoding="utf-8")

    (out_dir["root"] / "strategy_registry.json").write_text(
        json.dumps(registry_state, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir["root"] / "calibration_results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir["root"] / "calibration_report.md").write_text(
        _render_markdown(report), encoding="utf-8")

    return report


def _render_markdown(report: dict[str, Any]) -> str:
    meta, prov, mx = report["run_meta"], report["provider"], report["metrics"]
    a = mx["analysis"]
    reg = report["strategy_registry"]
    prof = report["profiles"]
    lines: list[str] = []
    lines += [
        "# Weaver Style Engine — 40-chunk 采样 LLM 标定报告",
        "",
        f"- 生成时间（UTC）：`{meta['generated_at']}`",
        f"- schema：`{meta['schema_version']}`  aggregation：`{meta['aggregation_version']}`  sampling：`{meta['sampling_version']}`",
        f"- llm_analyzer：`{meta['llm_analyzer_version']}`  narrative：`{meta['narrative_analyzer_version']}`  strategy_miner：`{meta['strategy_miner_version']}`",
        f"- blind：`{meta['blind']}`　chunk 数：`{meta['n_chunks']}`",
        "",
        "## Provider",
        "",
        f"- provider：`{prov['provider_id']}`　model：`{prov['model']}`　configured：`{prov['configured']}`",
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
        "### 分析层结果",
        "",
        "| 层 | 调用 | 成功 | unavailable | schema/JSON 失败 | 传输失败 |",
        "|---|---|---|---|---|---|",
        f"| Layer A（8 特征） | {a['layer_a']['calls']} | {a['layer_a']['ok']} | {a['layer_a']['unavailable']} | {a['layer_a']['schema_json']} | {a['layer_a']['transport']} |",
        f"| Layer B（叙事） | {a['layer_b']['calls']} | {a['layer_b']['ok']} | {a['layer_b']['unavailable']} | {a['layer_b']['schema_json']} | {a['layer_b']['transport']} |",
        f"| Layer C match | {a['layer_c_match']['calls']} | {a['layer_c_match']['ok']} | {a['layer_c_match']['unavailable']} | {a['layer_c_match']['schema_json']} | {a['layer_c_match']['transport']} |",
        f"| Layer C discover | {a['layer_c_discover']['calls']} | {a['layer_c_discover']['ok']} | {a['layer_c_discover']['unavailable']} | {a['layer_c_discover']['schema_json']} | {a['layer_c_discover']['transport']} |",
        "",
        "### 评估状态与证据",
        "",
        f"- assessment_status：observed={a['assessment']['observed']}，insufficient_evidence={a['assessment']['insufficient_evidence']}，not_observable={a['assessment']['not_observable']}",
        f"- 证据：verified={a['evidence']['verified']}，unverified={a['evidence']['unverified']}",
        f"- 叙事证据降级（高置信无已验证证据）：{a['layer_b']['downgraded']}",
        f"- 策略匹配：{a['strategy']['matches']}；发现：{a['strategy']['discoveries']}",
        f"- 因零验证证据被拒：{a['strategy']['rejected_zero_evidence']}；因未知策略被忽略：{a['strategy']['rejected_unknown_strategy']}",
        "",
        "## 策略注册表（生命周期）",
        "",
        f"- 策略总数：{reg['n_strategies']}　状态分布：{reg['by_status']}",
        "",
        "| strategy_id | status | evidence | works |",
        "|---|---|---|---|",
    ]
    for s in reg["strategies"]:
        works = sorted({e.get("work_id") for e in s.get("evidence", []) if e.get("work_id")})
        lines.append(
            f"| `{s['strategy_id']}` | {s['status']} | {len(s.get('evidence', []))} | {len(works)} |")

    lines += [
        "",
        "## 画像聚合",
        "",
        f"- chunk 画像：{prof['n_chunk_profiles']}　work 画像：{prof['n_work_profiles']}　author 画像：{prof['n_author_profiles']}",
        "",
        "### 作者画像（LLM 特征均值，type-aware）",
        "",
    ]
    _authors = author_ids()
    _display = author_display_names()
    _cols = []
    for aid in _authors:
        _cols.append(f" {_display.get(aid, aid)} mean | {aid} n |")
    lines.append("| feature |" + "".join(_cols))
    lines.append("|" + "---|" * (1 + 2 * len(_authors)))
    _features = sorted({
        fid for ap in prof["author_profiles"].values()
        for fid in ap.get("features", {})
    })
    for fid in _features:
        row = [f"| {fid} |"]
        for aid in _authors:
            ap = prof["author_profiles"].get(aid, {})
            f = ap.get("features", {}).get(fid, {})
            mean = f.get("mean")
            n = f.get("n_valid", 0)
            row.append(f" {mean if mean is None else round(mean, 4)} | {n} |")
        lines.append("".join(row))

    lines += [
        "",
        "> 本报告为 40-chunk 采样标定；LLM 特征来自受控采样，未对全语料运行。",
    ]
    return "\n".join(lines)


def main() -> None:
    report = run_sampled_calibration()
    mx = report["metrics"]
    a = mx["analysis"]
    print(f"calibration done: model={report['provider']['model']} "
          f"chunks={report['run_meta']['n_chunks']} "
          f"requests={mx['requests']['total']} success={mx['requests']['successful']} "
          f"retries={mx['requests']['retries']} cache_hit={mx['cache']['hits']}")
    print(f"  tokens: in={mx['token_usage']['prompt_tokens']} "
          f"out={mx['token_usage']['completion_tokens']} "
          f"total={mx['token_usage']['total_tokens']}")
    print(f"  schema_json_failures="
          f"{a['layer_a']['schema_json'] + a['layer_b']['schema_json'] + a['layer_c_match']['schema_json'] + a['layer_c_discover']['schema_json']} "
          f"downgrades={a['layer_b']['downgraded']} matches={a['strategy']['matches']} "
          f"discoveries={a['strategy']['discoveries']} "
          f"rejected_zero_evidence={a['strategy']['rejected_zero_evidence']}")
    print(f"  registry: {report['strategy_registry']['by_status']}")
    print("  artifacts: data/analysis/calibration/calibration_results.json + "
          "calibration_report.md + profiles/ + strategy_registry.json")


if __name__ == "__main__":
    main()
