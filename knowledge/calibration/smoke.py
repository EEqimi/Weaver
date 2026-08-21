# knowledge/calibration/smoke.py
"""Phase 4.2 之后：四 chunk 真实 provider 冒烟标定（第一阶段）。

目的（测量系统验证，不是文学调校）：
    - 验证真实 LLM 后端端到端可用（Layer A/B/C 全链路）；
    - 捕获运行期计量（请求数 / 成功 / 失败 / 重试 / 缓存命中 / 证据校验 /
      评估状态 / 叙事降级 / 策略匹配与发现 / 零验证证据拒绝）；
    - 产出可检视的 per-chunk 报告（JSON + Markdown）。

明确不做：
    - 不聚合 WorkProfile/AuthorProfile 结论（4 chunk 不足以推断作者风格）；
    - 不推进策略生命周期（用 smoke-only 注册表，绝不写回规范注册表）；
    - 不晋升 candidate_core 特征；
    - 不因某个结果"文学上意外"而修改 rubric/prompt。
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..analysis.base import AnalysisUnavailable, LLMResponseError
from ..analysis.narrative_analyzer import NarrativeAnalyzer
from ..analysis.strategy_miner import StrategyMiner
from ..analysis.style_analyzer import LLMFeatureAnalyzer
from ..config import data_layout, data_root as default_data_root
from ..corpus.metadata import CORPUS
from ..providers.llm_provider import (
    CacheBackedLLMProvider, LLMCache, LLMTransportError, OpenAICompatibleProvider,
)
from ..schema.feature_registry import FeatureDefinition, build_default_registry
from ..schema.narrative_schema import NarrativeObservation
from ..schema.rubrics import (
    ASSESSMENT_INSUFFICIENT, ASSESSMENT_NOT_OBSERVABLE, ASSESSMENT_OBSERVED,
)
from ..schema.strategy_schema import CreativeStrategy, StrategyEvidence
from ..schema.style_schema import FeatureValue
from ..schema.versions import (
    AGGREGATION_VERSION, LLM_ANALYZER_VERSION, NARRATIVE_ANALYZER_VERSION,
    SCHEMA_VERSION, STRATEGY_MINER_VERSION,
)
from ..strategies.registry import seed_default_registry

# 四 chunk 冒烟样本：每 TRAIN 作品 1 个，从已批准的 40-chunk 清单中选定
# （position × dialogue 档多样，不含 held-out）。见冒烟报告 §1。
SMOKE_CHUNKS: list[tuple[str, str]] = [
    ("pride_and_prejudice", "pride_and_prejudice__2000__0295"),
    ("emma", "emma__2000__0073"),
    ("great_expectations", "great_expectations__2000__0253"),
    ("david_copperfield", "david_copperfield__2000__0839"),
]

TARGET_CHARS = 2000
LLM_FEATURE_ANALYZER = "LlmFeatureAnalyzer"
CALIBRATION_DIRNAME = "calibration"


def calibration_layout(data_root_: Path | None = None) -> dict[str, Path]:
    base = Path(data_root_) if data_root_ is not None else default_data_root()
    d = base / "analysis" / CALIBRATION_DIRNAME
    return {"root": d, "cache": d / "llm_cache"}


def _author_by_work() -> dict[str, str]:
    return {m.work_id: m.author_id for m in CORPUS}


def _load_chunk(layout: dict[str, Path], work_id: str, chunk_id: str) -> dict:
    path = layout["chunks"] / f"{work_id}__{TARGET_CHARS}.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("chunk_id") == chunk_id:
            return rec
    raise FileNotFoundError(f"chunk {chunk_id} 不在 {path}")


def _run(fn) -> dict[str, Any]:
    """运行一个 analyzer 调用并归一化结果 / 异常（不在此序列化）。"""
    try:
        return {"status": "ok", "result": fn()}
    except LLMTransportError as e:
        return {"status": "error", "error_type": "transport", "error": str(e)}
    except (LLMResponseError, ValueError) as e:
        return {"status": "error", "error_type": "schema_json", "error": str(e)}
    except Exception as e:  # noqa: BLE001 —— 冒烟报表要看到一切意外
        return {"status": "error", "error_type": type(e).__name__, "error": str(e)}


def _serialize(result: Any) -> Any:
    if isinstance(result, FeatureValue):
        return result.to_dict()
    if isinstance(result, NarrativeObservation):
        return result.to_dict()
    if isinstance(result, AnalysisUnavailable):
        return result.to_dict()
    if isinstance(result, StrategyEvidence):
        return result.to_dict()
    if isinstance(result, CreativeStrategy):
        return result.to_dict()
    if isinstance(result, (str, int, float, bool)) or result is None:
        return result
    return result


def _bump(bucket: dict[str, int], key: str) -> None:
    """安全累加：未知错误类型也记录（动态建键），绝不因缺键而中断冒烟。

    冒烟报表要"看到一切意外"（见 `_run` 的 catch-all），因此不能假定错误类型只
    落在预置的 schema_json / transport 两个桶里；否则未知异常会在此处 KeyError。
    """
    bucket[key] = bucket.get(key, 0) + 1


def _feature_report(fid: str, out: dict[str, Any]) -> dict[str, Any]:
    """把 Layer A 单个特征结果拍平为便于报表检视的字段。

    仅接受 `{"status": "ok", "result": <FeatureValue.to_dict()>}`；error /
    unavailable 状态已在渲染层（`_render_markdown`）先行分流。
    """
    res = out["result"]
    d = res if isinstance(res, dict) else {"value": res}
    prov = d.get("provenance") or {}
    return {
        "feature": fid,
        "value": d.get("value"),
        "raw_value": d.get("raw_value"),
        "assessment_status": prov.get("assessment_status", ASSESSMENT_OBSERVED),
        "confidence": d.get("confidence"),
        "verified_evidence": d.get("evidence") or [],
        "unverified_evidence": prov.get("unverified_evidence") or [],
    }


def run_smoke_calibration(data_root_: Path | None = None) -> dict[str, Any]:
    layout = data_layout(data_root_)
    out_dir = calibration_layout(data_root_)
    out_dir["root"].mkdir(parents=True, exist_ok=True)
    out_dir["cache"].mkdir(parents=True, exist_ok=True)

    # ---- provider（真实后端 + 磁盘缓存）----
    provider = CacheBackedLLMProvider(
        OpenAICompatibleProvider(),
        LLMCache(out_dir["cache"]),
    )

    registry = build_default_registry()
    llm_features = sorted(
        (f for f in registry.all() if f.analyzer == LLM_FEATURE_ANALYZER),
        key=lambda f: f.id,
    )
    feature_analyzer = LLMFeatureAnalyzer(provider, blind=True)
    narrative_analyzer = NarrativeAnalyzer(provider, blind=True)
    smoke_registry = seed_default_registry()  # smoke-only，绝不写回规范注册表

    author_by_work = _author_by_work()

    # 计量累加器
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

    chunks_out: list[dict[str, Any]] = []

    for work_id, chunk_id in SMOKE_CHUNKS:
        rec = _load_chunk(layout, work_id, chunk_id)
        text = rec.get("text", "")
        author_id = author_by_work.get(work_id, "")
        chunk_rec = {
            "chunk_id": chunk_id, "work_id": work_id, "author_id": author_id,
            "chapter": rec.get("chapter", ""), "char_count": rec.get("char_count", 0),
            "layer_a": {}, "layer_b": {}, "layer_c": {},
        }

        # ---- Layer A：全部 LLM 派生特征（8 个）----
        for feat in llm_features:
            m["layer_a"]["calls"] += 1
            out = _run(lambda f=feat: feature_analyzer.analyze(
                text, f, chunk_id=chunk_id))
            chunk_rec["layer_a"][feat.id] = (
                _serialize(out["result"]) if out["status"] == "ok" else out)
            if out["status"] != "ok":
                _bump(m["layer_a"], out["error_type"])
                continue
            res = out["result"]
            if isinstance(res, AnalysisUnavailable):
                m["layer_a"]["unavailable"] += 1
                continue
            m["layer_a"]["ok"] += 1
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
        m["layer_b"]["calls"] += 1
        out = _run(lambda: narrative_analyzer.analyze(text, chunk_id=chunk_id))
        chunk_rec["layer_b"] = _serialize(out["result"]) if out["status"] == "ok" else out
        if out["status"] != "ok":
            _bump(m["layer_b"], out["error_type"])
        else:
            res = out["result"]
            if isinstance(res, AnalysisUnavailable):
                m["layer_b"]["unavailable"] += 1
            else:
                m["layer_b"]["ok"] += 1
                m["evidence"]["verified"] += len(res.observed_evidence or [])
                m["evidence"]["unverified"] += len(res.unverified_evidence or [])
                if "high_confidence_substantive_without_verified_evidence" in (
                        res.evidence_issues or []):
                    m["layer_b"]["downgraded"] += 1

        # ---- Layer C：策略匹配 + 发现（smoke-only，不写回注册表）----
        rejections: list[dict] = []
        miner = StrategyMiner(provider, smoke_registry, blind=True,
                              rejections=rejections)
        m["layer_c_match"]["calls"] += 1
        out = _run(lambda: miner.match(text, chunk_id=chunk_id, work_id=work_id,
                                       author_id=author_id))
        if out["status"] != "ok":
            _bump(m["layer_c_match"], out["error_type"])
            match_list = []
        else:
            res = out["result"]
            if isinstance(res, AnalysisUnavailable):
                m["layer_c_match"]["unavailable"] += 1
                match_list = []
            else:
                m["layer_c_match"]["ok"] += 1
                match_list = [{"strategy_id": sid, "evidence": ev.to_dict()}
                              for sid, ev in res]
                m["strategy"]["matches"] += len(match_list)
                for _, ev in res:
                    m["evidence"]["verified"] += len(ev.quotes or [])
                    m["evidence"]["unverified"] += len(ev.unverified_quotes or [])

        m["layer_c_discover"]["calls"] += 1
        out = _run(lambda: miner.discover(text, chunk_id=chunk_id, work_id=work_id,
                                          author_id=author_id))
        if out["status"] != "ok":
            _bump(m["layer_c_discover"], out["error_type"])
            disc_list = []
        else:
            res = out["result"]
            if isinstance(res, AnalysisUnavailable):
                m["layer_c_discover"]["unavailable"] += 1
                disc_list = []
            else:
                m["layer_c_discover"]["ok"] += 1
                disc_list = [s.to_dict() for s in res]
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

    # ---- 汇总 ----
    inner = provider._inner  # OpenAICompatibleProvider（暴露计量）
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
            "stage": "smoke_calibration",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "schema_version": SCHEMA_VERSION,
            "aggregation_version": AGGREGATION_VERSION,
            "llm_analyzer_version": LLM_ANALYZER_VERSION,
            "narrative_analyzer_version": NARRATIVE_ANALYZER_VERSION,
            "strategy_miner_version": STRATEGY_MINER_VERSION,
            "blind": True,
            "smoke_chunks": [{"work_id": w, "chunk_id": c} for w, c in SMOKE_CHUNKS],
        },
        "provider": {
            "provider_id": inner.provider_id,
            "model": inner.model,
            "configured": inner.is_configured(),
            "cache_enabled": True,
            "cache_dir": str(out_dir["cache"]),
        },
        "metrics": metrics,
        "chunks": chunks_out,
    }

    (out_dir["root"] / "smoke_results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir["root"] / "smoke_report.md").write_text(
        _render_markdown(report), encoding="utf-8")

    return report


def _render_markdown(report: dict[str, Any]) -> str:
    meta, prov, mx = report["run_meta"], report["provider"], report["metrics"]
    a = mx["analysis"]
    lines: list[str] = []
    lines += [
        "# Weaver Style Engine — LLM 冒烟标定报告（4 chunk）",
        "",
        f"- 生成时间（UTC）：`{meta['generated_at']}`",
        f"- schema_version：`{meta['schema_version']}`  aggregation_version：`{meta['aggregation_version']}`",
        f"- llm_analyzer：`{meta['llm_analyzer_version']}`  narrative：`{meta['narrative_analyzer_version']}`  strategy_miner：`{meta['strategy_miner_version']}`",
        f"- blind 模式：`{meta['blind']}`",
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
        f"| cost | 未估算（无可可靠定价；记录 token 用量） |",
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
        "### 评估状态（ordinal）与证据",
        "",
        f"- assessment_status：observed={a['assessment']['observed']}，insufficient_evidence={a['assessment']['insufficient_evidence']}，not_observable={a['assessment']['not_observable']}",
        f"- 证据校验：verified={mx['analysis']['evidence']['verified']}，unverified={mx['analysis']['evidence']['unverified']}",
        f"- 叙事证据降级（高置信无已验证证据）：{a['layer_b']['downgraded']}",
        f"- 策略匹配：{a['strategy']['matches']}；策略发现：{a['strategy']['discoveries']}",
        f"- 因零验证证据被拒的策略输出：{a['strategy']['rejected_zero_evidence']}；因未知策略被忽略：{a['strategy']['rejected_unknown_strategy']}",
        "",
    ]

    for c in report["chunks"]:
        lines += [
            f"## Chunk：{c['chunk_id']}",
            "",
            f"- work：`{c['work_id']}`（author `{c['author_id']}`）chapter `{c['chapter']}` chars `{c['char_count']}`",
            "",
            "### Layer A（LLM 特征）",
            "",
            "| feature | value | raw_value | assessment_status | confidence | verified | unverified |",
            "|---|---|---|---|---|---|---|",
        ]
        for fid, r in c["layer_a"].items():
            if isinstance(r, dict) and r.get("status") == "error":
                lines.append(
                    f"| {fid} | — | — | — | — | — | **{r.get('error_type')}**: {r.get('error','')} |")
                continue
            if isinstance(r, dict) and r.get("status") == "unavailable":
                lines.append(
                    f"| {fid} | — | — | — | — | — | unavailable: {r.get('reason','')} |")
                continue
            fr = _feature_report(fid, {"status": "ok", "result": r})
            lines.append(
                f"| {fr['feature']} | {fr['value']} | {fr['raw_value']} | "
                f"{fr['assessment_status']} | {fr['confidence']} | "
                f"{len(fr['verified_evidence'])} | {len(fr['unverified_evidence'])} |")

        lines += ["", "### Layer B（叙事）", ""]
        b = c["layer_b"]
        if isinstance(b, dict) and b.get("status") == "error":
            lines.append(f"- 错误：`{b.get('error_type')}` — {b.get('error','')}")
        elif isinstance(b, dict) and b.get("status") == "unavailable":
            lines.append(f"- 不可用：{b.get('reason','')}")
        else:
            lines.append(
                f"- pov=`{b.get('pov')}` focalization=`{b.get('focalization')}` "
                f"distance=`{b.get('narrative_distance')}` presence=`{b.get('narrator_presence')}` "
                f"info_access=`{b.get('information_access')}` temporal_order=`{b.get('temporal_order')}`")
            lines.append(
                f"- confidence=`{b.get('confidence')}`；verified_evidence={len(b.get('observed_evidence') or [])}，"
                f"unverified_evidence={len(b.get('unverified_evidence') or [])}")
            if b.get("proportion_issues"):
                lines.append(f"- proportion_issues：{b['proportion_issues']}")
            if b.get("evidence_issues"):
                lines.append(f"- evidence_issues：{b['evidence_issues']}")

        lines += ["", "### Layer C（策略）", ""]
        lc = c["layer_c"]
        lines.append(f"- 已知策略匹配：{len(lc['match'])}")
        for mch in lc["match"]:
            ev = mch["evidence"]
            lines.append(f"  - `{mch['strategy_id']}` conf={ev.get('confidence')} verified_quotes={ev.get('quotes')}")
        lines.append(f"- 候选发现：{len(lc['discover'])}")
        for s in lc["discover"]:
            lines.append(f"  - `{s['strategy_id']}`（{s.get('name')}）conf={s.get('confidence')}")
        lines.append(f"- 拒绝：{lc['rejections'] if lc['rejections'] else '（无）'}")
        lines.append("")

    lines.append("> 本报告仅验证测量系统端到端可用；4 chunk 不足以推断 Austen/Dickens 风格。")
    return "\n".join(lines)


def main() -> None:
    report = run_smoke_calibration()
    mx = report["metrics"]
    a = mx["analysis"]
    print(f"smoke done: model={report['provider']['model']} "
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
    print("  artifacts: data/analysis/calibration/smoke_results.json + smoke_report.md")


if __name__ == "__main__":
    main()
