# Weaver Style Engine — Project Status

Short current-state snapshot (≈1–2 min read). History lives in
[`STYLE_ENGINE_DEVLOG.md`](STYLE_ENGINE_DEVLOG.md); target design lives in
`STYLE_ENGINE_SPEC_V0.1.md`.

| Field | Value |
|---|---|
| **Current phase** | Phase 4.4 (40-chunk sampled LLM calibration) — COMPLETE |
| **Last completed checkpoint** | 40-chunk sampled calibration clean on DeepSeek (`deepseek-chat`): 440 requests, Layer A 318/320 (2 evidence-enforcement rejections), 148 matches / 71 discoveries / 17 zero-evidence rejections, registry 75 strategies (9 validated / 29 candidate / 37 discovered); 138 tests |
| **Current branch** | `feature/style-engine-v0.1` |

## What is functional
- Corpus pipeline: RAW → CLEAN → CHUNKS (1000/2000/4000) → METADATA/QC (deterministic, raw read-only).
- Feature Registry: 39 features, data-driven routing by analyzer name.
- Layer A deterministic analyzer: 22 features.
- Layer D stylometry: extraction + Burrows Delta + PCA + clustering + SVM/logreg validation (GroupKFold, held-out), leak-free grouped CV (per-fold vectorizer refit), function-word/word-unigram overlap audit.
- LLM provider abstraction (cacheable, unconfigured-safe).
- Real LLM backend: `DeepSeekProvider` (`deepseek-chat`, OpenAI-compatible, stdlib
  urllib, transport error body capture, runtime metering) + cache hit/miss counters.
- Smoke calibration: `knowledge/calibration/smoke.py` (4-chunk end-to-end run with
  JSON + Markdown report and rejection accounting).
- Profile aggregation: `ChunkProfile → WorkProfile → AuthorProfile`, type-aware, preserving uncertainty/evidence/provenance.
- Deterministic stratified sampling manifest (`seq`-ordered).
- Measurement rubrics: frequency vs ordinal protocols for all LLM-derived features.
- Shared evidence verification (NFC + punctuation normalization + substring match).
- Frequency features normalize programmatically to a rate per 1000 tokens (`raw_count`/`exposure` in provenance); ordinal features carry `assessment_status` (`observed`/`insufficient_evidence`/`not_observable`) with null values preserved.
- Narrative proportion validation (bounds/all-zero/unknown-key) and evidence-sufficiency downgrade.
- Strict strategy author/work consistency and zero-verified-evidence rejection.

## What is partially implemented
- Layer A judgment/hybrid, Layer B (narrative), Layer C (strategies) analyzers are **written and run on the 40-chunk calibration sample** (real `deepseek-chat`); not yet run on the full corpus.
- Strategy registry lifecycle implemented and exercised (75 strategies, 9 validated / 29 candidate / 37 discovered).

## What is not implemented yet
- Full-corpus LLM feature extraction (only the 40-chunk calibration sample has LLM features).
- Phase 5 (author-profile synthesis) and beyond (style mixing, planner, generation loop).
- NlpAnalyzer (POS) features — NLTK intentionally not installed.
- Mixed-effects / variance-decomposition model (deferred by spec).

## Current corpus
- **TRAIN:** Pride and Prejudice, Emma (Austen); Great Expectations, David Copperfield (Dickens).
- **HELD-OUT:** Persuasion (Austen); A Tale of Two Cities (Dickens).
- 6 works total; raw text outside the repo (`wensigongfang/text/`), `data/` gitignored.

## Current test status
- **138 tests passed** (was 136). Two new regression tests cover the Phase 4.4
  `StrategyEvidence.strategy_id` fix (field carried on the dataclass; work-profile
  strategy evidence counting by `strategy_id`). Earlier provider/cache/miner tests
  from Phase 4.3 remain.

## Latest experiment results (deterministic, no LLM)
- Layer A: 2,328 TRAIN chunks × 22 features.
- Layer D: 954 features (154 fw + 400 char-3gram + 400 word-unigram, function
  words excluded from word-unigram; `n_function_word_overlap=0`).
- Grouped leave-one-work-out CV (SVM, class-weighted), **leak-free**:
  `[0.819, 0.924, 0.794, 0.905]` (mean ≈ 0.861; was leaky 0.884).
- Held-out accuracy: `0.745` (was 0.756).
- Calibration sample: 40 chunks (4 × 10), `seq`-ordered deterministic stratified,
  held-out excluded.

### LLM smoke calibration (real `deepseek-chat`, 4 chunks) — CLEAN
- 44 requests (11 × 4), **44 success**, 0 schema/JSON failures, 0 retries, 0
  transport failures.
- Token usage 50,498 (39,174 in / 11,324 out); cache 0 hit / 44 miss (fresh backend).
- Assessment: observed=20, insufficient=0, not_observable=0; evidence 161 verified
  / 6 unverified; 0 narrative downgrades.
- Strategies: 9 matches, 7 discoveries, **1 zero-evidence rejection** (contract
  fired on real backend), 0 unknown-strategy rejections.
- Artifacts: `data/analysis/calibration/{smoke_results.json, smoke_report.md}`.

### LLM sampled calibration (real `deepseek-chat`, 40 chunks) — COMPLETE
- 440 requests (40 × 11); 44 smoke cache hits + 396 fresh. Token metering was lost in
  the first run's aggregation crash (report shows the cache-replay re-run: 440 cache
  hits / 0 requests); estimated ~455k fresh + 50.5k smoke ≈ ~505k total.
- Layer A: 320 calls → 318 success, **2 evidence-enforcement rejections** (high
  confidence without verified evidence — not JSON failures); B 40/40; C match 40/40,
  discover 40/40. 0 transport failures.
- Evidence: 1718 verified / 79 unverified; narrative downgrades 0; assessment
  observed=198, insufficient=0, not_observable=0.
- Strategies: 148 matches, 71 discoveries, 17 zero-evidence rejections, 0 unknown.
- Registry: 75 strategies = 9 validated / 29 candidate / 37 discovered.
- Author profiles (LLM feature means): simile_frequency austen 0.72 vs dickens 2.52;
  metaphor_frequency 13.95 vs 17.52; irony_frequency 8.92 vs 6.22.
- Artifacts: `data/analysis/calibration/{calibration_results.json, calibration_report.md,
  profiles/, strategy_registry.json}`.

### LLM smoke calibration (previous backend — real `qwen-plus`/DashScope, 4 chunks)
- Retained for history; superseded by the DeepSeek run above.
- 44 requests (11 × 4), 37 success, 0 schema/JSON failures, 0 retries.
- 7 failures — all `HTTP 400 Arrearage` (DashScope overdue payment), on the last
  chunk; account cut off mid-run. Environmental, not code.
- Token usage 42,349 (33,220 in / 9,129 out); 6 strategy matches, 0 discoveries,
  0 zero-evidence rejections, 0 narrative downgrades; 114 verified / 18 unverified
  evidence quotes.

## Current blockers / review items
- **discover 去重缺口**：`_to_strategy` 用 description hash 追加后缀，跨 chunk 同名
  策略生成不同 id，37 个 discovered 多为单 chunk 单例；Phase 5 前应改为稳定 id 去重。
- `candidate_core` 特征仍不得晋升（校准仅标定样本，不足以晋升）。

## Next planned action
- Phase 5 (author-profile synthesis): solve discover dedup (stable ids), then decide
  on full-corpus LLM feature extraction scale vs the 40-chunk sample before author-
  level synthesis / mixed model / planner / generation loop.
