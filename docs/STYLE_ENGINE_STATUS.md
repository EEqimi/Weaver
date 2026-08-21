# Weaver Style Engine — Project Status

Short current-state snapshot (≈1–2 min read). History lives in
[`STYLE_ENGINE_DEVLOG.md`](STYLE_ENGINE_DEVLOG.md); target design lives in
`STYLE_ENGINE_SPEC_V0.1.md`.

| Field | Value |
|---|---|
| **Current phase** | Phase 4.3 (LLM smoke calibration, 4 chunks) — COMPLETE |
| **Last completed checkpoint** | Phase 4.3 LLM smoke calibration: real `qwen-plus` backend end-to-end on 4 chunks (Layer A/B/C), 132 tests, 37/44 success — 7 failures are DashScope `Arrearage` (billing), not code |
| **Current branch** | `feature/style-engine-v0.1` |

## What is functional
- Corpus pipeline: RAW → CLEAN → CHUNKS (1000/2000/4000) → METADATA/QC (deterministic, raw read-only).
- Feature Registry: 39 features, data-driven routing by analyzer name.
- Layer A deterministic analyzer: 22 features.
- Layer D stylometry: extraction + Burrows Delta + PCA + clustering + SVM/logreg validation (GroupKFold, held-out), leak-free grouped CV (per-fold vectorizer refit), function-word/word-unigram overlap audit.
- LLM provider abstraction (cacheable, unconfigured-safe).
- Real LLM backend: `OpenAICompatibleProvider` (DashScope compatible-mode, stdlib,
  transport error body capture, runtime metering) + cache hit/miss counters.
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
- Layer A judgment/hybrid, Layer B (narrative), Layer C (strategies) analyzers are **written but not run** — they need a configured LLM and the calibration sample.
- Strategy registry lifecycle implemented; strategies still only seeded candidates (no evidence yet).

## What is not implemented yet
- Full LLM calibration on the 40-chunk sample.
- Phase 5 (author-profile synthesis) and beyond (style mixing, planner, generation loop).
- NlpAnalyzer (POS) features — NLTK intentionally not installed.
- Mixed-effects / variance-decomposition model (deferred by spec).

## Current corpus
- **TRAIN:** Pride and Prejudice, Emma (Austen); Great Expectations, David Copperfield (Dickens).
- **HELD-OUT:** Persuasion (Austen); A Tale of Two Cities (Dickens).
- 6 works total; raw text outside the repo (`wensigongfang/text/`), `data/` gitignored.

## Current test status
- **132 tests passed** (was 114). New regression tests cover: OpenAICompatibleProvider
  (configured/unconfigured, success+usage, 429 retry/backoff, permanent-4xx no-retry,
  HTTP error-body capture), cache hit/miss counters, StrategyMiner rejection collector,
  and smoke `_bump`/`_feature_report`.

## Latest experiment results (deterministic, no LLM)
- Layer A: 2,328 TRAIN chunks × 22 features.
- Layer D: 954 features (154 fw + 400 char-3gram + 400 word-unigram, function
  words excluded from word-unigram; `n_function_word_overlap=0`).
- Grouped leave-one-work-out CV (SVM, class-weighted), **leak-free**:
  `[0.819, 0.924, 0.794, 0.905]` (mean ≈ 0.861; was leaky 0.884).
- Held-out accuracy: `0.745` (was 0.756).
- Calibration sample: 40 chunks (4 × 10), `seq`-ordered deterministic stratified,
  held-out excluded.

### LLM smoke calibration (real `qwen-plus`, 4 chunks)
- 44 requests (11 × 4), 37 success, 0 schema/JSON failures, 0 retries.
- 7 failures — all `HTTP 400 Arrearage` (DashScope overdue payment), on the last
  chunk; account cut off mid-run. Environmental, not code.
- Token usage 42,349 (33,220 in / 9,129 out); 6 strategy matches, 0 discoveries,
  0 zero-evidence rejections, 0 narrative downgrades; 114 verified / 18 unverified
  evidence quotes.
- Artifacts: `data/analysis/calibration/{smoke_results.json, smoke_report.md}`.

## Current blockers / review items
- **DashScope account in arrears (`Arrearage`)** — top up before the 40-chunk
  calibration, or the run will be denied partway.
- **Awaiting review** of the Phase 3–4.2 contract checkpoint before the full
  sampled calibration. `candidate_core` features must remain un-promoted.

## Next planned action
- After topping up the account: run the sampled LLM calibration (Layer A
  judgment/hybrid, B, C on the 40-chunk sample) with the configured, cache-backed
  provider; then aggregate evidence and re-evaluate before Phase 5.
