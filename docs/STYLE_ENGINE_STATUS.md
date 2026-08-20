# Weaver Style Engine — Project Status

Short current-state snapshot (≈1–2 min read). History lives in
[`STYLE_ENGINE_DEVLOG.md`](STYLE_ENGINE_DEVLOG.md); target design lives in
`STYLE_ENGINE_SPEC_V0.1.md`.

| Field | Value |
|---|---|
| **Current phase** | Phase 3–4.2 (LLM calibration contract fix) — COMPLETE, review pending |
| **Last completed checkpoint** | Phase 3–4.2 LLM calibration contract fix (frequency rate normalization, ordinal assessment_status, narrative evidence/proportion contracts, strict strategy author consistency, expected-sample accounting) |
| **Current branch** | `feature/style-engine-v0.1` |

## What is functional
- Corpus pipeline: RAW → CLEAN → CHUNKS (1000/2000/4000) → METADATA/QC (deterministic, raw read-only).
- Feature Registry: 39 features, data-driven routing by analyzer name.
- Layer A deterministic analyzer: 22 features.
- Layer D stylometry: extraction + Burrows Delta + PCA + clustering + SVM/logreg validation (GroupKFold, held-out), leak-free grouped CV (per-fold vectorizer refit), function-word/word-unigram overlap audit.
- LLM provider abstraction (cacheable, unconfigured-safe).
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
- **114 tests passed** (was 97). New regression tests cover frequency rate
  normalization, ordinal assessment status, narrative evidence/proportion
  contracts, strict strategy author/work consistency, zero-verified-evidence
  rejection, and aggregation expected-sample accounting.

## Latest experiment results (deterministic, no LLM)
- Layer A: 2,328 TRAIN chunks × 22 features.
- Layer D: 954 features (154 fw + 400 char-3gram + 400 word-unigram, function
  words excluded from word-unigram; `n_function_word_overlap=0`).
- Grouped leave-one-work-out CV (SVM, class-weighted), **leak-free**:
  `[0.819, 0.924, 0.794, 0.905]` (mean ≈ 0.861; was leaky 0.884).
- Held-out accuracy: `0.745` (was 0.756).
- Calibration sample: 40 chunks (4 × 10), `seq`-ordered deterministic stratified,
  held-out excluded.

## Current blockers / review items
- **Awaiting review** of the Phase 3–4.2 LLM calibration contract checkpoint
  before any LLM spend. `candidate_core` features must remain un-promoted.

## Next planned action
- On approval: run the sampled LLM calibration (Layer A judgment/hybrid, B, C on
  the 40-chunk sample) with a configured, cache-backed provider; then aggregate
  evidence and re-evaluate before Phase 5.
