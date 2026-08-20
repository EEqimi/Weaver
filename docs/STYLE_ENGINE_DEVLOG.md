# Weaver Style Engine — Development Log

Canonical chronological record for the Style Engine / `knowledge/` subsystem.

**Role separation (do not conflate):**

| Document | Purpose |
|---|---|
| `STYLE_ENGINE_SPEC_V0.1.md` | What the system is *supposed* to become (target design). |
| `docs/STYLE_ENGINE_DEVLOG.md` | What has *actually* been implemented, tested, changed, and decided. |
| source code (`knowledge/`, `tests/`) | The implementation and its executable verification. |

Entries are append-only and ordered oldest → newest. Each entry records the state
at the time of its commit. Status values used: `COMPLETE`, `COMPLETE — REVIEW
PENDING`, `BLOCKED`, `SUPERSEDED`.

All experiment metrics below are reproducible from the committed code; generated
artifacts (`data/`) are gitignored. Reproducibility context (corpus role, chunk
size, feature set, analyzer/schema versions, seed/determinism) is recorded
inline where a metric appears.

---

## Checkpoint — Phase 1: Corpus Pipeline (RAW → CLEAN → CHUNKS → METADATA/QC)

**Status:** COMPLETE

### Goal
Deterministic, reproducible corpus ingestion that never mutates raw files and
emits three chunk granularities plus per-work metadata and QC.

### Implementation
- `knowledge/corpus/cleaner.py` — Gutenberg header/footer stripping, chapter
  heading detection, whitespace normalization.
- `knowledge/corpus/chunker.py` — paragraph-level chunking with overflow-merge
  (hard ceiling = `1.5 × target_chars`) and sentence-split carry-forward.
- `knowledge/corpus/discover.py` — maps files on disk to `WorkMetadata`.
- `knowledge/corpus/metadata.py` — explicit author/work/genre/year/role/filename
  table (no filename parsing; filenames contain typos).
- `knowledge/corpus/pipeline.py` — `build_corpus` orchestrator; writes
  `clean/`, `chunks/`, `metadata/`, `qc/` under `data/`.
- `knowledge/corpus/qc.py` — per-work + aggregate QC.
- `knowledge/config.py` — path derivation (`corpus_root` read-only, `data_root`)
  and `CHUNK_SIZES = (1000, 2000, 4000)`.

### Architecture decisions
- Raw corpus lives **outside** the repo (`wensigongfang/text/`), so copyrighted
  source text is never tracked. `data/` (all generated text) is gitignored.
- Corpus role (`train` / `held_out`) is explicit metadata, not inferred from
  filenames — this is the leakage-control anchor for later experiments.

### Corpus / data
- 6 pilot works: Austen — P&P (train), Emma (train), Persuasion (**held_out**);
  Dickens — Great Expectations (train), David Copperfield (train),
  A Tale of Two Cities (**held_out**).
- Chunk counts across sizes: `1000→5769`, `2000→2959`, `4000→1554`.

### Tests
- Chunker/cleaner/metadata/reproducibility/no-raw-mutation suites.
- See Phase 1–2.1 for the consolidated test count.

### Experimental results
- QC aggregate (after Phase 1–2.1 refinement): `residue_works=0`, `empty=0`,
  `tiny=1`, `oversized=0`, `duplicate=0`, `small_tail=419` (chapter-end tails,
  expected).

### Issues discovered / Fixes
- See Phase 1–2.1 refinement (chunk remainder handling).

### Known limitations / TODO
- One `tiny_chunks` (short final-chapter tail) remains; flagged by QC as
  expected, not auto-dropped.
- Manifest is hardcoded `CORPUS` tuple; to be migrated to a data-driven
  manifest file (JSON preferred, YAML if hand-authored) before scaling corpus.

---

## Checkpoint — Phase 2: Feature Registry (data-driven feature definitions)

**Status:** COMPLETE

### Goal
A single source of truth for candidate style features, so the analysis pipeline
never hard-codes feature branching.

### Implementation
- `knowledge/schema/feature_registry.py` — `FeatureDefinition` (frozen),
  `FeatureRegistry`, and `build_default_registry()`.
- Enums: `MeasurementType` (statistical/nlp/hybrid/judgment),
  `ValueType` (continuous/discrete/categorical/distribution),
  `ControlRole` (core/candidate_core/descriptive/diagnostic/experimental).
- `knowledge/schema/style_schema.py` — `FeatureValue` (value, raw_value,
  normalized_value, confidence, evidence, sample_count, variance, analyzer_id,
  analyzer_version, schema_version, provenance).

### Architecture decisions
- **Routing by `feature.analyzer` name** (MUST-3): `StatisticalAnalyzer`,
  `NlpAnalyzer`, `LlmFeatureAnalyzer`, `StylometricExtractor`. No `if feature_id
  == ...` branches in the pipeline.
- `core` role is reserved; V0.1 core candidates are all `candidate_core` and are
  **not** treated as validated (see Phase 3–4).

### Corpus / data
- 39 features total: 25 statistical (22 deterministic + 3 stylometric
  distribution), 6 nlp, 4 hybrid, 4 judgment.
- Control roles: 4 `candidate_core` (lexical_diversity, mean_sentence_length,
  mean_paragraph_length, dialogue_ratio), 18 descriptive, 14 experimental,
  3 diagnostic.

### Tests
- `tests/test_feature_registry.py` (registry integrity, candidate_core roles).

---

## Checkpoint — Phase 1–2.1 Refinement (chunker + terminology + portability)

**Status:** COMPLETE

### Goal
Nine review-driven refinements to close out Phases 1–2 before analysis begins.

### Implementation / changes
1. Chunk distribution audit (diagnosed small-chunk causes).
2. Long-paragraph regression tests (`tests/test_chunker.py`).
3. Chunker refinement: sentence-split remainders merge into the chapter-local
   paragraph buffer instead of becoming isolated small chunks.
4. Terminology: `char_count` (characters) kept distinct from `word_count`.
5. `candidate_core` provisional role (do not promote to `core` yet).
6. `FeatureValue.analyzer_version` added.
7. `source_path` made portable (`meta.filename`, relative to corpus root).
8. Cleaner extensibility documented.
9. Manifest YAML/JSON migration path documented (hardcoded `CORPUS` for now).

### Issues discovered
- Small chunks originated from **two** mechanisms: whole-paragraph greedy
  isolation (~63%) and sentence-split remainders (~37%).

### Fixes
- Paragraph overflow-merge (`hard = 1.5 × target`) plus sentence-split
  carry-forward; both mechanisms eliminated.

### Verification
- Added 4 long-paragraph tests; reran full Phase 1–2 suite and corpus QC.

### Tests
- **32 tests passed** (Phase 1–2 suite) at this checkpoint; QC clean.

---

## Checkpoint — Phase 3: Four-Layer Analysis Prototype

**Status:** COMPLETE — REVIEW PENDING (see current checkpoint below)

### Goal
Prototype the four analysis layers (A/B/C/D), build the LLM-provider
abstraction and the deterministic sampling mechanism **before** spending tokens
on LLM calibration.

### Implementation
- **Layer A deterministic:** `knowledge/analysis/statistical_analyzer.py`
  (22 features), `text_utils.py` (shared tokenizer/sentence/paragraph).
- **Layer A/B/C LLM analyzers:** `style_analyzer.py` (LLM feature judgment),
  `narrative_analyzer.py` (Layer B `NarrativeObservation`),
  `strategy_miner.py` (Layer C match + discover).
- **Layer D stylometry:** `knowledge/stylometry/{extract,delta,clustering,validation}.py`.
- **Provider abstraction:** `knowledge/providers/llm_provider.py`
  (`LLMProvider` protocol, `UnconfiguredLLMProvider`, `DummyLLMProvider`,
  `cache_key`, `LLMCache`, `CacheBackedLLMProvider`).
- **Strategies:** `knowledge/schema/strategy_schema.py`,
  `knowledge/strategies/registry.py` (monotonic lifecycle + 4 seeded candidates).
- **Sampling:** `knowledge/sampling/calibration.py` (deterministic stratified
  sample manifest).
- **Versions:** `knowledge/schema/versions.py` (analyzer/schema/aggregation/
  sampling version constants, all `0.1.0`).

### Architecture decisions
- **Blind analysis is the default** — author identity is never injected into
  LLM prompts (confirmation-bias safeguard, spec §11).
- **No LLM configured → explicit `AnalysisUnavailable`**, never fabricated
  results (spec §17.4).
- Cache key = SHA256(text) + analyzer_id + analyzer/schema versions + model +
  provider_id + prompt_name; results are cacheable and reproducible (spec §10.1).
- Strategy lifecycle is **monotonic** (`discovered → candidate → validated`),
  so seeded `candidate` strategies never regress to `discovered`.
- Only `reasoning_summary` + verbatim `evidence` are stored — no hidden
  chain-of-thought.

### Corpus / data
- Layer A deterministic + Layer D run on **TRAIN** works; held-out works are
  only ever `transform`-ed (never `fit`), and only as a validation test set.

### Experimental results (deterministic only — no LLM has run yet)
- Layer A: **2,328 TRAIN chunks × 22 deterministic features**.
- Layer D: `StylometricVectorizer` fit on 2,328 TRAIN chunks → **954 features**
  (154 function words + 400 char-3grams + 400 word-unigrams); 631 held-out
  chunks transformed.
- Grouped leave-one-work-out CV (SVM, class-weighted):
  **`[0.852, 0.922, 0.845, 0.916]`** (mean ≈ 0.884).
- Held-out accuracy (train→TRAIN, test→Persuasion + TOTC): **0.756**.
- Calibration sample: **4 works × 10 = 40 chunks**, stratified by
  position (early/middle/late) × dialogue (dialogue/mixed/narration),
  deterministic (no RNG). Held-out excluded.

### Issues discovered / Fixes
- **Class imbalance collapse:** initial grouped CV produced degenerate `0.0`
  fold accuracy because TRAIN chunk counts are imbalanced (Austen 833 vs
  Dickens 1,495). A linear SVM with default class prior collapsed to the
  majority class.
- **Fix:** `class_weight="balanced"` in `SVC`/`LogisticRegression`. This is a
  classifier-prior decision applied identically to train and held-out, not
  held-out tuning. Verified by rerun → CV above.

### Known limitations / TODO
- NlpAnalyzer (POS features) intentionally not implemented — NLTK not installed
  (spec §7.1 "POS optional/stub").

---

## Checkpoint — Phase 4: Profile Aggregation + Orchestration

**Status:** COMPLETE — REVIEW PENDING

### Goal
Feature-type-aware aggregation `ChunkProfile → WorkProfile → AuthorProfile` and
a deterministic orchestrator that runs everything *except* LLM calibration.

### Implementation
- `knowledge/profiles/aggregation.py` — type-aware aggregation:
  continuous → mean + variance + quartiles; categorical → category distribution;
  distribution → per-category mean; narrative enums → category distribution;
  narrative pace/detail → per-dimension mean.
- `knowledge/analysis/pipeline.py` — `run_deterministic_pipeline()` runs Layer A
  + Layer D + aggregation + sampling manifest and writes artifacts; **does not**
  call any LLM.

### Architecture decisions
- Aggregation preserves `sample_count` and distributions at every level — no
  "single-data-point-looks-authoritative" conclusions.
- Held-out works are excluded from work/author profiles (reserved for final
  validation only).

### Corpus / data
- 4 work profiles + 2 author profiles (TRAIN only).

### Experimental results
- See Phase 3 metrics; all produced by `run_deterministic_pipeline()`.

---

## Checkpoint — Phase 3–4 (verified deterministic state)

**Status:** SUPERSEDED (CV metrics replaced by the Phase 3–4.1 leak-free run
below). **STOPPED before the first sampled LLM calibration** (spec §13/§14). Do
not launch the 40-chunk LLM calibration or begin Phase 5 until reviewed.

Verified results (reproducible via `run_deterministic_pipeline()`):

- **Layer A deterministic:** 2,328 TRAIN chunks × 22 deterministic features. ✅
- **Layer D configuration:** `StylometricVectorizer` — 154 function words +
  char-3gram (`max_features=400`) + word-unigram (`max_features=400`); fit on
  TRAIN only.
- **Grouped leave-one-work-out CV (SVM):** `[0.852, 0.922, 0.845, 0.916]`.
- **Held-out accuracy:** `0.756`.
- **Class imbalance decision:** `class_weight="balanced"` (rationale above).
- **Calibration sample:** 40 chunks (4 × 10), deterministic stratified
  (position × dialogue), 9–10 chapters covered per work. **Held-out excluded.**
- **Profile aggregation:** 4 work + 2 author profiles, type-aware. ✅
- **Tests:** **85 passed** (32 Phase 1–2 + 53 Phase 3–4). ✅
- **No real LLM calls launched.** Provider unconfigured → `AnalysisUnavailable`.
- **Blind-analysis safeguard:** default on for all LLM analyzers. ✅
- **candidate_core:** 4 features remain `candidate_core`; **not** promoted to
  `core`. ✅
- **Generated artifacts (gitignored, under `data/analysis/`):**
  `chunk_profiles.jsonl`, `work_profiles.json`, `author_profiles.json`,
  `stylometry/{matrix.npz,index.json,baseline.json}`, `calibration_sample.json`.

---

## Current Checkpoint — Phase 3–4.1 Calibration Readiness Fix

**Status:** COMPLETE — REVIEW PENDING. **STOPPED before the first sampled LLM
calibration.** Do not launch the 40-chunk LLM calibration or begin Phase 5 until
reviewed. Do not merge or open a PR.

Code-review-driven fixes making the deterministic pipeline calibration-ready for
the sampled LLM run. Ten items addressed:

1. **LLM feature measurement scales** — `knowledge/schema/rubrics.py`:
   `MeasurementRubric` / `RubricRegistry` / `build_default_rubrics()`; two
   protocol families: `frequency` (LLM identifies instances, program counts
   *verified* instances) vs `ordinal` (anchored 0=absent … 4=dominant). Every
   LLM-derived feature declares `measurement_protocol` + `protocol_version`.
2. **Grouped-CV feature-selection leakage** — `grouped_cross_validation_texts()`
   refits `StylometricVectorizer` per fold (train works only); left-out work is
   transformed with that fold's vectorizer. Held-out Persuasion/TOTC stay
   train-fit-only.
3. **Calibration ordering** — sampling sorts by `seq` (globally monotonic), not
   the lexicographic chapter string (`"10" < "2"` bug).
4. **Aggregation preserves uncertainty/evidence** — work/author profiles keep
   `n_total` / `n_valid` / `n_missing`, an independent confidence summary (never
   averaged into value), evidence refs, analyzer ids/versions, schema versions,
   and chunk provenance.
5. **Evidence verification** — `knowledge/analysis/evidence.py` (NFC +
   punctuation normalization + whitespace collapse + substring match); applied
   to `LlmFeatureAnalyzer`, `NarrativeAnalyzer`, `StrategyMiner`; unverified
   quotes flagged, not silently dropped; high-confidence positives require
   verified evidence.
6. **Narrative validation** — proportions validated (numeric/range/keys/≈sum);
   explicit `unknown` / `insufficient_evidence` / `not_observable`; dataclass
   defaults no longer fabricate observations.
7. **Strategy evidence** — non-empty consistent author required before
   VALIDATED; match confidence + all valid quotes preserved; analyzer/schema
   provenance added.
8. **Stylometric family overlap** — word-unigram now excludes function words
   (`stop_words`), removing double-weighting; `family_overlap()` audit added.
9. **Regression tests** for every issue.
10. **Docs/Git** — this entry + STATUS update + regenerated manifest/baseline.

### Experimental results (deterministic, leak-free — no LLM run)
- Layer A: 2,328 TRAIN chunks × 22 deterministic features (unchanged).
- Layer D: `StylometricVectorizer` — 154 function words + char-3gram (400) +
  word-unigram (400, **function words excluded**); family-overlap audit:
  `n_word_unigram=400`, `n_function_word_overlap=0`, `overlap=[]`.
- Grouped leave-one-work-out CV (SVM, class-weighted):
  - **Old (leaky matrix CV, function-word overlap):** `[0.852, 0.922, 0.845, 0.916]`,
    mean ≈ 0.884.
  - **New (leak-free, per-fold vectorizer refit):** `[0.819, 0.924, 0.794, 0.905]`,
    mean ≈ 0.861.
  - Δ ≈ −0.023. The leak-free estimate is the honest bound; the difference
    reflects (a) per-fold refit (no train-vocab leakage) and (b) function-word
    de-duplication in the word-unigram family.
- Held-out accuracy (train→TRAIN, test→Persuasion + TOTC): **0.745**
  (was 0.756).
- Calibration sample: 40 chunks (4 × 10), `seq`-ordered deterministic stratified
  sampling; held-out excluded.

### Tests
- **97 tests passed** (was 85). New regression tests cover: measurement-protocol
  classification, frequency-vs-ordinal LLM contracts, evidence verification,
  calibration `seq` ordering, aggregation uncertainty/evidence preservation,
  word-unigram function-word exclusion, leak-free grouped CV.

### No real LLM calls launched.

---

## Current Checkpoint — Phase 3–4.2 LLM Calibration Contract Fix

**Status:** COMPLETE — REVIEW PENDING. **STOPPED before the first sampled LLM
calibration.** Do not launch the 40-chunk LLM calibration or begin Phase 5 until
reviewed. Do not merge or open a PR.

Focused contract fix tightening the LLM calibration semantics so that a future
sampled run cannot silently normalize away real signal, fabricate absence as
zero, or count unverified claims as evidence. Eight items addressed:

1. **True frequency normalization** — `metaphor_frequency`, `simile_frequency`,
   `irony_frequency` now store `raw_count` (verified instance count) and
   `exposure` (token count via the deterministic project tokenizer
   `text_utils.tokens`); `value = raw_count / exposure × 1000` (instances per
   1000 tokens). The program — never LLM output — performs normalization.
   `raw_count` / `exposure_tokens` / `unit` are preserved in provenance, and the
   unit (`instances per 1000 tokens`) is documented in the rubric.
2. **Distinguish absence from not-observable for ordinal features** — added
   `assessment_status ∈ {observed, insufficient_evidence, not_observable}`.
   When status ≠ observed, `level` must be null and `FeatureValue` preserves the
   state with `value = None` (never coerced to zero); level `0` means an actually
   observed absence only. Applied to `irony_intensity`,
   `narrator_evaluative_intervention`, `psychological_representation`,
   `emotional_restraint`, `emotional_intensity`. Aggregation counts unobservable/
   insufficient samples without pulling their means toward zero.
3. **Narrative evidence contract** — high-confidence (≥0.9) substantive
   narrative judgments must carry ≥1 verified evidence quote; otherwise the
   observation is deterministically downgraded to `confidence = 0.0` and tagged
   `high_confidence_substantive_without_verified_evidence` (never silently
   retaining 0.9+ with no verified evidence).
4. **Narrative proportion validation** — `temporal_pace` / `scene_detail`: every
   value ∈ [0,1]; non-empty distribution sums ≈1; all-zero non-empty is invalid/
   insufficient; unknown keys are reported (never silently renormalized or
   dropped). Regression tests cover value>1, negative, all-zero, sum≪1, and valid
   approximate.
5. **Strict strategy author consistency** — VALIDATED requires every counted
   evidence to have non-empty `author_id` **and** non-empty `work_id`, all counted
   author ids identical, and ≥2 distinct works. Missing author/work metadata no
   longer contributes to author-level validation. Regression test: one Austen +
   one empty-author across two works must **not** validate.
6. **Strategy evidence sufficiency** — a positive match/discovery with zero
   verified evidence never counts as lifecycle evidence regardless of confidence.
7. **Aggregation expected-sample accounting** — `n_total = n_expected` (chunks
   expected to receive feature analysis); preserved fields `n_expected`,
   `n_valid`, `n_missing`, `n_unobservable`, `n_insufficient`. Missing
   `FeatureValue`s are never fabricated.
8. **Tests + docs** — regression tests for all of the above; this entry + STATUS
   update.

### Version bumps (schema/analyzer/aggregation — analyzer_version stays separate
### from schema_version per spec §17.6)
- `NARRATIVE_SCHEMA_VERSION`, `LLM_ANALYZER_VERSION`, `NARRATIVE_ANALYZER_VERSION`,
  `STRATEGY_MINER_VERSION`, `AGGREGATION_VERSION` → `0.2.0`.

### Tests
- **114 tests passed** (was 97). New regression tests cover: frequency rate
  normalization (raw_count/exposure/unit), ordinal assessment status
  (observed/insufficient/not_observable + null-level contract + invalid status),
  narrative high-confidence downgrade/keep, narrative proportion bounds/all-zero/
  unknown-key, strict strategy author/work consistency, zero-verified-evidence
  rejection, and aggregation expected-sample accounting.

### No real LLM calls launched.

---

## Workflow (going forward)

1. Implement → 2. run tests → 3. run experiment if applicable → 4. inspect git
diff → 5. update this dev log → 6. update `docs/STYLE_ENGINE_STATUS.md` →
7. commit code + tests + docs together → 8. push the `feature/style-engine-v0.1`
branch → 9. STOP at any explicitly requested review checkpoint.

Never commit: raw corpus, generated clean/chunk text, secrets, caches, vector
stores, or machine-specific files (see `.gitignore`).
