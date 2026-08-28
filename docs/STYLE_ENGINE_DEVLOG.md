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

## Checkpoint — Phase 4.3: LLM Smoke Calibration (4 chunks)

**Status:** COMPLETE. Measurement-system validation only — **not** the 40-chunk
calibration. Still stopped before the full sampled calibration (spec §13/§14).

### Goal
Before spending on the full 40-chunk calibration, validate the real LLM backend
end-to-end on a deterministic 4-chunk sample (Layer A judgment/hybrid, B
narrative, C strategy match+discover) and produce an inspectable per-chunk report.

### Implementation
- `knowledge/providers/llm_provider.py` — `OpenAICompatibleProvider` (stdlib
  `urllib` DashScope compatible-mode backend, no third-party deps; key from env,
  never logged/persisted), `LLMTransportError` (transport vs schema-failure
  separation), `_error_detail` (reads HTTP error body), runtime metering
  (`n_calls`/`n_success`/`n_retries`/`usage`); `CacheBackedLLMProvider` counts
  `cache_hits`/`cache_misses`.
- `knowledge/analysis/strategy_miner.py` — optional `rejections` collector
  recording zero-verified-evidence and unknown-strategy rejections (default
  `None` = unchanged behavior).
- `knowledge/calibration/smoke.py` — `run_smoke_calibration()`: 4 chunks (one per
  TRAIN work, position×dialogue diverse, held-out excluded), smoke-only strategy
  registry (never writes back to the canonical registry), JSON + Markdown report.

### Architecture decisions
- Smoke is a **measurement-system check, not literary calibration**: no
  Work/Author profile synthesis, no strategy lifecycle promotion, no
  `candidate_core` promotion, no rubric/prompt edits driven by "literarily
  surprising" results.
- Blind analysis on (default), cache-backed, unconfigured-safe.

### Experimental results (real LLM — `qwen-plus`)
- 44 requests (11 × 4 chunks), **37 success**, 0 schema/JSON failures, 0 retries.
- **7 failures — all `HTTP 400 Arrearage`** (DashScope account overdue payment),
  on the last chunk; the account was cut off mid-run. Environmental, not code.
- Token usage: 42,349 total (33,220 input / 9,129 output).
- 6 strategy matches, 0 discoveries, 0 zero-evidence rejections, 0 narrative
  downgrades; evidence accounting: 114 verified / 18 unverified.
- Artifacts: `data/analysis/calibration/{smoke_results.json, smoke_report.md}`.

### Issues discovered / Fixes
- Provider dropped the HTTP error body → added `_error_detail`; an opaque
  "400 Bad Request" became a readable "Arrearage / overdue payment". A cache-backed
  re-run (37 hits) confirmed the 7 failures are deterministic billing denials.
- Smoke metric accumulator would `KeyError` on unexpected error types → `_bump`
  helper (safe dynamic-key increment).

### Tests
- **132 passed** (was 114). New tests: `OpenAICompatibleProvider` (configured/
  unconfigured, success+usage, 429 retry/backoff, permanent-4xx no-retry, error
  body capture), cache hit/miss counters, `StrategyMiner` rejection collector,
  smoke `_bump`/`_feature_report`.

### Blocker (next step)
- **DashScope account is in arrears (`Arrearage`)** — top up before the 40-chunk
  calibration, or the run will be denied partway.

---

## Checkpoint — Phase 4.3.1: Provider Switch (DashScope → DeepSeek)

DashScope（百炼）账户欠费（`Arrearage`）阻塞了标定；换成 DeepSeek（OpenAI 兼容，
`deepseek-chat`），无需改动任何 analyzer / rubric / prompt。架构本就厂商无关
（`LLMProvider` 协议 + 注入），这次只是把具体 provider 预设从 DashScope 换到
DeepSeek。

### Implementation
- `knowledge/providers/llm_provider.py` — 把 `OpenAICompatibleProvider` 泛化为
  通用 OpenAI 兼容传输层（显式参数 > 环境变量 `{ENV_PREFIX}_*` > 类默认），新增
  两个预设子类：`DeepSeekProvider`（`deepseek-chat` @ `https://api.deepseek.com`，
  环境变量 `DEEPSEEK_API_KEY`）与 `DashScopeProvider`（保留旧后端，供回退/对照）。
- `knowledge/calibration/smoke.py` — `run_smoke_calibration()` 改用
  `DeepSeekProvider()`（缓存键因 `provider_id`/`model` 变化而自动隔离，不混用旧
  DashScope 缓存）。
- `AI_coding/utils/config.py` 是独立的旧 App 配置，仍指向 DashScope，本次未动。

### Tests
- **136 passed**（was 132）。新增 `DeepSeekProvider`（默认值 / 无 key 不可用 /
  读 `DEEPSEEK_API_KEY`）与 `DashScopeProvider`（保留旧默认）预设测试。

### Blocker (next step)
- 需要设置 `DEEPSEEK_API_KEY`（当前未设），否则 `DeepSeekProvider.is_configured()`
  为 False，冒烟/标定会显式不可用。设好后重跑 4-chunk 冒烟，再跑 40-chunk 标定。

---

## Checkpoint — Phase 4.3.2: Smoke re-run on DeepSeek (clean)

**Status:** COMPLETE。4-chunk 冒烟在 DeepSeek 后端干净重跑通过，测量系统端到端
验证完成。仍 **不** 是 40-chunk 标定（spec §13/§14 仍停在标定前）。

### Goal
Phase 4.3.1 的阻塞是 `DEEPSEEK_API_KEY` 未设；设好后在 DeepSeek `deepseek-chat`
上重跑 4-chunk 冒烟，验证换后端后全链路（Layer A judgment/hybrid、B 叙事、C
策略 match+discover）端到端可用，并产出可检视报告。

### Experimental results (real LLM — `deepseek-chat`)
- 44 请求（11 × 4），**44 成功**，0 schema/JSON 失败，0 重试，0 传输失败。
- Token：39,174 in / 11,324 out / **50,498 total**（缓存 0 命中 / 44 未命中，全新后端）。
- 评估状态：observed=20，insufficient_evidence=0，not_observable=0。
- 证据：**161 verified / 6 unverified**（未静默丢弃）；叙事证据降级 0。
- 策略：9 匹配、7 发现、**1 次 zero-verified-evidence 拒绝**、0 未知策略拒绝。
  被拒项：`Physical gesture as psychological revelation`（conf=0.9，无已验证引文）
  ——Phase 3–4.2 的零证据拒绝契约在真实后端上首次生效。
- 对比旧 DashScope 跑：37/44（7 次欠费失败）→ **44/44**；0 发现 → 7 发现。
- Artifacts：`data/analysis/calibration/{smoke_results.json, smoke_report.md}`。

### Notes
- DeepSeek 缓存键与旧 DashScope 缓存隔离（`provider_id`/`model` 进入 `cache_key`），
  本次 0 命中 / 44 未命中属预期（全新后端首次跑）。
- 冒烟仍只验证测量系统，不做 Work/Author 聚合、不推进策略生命周期、不晋升
  `candidate_core`。

### Next step
- 40-chunk 采样标定（Layer A judgment/hybrid、B、C）——仍停在 spec §13/§14 复审
  检查点之后才启动。

---

## Checkpoint — Phase 4.4: 40-chunk Sampled LLM Calibration

**Status:** COMPLETE。第一次真实采样标定跑通——Layer A judgment/hybrid、B 叙事、
C 策略 match+discover 在 40-chunk 采样清单上端到端运行，策略生命周期写回规范注册表，
聚合出 Work/Author 画像。spec §13/§14 的"标定前复审"检查点已由用户放行。

### Goal
在 4-chunk 冒烟（测量系统验证）通过后，用缓存后端的 DeepSeek `deepseek-chat` 对
40-chunk 采样清单跑完整标定，产出可检视的 JSON/MD 报告、Chunk/Work/Author 画像与
策略注册表，为 Phase 5（作者级综合）铺路。

### Implementation
- `knowledge/calibration/calibrate.py` — `run_sampled_calibration()`：读采样清单，
  逐 chunk 跑 Layer A（8 LLM 特征）/ B / C（match+discover 经规范注册表，discover
  写回生命周期），构建 `ChunkProfile`，聚合 Work/Author 画像，产出
  `calibration_results.json` + `calibration_report.md` + `profiles/` +
  `strategy_registry.json`。复用 smoke.py 的 provider/缓存/计量与辅助函数。
- `knowledge/schema/strategy_schema.py` — `StrategyEvidence` 补 `strategy_id`
  （默认空串，兼容旧位置参数构造），`to_dict()` 一并序列化。
- `knowledge/analysis/strategy_miner.py` — match/discover 构造证据时传入
  `strategy_id`。

### Architecture decisions
- discover 注册策略时先以 `evidence=[]` 注册（避免证据双计），再把逐条证据经
  `record_evidence` 写回；否则策略已注册时证据不会被登记。
- 采样清单固定（position × dialogue 分层、`seq` 有序、held-out 排除），4 works × 10。

### Experimental results (real LLM — `deepseek-chat`)
- 440 请求（40 chunk × 11）；44 命中冒烟缓存、396 全新。token 未直接计量——首次
  运行在聚合阶段崩溃、计量未落盘；重跑为 100% 缓存命中（report 里 `requests=0` 即
  缓存重放）。按冒烟每请求均值（50,498/44 ≈ 1,148）估算：**~455k 全新 + 50.5k
  冒烟已付 ≈ ~505k total**。
- Layer A：320 调用 → 318 成功；**2 次高置信正向判定无已验证证据被拒**（非 JSON
  解析失败，是 Phase 3–4.2 task item 5/6 的证据强制契约）；0 unavailable、0 传输失败。
- Layer B：40/40；Layer C：match 40/40，discover 40/40。
- assessment_status：observed=198，insufficient_evidence=0，not_observable=0。
- 证据：**verified=1718，unverified=79**（未静默丢弃）；叙事证据降级 0。
- 策略：**148 匹配、71 发现、17 零证据拒绝、0 未知策略拒绝**。
- 注册表：**75 策略 = 9 validated / 29 candidate / 37 discovered**（自 4 个 seed
  candidate 起）。9 个 validated 例：`objectification_of_emotion`、
  `narrative_irony_through_free_indirect_discourse`、
  `free_indirect_discourse_for_moral_self_assessment`、
  `delayed_revelation_through_character_reaction` 等。
- 作者画像 LLM 特征均值（type-aware；个别 n=19 因上述 2 次证据拒绝）初步分化：
  `simile_frequency` austen 0.72 vs dickens 2.52；`metaphor_frequency` 13.95 vs 17.52；
  `irony_frequency` 8.92 vs 6.22。（40-chunk 仅标定样本，不足以下作者级结论。）

### Issues discovered / Fixes
- **首次运行在聚合阶段崩溃**：`Aggregator._count_strategy_evidence` 引用
  `StrategyEvidence.strategy_id`，但该字段此前不存在（确定性流水线从不填充
  strategy_evidence，故从未触发）。修复：补 `strategy_id` 字段并在 match/discover
  两处构造时传入。重跑（全部缓存命中、0 新增 token）通过。
- 计量未持久化：崩溃发生在报告写出前，token 用量未落盘；已按估算记录（见上）。

### Known limitations / TODO
- **discover 去重缺口**：`_to_strategy` 对同名策略用 description hash 追加后缀，
  跨 chunk 重新发现同一策略会生成不同 id，无法干净聚合（37 个 discovered 多为单
  chunk 单例）。Phase 5 前应改为稳定 id 去重。
- 40-chunk 仅标定样本，LLM 特征未对全语料运行。

### Tests
- **138 passed**（was 136）。新增 `StrategyEvidence.strategy_id` 携带 + work 画像按
  策略计数回归测试。

### Next step
- Phase 5（author-profile synthesis / 混合模型 / planner / generation loop），先解决
  discover 去重与全量 LLM 特征的可扩展性。

---

## Checkpoint — Phase 4.5: Author-scoped Strategy Consolidation（基础设施 + 输入产物）

**Status:** COMPLETE — **停在真正付费 consolidation 之前**（等待 review）。本阶段只建
两层 schema、consolidation 基础设施与作者级输入产物；**未调用任何付费 LLM**。

### Goal
Phase 4.4 暴露了策略发现的结构性问题：同一文学机制在不同 chunk 因 name/description/
trigger/operation/effect 措辞不同而被登记为多个 Strategy。本阶段引入"raw → canonical"
两层结构，把作者级合并做成**严格 author-scoped、不删除原始数据、可追溯**的基础设施，
并产出作者级 consolidation 输入产物供 review。

### Implementation
- `knowledge/schema/strategy_schema.py` — 新增两层结构：
  - `RawStrategy`（作者归一后的合并输入单元，携带作者范围内证据与 `source_strategy_ids`）；
  - `ConsolidationGroup`（LLM 返回的结构化分组映射，`from_dict`/`to_dict`）；
  - `CanonicalStrategy`（作者级规范化策略：`source_strategy_ids` / `supporting_chunk_ids`
    / `supporting_work_ids` / `evidence` / 支持计数 / `support_status`）；
  - `canonical_strategy_id(author_id, name)` → `"{author_id}::{slug}"`（只从 name 派生，
    不依赖 description 自由文本 hash）。
- `knowledge/strategies/consolidation.py` — `StrategyConsolidator`：
  确定性预处理（NFC + name/whitespace 归一 + 精确结构重复折叠）、`validate_author_scope`
  （越界/缺作者拒绝）、`validate_mapping`（恰好一次 / 无幻觉 / 无重复 / 无丢失）、
  `build_canonicals`（纯函数、完全可追溯）、`build_prompt`、`consolidate`（全流程，
  未配置 provider 显式不可用）。
- `knowledge/calibration/consolidation_input.py` — `build_consolidation_inputs()`：从
  `strategy_registry.json` 按作者分区，写出每作者 `consolidation_input.json`（含 prompt
  与 token/请求估算）+ 汇总 + 异常标注，**复用 Phase 4.4 结果，绝不重跑 analyzer**。
- `knowledge/schema/versions.py` — 新增 `CANONICAL_STRATEGY_SCHEMA_VERSION` 与
  `STRATEGY_CONSOLIDATOR_VERSION`（**不 bump** `STRATEGY_SCHEMA_VERSION`，以免让 Phase 4.4
  的 strategy match/discover 缓存键失效）。

### Architecture decisions（本阶段必须固化的边界）
1. **每个作者独立构建知识库** —— 不共享、不交叉；
2. **Raw Strategy 永久保留** —— `CreativeStrategy`/`StrategyEvidence` 是原始观察，绝不覆盖；
3. **LLM consolidation 只创建 canonical mapping，不删除原始数据** —— 输出结构化分组，
   不直接改注册表；
4. **consolidation 严格 author-scoped** —— 一次只处理一位作者，越界/缺作者一律拒绝；
5. **不同作者允许同名 canonical strategy** —— `austen::dramatic_irony` 与
   `dickens::dramatic_irony` id 不冲突；
6. **作者级 lifecycle 独立** —— canonical 的 `support_status` 只按该作者证据重算；
7. **新作者加入不要求重算已有作者** —— 作者知识库边界独立；
8. **本阶段不引入 vector database** —— FAISS/Chroma/Milvus 等属未来检索层，非 Phase 4.5；
9. **Phase 4.4 的 40-chunk 结果复用** —— 不重跑 analyzer。

### Data（复用 Phase 4.4 注册表，按作者分区）
- Austen：**51** raw strategies（candidate 24 / discovered 22 / validated 5 legacy），
  11 个跨 ≥2 作品；Dickens：**44**（candidate 23 / discovered 15 / validated 6 legacy），
  12 个跨 ≥2 作品。精确去重折叠：0（近重复策略因 `_to_strategy` 的 description-hash
  后缀而 description 不同，故不被"精确一致"折叠——这正是 LLM 语义合并要解决的）。
- 估算（单次 shot，DeepSeek `deepseek-chat`）：
  - Austen 1 请求，~13.1k input + ~3.1k output token；
  - Dickens 1 请求，~11.0k input + ~2.6k output token；
  - 无既有 consolidation 缓存。

### Data anomaly（发现并标注）
- 全局注册表生命周期**单调且跨作者**：20 个策略在两位作者名下都有证据；其中
  **2 个策略虽标 `validated`，却残留跨作者证据**（`delayed_revelation_through_character_reaction`、
  `interrogative_escalation_through_repeated_conditional_challenges`）——先在 Dickens 内
  达 validated，之后混入 Austen 证据因单调不降而残留。这印证了"必须先按作者分区再定级"；
  `author_scoped_support_status` 已在产物中按作者重算。

### Tests
- **151 passed**（was 138，+13）。覆盖：author scope 隔离 / 缺失作者拒绝 / 完整覆盖 /
  重复赋值拒绝 / 幻觉 id 拒绝 / 丢失 id 拒绝 / canonical 溯源（raw→chunk→work→evidence）/
  跨作者同名 canonical id 不冲突 / canonical id 稳定不依赖 description / 精确去重折叠 /
  不按名称近似做语义合并 / dummy 端到端 / consolidate 在调用 LLM 前拒绝跨作者输入。

### Next step（停在 review 检查点）
- Review `data/analysis/consolidation/{austen,dickens}_consolidation_input.json` 与
  `consolidation_summary.json`，确认 raw 数量、prompt、估算后，再放行真正付费
  consolidation（单次/作者）。尚未调用任何 LLM。

---

## Checkpoint — Phase 4.5.1: Consolidation-quality fix（输入提示词 + 严格校验 + id 稳定性文档）

**Status:** COMPLETE — REVIEW PENDING（仍未调用任何付费 LLM，停在 review 检查点）

### Goal
放行付费 consolidation 前的一轮质量修复：让 LLM 能基于「定义 + 真实文本证据」判断语义
同一性；让输出校验对非法字段显式报错而非静默兜底；并固化 canonical id 的长期稳定性约定。

### Implementation
1. **提示词加入紧凑支持/证据上下文**（`StrategyConsolidator.build_prompt`）：每个发送的
   `RawStrategy` 追加 `support: chunks=N works=M status=… [confidence=…]`，并附**至多 2 条
   已验证短引文**（每条 ≤80 字符、压缩空白、去重；`unverified_quotes` 绝不进入提示词）。
   让 LLM 从「触发/操作/效果」定义与「真实文本证据」两方面判断语义同一性，token 增长
   控制在 ~2–3k/作者。
2. **严格输出校验**（`ConsolidationGroup.from_dict`）：`canonical_name` /
   `canonical_description` 必须非空、`source_strategy_ids` 必须非空字符串列表、
   `confidence` 必须是 [0,1] 内数值（int/float，布尔除外）。缺失/非法字段抛 `ValueError`，
   `consolidate` 捕获后转 `LLMResponseError`；**绝不静默把非法 confidence 转成 None**
   （删除 `_safe_confidence` 静默兜底）。`reasoning_summary` 仍可选/简洁。
3. **固化 canonical id 长期稳定性**（`canonical_strategy_id` / `CanonicalStrategy` docstring）：
   `author_id::slug(canonical_name)` 仅用于首次派生；作者新增作品时**不得从零重建已持久化
   id**，正确流程是 reconciliation（persisted CanonicalStrategies + 新 RawStrategies →
   匹配旧 canonical 或新建，旧 id 稳定）。reconciliation 系统本阶段未实现。

### Data（重新生成输入产物，无 LLM 调用）
- 新增的 support/evidence 上下文只增 input 估算，output 估算不变（仍 = `n_prepared × 60`）：
  - Austen：**15,870 input** / 3,060 output（was ~13.1k / ~3.1k）；
  - Dickens：**13,242 input** / 2,640 output（was ~11.0k / ~2.6k）。
- raw 数量不变：Austen 51 / Dickens 44，精确去重折叠 0。

### Tests
- **158 passed**（was 151，+7）：prompt 含紧凑支持上下文与已验证引文、引文上限 2 条、
  空 canonical name 拒绝、空 canonical description 拒绝、confidence 非数值类型拒绝、
  confidence 越界（<0 / >1）拒绝、consolidate 把非法字段包装为 LLMResponseError。

### Next step
- Review 重新生成的三件产物，确认后放行付费 consolidation（单次/作者）。仍未调用任何 LLM。

---

## Checkpoint — Phase 4.5 (run): Author-scoped paid consolidation（真实 LLM 执行）

**Status:** COMPLETE

### Goal
放行付费 consolidation：对两位作者执行真实 `deepseek-chat` 请求，产出作者级 canonical
strategy 集合，完成 Phase 4.5 的落地产物（复用 Phase 4.4 注册表，绝不重跑 analyzer）。

### Result
- Austen：**51 raw → 26 canonical**（validated 7 / candidate 2 / discovered 17）。
- Dickens：**44 raw → 36 canonical**（validated 12 / candidate 5 / discovered 19）。
- 覆盖完整：Austen 51 / Dickens 44 个 raw id 全部映射，无遗漏、无幻觉、无重复。
- 产物：`data/analysis/consolidation/{austen,dickens}_canonical_strategies.json` +
  `consolidation_results.json` + `consolidation_report.md`。

### Bugs found & fixed（执行中暴露，均加回归测试）
1. **max_tokens 截断**：provider 默认 `max_tokens=2048` 把两份 consolidation JSON 截断
   （输出需 ~3k+ token）。修复：`StrategyConsolidator(max_tokens=8192)`，并把 `max_tokens`
   纳入 cache key（`extra`），避免同 prompt 不同上限互相污染缓存。
2. **LLM 覆盖遗漏**：Austen 首轮漏掉 2/51 个 raw id（`characterization_via_possessions`、
   `narrative_irony_through_free_indirect_discourse`）。修复：确定性覆盖修复 pass
   （`repair()` + `_merge_repair()`）——仅对遗漏 id 二次请求，按 canonical_name 精确
   匹配 merge 进已有组或新建组，再复验。修复后前者并入
   `austen::free_indirect_discourse_for_psychological_depth`，后者新建
   `austen::characterization_through_possessions`。

### Token / cost
- 最终运行计量（报告内）：1 次修复请求 + 2 次缓存命中，1690 in / 500 out。
- 含两次调试运行（截断 + 遗漏修复）实际累计 ~68k token（输入 ~50.3k / 输出 ~18.0k）。

### Tests
- **161 passed**（was 158，+3：max_tokens 透传、repair 并入已有组、repair 新建组）。

### Next step
- Canonical 作者策略集已就绪 → 进入 Phase 5（作者画像合成）。

---

## Checkpoint — Phase 4.5 (repair hardening): canonical identity keyed by id, not name

**Status:** COMPLETE

### Goal
收尾加固 Phase 4.5 的覆盖修复（`repair()`）：把「按 `canonical_name` 精确匹配」升级为
「按稳定 `canonical_strategy_id` 引用」，消除 name paraphrase 可能造成「误建新 canonical」
的长期稳健性风险。仅本地加固 patch；不改动已确认的 Austen / Dickens consolidation 结果，
不重跑任何真实 LLM，不启动 Phase 5。

### Implementation
- 新增 `RepairAssignment` dataclass + 严格 `from_dict`：显式区分 `merge_existing`
  （仅 `target_canonical_id`）与 `create_new`（完整 canonical 定义）；字段非法即抛
  `LLMResponseError`，绝不静默。
- `repair()` 改向已有组暴露 `canonical_strategy_id`（+ name / description / trigger /
  operation / effect 摘要），响应契约改为 `{"assignments": [...]}`。
- 新增 `_apply_repair()`（替换 `_merge_repair()`）：按 `canonical_strategy_id` 匹配 merge
  目标，**绝不按 name 匹配**；并确定性校验：跨作者 target 拒绝、幻觉 target 拒绝、重复
  分配拒绝、遗漏未覆盖拒绝、幻觉 raw id 拒绝。修复后仍跑 `validate_mapping` 复验。
- 约束保持：首次合并契约/缓存不变；author-scope 隔离 / provenance / support_status /
  canonical id 稳定性均未改动。

### Cache / version
- repair 的 prompt 与响应契约改变，故在 repair cache key 的 `extra` 加入
  `repair_contract_version=2.0`，杜绝旧 repair 缓存被新代码复用。
- 首次合并的 cache key（`prompt_name=strategy_consolidation:...`）完全不变 → 不无效化
  已有的 Austen/Dickens 首轮合并缓存（内容寻址本已区分，显式版本为双保险）。

### Tests
- **167 passed**（was 161）：新增/改写 8 个 repair 稳健性测试（merge_by_id、paraphrase
  不新建、create_new、幻觉 target id 拒绝、重复分配拒绝、遗漏未覆盖拒绝、幻觉 raw id 拒绝、
  跨作者 target 拒绝）；首轮合并行为不变由既有
  `test_consolidate_end_to_end_with_dummy_provider` 覆盖。

### Non-goals（本次明确不做）
- 未重跑 Austen/Dickens consolidation；未调用真实 LLM（测试全确定性）；未改动任何
  `data/analysis/consolidation/` 产物；未开始 Phase 5。

---

## Checkpoint — Phase 5: Author Profile Synthesis（作者风格画像合成）

**Status:** COMPLETE — REVIEW PENDING（最后停止，等待人工 review；未进入 Phase 6）

### Goal
把 Phase 1–4.5 已有测量结果**确定性合成**为统一的 `AuthorStyleProfile`，供未来 Style
Planner 使用。纯合成：不重新分析文本、不调用任何 LLM、不生成文章、不做风格混合、不修改
既有 consolidation 产物、不把 held-out 作品混入画像。

### Implementation
- 新模块 `knowledge/profiles/style_profile.py`：
  - `ProfileControlRole`（direct_control / conditional_control / diagnostic /
    reference_only）——**派生自**既有 `FeatureRegistry.control_role`，经确定性映射
    （core/candidate_core/descriptive→direct_control、diagnostic→diagnostic、
    experimental→reference_only、未知→reference_only），不新建第二套冲突 role system。
  - `GenerationControl` / `NarrativeControl` / `StrategyControl` / `AuthorStyleProfile`
    dataclass；generation control 携带 feature_id/value/control_role/confidence/sample
    support/variance/source/provenance。
  - `AuthorStyleProfileSynthesizer.synthesize(...)`：纯函数，无 I/O、无随机、无时间戳。
- 新 runner `knowledge/calibration/synthesize.py`：读回 4 份产物（全语料 author_profiles、
  sampled author_profiles、`{author}_canonical_strategies.json`、stylometry baseline+index），
  合成并落盘 `data/analysis/style_profiles/`。

### 关键设计决策
- **stylometric 指纹绝不进入 generation controls**：char 3-gram / function-word / PCA /
  Delta 统一归入 `diagnostics.stylometry`（control_role=diagnostic），只做生成后相似度诊断。
- **不确定性一等**：n_expected/n_valid/n_missing/n_unobservable/n_insufficient 全保留；
  not_observable / insufficient_evidence / missing **绝不伪造为 0**（mean 保持 null）。
- **sampled LLM 结果带 scope**：`source_scope` 区分 `full_train_corpus`（22 统计特征）与
  `calibration_sample`（8 LLM 特征 + 10 叙事维度），不表述为全语料确定真值。
- **canonical strategies → conditional_control + 确定性 control_priority**：排序键
  （降序）= support_status tier（validated>candidate>discovered）→ 跨作品数 → 跨 chunk 数 →
  confidence → raw observations → canonical id（稳定兜底）。绝不简单 1/0.5/0.1，绝不调用 LLM。
- **provenance**：canonical → raw strategies → chunks → works → evidence 链保留；明确
  global strategy_registry.status 为跨作者单调生命周期，不作为作者级结论。
- **确定性复现**：无随机/时间戳；每份画像带 `reproducibility_hash`（sha256，覆盖除 hash
  外的全部内容）。同输入重复合成 → 结构与 hash 完全一致。
- **held-out 隔离**：`author_scope.held_out_isolation` 显式校验 profile work_ids 与
  strategy supporting_work_ids 均 ⊆ train；Persuasion / A Tale of Two Cities 绝不进入画像。

### 版本
- `knowledge/schema/versions.py` 新增 `AUTHOR_STYLE_PROFILE_SCHEMA_VERSION = "0.1.0"`（独立
  版本，不影响既有 aggregation/consolidation 缓存与产物）。

### 产物（确定性合成结果）
- `data/analysis/style_profiles/{austen,dickens}_style_profile.json` +
  `style_profile_summary.json` + `style_profile_report.md`。
- Austen：generation 30（22 direct + 8 reference）、narrative 10、strategy 26（validated 7 /
  candidate 2 / discovered 17）、diagnostic 3 族、heldout_clean=True。
- Dickens：generation 30（22 direct + 8 reference）、narrative 10、strategy 36（validated 12 /
  candidate 5 / discovered 19）、diagnostic 3 族、heldout_clean=True。
- canonical 数量与 support_status 与 Phase 4.5 产物**完全一致**（26 / 36，未丢失任何策略）。

### Tests
- **182 passed**（was 167）：新增 15 个 Phase 5 测试（control-role 映射、diagnostic 不进
  generation、direct/conditional/reference-only 分桶、canonical 数量保持、support_status
  保持、不确定性不伪造 0、narrative not_observable 保留、full-corpus vs sampled scope、
  held-out 隔离（clean + 双通道污染检出）、strategy 优先级确定性 + 跨轮稳定、字节级复现）。

### Non-goals（本次明确不做）
- 未调用 DeepSeek / Qwen / 任何 LLM；未重跑 40-chunk calibration；未运行 full-corpus LLM；
  未做 Style Mixing / Planner / Prompt Compiler / 生成 / Revision Loop；未修改 Austen/Dickens
  canonical consolidation 结果；未清洗旧 global registry；未 candidate_core promotion；
  未安装 NLTK；未建 mixed-effects model。

---

## Checkpoint — Phase 5.1: AuthorStyleProfile 完整性 + 作者专属文体学目标

**Status:** COMPLETE — REVIEW PENDING（最后停止，等待人工 review；未进入 Phase 6）

### Goal
按 review 意见补强 Phase 5：(1) 作者专属 stylometric 目标（质心/离散度），而非全局实验
元数据；(2) held-out 隔离 fail-closed（污染即拒绝写出画像）；(3) provenance 路径去占位符；
(4) 稳定反序列化 + 校验 + 往返一致性 + hash 复核；(5) control-role 语义显式文档化。

### Implementation
- `knowledge/profiles/style_profile.py`：
  - 新增 `ProfileSynthesisError`（fail-closed）与 `ProfileSchemaError`（反序列化校验）。
  - `synthesize()` 增加 held-out 隔离 fail-closed：`clean=False` 即抛 `ProfileSynthesisError`，
    **绝不**返回/写出画像；隔离元数据仍保留在 clean 画像内。
  - `_build_diagnostics` 拆为 `diagnostics.stylometry.author_target`（作者专属）+
    `validation_metadata`（全局共享）；diagnostic 角色不变。
  - `_build_strategy_controls` / `_build_provenance` 接收 `author_id`，把
    `{author_id}` 占位符解析为具体 `austen` / `dickens` 路径（非绝对路径）。
  - 新增 `AuthorStyleProfile.from_dict` + 嵌套 `GenerationControl/NarrativeControl/
    StrategyControl.from_dict`（`_require` 校验必填字段；schema_version 不匹配即抛
    `ProfileSchemaError`）；新增 `verify_reproducibility_hash()`。
  - control-role 语义显式注释（core/candidate_core/descriptive/experimental/diagnostic
    对 Phase 6 的激活约定），**不实现 Style Planner**。
- `knowledge/calibration/synthesize.py`：
  - 新增纯函数 `_author_targets_from_matrix(X_train, train_authors, train_works,
    stylometry_version)`：签名只收 TRAIN 侧数据，从类型上杜绝 held-out 参与；计算作者
    质心（raw 均值）、离散度（raw 标准差）、`mean_within_author_cosine_distance`，产出
    `full`（落盘载体，954 维质心/离散度）与 `compact`（画像内紧凑标量 + 产物引用）。
  - `_compute_stylometric_author_targets` 只读 `matrix.npz` 的 `X_train` + `index.json`
    的 `train_*` 字段，绝不触碰 `X_heldout` / `heldout_*`。
  - `synthesize_style_profiles` 改为**先全部内存合成，成功后统一落盘**：任一作者
    fail-closed 则整体不写出任何画像/汇总/报告/目标产物。
  - 落盘新增 `stylometric_author_targets.json`（`no_held_out=True`、`fit_scope=train_only`）。

### 作者专属文体学目标表示（compact，profile 内）
`diagnostics.stylometry.author_target`：author_id / n_samples / source_work_ids /
stylometry_version / feature_dim / fit_scope=train_only / normalization / vectorizer
provenance / centroid_norm / mean_dispersion / mean_within_author_cosine_distance /
artifact（引用 `stylometric_author_targets.json`）+ artifact_keys（centroid/dispersion）。
954 维质心/离散度向量不进入画像 JSON，也不进入人类可读 report（仅落独立产物）。

### 产物（确定性重建结果）
- Austen：n_samples=833、source_work_ids=[emma, pride_and_prejudice]、centroid_norm=
  0.098128、within_cosine=0.174174；Dickens：n_samples=1495、source_work_ids=
  [david_copperfield, great_expectations]、centroid_norm=0.103634、within_cosine=0.155668。
- 两作者文体学目标**互异**（centroid/within-distance 均不同）；canonical 26 / 36 保持；
  held-out 隔离 clean=True；画像内无 `{author_id}` 占位符；`verify_reproducibility_hash()`
  = True。

### Tests
- **191 passed**（was 182）：新增 9 个 Phase 5.1 测试（diagnostics 拆分、作者目标互异、
  concrete provenance 无占位符、纯函数 TRAIN-only 目标计算、真实产物 Austen/Dickens 目标
  互异、往返序列化精确相等、reload 后 hash 复核、错误 schema_version 拒绝、缺字段拒绝）。

### Non-goals（本次明确不做）
- 未调用任何 LLM / 未重跑 chunk analysis / 未重跑 calibration / 未修改 Phase 4.5 canonical
  结果 / 未实现 Style Planner / Prompt Compiler / style mixing / 生成 / revision loop。

---

## Checkpoint — Phase 6: Style Planner & Prompt Compiler（画像 → 计划 → 提示词）

**Status:** COMPLETE — REVIEW PENDING（停在 review 检查点，未进入 Phase 7）

### Goal
建立三层严格分离的确定性管线：`AuthorStyleProfile`（观察到了什么，只读）→
`StylePlanner` → `StylePlan`（本次激活哪些控制）→ `PromptCompiler` → 生成提示词。
绝不把画像 JSON 直接塞进提示词；绝不在提示词中提作者名 / 模仿；绝不写微观 stylometric
指令；绝不改写用户 core story facts / 人物关系 / 约束。

### Implementation（`knowledge/planning/`）
- `schema.py`：`WritingRequest`（content/desired_length/target_words/language/pov/
  constraints，`__post_init__` 校验非空）、`ActivationLevel`（strong/medium/weak/
  reference/suppressed 有限枚举，无伪连续权重）、`PlannedControl` / `PlannedNarrativeControl`
  / `PlannedStrategy`、`PlannerPolicy`（可配置预算 + candidate_core 门槛）、`StylePlan`
  （style_plan_id/schema_version/author_id/source_profile_hash/writing_request/
  language_controls/narrative_controls/strategy_controls/reference_controls/
  reference_strategy_controls/suppressed_controls/warnings/planner_metadata），
  `make_style_plan_id`（sha256 确定性 id）。
- `policy.py`（纯函数，唯一政策权威）：
  - `language_activation`：diagnostic→suppressed（绝不控制生成）；core→strong（预留）；
    experimental→reference（仅 40-chunk LLM 采样）；candidate_core→确定性门槛
    `_gate_candidate_core`（n_valid/n_expected 完整度、missing/insufficient/unobservable、
    zero-variance、source_scope、相对离散度），**绝不晋升 core**；descriptive→weak。
  - `assign_language_buckets`：primary/secondary/reference/suppressed 四桶 + 预算，
    超出进 suppressed 且 reason=`suppressed_due_to_budget`（不静默）。
  - `narrative_activation` + `apply_narrative_budget`：sampled（40-chunk）→ 不超 medium；
    用户显式 pov 覆盖作者倾向（overridden=True，不 reject）。
  - `select_strategies`：validated > candidate > discovered；discovered 默认 reference，
    溢出进 reference 不丢弃。
  - 数值→自然语言 banding（`describe_feature` / `describe_narrative`，English），
    如 dialogue_ratio 0.43→"dialogue is prominent…"、0.10→"dialogue is relatively sparse…"。
- `planner.py`：`StylePlanner.plan` 先做画像完整性校验（`verify_reproducibility_hash()`
  + held-out 隔离 clean，否则抛 `PlanningError` fail-closed），再合成三路控制与 warnings
  （candidate_core≠core 提示、pov 冲突提示）、planner_metadata（预留 conflicts/
  resolution_required，单作者恒空）。
- `compiler.py`：`PromptCompiler.compile` → `CompiledPrompt`（ROLE / CONTENT / STYLE
  CONTROL / NARRATIVE / CONDITIONAL STRATEGIES / IMPORTANT 六段）。策略渲染为
  `WHEN trigger → THEN operation → TO effect` 条件规则（剥离 trigger_summary 开头的
  "When"）；确定性预算截断（超出 max_prompt_chars 从末尾策略起丢弃并记录）。
- `run.py`：加载 Austen/Dickens 画像（from_dict + hash + held-out 校验）→ 同一中性
  WritingRequest → plan + compile → 落盘 `data/analysis/planning/` 对比产物 + 报告。
- `versions.py`：新增 `WRITING_REQUEST_SCHEMA_VERSION` / `STYLE_PLAN_SCHEMA_VERSION` /
  `STYLE_PLANNER_VERSION` / `PROMPT_COMPILER_VERSION`（均 0.1.0，独立，不影响既有缓存）。

### 激活政策（关键约定）
- core → strong（未来正式核心；V0.1 无 core）。
- candidate_core → 门槛 gate（不晋升 core）：full-corpus + 完整 + 稳定 → strong；
  sampled scope / 高相对离散 → medium；完整度 < 0.5 / 证据不足 → weak 或 suppressed。
- descriptive → weak（辅助）；experimental → reference_only；diagnostic → 绝不在提示词。
- 策略：validated/candidate → active（≤ max_strategies）；discovered → reference（不主动）。

### 产物（确定性，同一 brief、不同画像）
- Austen：10 激活语言控制（4 candidate_core strong + 6 descriptive weak）、4 叙事激活
  （third-person / low narrator presence / internal focalization / close distance）、
  6 active 策略（Free Indirect Discourse 等）、8 reference、12 suppressed；
  提示词 4834 chars、未截断。
- Dickens：同样 10/4/6 结构，但方向相反——dialogue_ratio "sparse"（vs Austen
  "prominent"）、first-person（vs third）、comma_density "dense"（vs moderate）、
  mean_paragraph_length "medium"（vs "longer paragraphs"）；策略为
  Character revelation through dialogue / Objectification of emotion 等；5098 chars。
- 落盘：`data/analysis/planning/{austen,dickens}_{style_plan,compiled_prompt}.json`、
  `{austen,dickens}_compiled_prompt.md`、`planning_comparison_report.md`、
  `planning_summary.json`。

### Tests
- **223 passed**（was 191）：新增 32 个 Phase 6 测试（schema 往返、空 content 拒绝、
  激活政策 7 例、语言/叙事/策略预算、experimental→reference、POV 覆盖 + 无冲突时不警告、
  hash/held-out fail-closed、plan/prompt 确定性、提示词六段、不提作者名 / 不写微观
  stylometric、保留用户 brief、POV 覆盖写入提示词、预算截断、真实产物 Austen/Dickens
  计划互异 + 提示词不提作者名）。

### Non-goals（本次明确不做）
- 未调用任何 LLM（DeepSeek/Qwen/其他）；未生成任何小说正文；未实现 evaluation /
  revision loop（Phase 7+）；未实现多作者风格混合（conflicts 结构仅预留）。

---

## Checkpoint — Phase 6.1: Evidence-Grounded Guidance & Prompt Budget Integrity

**Status:** COMPLETE — REVIEW PENDING（停在 review 检查点，未进入 Phase 7）

### Goal
在 Phase 6 已通过架构 review 之后、任何真实生成调用之前，做一次聚焦修正：把语言控制
guidance 从"人工绝对阈值 + 超出测量的文学解释"改为"TRAIN-only 经验 band + 字面指令"；
把提示词预算从"硬截断"改为"确定性降级（绝不截断用户内容）"；并补齐 section 一致性
与缺失值文档。**未调用任何 LLM，未生成任何正文，未修改 AuthorStyleProfile 测量。**

### 变更 1：移除人工阈值 → TRAIN-only 经验 band（spec §1）
- 删除 `policy.py` 里的 `_FEATURE_BANDS`（dialogue_ratio 0.2/0.4、mean_sentence_length
  15/22、comma_density 60/90 等手选绝对阈值）与 `describe_feature`。
- 新增 `knowledge/planning/bands.py`：
  - `compute_band_thresholds(chunks, train_work_ids)`：TRAIN chunk-level 分布 →
    确定性分位数 → band 阈值（`low < Q33`、`medium ∈ [Q33, Q67]`、`high > Q67`）。
    分位数用线性插值（等价 numpy `linear`），值 round 到 8 位保证持久化稳定。
  - **TRAIN-only 保证**：`train_work_ids` 白名单在签名层显式排除非 TRAIN chunk；
    held-out 绝不参与；`run.py` 用两位作者画像的 `author_scope.train_work_ids` 并集
    作为白名单，fail-closed。
  - 阈值持久化 + 版本化：`data/analysis/planning/band_thresholds.json`
    （`schema_version=BAND_SCHEMA_VERSION=0.1.0`，独立版本不影响既有缓存）。
  - `band_label` / `describe_feature`：无阈值或未知 feature → 返回 `None`
    （`not_compilable`），绝不编造标签。
- **跨作者合并阈值**（关键决策）：不是 per-author，而是 4 部 TRAIN 作品的 2,328 个 chunk
  **合并**求 Q33/Q67。原因：per-author 分位数会让 Austen/Dickens 各自都读成 "medium"，
  抹掉对比的目的；合并阈值才能保留两位作者"同一 band 系统下方向相反"的区分。

### 变更 2：guidance 只字面描述测得什么（spec §2）
- 新增 `_LITERAL_GUIDANCE`（22 个统计特征 × {low/medium/high} 字面指令）：
  `comma_density high → "Use commas relatively frequently."`、
  `semicolon_density high → "Use semicolons relatively frequently."`、
  `mean_sentence_length high → "Favor relatively long sentences."`、
  `dialogue_ratio high → "Use dialogue relatively often."` 等。
- **移除全部未测量的文学机制**：不再有 "many subordinate clauses and parenthetical
  insertions"、"paired and antithetical constructions"；不再硬编码 Austen/Dickens
  专属文学知识。
- `describe_feature` 返回 `None` 时，`planner.py` 把本可激活（strong/medium/weak）的
  控制**降级为 reference**，reason=`not_compilable: no empirical band threshold`。

### 变更 3：预算绝不截断用户内容（spec §3）
- 删除 `compiler.py` 里的 `text = text[:max_prompt_chars]` 硬截断。
- 确定性降级顺序（每步都记录进 `removed_controls`，绝不静默）：
  1. 丢弃最低优先级条件策略（从末尾起）；
  2. 丢弃 secondary 语言控制（从末尾起）；
  3. 丢弃最弱语言控制（weak → medium → strong，同层取末位）；
  4. 移除可选解释措辞（ROLE / IMPORTANT 的精简变体）；
  5. 强制内容仍放不下 → 抛 `PromptBudgetError`（绝不切 CONTENT）。
- `CompiledPrompt` 字段改为 `degraded` / `removed_controls` / `degradation_note`
  （取代误导性的 `truncated` / `truncation_note`）。

### 变更 4：section 一致性（spec §4）
- `compile` 现在把 6 段全存入 `sections`，`text` 直接由 `_assemble(sections)` 派生，
  因此 `_assemble(prompt.sections) == prompt.text` 恒成立（有回归测试）。

### 变更 5：POV 移到 CONTENT-only + 缺失值文档（spec §5）
- 移除 NARRATIVE 段里重复的 POV 覆盖行（"explicit user requirement"）；POV 只出现在
  CONTENT。缺失值语义在 `bands.py` 与 `planner.py` docstring 明确：全缺 / `n_valid=0`
  → suppressed；部分缺失经 `completeness` 贡献（不自动 suppress）。

### 产物（确定性，同一 brief、不同画像，无 LLM）
- `data/analysis/planning/band_thresholds.json`：22 特征 × {q33, q67, n, min, median,
  max}，train_only=True，2,328 TRAIN chunks，4 部 TRAIN 作品，held-out 排除。
- Austen 语言控制 guidance（经验 band）：dialogue_ratio "Use dialogue relatively
  often."（high）、lexical_diversity "Vary vocabulary relatively widely."（high）、
  mean_paragraph_length "Favor relatively long paragraphs."（high）、comma_density
  "Use commas relatively rarely."（low）、semicolon_density "Use semicolons
  relatively frequently."（high）。
- Dickens 语言控制 guidance：dialogue_ratio "Use dialogue in moderate proportion."、
  lexical_diversity "Use moderate lexical variety."、mean_paragraph_length "Use a
  moderate paragraph length."、comma_density "Use commas moderately."、
  semicolon_density "Use semicolons moderately."
- 与 Phase 6 对比：原来 Austen dialogue "prominent"/Dickens "sparse"（文学解释）→ 现为
  字面 "relatively often"/"in moderate proportion"；原来 "dense commas, many
  subordinate clauses…" → 现为 "Use commas moderately/frequently."
- 提示词：Austen 4826 chars、Dickens 5055 chars，均未降级（degraded=False）。

### Tests
- **236 passed**（was 223）：新增 13 个 Phase 6.1 测试 —— TRAIN-only band（held-out
  排除 + 不改变阈值）、确定性、跨作者合并（一份阈值非 per-author）、band_label 三档
  边界、字面 guidance 无未测机制（comma/semicolon/long-sentence）、无 band → None、
  not_compilable→reference、长内容永不硬截断（多档预算循环）、低优先级先于强制内容
  移除、强制溢出抛 `PromptBudgetError`、sections 精确重构 text、真实 band_thresholds
  TRAIN-only 校验。

### Non-goals（本次明确不做）
- **未调用任何 LLM；未生成任何正文；未修改 AuthorStyleProfile 测量；未启动 Phase 7。**

---

## Checkpoint — Phase 7: Style-Conditioned Generation（第一次真实作者风格生成实验）

**Status:** COMPLETE

### Goal
把 Phase 6 的 CompiledPrompt 交给真实生成模型，产出 Austen 条件 / Dickens 条件两段
正文（GeneratedPassage）。同一 WritingRequest、同一模型、同一生成参数；**唯一变量**是
画像导出的风格控制。绝不自动评价（Phase 8）、绝不自动改写、绝不在 prompt 注入作者名。

### Implementation
- `knowledge/generation/schema.py` — `GeneratedPassage` / `GenerationResult` /
  `GenerationUsage` / `GenerationParameters`；`compiled_prompt_hash`（sha256）、
  `assert_no_author_leakage`（fail-closed）、`make_generation_id`（确定性）。与
  analysis 的测量 schema **严格分离**（spec §5）。
- `knowledge/generation/provider.py` — `GenerationProvider`（复用
  `OpenAICompatibleProvider.complete_with_metadata` 的同一 HTTP 传输，绝不另写第二套
  client）+ `DummyGenerationProvider`（测试，零 token）。
- `knowledge/generation/run.py` — `run_plumbing`（exactly ONE Austen 验证请求）与
  `run_generation`（正式 Austen + Dickens，fresh request，不藏 cache）+ 对比报告 +
  汇总。
- `knowledge/providers/llm_provider.py` — 拆出 `complete_with_metadata`（返回 content
  + finish_reason + per-call usage），新增 `top_p` 透传与只读 `base_url`。
- `knowledge/planning/compiler.py` — `IMPORTANT` 段改写，实际 prompt 不再含
  `imitate` / `write like` / `in the style of`（守卫语义保留：不命名 / 不复现具名作者
  风格）。
- `knowledge/schema/versions.py` — `GENERATION_SCHEMA_VERSION=0.1.0`、
  `GENERATION_VERSION=0.1.0`。

### 生成参数（两位作者严格一致）
- provider `deepseek` / model `deepseek-chat` / endpoint
  `https://api.deepseek.com/chat/completions`；`temperature=0.8`、`top_p=0.9`、
  `max_tokens=2048`；独立 `experiment_id=phase7-generation-v0.1`，无 LLMCache（每次
  均为 fresh request）。

### 无作者名注入（铁律，spec §8）
- 实际 prompt 不含 `Jane Austen` / `Charles Dickens` / `write like` / `imitate` /
  `in the style of`；作者 ID 只在 metadata。`assert_no_author_leakage` 在发送前
  fail-closed 校验。

### Plumbing request（真实请求前，exactly ONE Austen）
- finish_reason=`stop`、`n_retries=0`、fresh_request=true。
- token：prompt 917 / completion 797 / **total 1714**；正文 648 words（500–800 区间）。

### 正式生成（Austen + Dickens，各 1 次 fresh request）
- Austen：702 words，finish=`stop`，token prompt 917 / completion 871 / total 1788。
- Dickens：503 words，finish=`stop`，token prompt 982 / completion 620 / total 1602。
- 两段 prompt 互异（Austen hash `338faf8e…`、Dickens `b8757f34…`）；风格控制不同
  （同一 brief）。
- 基本检查（非文学评价）：Austen 第三人称叙事、Dickens 第一人称叙事，均原创内容、
  无作者名、无复制源文本。

### 产物（`data/analysis/generation/`，gitignored）
- `generation_experiment.json`、`{austen,dickens}_generation.json`、
  `{austen,dickens}_passage.md`、`generation_comparison_report.md`、
  `generation_summary.json`、`generation_plumbing.json`。

### Tests
- **254 passed**（was 236）：新增 18 个 Phase 7 测试（Dummy provider，零 token）——
  GenerationResult/usage/参数序列化、GeneratedPassage 往返 + finish_reason、空生成
  拒绝、prompt hash 正确 + 敏感、generation_id 确定性、作者名/模仿令牌泄露检出、
  编译 prompt 无作者名无模仿、provenance 保存、同一 WritingRequest 共享、
  provider/model/参数一致、未配置 provider fail-closed、artifact 布局 + 无自动评价
  / 无自动改写 + 铁律令牌集合。

### Non-goals（本次明确不做）
- **不自动评价（Phase 8）、不自动改写正文、不写新故事、不用 Austen/Dickens 原文情节、
  不打印/保存/提交 `DEEPSEEK_API_KEY`。**

---

## Checkpoint — Phase 7.1: Provenance / Integrity Hardening（零 token）

Phase 7 review 通过后、Phase 8 之前的 provenance / 完整性加固。**零 token、不调用
DeepSeek、不生成/不评价/不改写任何正文**；既有 Austen/Dickens 生成产物保持原样、
**绝不重生成**。

### 1. 生成身份模型（条件 vs 结果分离）
- `generation_condition_id`：作者 / style_plan / prompt hash / provider / model /
  参数的确定性 hash——标识"这次生成的条件"（同一条件多次随机抽样，条件 id 不变）。
- `generation_id`：具体生成结果的 identity = `condition_id + experiment_id + output
  hash (+ provider request id)`。同条件下两次 fresh 抽样若正文不同 → output hash
  不同 → `generation_id` 不同。**绝不依赖当前时间**。
- `GenerationResult` 新增 `request_id`；`GeneratedPassage` 新增
  `generation_condition_id`（必填）与 `request_id`。
- 向后兼容：`from_dict` 对缺 `generation_condition_id` 的 Phase 7 旧产物回退到旧
  `generation_id`（当时它其实就是条件 id），**绝不要求重生成**。

### 2. Plumbing gate（fail-closed）
- `run_generation()` 在正式生成前强制校验 plumbing 记录：文件存在、`plumbing` /
  `success` 为 True、正文非空、`finish_reason ∈ {"stop"}`、provider/model 匹配、
  `generation_parameters` 一致、`fresh_request=true`、`cache_hit=false`。任一违反 →
  `GenerationError`（绝不 "merely record plumbing=None"）。

### 3. Markdown 渲染修复
- `_render_passage_md` 的 `{p.experiment_id}` / `{p.generation_id}` /
  `{p.schema_version}` / `{p.generation_version}` 行补 `f` 前缀（此前漏了 f-string，
  渲染出字面量占位符），并新增 `generation_condition_id` / `request_id`。
- 新增测试：渲染 Markdown 含真实 ID、无未解析 `{p.` 占位符。

### 4. 泄露守卫 A/B 分离（数据驱动，非硬编码）
- 删 `BANNED_AUTHOR_LEAK_TOKENS` / `assert_no_author_leakage`，拆为：
  - A. `assert_no_imitation_instruction`（`write like` / `imitate` / `in the style
    of`）——**只查我们生成的风格控制指令**（非 CONTENT section），绝不查用户 brief
    正文（`imitate` 是普通英语动词，可合法出现在故事情节里）。
  - B. `assert_no_author_identity`（作者显示名名单）——名单来自 author metadata
    （`author_display_names()`），非硬编码 Austen/Dickens，新增作者无需改守卫代码。
- 风格控制指令在 `_build_passage` 时入 `provenance["style_control_text"]`，渲染阶段
  据此复检，无需重跑 planner/compiler。

### Tests（全确定性，Dummy provider，零 token）
- **267 passed**（was 254）：+13 Phase 7.1 测试——同条件不同正文 → 不同
  `generation_id`；同 prompt/参数 → 同 `generation_condition_id`；缺 plumbing 阻塞
  正式生成；失败/不匹配 plumbing 阻塞；合法 plumbing 放行；Markdown 含已解析元数据；
  泄露守卫支持未来作者身份；用户正文合法 "imitate" 不误报作者身份泄露。

### Non-goals（本次明确不做）
- **不调用 DeepSeek、不生成新正文、不评价/不改写既有 Austen/Dickens 段落；
  既有 Phase 7 产物保持原样（generation_condition_id 通过向后兼容回填，绝不重生成）。**

---

## Checkpoint — Phase 8: Style Feedback Loop + 独立 LLM 文学评价（真实 deepseek-chat）

**Status:** COMPLETE

### Goal
闭合 spec §15 反馈环：对 Phase 7 生成的 Austen/Dickens 正文再测量为 Actual Style
Profile（Layer A 统计+判断 / B 叙事 / C 策略 / D stylometric）→ 目标 vs 实际偏差 →
优先化改写计划（P0–P4）→ 最小编辑改写 → 再分析 → stylometric 诊断 → 确定性
Accept / Continue / Roll Back；并加**独立**的 6 维 LLM 文学评价（1–10 + 证据引文）。

### New files
- `knowledge/evaluation/schema.py` — 纯数据 schema（to_dict/from_dict + schema-version
  守卫，镜像 planning/schema.py）：`ActualStyleProfile` / `FeatureDeviation` /
  `NarrativeDeviation` / `StrategyCoverage` / `ComparisonResult` / `DimensionScore` /
  `LiteraryEvaluation` / `RevisionItem` / `RevisionPlan` / `RevisionResult` / `EvalError`；
  常量 `LITERARY_DIMENSIONS`（6 维）、`DEFAULT_DIMENSION_WEIGHTS`、`REVISION_PRIORITIES`、
  `CATEGORY_TO_PRIORITY`。
- `knowledge/evaluation/literary.py` — `LiteraryEvaluator`（LLM，盲测）：6 维各
  score 1–10 + summary + strength + weakness + evidence（逐字校验，未验证引文显式
  丢弃）；加权总分；盲测 system prompt。
- `knowledge/evaluation/analyze.py` — `measure_actual_profile`：复用既有 analyzer
  （`StatisticalAnalyzer` 22 维 + `LLMFeatureAnalyzer` 8 维 + `NarrativeAnalyzer` +
  `StrategyMiner`），Layer D 在 TRAIN chunk 上**重拟合** `StylometricVectorizer` 并与
  持久化 `index.json` 的 `feature_names` 严格比对（fail-closed），对作者质心算余弦距离。
- `knowledge/evaluation/compare.py` — `compare_target_actual`（纯函数）：语言 band
  偏差（low/medium/high → on/above/below/not_measurable）、叙事字段偏差、策略覆盖。
- `knowledge/evaluation/revision.py` — `build_revision_plan`（纯函数，P0>P1>P2>P3 有序，
  P4 恒无改写项）+ `RevisionRewriter`（LLM 最小编辑改写，盲测，P0 保护强制入 prompt，
  A/B 泄露守卫 fail-closed）。
- `knowledge/evaluation/run.py` — `run_evaluation` 编排 + report/summary 渲染 +
  `decide_feedback_outcome`（纯函数，确定性 Accept/Continue/Roll Back）。
- `knowledge/schema/versions.py` — 新增 `EVALUATION_SCHEMA_VERSION` /
  `LITERARY_EVALUATOR_VERSION` / `REVISION_REWRITER_VERSION`（独立版本，绝不 bump 既有
  cache 影响版本）。

### 设计铁律（测试强制断言）
- 盲测：评价与改写 prompt 绝不含作者名或 "write like"/"imitate"/"in the style of"
  （复用 `generation/schema.py` A/B 守卫，fail-closed）。
- P0 绝不因低优先级风格编辑被破坏：改写指令显式禁止改动情节/事实/人物/前提。
- 改写指令只含可解释自然语言（字面 guidance / trigger→operation→effect），绝不含
  作者名、原始数值、微观 stylometric 指纹（"char 3-gram" 等）。
- stylometric 余弦距离只诊断，绝不进改写指令、绝不进决策。
- 确定性：compare / 优先级排序 / accept/roll_back 均为纯函数；LLM 步骤隔离在注入
  provider 后（测试用 `DummyLLMProvider`，零 token）。

### 真实运行结果（deepseek-chat，读真实 DEEPSEEK_API_KEY，读 Phase 7 产物，绝不覆盖）
- **Austen**：文学评价 8.5 → 8.5；改写项 9（P2×2 + P3×7）；改写做了 13 处局部编辑
  （内部声音 / 身体动作 / 延迟揭示 / 破折号 / 对话 / 分号等）；高优先级偏差 9 → 8 →
  **continue**；stylometric 余弦距离 0.16785717 → 0.16798929（几乎不动，说明改写
  未造成指纹漂移）。
- **Dickens**：文学评价 8.5 → 8.5；改写项 9（P2×2 + P3×7）；改写器判定"已符合清单，
  无需改动"（合法 LLM 行为）→ 偏差 9 → 9 → **roll_back**（保留原文）。
- token：61,117（46,563 in / 14,554 out）。blind=True；6 维文学评价每维含 score +
  strength + weakness；evidence 经逐字校验（Austen plot_logic 的模型引文非逐字，
  被 fail-closed 丢弃 → evidence=0，属预期而非错误）。
- 产物：`data/analysis/evaluation/`（{author}_{actual_profile, literary_evaluation,
  revision_plan, revision_result, revised_actual_profile, revised_literary_evaluation}
  .json + `evaluation_summary.json` + `evaluation_report.md`）。

### Tests
- **287 passed**（was 267）：+20 Phase 8 测试（全确定性，Dummy provider，零 token）——
  schema 往返 + 版本守卫；compare band 分类；改写优先级 P0>P1>P2>P3 且指令无作者名/
  无数值/无微观指纹；rewriter P0 保护 + 盲测 + 空计划短路 + 作者名泄露 fail-closed；
  `decide_feedback_outcome` 规则；`measure_actual_profile` 的 Layer A 统计（22）+ Layer D
  重拟合诊断（合成 fixture）在未配置 provider 下 LLM 层显式 unavailable、绝不伪造。

### Non-goals（本次明确不做）
- `max_iterations > 2` 的多轮循环；真正段级 stylometric 漂移定位（spec §15.4）——
  首版用整段最小编辑 + 变更说明局部性映射。
- §19.5 生成可控性实验（low/medium/high 再生成）——后续独立实验，非反馈环内容。

---

## Checkpoint — Phase 8.1: Evaluation Decision Integrity & Revision Safety

**Status:** COMPLETE（真实 deepseek-chat 验证已跑）

### Goal
修复 Phase 8 决策与改写的四类完整性缺陷（spec §一）：
1. 决策只看风格偏差数量、忽略文学质量（风格改善可能掩盖文学分 8.2→5.1 的崩塌）；
2. 文学分在无逐字验证证据时仍算有效；
3. 改写只"告诉"模型别改情节/事实/角色，却不验证它确实没改；
4. 空改写计划被误判为 roll_back，而非 `no_action`。

范围（明确**非** Phase 9，无多轮自动循环）：决策质量 gate、文学评价证据契约、
改写后内容完整性检查、no_action 语义。

### 决策 gate 顺序（spec §四/§五，Style 与 Literary **分别报告**，绝不合并加权分）
- **STEP 1 — Content Integrity（最高）**：改写破坏内容 → `roll_back`。
- **STEP 2 — Literary Quality guard**：`max_literary_drop` 可配置容忍度（默认 0.5，
  非科学真值）；超限 → `roll_back`。**fail-closed 边界**：基线有效但改写后文学评价
  unavailable（如 6 维证据契约全失败）→ 无法验证文学保留，绝不单凭 Style Fidelity
  接受，直接 `roll_back`（"post-revision literary evaluation unavailable"）；基线本身
  unavailable → 不伪造基线，可走 Style Fidelity，但 `literary_quality.guard="unavailable"`
  显式标记。
- **STEP 3 — Style Fidelity**：改善 → `continue`/`accept`；未改善 → `roll_back`。
- `no_action`：改写计划为空 → 独立于 `roll_back`。

### Implementation
- `knowledge/evaluation/schema.py` — `EvaluationPolicy`（`max_literary_drop` /
  `weak_score_threshold`）、`ContentIntegrityViolation` / `ContentIntegrityResult`、
  `FeedbackDecision`（`style_fidelity` 与 `literary_quality` 分开）；`DimensionScore`
  加 `assessment_status`（observed / insufficient_evidence）+ `verified_evidence_count`
  （向后兼容默认）；新增 `LITERARY_EVALUATION_SCHEMA_VERSION=0.2.0` /
  `CONTENT_INTEGRITY_VERSION` / `FEEDBACK_DECISION_SCHEMA_VERSION` 常量。
- `knowledge/evaluation/literary.py` — evidence contract：每维 ≥1 条逐字验证证据 →
  observed，否则 insufficient_evidence（不进加权总分）；全维 insufficient → 整体
  unavailable（拒绝伪总分）；严格 exactly-six（缺/多维度均 reject）。版本 bump
  `LITERARY_EVALUATOR_VERSION=0.1.0→0.2.0` 以作废旧文学缓存。
- `knowledge/evaluation/integrity.py` — `ContentIntegrityChecker`（盲测，无作者名、
  不讨论风格）：确定性短路（零 token：一致→pass，空→fail）+ LLM 语义层严格 JSON
  校验（boolean 必须 bool、kind/severity 枚举）；只存简短 reasoning_summary。
- `knowledge/evaluation/run.py` — `decide_feedback_outcome` 返回 `FeedbackDecision`
  （三阶 gate + no_action + fail-closed 文学边界）；`run_evaluation` 改写后**先**
  Integrity 再重测（省 token），产物写 `data/analysis/evaluation_v2/`，LLM cache 复用
  v1（原文再测量缓存命中，0 token）。
- `knowledge/evaluation/revision.py` — `DEFAULT_WEAK_SCORE_THRESHOLD` 改从
  `EvaluationPolicy().weak_score_threshold` 取（不再散落硬编码 5.0）。

### Tests
- **311 passed**（Phase 8 后 287 → 308 → +3 决策完整性边界 = 311）：+24 Phase 8.1 测试
  （全确定性，Dummy provider，零 token）——新 schema 往返/版本守卫；证据契约
  （insufficient 维不进总分 / 全维 insufficient→unavailable / 多余维度 reject）；
  完整性（一致短路 pass / 空 fail / LLM 解析 / 非 bool reject / 违规 kind reject /
  盲测 / 作者名 fail-closed）；三阶 gate（integrity 优先于风格 / 文学超限 roll_back /
  容忍度可配置 / Style 与 Literary 分开报告 / no_action）；决策完整性边界
  （基线有效+改写后文学 unavailable → roll_back，即便风格改善或 perfect；基线
  unavailable → 走风格但 guard 标记 unavailable）。

### 真实运行结果（deepseek-chat，读真实 DEEPSEEK_API_KEY，产物写 `evaluation_v2`）
- **Austen**：文学评价 8.5 → 8.7（drop=-0.2，未触发文学护栏）；改写项 9（P2×2 + P3×7）；
  改写 13 处局部编辑；内容完整性 LLM 语义检查 **passed**（4 项 preserved 全真、无事件
  增删、0 违规，deterministic=False）；高优先级偏差 9 → 8 → **continue**；stylometric
  余弦距离 0.16785717 → 0.16798929（几乎不动，无指纹漂移）。
- **Dickens**：文学评价 8.5 → 8.5（drop=0.0）；改写项 9；改写器判定"已符合清单，无需改动"
  → revised_text==original → 内容完整性**确定性短路** passed（deterministic=True，0 token）；
  偏差 9 → 9 → **roll_back**（保留原文）；stylometric 0.14843544 不变。
- **证据契约**：两位作者 12 个维度（6×2）全部 `assessment_status=observed`，每维
  `verified_evidence_count ∈ [2,3]`；文学评价 schema=0.2.0（版本隔离生效）。
- **token：8,594**（5,307 in / 3,287 out）；**cache 43 hit / 4 miss**（原文再测量、
  改写、改写后重测全部命中 v1 缓存；仅 2×原文文学评价 + Austen 完整性 + Austen 改写后
  文学评价 4 次 fresh）。对比 Phase 8 的 61,117 token——cache 复用 + 完整性前置省 token
  显著。cache-replay 复跑确认 47/47 全命中、0 token（幂等）。
- 产物：`data/analysis/evaluation_v2/`（{author}_{actual_profile, literary_evaluation,
  revision_plan, revision_result, content_integrity, revised_actual_profile,
  revised_literary_evaluation}.json + `evaluation_summary.json` + `evaluation_report.md`）。
  `data/analysis/evaluation/`（Phase 8 v1）与 `data/analysis/generation/`（Phase 7）未动。

### Non-goals / 后续
- `max_iterations > 2` 多轮循环、段级 stylometric 漂移定位（spec §15.4）、§19.5
  生成可控性实验——均属 Phase 9 及之后，非本增量。

---

## Phase 8.1 Post-Run Audit — COMPLETE (NEEDS_FIX)

- 新增 `knowledge/evaluation/audit.py`（确定性审计 runner，零 LLM）+ `tests/test_audit.py`
  （21 测试）。从既有 `evaluation_v2/` + `generation/` 产物独立重建：逐项偏差对照
  （A/B/C/D/E 分类）、确定性文本 diff（identical/punctuation_only/minimal/substantive）、
  证据契约审计（逐字引文存在性/复用/过短）、决策重构（复用 `decide_feedback_outcome`）。
- 产物：`data/analysis/evaluation_v2/phase8_1_postrun_audit.json` + `.md`。
- **结论 NEEDS_FIX**。核心发现（进入 Phase 8.2 修）：
  1. Austen 改写是**标点归一化 no-op**（词数 703→703、改动词 token 0），却自报多个实质
     `change_descriptions`（幻觉）；
  2. Austen 9→8 的"改善"来自 Layer C strategy match 的 LLM 翻转（文本无实质改动）；
  3. Austen 文学 8.5→8.7 来自 LLM evaluator 噪声（文本无实质改动仍漂移 0.2）；
  4. Austen/Dickens `revision_items_applied=9` 与文本词级零改动矛盾；
  5. Dickens 字节级相等却落 `roll_back` 而非更贴切的 `no_effect`。
- 三阶 gate 决策逻辑本身重建正确（stored==reconstructed）。

## Phase 8.2 (Revision Effect & Measurement Validity) — COMPLETE（确定性，未运行真实 LLM）

修复 Phase 8.1 Post-Run Audit 的真实 feedback 缺陷。**本轮只做代码/测试/文档 + 确定性
验证，绝不调用 DeepSeek/Qwen/任何真实 LLM，绝不重新生成 Austen/Dickens 正文，绝不进入
Phase 9。**

- 新增 **Gate 0（Revision Effect）**，置于 Content Integrity 之前：`RevisionEffectAnalyzer`
  （`knowledge/evaluation/effect.py`，确定性零 token）用 `normalize_for_revision_comparison`
  canonical 归一化（Unicode 弯引号/弯连字符/省略号 + 空白/NBSP/换行，**绝不**改词形或词序）
  判断改写是否产生实质词级变化。
- 有限 `effect_status` 枚举：`identical` / `formatting_only` / `punctuation_only` /
  `substantive`。只有 `substantive_edit == True` 才允许 after-measurement 参与比较
  （杜绝 LLM 测量噪声被记为改善）；`word_change_count` / `word_change_ratio` /
  `sentence_change_count` + 四个 sha256 hash（原文/改后/canonical 原文/canonical 改后）。
- `no_effect` 独立于 `no_action`（空计划）与 `roll_back`（实质改写被拒）；短路后续一切
  昂贵步骤（不调 ContentIntegrityChecker / LiteraryEvaluator(after) / NarrativeAnalyzer /
  StrategyMiner / after-style 比较）。
- 改写器自报字段降级：`change_descriptions`→`claimed_change_descriptions`、
  `revision_items_applied`→`claimed_revision_items`（best-effort，绝不作为权威证据）；
  真实有效性由 deterministic `revision_effect` 给出。`RevisionResult.from_dict` 向后兼容
  旧字段名（Phase 8.1 产物仍可解析）。
- `FeedbackDecision` 新增 `revision_effect` / `literary_guard_status`（含
  `not_applicable_no_effect`）/ `style_comparison_performed`；`FEEDBACK_NO_EFFECT` 加入
  结果枚举。
- 版本隔离：新增 `REVISION_EFFECT_SCHEMA_VERSION` / `REVISION_EFFECT_ANALYZER_VERSION` /
  `REVISION_RESULT_SCHEMA_VERSION`；bump `REVISION_REWRITER_VERSION=0.1.0→0.2.0`、
  `FEEDBACK_DECISION_SCHEMA_VERSION=0.1.0→0.2.0`（绝不复用 Phase 8.1 decision/rewriter
  cache）。Phase 8.2 未来真实运行写 `data/analysis/evaluation_v3/`（v1/v2 绝不覆盖）。
- 确定性 dry-run（读 Phase 8.1 既有产物，零 LLM）：Austen → `punctuation_only`、
  `substantive_edit=False` → **no_effect**；Dickens → `identical` → **no_effect**。
- Tests：**356 passed**（新增 `tests/test_effect.py` 24 项确定性回归：分类/归一化/词序
  不变性/no_effect 短路语义/`run_evaluation` Gate 0 短路（断言 checker 0 调用、literary
  evaluator 仅 before 1 次、重测仅 before 1 次、provider.complete 0 调用）/schema 往返 +
  版本 guard + 旧字段向后兼容）。绝无真实 LLM、绝无真实 `data/` 写入。

---

## Workflow (going forward)

1. Implement → 2. run tests → 3. run experiment if applicable → 4. inspect git
diff → 5. update this dev log → 6. update `docs/STYLE_ENGINE_STATUS.md` →
7. commit code + tests + docs together → 8. push the `feature/style-engine-v0.1`
branch → 9. STOP at any explicitly requested review checkpoint.

Never commit: raw corpus, generated clean/chunk text, secrets, caches, vector
stores, or machine-specific files (see `.gitignore`).
