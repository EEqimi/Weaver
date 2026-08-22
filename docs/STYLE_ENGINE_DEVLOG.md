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

## Workflow (going forward)

1. Implement → 2. run tests → 3. run experiment if applicable → 4. inspect git
diff → 5. update this dev log → 6. update `docs/STYLE_ENGINE_STATUS.md` →
7. commit code + tests + docs together → 8. push the `feature/style-engine-v0.1`
branch → 9. STOP at any explicitly requested review checkpoint.

Never commit: raw corpus, generated clean/chunk text, secrets, caches, vector
stores, or machine-specific files (see `.gitignore`).
