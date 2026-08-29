# Weaver Style Engine — Project Status

Short current-state snapshot (≈1–2 min read). History lives in
[`STYLE_ENGINE_DEVLOG.md`](STYLE_ENGINE_DEVLOG.md); target design lives in
`STYLE_ENGINE_SPEC_V0.1.md`.

| Field | Value |
|---|---|
| **Current phase** | Phase 9.2/9.3 — COMPLETE（含 §19.5 真实生成 + repeated-sampling n=3）：§15.4 段级 stylometric 漂移定位 + §19.5 生成可控性实验（low/medium/high 真实 `deepseek-chat` 重生成 + 单调性观测 + 重复采样均值/中位数判定）；382 tests |
| **Last completed checkpoint** | §19.5 repeated-sampling（每档 n=3，12 新 fresh 请求，20,799 tokens）：Dickens 单调递减 n=3 仍成立（mean 0.154→0.127→0.120）；Austen medium 偏高未被平均掉（mean 0.172→0.216→0.172，仍 non-monotonic）——小样本弱证据、非硬门 |
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
- Strategy registry lifecycle implemented and exercised but **global/cross-author** (75 strategies, 9 validated / 29 candidate / 37 discovered); author-scoped canonical consolidation supersedes it for author profiles.
- Author-scoped strategy consolidation **run** (`StrategyConsolidator` + repair pass): Austen 51→26, Dickens 44→36 canonical strategies (full coverage), persisted under `data/analysis/consolidation/`.

## What is not implemented yet
- Full-corpus LLM feature extraction (only the 40-chunk calibration sample has LLM features).
- §19.5 >3 样本的正式统计（当前 n=3 仍小，未做假设检验/功效分析/多轮方差分解）。
- 把段级漂移图**接入改写 planner 做段级目标编辑**（漂移图本次仅产出 + 持久化，仍整段最小编辑）。
- Multi-author style mixing (the `conflicts` / `resolution_required` structure is reserved in
  `StylePlan.planner_metadata`, currently empty for single-author planning).
- NlpAnalyzer (POS) features — NLTK intentionally not installed.
- Mixed-effects / variance-decomposition model (deferred by spec).

## Phase 5 (author-profile synthesis) — COMPLETE (REVIEW PENDING)
- `knowledge/profiles/style_profile.py`（schema + 确定性合成器）+
  `knowledge/calibration/synthesize.py`（runner）。
- `AuthorStyleProfile`：generation_controls / narrative_controls / strategy_controls /
  diagnostics.stylometry / uncertainty / provenance / reproducibility_hash。
- control_role 复用既有 registry 角色：core/candidate_core/descriptive→direct_control、
  diagnostic→diagnostic、experimental→reference_only；stylometric 指纹绝不进 generation。
- canonical strategies → conditional_control + 确定性 control_priority（support_status →
  跨作品 → 跨 chunk → confidence → raw → id）。
- 不确定性一等（missing/insufficient/not_observable 绝不伪造 0）；sampled LLM 结果带
  `source_scope`；held-out 隔离显式校验；`reproducibility_hash` 保证字节级可复现。
- 产物：`data/analysis/style_profiles/`（Austen 26 / Dickens 36 canonical 数量与 support_status
  与 Phase 4.5 完全一致）。**停止等待人工 review，未进入 Phase 6。**

## Phase 5.1 (profile integrity + author-specific stylometric targets) — COMPLETE (REVIEW PENDING)
- `diagnostics.stylometry` 拆为 `author_target`（作者专属质心/离散度，compact 标量 + 引用
  `stylometric_author_targets.json`）与 `validation_metadata`（全局实验元数据）。
- 作者目标纯函数 `_author_targets_from_matrix` 只收 TRAIN 侧数据（X_train / train_authors /
  train_works），从签名杜绝 held-out；`fit_scope=train_only`、`no_held_out=True`。
- held-out 隔离 **fail-closed**：`clean=False` 抛 `ProfileSynthesisError`，先内存合成、成功
  后统一落盘，污染即不写任何产物。
- provenance 路径解析为具体 `austen`/`dickens`（无 `{author_id}` 占位符，非绝对路径）。
- `AuthorStyleProfile.from_dict` + 嵌套 from_dict + `verify_reproducibility_hash()`；错误
  schema_version / 缺字段抛 `ProfileSchemaError`。
- Austen target: n=833 / [emma, p&p] / centroid_norm 0.098128 / within_cosine 0.174174；
  Dickens target: n=1495 / [dc, ge] / centroid_norm 0.103634 / within_cosine 0.155668（互异）。

## Phase 6 (Style Planner & Prompt Compiler) — COMPLETE (REVIEW PENDING)
- `knowledge/planning/`：`schema.py`（WritingRequest / StylePlan / PlannerPolicy /
  ActivationLevel）、`policy.py`（激活政策 + 控制预算 + 数值→自然语言 banding，纯函数）、
  `planner.py`（`StylePlanner.plan`，画像完整性 fail-closed）、`compiler.py`
  （`PromptCompiler.compile` → 六段提示词）、`run.py`（Austen/Dickens 对比产物）。
- 三层严格分离：画像（观察）→ StylePlan（本次激活哪些控制）→ 提示词（可执行指令）；
  绝不 dump 画像 JSON，绝不提作者名，绝不写微观 stylometric 指令，绝不改写用户 brief。
- 激活政策：candidate_core 确定性门槛（完整度/证据/source_scope/离散度），**绝不晋升
  core**；descriptive→weak；experimental→reference_only；diagnostic→绝不在提示词。
  激活级别 strong/medium/weak/reference/suppressed（有限枚举，无伪连续权重）。
- 控制预算可配置（primary≤6 / secondary≤4 / narrative≤4 / strategies≤6），超出显式记
  `suppressed_due_to_budget`（绝不静默丢弃）；用户显式 pov 覆盖作者倾向（warning，不 reject）。
- 策略选择 validated > candidate > discovered（discovered 保持 reference）；条件规则
  `WHEN → THEN → TO`。
- 产物：`data/analysis/planning/`（Austen/Dickens style_plan + compiled prompt + 对比报告
  + 汇总）。同一中性 brief：Austen dialogue "prominent" + third-person + low narrator
  presence vs Dickens dialogue "sparse" + first-person + medium narrator presence。

## Phase 6.1 (evidence-grounded guidance + prompt budget integrity) — COMPLETE (REVIEW PENDING)
- `knowledge/planning/bands.py`：TRAIN-only 经验 band 阈值（`low < Q33`、`medium ∈
  [Q33, Q67]`、`high > Q67`，线性插值分位数）；`compute_band_thresholds` 纯函数 +
  `band_label` + `describe_feature`（无阈值/未知特征 → `None`，绝不编造）+ 22 特征
  字面 guidance（无未测文学机制）。阈值持久化为 `band_thresholds.json`（独立 schema
  版本）。跨作者**合并** TRAIN 分位数（非 per-author）以保留 Austen/Dickens 区分。
- `policy.py`：删除 `_FEATURE_BANDS` 人工阈值与 `describe_feature`；激活/预算/叙事逻辑不变。
- `planner.py`：`StylePlanner(policy, band_thresholds)`；guidance `None` 且本可激活
  （strong/medium/weak）→ 降级 reference（reason=`not_compilable`）。
- `compiler.py`：预算**确定性降级**（策略→secondary→weak→措辞→`PromptBudgetError`），
  **绝不硬截断用户内容**；`removed_controls` 记录每次移除；`sections` 6 段全保留且
  `_assemble(sections) == text`；POV 移至 CONTENT-only（NARRATIVE 不再重复）。
- 缺失值语义明确：全缺 / `n_valid=0` → suppressed；部分缺失经 `completeness` 贡献
  （不自动 suppress）。
- 产物：`band_thresholds.json`（22 特征、2,328 TRAIN chunks、held-out 排除）；Austen
  dialogue_ratio → "Use dialogue relatively often."、Dickens → "Use dialogue in
  moderate proportion."（字面，无 "prominent/sparse" 文学解释）。

## Phase 7 (style-conditioned generation) — COMPLETE
- `knowledge/generation/`：`schema.py`（`GeneratedPassage` / `GenerationResult` /
  `GenerationUsage` / `GenerationParameters` / `compiled_prompt_hash` /
  `assert_no_author_leakage` / `make_generation_id`）、`provider.py`
  （`GenerationProvider` 复用 OpenAI 兼容传输 + `DummyGenerationProvider`）、
  `run.py`（`run_plumbing` + `run_generation` + 对比报告 + 汇总）。
- provider `deepseek` / model `deepseek-chat` / endpoint
  `https://api.deepseek.com/chat/completions`；`temperature=0.8`、`top_p=0.9`、
  `max_tokens=2048`；两位作者参数一致，唯一变量是画像导出的风格控制。
- 无作者名注入：实际 prompt 不含 `Jane Austen` / `Charles Dickens` / `write like` /
  `imitate` / `in the style of`（`assert_no_author_leakage` fail-closed；compiler
  `IMPORTANT` 段已改写，守卫语义保留）。
- 复用 OpenAICompatibleProvider 的 `complete_with_metadata`（单 client，记录
  finish_reason + per-call usage），不另写第二套 HTTP；独立 experiment_id / 无缓存，
  每次生成都是 fresh request。
- 产物：`data/analysis/generation/`（`generation_experiment.json`、
  `{austen,dickens}_generation.json`、`{austen,dickens}_passage.md`、
  `generation_comparison_report.md`、`generation_summary.json`、
  `generation_plumbing.json`，gitignored）。
- 无自动评价（Phase 8）、无自动改写。

## Phase 7.1 (provenance / integrity hardening) — COMPLETE (zero token)
- **身份模型**：`generation_condition_id`（作者/计划/prompt/provider/model/参数 的确定性
  hash，标识"条件"）与 `generation_id`（`condition_id + experiment_id + output hash
  (+ request id)`，标识"具体结果"）分离。同条件不同正文 → 不同 `generation_id`；同条件
  同正文 → 同 id；**绝不依赖当前时间**。`GeneratedPassage` 新增 `generation_condition_id`
  + `request_id`；`from_dict` 对 Phase 7 旧产物向后兼容回填，绝不要求重生成。
- **Plumbing gate**：`run_generation` 正式生成前强制 `_require_valid_plumbing`（文件存在
  / success / 正文非空 / finish_reason=stop / provider+model 匹配 / 参数一致 /
  fresh_request=true / cache_hit=false），任一违反 → `GenerationError`（fail-closed）。
- **Markdown 修复**：`_render_passage_md` 补 f-string 前缀（`{p.experiment_id}` 等不再
  渲染成字面量），新增 `generation_condition_id`/`request_id`；测试保证无未解析 `{p.`。
- **泄露守卫 A/B 分离**：A. `assert_no_imitation_instruction` 只查风格控制指令（非
  CONTENT），用户 brief 正文合法的 "imitate" 绝不误报；B. `assert_no_author_identity`
  作者名单来自 author metadata（`author_display_names()`），支持未来作者身份，非硬编码。
- **零 token**：不调用 DeepSeek、不生成/不评价/不改写正文；既有 Austen/Dickens 产物
  保持原样（未重生成）。

## Current corpus
- **TRAIN:** Pride and Prejudice, Emma (Austen); Great Expectations, David Copperfield (Dickens).
- **HELD-OUT:** Persuasion (Austen); A Tale of Two Cities (Dickens).
- 6 works total; raw text outside the repo (`wensigongfang/text/`), `data/` gitignored.

## Current test status
- **267 tests passed** (was 254). +13 Phase 7.1 tests (Dummy provider, zero token):
  同条件不同正文 → 不同 `generation_id`、同 prompt/参数 → 同 `generation_condition_id`、
  缺 plumbing 阻塞正式生成、失败/不匹配 plumbing 阻塞、合法 plumbing 放行、Markdown
  含已解析元数据（无 `{p.` 占位符）、泄露守卫支持未来作者身份、用户正文合法 "imitate"
  不误报作者身份泄露、旧产物向后兼容（缺 `generation_condition_id` 回填）。
- **254 tests passed** (was 236). +18 Phase 7 tests (Dummy provider, zero token):
  GenerationResult/usage/参数序列化、GeneratedPassage 往返 + finish_reason、空生成拒绝、
  prompt hash 正确 + 敏感、generation_id 确定性、作者名/模仿令牌泄露检出、编译 prompt
  无作者名无模仿、provenance 保存、同一 WritingRequest 共享、provider/model/参数一致、
  未配置 provider fail-closed、artifact 布局 + 无自动评价/无自动改写 + 铁律令牌集合。
- **236 tests passed** (was 223). +13 Phase 6.1 tests: TRAIN-only band（held-out 排除 /
  不改变阈值）、band 确定性、跨作者合并阈值、band_label 三档边界、字面 guidance 无未测
  机制、无 band → None、not_compilable→reference、长内容永不硬截断（多档预算）、低优先级
  先于强制内容移除、强制溢出抛 `PromptBudgetError`、sections 精确重构 text、真实
  band_thresholds TRAIN-only 校验。
- **223 tests passed** (was 191). +32 Phase 6 tests: schema 往返（WritingRequest/Policy/
  PlannedControl/PlannedStrategy/StylePlan）、空 content 拒绝、激活政策 7 例（candidate_core
  strong / insufficient / sampled / descriptive / experimental / diagnostic）、语言/叙事/策略
  预算（不静默丢弃）、experimental→reference、POV 覆盖 + 同视角不警告、hash/held-out
  fail-closed、plan/prompt 确定性、提示词六段、不提作者名 / 不写微观 stylometric、保留用户
  brief、POV 覆盖写入提示词、预算截断、真实产物 Austen/Dickens 计划互异 + 提示词不提作者名。
- **191 tests passed** (was 182). +9 Phase 5.1 tests: diagnostics 拆分（author_target /
  validation_metadata）、作者目标互异、concrete provenance 无 `{author_id}` 占位符、纯函数
  TRAIN-only 目标计算、真实产物 Austen/Dickens 目标互异、往返序列化精确相等、reload 后 hash
  复核、错误 schema_version 拒绝、缺字段拒绝。
- **182 tests passed** (was 167). +15 Phase 5 tests: control-role 映射、diagnostic 不进
  generation、direct/conditional/reference-only 分桶、canonical 数量保持、support_status
  保持、不确定性不伪造 0、narrative not_observable 保留、full-corpus vs sampled scope、
  held-out 隔离（clean + 双通道污染检出）、strategy 优先级确定性 + 跨轮稳定、字节级复现。
- **167 tests passed** (was 161). +13 Phase 4.5 tests: author-scope isolation,
  missing-author rejection, complete source coverage, duplicate-assignment /
  hallucinated / missing source-id rejection, canonical provenance
  (raw→chunk→work→evidence), cross-author same-name ids, canonical-id stability,
  exact-dup fold, no name-similarity merge, dummy end-to-end consolidate.
  +7 Phase 4.5.1 tests: prompt support/evidence context, 2-quote cap, empty-name /
  empty-description rejection, non-numeric confidence rejection, confidence
  out-of-range (<0 / >1) rejection, LLMResponseError wrap on invalid fields.
  +3 Phase 4.5-run tests: max_tokens propagation, repair-into-existing-group,
  repair-creates-new-group.
  +8 repair-hardening tests (replacing the 2 name-based repair tests): merge-by-
  `canonical_strategy_id`, paraphrase-does-not-spawn-new-canonical, create_new,
  hallucinated-target-id reject, duplicate-assignment reject, incomplete-coverage
  reject, hallucinated-raw-id reject, cross-author-target reject.

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

## Phase 4.5 consolidation input (infrastructure only; no LLM called)
- Austen **51** raw strategies; Dickens **44** (author-partitioned from the Phase 4.4
  registry; 20 strategies appear in both authors → partitioned, never merged).
- Exact-duplicate fold: 0 (near-duplicates differ in description due to the discover
  hash suffix — the LLM semantic merge is what collapses them).
- Estimates (single-shot, DeepSeek `deepseek-chat`): Austen 1 req ~15.9k in / ~3.1k out;
  Dickens 1 req ~13.2k in / ~2.6k out. No existing consolidation cache.
  (Phase 4.5.1 added per-strategy `support:` + ≤2 verified evidence quotes — input
  grew ~2–3k/author; output unchanged.)
- Artifacts: `data/analysis/consolidation/{austen,dickens}_consolidation_input.json`,
  `consolidation_summary.json`.

## Phase 4.5 consolidation results (real `deepseek-chat` run)
- Austen **51 → 26** canonical (validated 7 / candidate 2 / discovered 17);
  Dickens **44 → 36** (validated 12 / candidate 5 / discovered 19).
- Full coverage: every raw id mapped exactly once, no hallucination / duplicate / missing.
- Two run-time bugs fixed (both with regression tests): (1) `max_tokens=2048` truncated
  the JSON → raised to 8192 and added to the cache key; (2) Austen first pass omitted
  2/51 ids → deterministic coverage-repair pass merged them back.
- Artifacts: `data/analysis/consolidation/{austen,dickens}_canonical_strategies.json`,
  `consolidation_results.json`, `consolidation_report.md`.

## Current blockers / review items
- **Data anomaly (flagged, not yet fixed):** 20 registry strategies carry cross-author
  evidence; 2 stay `validated` despite it (monotonic lifecycle masked the crossover).
  Author-scoped `support_status` recomputes correctly; the global registry `status` must
  not be read as an author-level claim.
- `candidate_core` 特征仍不得晋升（校准仅标定样本，不足以晋升）——Phase 6 已把这一条
  落实为确定性门槛 gate 并写入 warning：strong 激活的 candidate_core 仍是 CANDIDATE。

## Phase 8 (Style Feedback Loop + 独立 LLM 文学评价) — COMPLETE
- `knowledge/evaluation/`：schema / literary / analyze / compare / revision / run；
  `knowledge/schema/versions.py` 新增 `EVALUATION_SCHEMA_VERSION` /
  `LITERARY_EVALUATOR_VERSION` / `REVISION_REWRITER_VERSION`（独立，不 bump 既有版本）。
- 闭环：再测量（Actual Style Profile，Layer A 统计 22 + 判断 8 LLM + B 叙事 + C 策略
  + D stylometric 重拟合诊断）→ 目标 vs 实际（band 偏差 / 叙事 / 策略覆盖）→ 优先化
  改写计划（P0–P4）→ 最小编辑改写 → 再分析 → 确定性 Accept / Continue / Roll Back。
- 独立 LLM 文学评价：6 维 1–10 + strength/weakness + 逐字校验证据引文，加权总分。
- 铁律落实为测试断言：盲测（评价/改写 prompt 无作者名/模仿令牌，A/B 守卫 fail-closed）、
  P0 保护强制入改写 prompt、改写指令只含可解释自然语言（无作者名/原始数值/微观指纹）、
  stylometric 距离仅诊断绝不进指令或决策、compare/优先级/决策均为纯函数。
- 真实运行（deepseek-chat，61,117 token）：Austen 评价 8.5→8.5、9 改写项、偏差 9→8
  → **continue**；Dickens 评价 8.5→8.5、改写器判定无需改动 → 偏差 9→9 → **roll_back**。
- 产物：`data/analysis/evaluation/`（actual_profile / literary_evaluation /
  revision_plan / revision_result / revised_actual_profile / revised_literary_evaluation
  + evaluation_summary.json + evaluation_report.md）。绝不覆盖 `data/analysis/generation/`。
- Tests：**287 passed**（was 267，+20 全确定性 Dummy-provider 零 token）。

## Phase 8.1 (Evaluation Decision Integrity & Revision Safety) — COMPLETE
- 决策三阶 gate（spec §四/§五）：STEP 1 Content Integrity（最高，破坏内容 → roll_back）
  → STEP 2 Literary Quality guard（`max_literary_drop` 可配置容忍度，默认 0.5）→ STEP 3
  Style Fidelity。**Style Fidelity 与 Literary Quality 分别报告**，绝不合并成单一加权分。
- `no_action` 独立于 `roll_back`（空改写计划 = 未执行改写）。
- `ContentIntegrityChecker`：改写后**先**跑（省 token），盲测（无作者名、不讨论风格），
  确定性短路（一致→pass、空→fail，零 token）+ LLM 语义层严格 JSON 校验。
- 文学评价证据契约：每维 ≥1 条逐字验证证据 → observed，否则 insufficient_evidence（不进
  加权总分）；全维 insufficient → 整体 unavailable（拒绝伪总分）；严格 exactly-six。
- **fail-closed 决策完整性边界**：基线有效但改写后文学评价 unavailable（如证据契约全失败）
  → roll_back（"post-revision literary evaluation unavailable"，即便风格改善或 perfect）；
  基线本身 unavailable → 不伪造基线，可走 Style Fidelity，但 `literary_quality.guard=
  "unavailable"` 显式标记。绝不把 unavailable 分数转成 0。
- 版本隔离：`LITERARY_EVALUATOR_VERSION=0.1.0→0.2.0`（作废旧文学缓存）；
  `LITERARY_EVALUATION_SCHEMA_VERSION` / `CONTENT_INTEGRITY_VERSION` /
  `FEEDBACK_DECISION_SCHEMA_VERSION` 新增；原文再测量缓存复用 v1（命中）。
- 产物隔离：Phase 8.1 写 `data/analysis/evaluation_v2/`；`data/analysis/evaluation/` 与
  `data/analysis/generation/` 绝不覆盖、Phase 7 绝不重生成。
- 真实验证（deepseek-chat，8,594 token / 43 cache hit / 4 miss）：Austen 文学 8.5→8.7 +
  完整性 LLM 语义 passed → 偏差 9→8 → **continue**；Dickens 文学 8.5→8.5 + 完整性确定性
  短路 passed → 偏差 9→9 → **roll_back**；12 维度证据契约全 observed（每维 2–3 条逐字证据）。
- Tests：**311 passed**（+24 全确定性 Dummy-provider 零 token）。

## Phase 8.1 Post-Run Audit — COMPLETE (NEEDS_FIX)
- 确定性审计（`knowledge/evaluation/audit.py`，零 LLM，21 测试）：从既有产物独立重建
  逐项偏差对照 + 文本 diff + 证据契约审计 + 决策重构。产物
  `data/analysis/evaluation_v2/phase8_1_postrun_audit.json` / `.md`。
- 三阶 gate 决策逻辑重建正确（stored==reconstructed），但发现真实 feedback 缺陷：
  Austen 改写为标点 no-op（词级零改动）却自报实质 `change_descriptions`（幻觉）；9→8 的
  "改善"与 8.5→8.7 的文学提升均来自 LLM 测量噪声（文本无实质改动）；`revision_items_applied=9`
  与文本词级零改动矛盾；Dickens 字节级相等却落 `roll_back` 而非 `no_effect`。→ Phase 8.2 修。

## Phase 8.2 (Revision Effect & Measurement Validity) — COMPLETE（确定性，未运行真实 LLM）
- 新增 **Gate 0（Revision Effect）**（`RevisionEffectAnalyzer`，确定性零 token）：canonical
  归一化（排版标点 + 空白，绝不改词形/词序）→ `identical` / `formatting_only` /
  `punctuation_only` / `substantive`；只有 `substantive_edit==True` 才允许 after 比较。
- `no_effect` 独立于 `no_action` 与 `roll_back`；短路后续一切昂贵步骤（完整性/重测/文学
  评价/策略匹配 0 调用）。
- 改写器自报字段降级 `claimed_*`（best-effort）；`RevisionResult` 附着 deterministic
  `revision_effect`；`FeedbackDecision` 新增 `revision_effect` / `literary_guard_status` /
  `style_comparison_performed`。
- 版本隔离：新增 `REVISION_EFFECT_*` / `REVISION_RESULT_SCHEMA_VERSION`；bump
  `REVISION_REWRITER_VERSION` 与 `FEEDBACK_DECISION_SCHEMA_VERSION` 至 0.2.0；未来真实运行
  写 `evaluation_v3/`（v1/v2 绝不覆盖）。
- 确定性 dry-run（Phase 8.1 既有产物）：Austen `punctuation_only` → no_effect；Dickens
  `identical` → no_effect。
- Tests：**356 passed**（+24 确定性回归，含 `run_evaluation` Gate 0 短路断言 provider 0 调用）。

## Phase 8.2 Real End-to-End Validation — COMPLETE（真实 deepseek-chat）
- 全新生成 austen_02 / dickens_02（experiment_id=`phase8_2-generation-v0.1`，fresh，不读
  Phase 7 缓存，不覆盖 `generation/` / `evaluation/` / `evaluation_v2/`）。
- 实验身份参数化：`run_generation(experiment_id)` 写 `generation/{id}/`；`run_evaluation(
  generation_experiment_id, run_tag, summary_prefix, max_iterations)` 写 `evaluation_v3/
  {author}_02/`。新增 `knowledge/evaluation/phase8_2.py` runner；`max_iterations=1`（§二十 单轮）。
- 结果：austen_02 实质改写（6 项/20 词）→ 文学 +0.35、风格 6→6 无改善 → **roll_back**；
  dickens_02 实质改写（11 项/17 词）→ 文学 +0.2、风格 11→8 改善 → **accept**。两位作者均
  实质改写（修正 Phase 8.1 的"自报实质/实际 no-op"幻觉）。
- 成本：generation 3,346 + evaluation 79,419 = **82,765 token**（48 real requests，0 hit）。
- Tests：**361 passed**（+5）。

## Phase 9.1 (Multi-Round Feedback Loop) — COMPLETE（确定性实现，未运行真实 LLM）
- 把 `run_evaluation` 的单次 if/else 决策重构为有界 `while` 闭环：`continue` 真正迭代
  revise → measure → decide；`decide_feedback_outcome` 纯函数不变（仍编码全部停止条件）。
- 每轮 delta 语义：第 N 轮 "before" = 第 N−1 轮接受的改写（爬坡当前最佳，不重改原文）；
  `roll_back` 保留 best-so-far（第 N 轮 before），绝不回退原文。
- 铁律：改写器每轮编辑当前最佳正文；内容完整性每轮对照**不可变原文**（P0 每轮保留）；
  Revision Effect 每轮对照当前正文（中途 no-op → no_effect 停）；stylometric 仍仅诊断。
- 产物：第 1 轮沿用旧文件名（向后兼容），第 N≥2 轮 `_iter{N}` 后缀；新增
  `{author}_iterations.json`（loop_version + 逐轮 history + final_*）；summary 新增
  `iterations`/`final_outcome`/`final_iteration`/`final_text_hash`/`n_iterations`（单轮同样填充）。
- 新增 `FEEDBACK_LOOP_VERSION=0.1.0`（独立版本，绝不 bump 任何 cache 相关版本）。
- Tests：**368 passed**（+7 多轮确定性，零 token，零真实 LLM）。

## Phase 9.2 (Segment-Level Stylometric Drift Localization, §15.4) — COMPLETE（确定性）
- `knowledge/evaluation/analyze.py`：抽 `_refit_layer_d`（重拟合 + fail-closed feature_names
  比对，整段/段级共享**同一次**重拟合）；新增 `_segment_drift`（句子粒度逐句余弦距离到作者
  质心，降序漂移图）与公开 `stylometric_distance`（整段距离薄封装）。
- `_layer_d_diagnostic` 返回值**新增** `segment_drift` 键（加法字段，`layer_d_stylometric`
  已是 dict，不 bump 任何 schema）。漂移图含每句 `segment_index`/`char_start`/`char_end`/
  `paragraph_index`/`n_tokens`/`cosine_distance`/`text_preview`，过短句（<3 词）标
  `skipped`（短文本余弦噪声过大）。
- 铁律不变：stylometric 指纹仅诊断，绝不生成改写指令；短段距离是段内**相对排序**。
- 新增 `SEGMENT_DRIFT_VERSION=0.1.0`（独立，不 bump STYLOMETRY/EVALUATION 版本）。

## Phase 9.3 (Generation Controllability, §19.5) — COMPLETE（确定性 harness + 真实生成 + repeated-sampling）
- 新建 `knowledge/generation/controllability.py`：`apply_intensity`（纯函数，把已激活语言
  控制统一重标 low→weak / medium→medium / high→strong，互异确定性 plan id 经
  `make_intensity_plan_id`）、`check_monotonic`（low≥medium≥high 容差判定，**报告观测非
  硬门**）、`run_controllability`（镜像 `run_generation`：plumbing gate + 泄露/预算守卫 +
  `_build_passage` 复用 + `stylometric_distance` 测量，落盘 `{author}_{intensity}_*` +
  `controllability_summary/report`）。
- repeated-sampling：`run_controllability_repeated` 读首样本 + 每档 2 个 fresh → n=3，
  汇总 mean/median/std/min/max + `monotonic_on_mean` / `monotonic_on_median` /
  `effect_direction` / 区间重叠；落盘 `controllability_repeated_summary/report` +
  `{author}_{intensity}_rep{{1,2}}_*`（**不覆盖**原单次实验）。
- 强度旋钮 = 语言控制 `activation`（compiler `_ACTIVATION_PREFIX` 唯一带强度措辞的层）。
- `knowledge/planning/schema.py` 新增 `make_intensity_plan_id`；新增
  `CONTROLLABILITY_VERSION=0.1.0`、`CONTROLLABILITY_REPEATED_VERSION=0.1.0`（均独立）。
- Tests：**382 passed**（+14：3 段级 + 11 可控性/repeated，零 token，零真实 LLM）。

### §19.5 真实生成结果（deepseek-chat，6 次 fresh 请求，10,149 tokens）

- **Dickens**：`low=0.164 → medium=0.125 → high=0.111`，**单调递减**（强度↑ → 距离↓，符合预期）。
- **Austen**：`low=0.157 → medium=0.270 → high=0.193`，**非单调**（medium 反而最远，high 未超 low）。

### §19.5 repeated-sampling 结果（deepseek-chat，每档 n=3，12 新 fresh 请求，20,799 tokens）

- **Austen**（n=3）：mean `low=0.172 / medium=0.216 / high=0.172`，median `0.159/0.194/0.165`；
  **mean 与 median 均 non-monotonic**——medium 偏高未被重复采样平均掉，且 medium std 最大
  （0.038）；effect_size(low−high)=0.0006 ≈ 0（low≈high，趋势实质是"medium 异常偏高"）。
- **Dickens**（n=3）：mean `low=0.154 / medium=0.127 / high=0.120`，median `0.153/0.125/0.123`；
  **mean 与 median 均单调递减**——原单调趋势 n=3 后仍成立，effect_size(low−high)=0.034。
- 结论：单次结论在 n=3 下**部分稳健**（Dickens 复现、Austen medium 系统性偏高非噪声）；但
  三者区间仍两两重叠、档间差异与档内散布同阶，仍是**小样本弱证据**，不设硬 pass/fail。

## Next planned action
- **STOP，报告，等待人工 review**：Phase 9.2/9.3 已跑完（确定性实现 + §19.5 真实生成 +
  repeated-sampling n=3）。Austen medium 系统性偏高、Dickens 单调递减 n=3 复现，均为小样本
  弱证据；可选下一步：>3 样本正式统计或段级漂移接入 planner。绝不自动进入进一步真实 LLM
  运行、不合并 main、不提 PR。
