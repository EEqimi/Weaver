# Weaver Style Engine — Specification V0.1（重建 + 状态标注）

> 本文档是 Weaver Style Engine 的**目标设计**（target design），并标注每项在 V0.1
> 冻结时的**实现/验证状态**。实现与测试的完整历史见
> [`STYLE_ENGINE_DEVLOG.md`](STYLE_ENGINE_DEVLOG.md)；当前快照见
> [`STYLE_ENGINE_STATUS.md`](STYLE_ENGINE_STATUS.md)。三者关系：
>
> | 文档 | 职责 |
> |---|---|
> | `STYLE_ENGINE_SPEC_V0.1.md`（本文） | 系统**应当**变成什么（目标设计）+ 每项状态 |
> | `STYLE_ENGINE_DEVLOG.md` | 实际**已经**实现/测试/变更/决策了什么 |
> | `knowledge/` + `tests/` | 实现本身及其可执行验证 |

## 状态标注图例

| 状态 | 含义 |
|---|---|
| `IMPLEMENTED` | 代码已实现，确定性测试通过（零 LLM） |
| `VALIDATED` | 已真实 LLM 运行并记录结果，行为符合设计 |
| `PARTIALLY_SUPPORTED` | 已实现/已验证，但结论部分成立（小样本弱证据或作者间不一致） |
| `PLANNED` | spec 已定义、尚未实现 |
| `V0.2 BACKLOG` | 明确推迟到 V0.2（不在 V0.1 冻结范围） |

---

## 1. 系统目标（V0.1）

确定性、可复现的文学风格分析/合成引擎。给定作者语料（train / held_out 分离），
系统应能：

1. 摄取语料（清洗→分块→QC）——确定性、原始文件只读。
2. 分层提取风格特征（统计 / NLP / LLM 判断 / 文体学 stylometric）。
3. 标定并聚合作者画像（Chunk → Work → Author，类型感知、不确定性一等）。
4. 把画像合成为可执行风格控制（StylePlan + PromptCompiler）。
5. 在**盲测**（不注入作者名 / 不写模仿指令）下按画像条件生成风格化文本。
6. 测量、评价、最小编辑改写，形成有界反馈闭环（内容完整性 P0 永不破坏）。
7. **新增作者不修改核心分析代码**，只需 author manifest + 语料（corpus registry 驱动）。

铁律（贯穿全文、落实为测试断言）：盲测 fail-closed；P0（故事情节/语义一致）永不
被低优先级风格编辑破坏；stylometric 指纹仅诊断、绝不生成改写指令；`DEEPSEEK_API_KEY`
只读（绝不打印/保存/提交）；绝不提交原始语料 / 生成正文 / 密钥 / 缓存 / 向量库；
不 merge main / 不提 PR。

---

## 2. 顶层章节索引（§一–§二十四）+ 状态

> 章节编号沿用既有代码/doc 中的引用（§一…§二十四，子节用阿拉伯数字）。此处给出每节
> 的范围与 V0.1 状态；详细设计见 DEVLOG/STATUS 对应 checkpoint。

| § | 范围 | V0.1 状态 |
|---|---|---|
| 一 | 反馈闭环正确性（决策/改写的四类完整性缺陷） | `IMPLEMENTED` + `VALIDATED`（Phase 8.1/8.2） |
| 二 | 决策策略统一配置（STEP 2）+ 生成参数一致性（两位作者一致） | `IMPLEMENTED` |
| 三 | 四阶决策 gate 顺序：Revision Effect → Content Integrity → Literary Quality → Style Fidelity | `IMPLEMENTED` |
| 四 | 生成顺序：plumbing（单次验证，Austen）→ 正式生成 | `IMPLEMENTED` + `VALIDATED` |
| 五 | 决策 gate 语义：Style 与 Literary **分别报告**，绝不合并加权分 | `IMPLEMENTED` |
| 六 | 改写效果的 canonical 归一化（排版标点+空白，不改词形/词序） | `IMPLEMENTED` |
| 七 | 策略合并的结构化映射（恰好一次、无幻觉/重复/丢失） | `IMPLEMENTED` + `VALIDATED`（Phase 4.5） |
| 八 | （§8.1 子节）决策完整性边界（unavailable 不伪造 0） | `IMPLEMENTED` |
| 九 | 确定性短路（integrity gate，省 token） | `IMPLEMENTED` |
| 十 | 首轮目标长度（500–800 English words）+ integrity | `IMPLEMENTED` |
| 十一 | token 节省：跳过昂贵 Layer B/C/Literary 重测 | `IMPLEMENTED` |
| 十二 | （PLANNED，未在既有产物中单独引用） | `PLANNED` |
| 十三 | 段级漂移定位（§15.4 的父节） | `IMPLEMENTED`（Phase 9.2） |
| 十四 | 决策 CASE 语义 + token 成本保护 + substantive_edit 门 | `IMPLEMENTED` |
| 十五 | 文体学诊断（§15.1/§15.4：段级 stylometric 漂移定位） | `IMPLEMENTED` |
| 十六 | 成本预检（真实 LLM 运行前显式批准） | `IMPLEMENTED`（门控流程） |
| 十七 | 测试契约（§17.1–17.6 逐条测试） | `IMPLEMENTED`（393 tests） |
| 十八 | 生成顺序（与 §四 一致） | `IMPLEMENTED` |
| 十九 | Post-run 审计（只记录+建议，不改核心逻辑） | `IMPLEMENTED` |
| 二十 | 合规（`max_iterations=1` 单轮） | `IMPLEMENTED` + `VALIDATED` |
| 二十一 | （PLANNED） | `PLANNED` |
| 二十二 | （PLANNED） | `PLANNED` |
| 二十三 | （PLANNED） | `PLANNED` |
| 二十四 | 收尾（§三–§二十四 全流程串接） | `IMPLEMENTED` |

> 注：§十二、§二十一、§二十二、§二十三在既有代码/文档中未被单独引用，重建时保守
> 标为 `PLANNED`，不臆造其内容；如需精确文本以原 spec 源为准（若存在）。

---

## 3. 核心组件状态明细

### 3.1 语料管线（Phase 1）— `IMPLEMENTED` + `VALIDATED`
RAW → CLEAN → CHUNKS（1000/2000/4000）→ METADATA/QC，确定性、原始只读、`data/` gitignore。
6 部 pilot：Austen P&P/Emma(train) + Persuasion(held_out)；Dickens GE/DC(train) + Tale(held_out)。

### 3.2 Corpus Registry（Generic Author Onboarding）— `IMPLEMENTED`（本轮新增）
**V0.1 核心验收项**：新增第三位作者**无需修改 Style Engine 核心分析代码**，只需
author manifest + 语料。

- `knowledge/corpus/manifest.py`：manifest schema（`MANIFEST_SCHEMA_VERSION=0.1.0`，
  `AuthorManifest`/`WorkManifest` + `parse_manifest`/`load_manifest_file`，JSON 规范、
  YAML 可选）。字段：`author_id`/`display_name`/`language`/`works[work_id/title/year/
  genre/filename/role(train|held_out)]`。校验 fail-closed（schema_version、必填非空、
  author_id/work_id 全局唯一、role ∈ {train,held_out}）。
- `knowledge/corpus/metadata.py`：`CORPUS` 由 `manifests/*.json` 数据驱动
  （`load_corpus()`）；`author_ids()` 派生作者全集；`works_from_manifest()` 展开单作者；
  `manifest_dir()` 支持 `WEAVER_MANIFEST_DIR` 覆盖。`WorkMetadata.year` 放宽为 `int|None`。
- `knowledge/ingestion/`（新包）：
  - `validate_author` / `register_author` / `build_author` / `onboard_author`。
  - 状态协议：`INVALID` / `READY_FOR_NEXT_STEP` / `REQUIRES_LLM_APPROVAL`。
  - 确定性部分（discover→clean→chunk→QC→metadata）复用 corpus 管线，零 LLM；
    需要 LLM 的后续步骤（采样→特征分析→聚合→策略合并→画像合成）**绝不自动执行、
    绝不自费调用**，返回 `REQUIRES_LLM_APPROVAL`。
  - CLI：`python -m knowledge.ingestion.add_author path/to/author_manifest.{json,yaml}`。
- 泛化：`planning/run.py:AUTHOR_IDS = author_ids()`；规划/生成/标定/评估报告表按注册作者
  循环（不再硬编码 Austen/Dickens 表头）；`discover`/`build_works` 支持作品子集。
- 兼容：既有 Austen/Dickens 行为不变（`author_ids()` → `("austen","dickens")`，
  6 作品元数据逐字段一致，`test_metadata.py` 原断言全绿）。

### 3.3 特征注册表（Phase 2）— `IMPLEMENTED`
39 特征、按 analyzer 名数据驱动路由（`MUST-3`）；`core` 保留，候选均为 `candidate_core`。

### 3.4 四层分析（Phase 3）— `IMPLEMENTED`（Layer A 统计 22 确定性 / Layer D 文体学）
Layer B（叙事）/ Layer C（策略）/ Layer A 判断类为 LLM，已在 40-chunk 采样上真实运行。

### 3.5 标定与聚合作者画像（Phase 4–5.1）— `IMPLEMENTED` + `VALIDATED`
40-chunk 采样标定（真实 deepseek-chat）；策略合并 Austen 51→26 / Dickens 44→36；
作者专属文体学目标（TRAIN-only，held-out fail-closed）；`reproducibility_hash` 字节级复现。

### 3.6 风格规划与提示词编译（Phase 6/6.1）— `IMPLEMENTED`
三层分离（画像→StylePlan→提示词）；TRAIN-only 经验 band 阈值；预算确定性降级、绝不硬截断。

### 3.7 风格条件生成（Phase 7/7.1）— `IMPLEMENTED` + `VALIDATED`
plumbing gate + 泄露守卫 A/B 分离 + `generation_condition_id`/`generation_id` 分离。

### 3.8 反馈闭环与文学评价（Phase 8–8.2）— `IMPLEMENTED` + `VALIDATED`
四阶 gate、Content Integrity 确定性短路、Revision Effect、真实端到端（austen_02 roll_back /
dickens_02 accept）。

### 3.9 多轮反馈（Phase 9.1）— `IMPLEMENTED`（确定性，未运行真实 LLM）
有界 while 闭环，`continue` 真正迭代，`roll_back` 保留 best-so-far。

### 3.10 段级 stylometric 漂移定位（Phase 9.2 / §15.4）— `IMPLEMENTED`
句子粒度漂移图（`segment_drift`），stylometric 仅诊断、绝不生成改写指令。

### 3.11 生成可控性（Phase 9.3 / §19.5）— `PARTIALLY_SUPPORTED`（如实记录）
强度旋钮 = 语言控制 `activation`（low→weak / medium→medium / high→strong）；重复采样
每档 n=3（首样本 + 2 fresh）。详见 §4。

---

## 4. Phase 9.3 最终结论 — `PARTIALLY_SUPPORTED`

**结论定性为 PARTIALLY_SUPPORTED**（Phase 9 到此封板）：

- **Dickens**：mean `low=0.154 → medium=0.127 → high=0.120`、median `0.153/0.125/0.123`，
  **mean 与 median 均单调递减**；n=3 后原趋势稳定复现，effect_size(low−high)=0.034。
- **Austen**：mean `low=0.172 → medium=0.216 → high=0.172`、median `0.159/0.194/0.165`，
  **mean 与 median 均 non-monotonic**——medium 档**系统性偏高**（low≈high，非干净单调），
  重复采样未把 medium 异常平均掉，且 medium std 最大（0.038）。
- **证据强度**：三者区间两两重叠、档间差异与档内散布同阶，n=3 仍是**小样本弱证据**；
  不设硬 pass/fail，不视为"失败"，是诚实的混合观测。

**后续决策（冻结，绝不自动执行）**：不扩大样本；不修改 intensity 控制算法；不修改
Austen medium 异常；`>3 样本正式统计` 与 `段级 drift 接入 RevisionPlanner` 进入
**V0.2 BACKLOG**。

---

## 5. V0.2 Backlog（明确推迟，不在 V0.1 冻结范围）

- >3 样本的正式统计（假设检验 / 功效分析 / 多轮方差分解）。
- 段级漂移图接入改写 planner 做**真段级目标编辑**（当前仍整段最小编辑，漂移图仅产出）。
- 全语料 LLM 特征提取（当前仅 40-chunk 采样有 LLM 特征）。
- Multi-author style mixing（`StylePlan.planner_metadata.conflicts` 已预留结构，未启用）。
- NlpAnalyzer（POS）特征（NLTK 有意不安装）。
- Mixed-effects / variance-decomposition 模型。

---

## 6. V0.1 冻结验收清单

| 验收项 | 状态 |
|---|---|
| 新增第三作者不修改核心分析代码（manifest + 语料即可） | ✅ `IMPLEMENTED`（第三作者 synthetic smoke test 通过） |
| Austen/Dickens 既有行为/画像/策略/规划/生成/评价兼容 | ✅（`author_ids()` 派生仍为两位，报告表泛化后逐字段一致） |
| 盲测 / P0 保护 / stylometric 仅诊断 / 密钥只读 | ✅（测试断言覆盖） |
| 需 LLM 的步骤返回 `READY_FOR_NEXT_STEP`/`REQUIRES_LLM_APPROVAL`/`INVALID`，绝不自动计费 | ✅ |
| Phase 9.3 如实记录 `PARTIALLY_SUPPORTED` | ✅（§4） |
| 完整测试通过 | ✅ 393 passed（含 11 新 onboarding 测试，零 LLM） |

**冻结判定：`READY_FOR_V0_1_FREEZE`** —— 见最终报告。
