# knowledge/planning/compiler.py
"""Phase 6 PromptCompiler：StylePlan → CompiledPrompt（生成提示词）。

三层分离的最后一步：把"本次要激活的控制"翻译成生成模型可执行的指令。
铁律（spec §19–§21）：
    - 绝不只写 "write like <author>"；绝不提作者名或"模仿"。
    - 绝不 dump 画像 JSON；绝不写微观 stylometric 指令（功能词 / 字符 n-gram /
      PCA / centroid）。
    - 绝不改写用户 core story facts / 人物关系 / 约束（CONTENT 原样保留）。

Phase 6.1 预算铁律（spec §3）：
    - **绝不硬截断用户内容**。超出 `max_prompt_chars` 时按确定性顺序降级：
      (1) 丢弃最低优先级条件策略 → (2) 丢弃 secondary 语言控制 → (3) 丢弃最弱语言
      控制 → (4) 移除可选解释措辞 → (5) 强制内容仍放不下则抛 `PromptBudgetError`。
    - 每次移除都记录进 `removed_controls`（绝不静默）。
    - `sections` 精确重构 `text`（6 段全保留，`_assemble(sections) == text`）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..schema.versions import PROMPT_COMPILER_VERSION
from .policy import _NARRATIVE_VALUE_LABELS
from .schema import PlannerPolicy, PromptBudgetError, StylePlan

# 叙事字段 → 人类可读维度名（供 NARRATIVE 段用）
_NARRATIVE_FIELD_LABELS: dict[str, str] = {
    "pov": "Point of view",
    "narrator_presence": "Narrator presence",
    "focalization": "Focalization",
    "narrative_distance": "Narrative distance",
    "perspective_stability": "Viewpoint stability",
    "information_access": "Information access",
    "temporal_order": "Temporal order",
    "narrator_evaluative_intervention": "Narrator intervention",
    "temporal_pace": "Pacing",
    "scene_detail": "Scene emphasis",
}

_ACTIVATION_PREFIX: dict[str, str] = {
    "strong": "Strongly prefer: ",
    "medium": "Tend toward: ",
    "weak": "As a general tendency: ",
}

# 语言控制"最弱者"优先被丢弃的强度序（数字越大越先丢）。
_ACTIVATION_DROP_RANK: dict[str, int] = {"weak": 2, "medium": 1, "strong": 0}


@dataclass
class CompiledPrompt:
    """compiler 输出：分节 + 拼装好的文本（不生成正文，只生成提示词）。

    Phase 6.1：`degraded` / `removed_controls` / `degradation_note` 取代原先的
    `truncated` / `truncation_note`——预算问题用"降级"解决，绝不截断用户内容。
    """
    author_id: str
    style_plan_id: str
    compiler_version: str
    sections: list[dict[str, str]] = field(default_factory=list)
    text: str = ""
    char_count: int = 0
    degraded: bool = False
    removed_controls: list[str] = field(default_factory=list)
    degradation_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "author_id": self.author_id,
            "style_plan_id": self.style_plan_id,
            "compiler_version": self.compiler_version,
            "sections": [dict(s) for s in self.sections],
            "text": self.text,
            "char_count": self.char_count,
            "degraded": self.degraded,
            "removed_controls": list(self.removed_controls),
            "degradation_note": self.degradation_note,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CompiledPrompt":
        return cls(
            author_id=d["author_id"], style_plan_id=d["style_plan_id"],
            compiler_version=d["compiler_version"],
            sections=[dict(s) for s in d["sections"]], text=d["text"],
            char_count=d["char_count"], degraded=d["degraded"],
            removed_controls=list(d["removed_controls"]),
            degradation_note=d["degradation_note"],
        )


class PromptCompiler:
    """确定性编译器。"""

    def __init__(self, policy: PlannerPolicy | None = None) -> None:
        self.policy = policy or PlannerPolicy()

    # ------------------------------------------------------------------ #
    def compile(self, plan: StylePlan) -> CompiledPrompt:
        budget = self.policy.max_prompt_chars
        removed: list[str] = []
        degraded = False

        strategies = [s for s in plan.strategy_controls if s.activation == "active"]
        strategy_bullets = [self._strategy_bullet(s) for s in strategies]
        # plan.language_controls 已按 primary → secondary 稳定排序。
        language = list(plan.language_controls)
        narrative = [n for n in plan.narrative_controls if n.activation == "medium"]

        def build(verbose: bool = True) -> tuple[list[dict[str, str]], str]:
            sections = [
                {"heading": "ROLE", "body": self._role(verbose)},
                {"heading": "CONTENT", "body": self._content(plan)},
                {"heading": "STYLE CONTROL", "body": self._style_control(language)},
                {"heading": "NARRATIVE", "body": self._narrative(narrative)},
                {"heading": "CONDITIONAL STRATEGIES", "body": self._strategies_body(strategy_bullets)},
                {"heading": "IMPORTANT", "body": self._important(verbose)},
            ]
            return sections, self._assemble(sections)

        sections, text = build(verbose=True)

        # (1) 丢弃最低优先级条件策略（从末尾起，即 priority 最低者）
        while len(text) > budget and strategy_bullets:
            dropped = strategies[len(strategy_bullets) - 1]
            strategy_bullets.pop()
            removed.append(f"strategy:{dropped.canonical_strategy_id}")
            degraded = True
            sections, text = build()

        # (2) 丢弃 secondary 语言控制（从末尾起）
        while len(text) > budget and any(c.bucket == "secondary" for c in language):
            idx = max(i for i, c in enumerate(language) if c.bucket == "secondary")
            removed.append(f"language:{language[idx].feature_id}")
            language.pop(idx)
            degraded = True
            sections, text = build()

        # (3) 丢弃最弱语言控制（weak → medium → strong，同层取末位者）
        while len(text) > budget and language:
            idx = self._drop_weakest(language)
            removed.append(f"language:{language[idx].feature_id}")
            language.pop(idx)
            degraded = True
            sections, text = build()

        # (4) 移除可选解释措辞（ROLE / IMPORTANT 的精简变体）
        if len(text) > budget:
            sections, text = build(verbose=False)
            if len(text) <= budget:
                removed.append("wording:verbose")
                degraded = True

        # (5) 强制内容仍放不下 → 显式失败，绝不硬截断用户内容
        if len(text) > budget:
            raise PromptBudgetError(
                f"mandatory prompt content ({len(text)} chars) cannot fit within "
                f"max_prompt_chars={budget}; removed={removed}")

        return CompiledPrompt(
            author_id=plan.author_id,
            style_plan_id=plan.style_plan_id,
            compiler_version=PROMPT_COMPILER_VERSION,
            sections=sections,
            text=text,
            char_count=len(text),
            degraded=degraded,
            removed_controls=removed,
            degradation_note=self._degradation_note(removed),
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _assemble(sections: list[dict[str, str]]) -> str:
        return "\n\n".join(
            f"## {s['heading']}\n{s['body']}" for s in sections)

    @staticmethod
    def _drop_weakest(language: list[Any]) -> int:
        """返回最弱语言控制的索引（weak 最弱先丢；同层取末位，保序稳定）。"""
        weakest = max(_ACTIVATION_DROP_RANK.get(c.activation, 0) for c in language)
        return max(i for i, c in enumerate(language)
                   if _ACTIVATION_DROP_RANK.get(c.activation, 0) == weakest)

    @staticmethod
    def _degradation_note(removed: list[str]) -> str:
        if not removed:
            return ""
        n_strategy = sum(1 for r in removed if r.startswith("strategy:"))
        n_lang = sum(1 for r in removed if r.startswith("language:"))
        n_wording = sum(1 for r in removed if r.startswith("wording:"))
        parts: list[str] = []
        if n_strategy:
            parts.append(f"dropped {n_strategy} conditional "
                         f"strateg{'y' if n_strategy == 1 else 'ies'}")
        if n_lang:
            parts.append(f"dropped {n_lang} language "
                         f"control{'s' if n_lang != 1 else ''}")
        if n_wording:
            parts.append("removed optional explanatory wording")
        return "; ".join(parts) + " to fit prompt budget (no content truncated)"

    # ------------------------------------------------------------------ #
    @staticmethod
    def _role(verbose: bool = True) -> str:
        if verbose:
            return (
                "You are an accomplished prose writer. Write the passage requested in "
                "CONTENT. Follow the stylistic tendencies in STYLE CONTROL and NARRATIVE "
                "as general guidance. Apply a CONDITIONAL STRATEGY only when its stated "
                "trigger actually occurs in the scene; otherwise ignore it. Do not invent "
                "facts, characters, or relationships beyond the brief."
            )
        return (
            "You are a prose writer. Write the passage in CONTENT, following STYLE "
            "CONTROL and NARRATIVE as guidance, and apply CONDITIONAL STRATEGIES only "
            "when their trigger occurs."
        )

    def _content(self, plan: StylePlan) -> str:
        req = plan.writing_request
        lines: list[str] = [str(req.get("content", "")).strip()]
        length = f"Length: {req.get('desired_length', 'short_scene')}"
        if req.get("target_words") is not None:
            length += f", approximately {req['target_words']} words"
        lines.append(length)
        lines.append(f"Language: {req.get('language', 'english')}")
        pov = req.get("pov")
        if pov is not None:
            labels = _NARRATIVE_VALUE_LABELS.get("pov", {})
            lines.append(f"Point of view: {labels.get(pov, str(pov))}")
        constraints = req.get("constraints") or []
        if constraints:
            lines.append("Constraints:")
            lines.extend(f"- {c}" for c in constraints)
        return "\n".join(lines)

    @staticmethod
    def _style_control(language: list[Any]) -> str:
        lines: list[str] = []
        for c in language:
            prefix = _ACTIVATION_PREFIX.get(c.activation, "- ")
            lines.append(f"- {prefix}{c.guidance}")
        return "\n".join(lines) if lines else "(no activated language controls)"

    @staticmethod
    def _narrative(narrative: list[Any]) -> str:
        lines: list[str] = []
        for nc in narrative:
            label = _NARRATIVE_FIELD_LABELS.get(nc.field, nc.field)
            lines.append(f"- {label}: {nc.guidance}")
        return "\n".join(lines) if lines else "(no activated narrative controls)"

    @staticmethod
    def _strategies_body(bullets: list[str]) -> str:
        return "\n\n".join(bullets) if bullets else "(none)"

    @staticmethod
    def _strip_trigger_prefix(s: str) -> str:
        t = s.strip()
        for prefix in ("Whenever ", "whenever ", "When ", "when ", "If ", "if "):
            if t.startswith(prefix):
                return t[len(prefix):]
        return t

    @classmethod
    def _lcfirst(cls, s: str) -> str:
        t = s.strip()
        return t[:1].lower() + t[1:] if t else t

    @classmethod
    def _strategy_bullet(cls, s: Any) -> str:
        # trigger_summary 常以 "When ..." 开头，剥离后用小写接 "WHEN" 标记，避免 "WHEN When"。
        trigger = cls._lcfirst(cls._strip_trigger_prefix(s.trigger))
        operation = cls._lcfirst(s.operation)
        effect = cls._lcfirst(s.effect)
        return (
            f"- {s.canonical_name}\n"
            f"  WHEN {trigger}\n"
            f"  THEN {operation}\n"
            f"  TO {effect}"
        )

    @staticmethod
    def _important(verbose: bool = True) -> str:
        if verbose:
            return (
                "- Do not mention any author's name, and do not attempt to imitate or "
                "\"write like\" a named author.\n"
                "- Do not copy or reproduce wording from any source text; write original "
                "sentences.\n"
                "- Preserve the plot, characters, facts, and constraints in CONTENT exactly "
                "as given.\n"
                "- The stylistic guidance describes broad tendencies, not micro-level rules; "
                "do not apply mechanical, statistical, or character-level constraints of any "
                "kind.\n"
                "- Apply CONDITIONAL STRATEGIES only when their trigger actually occurs."
            )
        return (
            "- Do not mention or imitate any named author.\n"
            "- Do not copy any source text; write original sentences.\n"
            "- Preserve CONTENT exactly.\n"
            "- Apply CONDITIONAL STRATEGIES only when their trigger occurs."
        )
