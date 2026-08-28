# knowledge/planning/compiler.py
"""Phase 6 PromptCompiler：StylePlan → CompiledPrompt（生成提示词）。

三层分离的最后一步：把"本次要激活的控制"翻译成生成模型可执行的指令。
铁律（spec §19–§21）：
    - 绝不只写 "write like <author>"；绝不提作者名或"模仿"。
    - 绝不 dump 画像 JSON；绝不写微观 stylometric 指令（功能词 / 字符 n-gram /
      PCA / centroid）。
    - 绝不改写用户 core story facts / 人物关系 / 约束（CONTENT 原样保留）。
    - 确定性与预算：同一 plan + policy → 同一 prompt；超出 max_prompt_chars 时
      确定性截断（从末尾策略起丢弃）并记录，绝不静默。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..schema.versions import PROMPT_COMPILER_VERSION
from .policy import _NARRATIVE_VALUE_LABELS
from .schema import PlannerPolicy, StylePlan

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


@dataclass
class CompiledPrompt:
    """compiler 输出：分节 + 拼装好的文本（不生成正文，只生成提示词）。"""
    author_id: str
    style_plan_id: str
    compiler_version: str
    sections: list[dict[str, str]] = field(default_factory=list)
    text: str = ""
    char_count: int = 0
    truncated: bool = False
    truncation_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "author_id": self.author_id,
            "style_plan_id": self.style_plan_id,
            "compiler_version": self.compiler_version,
            "sections": [dict(s) for s in self.sections],
            "text": self.text,
            "char_count": self.char_count,
            "truncated": self.truncated,
            "truncation_note": self.truncation_note,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CompiledPrompt":
        return cls(
            author_id=d["author_id"], style_plan_id=d["style_plan_id"],
            compiler_version=d["compiler_version"],
            sections=[dict(s) for s in d["sections"]], text=d["text"],
            char_count=d["char_count"], truncated=d["truncated"],
            truncation_note=d["truncation_note"],
        )


class PromptCompiler:
    """确定性编译器。"""

    def __init__(self, policy: PlannerPolicy | None = None) -> None:
        self.policy = policy or PlannerPolicy()

    # ------------------------------------------------------------------ #
    def compile(self, plan: StylePlan) -> CompiledPrompt:
        active = [s for s in plan.strategy_controls if s.activation == "active"]
        bullets = [self._strategy_bullet(s) for s in active]

        sections: list[dict[str, str]] = [
            {"heading": "ROLE", "body": self._role()},
            {"heading": "CONTENT", "body": self._content(plan)},
            {"heading": "STYLE CONTROL", "body": self._style_control(plan)},
            {"heading": "NARRATIVE", "body": self._narrative(plan)},
        ]

        text = self._assemble(sections + [
            {"heading": "CONDITIONAL STRATEGIES", "body": self._strategies_body(bullets)},
            {"heading": "IMPORTANT", "body": self._important()},
        ])

        # 确定性预算截断：从末尾策略起丢弃；仍超则硬截断（记录，不静默）。
        truncated = False
        truncation_note = ""
        while bullets and len(text) > self.policy.max_prompt_chars:
            bullets.pop()
            text = self._assemble(sections + [
                {"heading": "CONDITIONAL STRATEGIES", "body": self._strategies_body(bullets)},
                {"heading": "IMPORTANT", "body": self._important()},
            ])
            truncated = True
            truncation_note = (
                f"dropped {len(active) - len(bullets)} trailing strategy/strategies "
                "to fit prompt budget")
        if len(text) > self.policy.max_prompt_chars:
            text = text[:self.policy.max_prompt_chars]
            truncated = True
            truncation_note = (truncation_note + "; hard-truncated").strip("; ")

        return CompiledPrompt(
            author_id=plan.author_id,
            style_plan_id=plan.style_plan_id,
            compiler_version=PROMPT_COMPILER_VERSION,
            sections=sections,
            text=text,
            char_count=len(text),
            truncated=truncated,
            truncation_note=truncation_note,
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _assemble(sections: list[dict[str, str]]) -> str:
        return "\n\n".join(
            f"## {s['heading']}\n{s['body']}" for s in sections)

    @staticmethod
    def _role() -> str:
        return (
            "You are an accomplished prose writer. Write the passage requested in "
            "CONTENT. Follow the stylistic tendencies in STYLE CONTROL and NARRATIVE "
            "as general guidance. Apply a CONDITIONAL STRATEGY only when its stated "
            "trigger actually occurs in the scene; otherwise ignore it. Do not invent "
            "facts, characters, or relationships beyond the brief."
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
    def _style_control(plan: StylePlan) -> str:
        lines: list[str] = []
        for c in plan.language_controls:
            prefix = _ACTIVATION_PREFIX.get(c.activation, "- ")
            lines.append(f"- {prefix}{c.guidance}")
        return "\n".join(lines) if lines else "(no activated language controls)"

    def _narrative(self, plan: StylePlan) -> str:
        req = plan.writing_request
        lines: list[str] = []
        pov = req.get("pov")
        for nc in plan.narrative_controls:
            if nc.overridden and nc.field == "pov":
                labels = _NARRATIVE_VALUE_LABELS.get("pov", {})
                lines.append(
                    f"- Point of view: {labels.get(pov, str(pov))} "
                    "(explicit user requirement)")
            elif nc.activation == "medium":
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
    def _important() -> str:
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
