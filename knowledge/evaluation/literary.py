# knowledge/evaluation/literary.py
"""Phase 8 独立 LLM 文学评价器（6 维 1–10 + 证据引文）。

与目标画像测量（Layer A/B/C）完全分离：这是**独立的文学评价**，不是风格特征测量。
README_AGENTS "评价迭代器" 的 6 维：情节逻辑 / 人物塑造 / 语言质地 / 主题表达 /
节奏 / 情感共鸣。

铁律：
    - 盲测默认：prompt 绝不含作者名或 "write like"/"imitate"/"in the style of"；
    - 每个维度至少一个优点 + 至少一个缺点 + 逐字证据引文（经 verify_evidence_quotes
      校验，未验证引文显式丢弃）；
    - 无 provider 时返回 AnalysisUnavailable，绝不伪造分数；
    - 结果按稳定缓存键缓存（text hash + 版本 + model/provider）。
"""
from __future__ import annotations

from ..analysis.base import AnalysisUnavailable, LLMResponseError, parse_json_response
from ..analysis.evidence import verify_evidence_quotes
from ..providers.llm_provider import LLMProvider, cache_key
from ..schema.versions import EVALUATION_SCHEMA_VERSION, LITERARY_EVALUATOR_VERSION
from .schema import (
    DEFAULT_DIMENSION_WEIGHTS, LITERARY_DIMENSIONS, DimensionScore,
    LiteraryEvaluation,
)

ANALYZER_ID = "LiteraryEvaluator"
ANALYZER_VERSION = LITERARY_EVALUATOR_VERSION

_SCORE_MIN = 1.0
_SCORE_MAX = 10.0


def _build_system_prompt() -> str:
    dims = "\n".join(f'  "{dim_id}": an object for "{label}"'
                     for dim_id, label in LITERARY_DIMENSIONS)
    return (
        "You are a literary critic. Evaluate a single text passage on exactly six "
        "dimensions of literary merit. Judge the passage on its own terms as a piece "
        "of writing; do NOT assume or mention the author's identity, and do NOT "
        "compare it to any specific author.\n"
        "For EACH dimension:\n"
        '  "score": a number from 1 (very weak) to 10 (outstanding) for THAT dimension,\n'
        '  "summary": 1-2 sentences explaining the score,\n'
        '  "strength": the passage\'s strongest aspect on that dimension,\n'
        '  "weakness": its clearest weakness on that dimension,\n'
        '  "evidence": an array of 1-3 SHORT VERBATIM quotes from the passage that '
        "support the judgment.\n"
        "Return ONLY a JSON object (no prose, no markdown fences) with exactly these "
        "keys:\n"
        '  "dimensions": an object with exactly these six keys:\n'
        f"{dims}\n"
        '  "summary": a concise 2-3 sentence overall assessment of the passage.\n'
        "Every evidence quote must be copied verbatim from the passage; do not invent "
        "or paraphrase quotes."
    )


class LiteraryEvaluator:
    """独立 LLM 文学评价器（盲测默认）。"""

    def __init__(self, provider: LLMProvider, blind: bool = True,
                 weights: dict[str, float] | None = None):
        self._provider = provider
        self.blind = blind
        self.weights = dict(weights if weights is not None else DEFAULT_DIMENSION_WEIGHTS)

    def evaluate(self, text: str, author_id: str = "",
                 passage_id: str = "") -> LiteraryEvaluation | AnalysisUnavailable:
        if not self._provider.is_configured():
            return AnalysisUnavailable("literary_evaluation", ANALYZER_ID,
                                       ANALYZER_VERSION, "未配置 LLM provider")
        messages = [
            {"role": "system", "content": _build_system_prompt()},
            {"role": "user", "content": f'PASSAGE:\n"""{text}"""'},
        ]
        key = cache_key(
            text=text, analyzer_id=ANALYZER_ID, analyzer_version=ANALYZER_VERSION,
            schema_version=EVALUATION_SCHEMA_VERSION, model=self._provider.model,
            provider_id=self._provider.provider_id,
            prompt_name=f"literary_evaluation:blind={self.blind}",
        )
        raw = self._provider.complete(messages, cache_hint=key)
        data = parse_json_response(raw)
        return self._to_evaluation(author_id, passage_id, text, data)

    # ------------------------------------------------------------------ #
    def _to_evaluation(self, author_id: str, passage_id: str, text: str,
                       data: dict) -> LiteraryEvaluation:
        raw_dims = data.get("dimensions")
        if not isinstance(raw_dims, dict):
            raise LLMResponseError("literary_evaluation 的 dimensions 必须是对象")

        dimensions: dict[str, DimensionScore] = {}
        for dim_id, label in LITERARY_DIMENSIONS:
            if dim_id not in raw_dims:
                raise LLMResponseError(
                    f"literary_evaluation 缺少维度 {dim_id}")
            dimensions[dim_id] = self._dimension(dim_id, label, raw_dims[dim_id], text)

        summary = data.get("summary", "")
        if not isinstance(summary, str) or not summary.strip():
            raise LLMResponseError("literary_evaluation 的 summary 必须是非空字符串")

        total = self._weighted_total(dimensions)
        return LiteraryEvaluation(
            schema_version=EVALUATION_SCHEMA_VERSION,
            author_id=author_id, passage_id=passage_id,
            dimensions=dimensions, weights=self.weights,
            total_score=total, summary=summary.strip(),
            blind=self.blind, evaluator_version=ANALYZER_VERSION,
        )

    def _dimension(self, dim_id: str, label: str, raw: dict,
                   text: str) -> DimensionScore:
        if not isinstance(raw, dict):
            raise LLMResponseError(f"维度 {dim_id} 必须是对象")
        score = raw.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise LLMResponseError(f"维度 {dim_id} 的 score 必须是数值")
        score = float(score)
        if not _SCORE_MIN <= score <= _SCORE_MAX:
            raise LLMResponseError(
                f"维度 {dim_id} 的 score={score} 越界（应 ∈ [1,10]）")

        def _text_field(key: str) -> str:
            v = raw.get(key)
            if not isinstance(v, str) or not v.strip():
                raise LLMResponseError(f"维度 {dim_id} 的 {key} 必须是非空字符串")
            return v.strip()

        summary = _text_field("summary")
        strength = _text_field("strength")
        weakness = _text_field("weakness")

        raw_evidence = raw.get("evidence", [])
        if not isinstance(raw_evidence, list) or not all(isinstance(e, str)
                                                         for e in raw_evidence):
            raise LLMResponseError(f"维度 {dim_id} 的 evidence 必须是字符串列表")
        check = verify_evidence_quotes(raw_evidence, text)
        evidence = [e for e in check.verified if isinstance(e, str)]

        return DimensionScore(
            dimension=dim_id, label=label, score=score, summary=summary,
            strength=strength, weakness=weakness, evidence=evidence,
        )

    def _weighted_total(self, dimensions: dict[str, DimensionScore]) -> float:
        total_w = 0.0
        weighted = 0.0
        for dim_id, dim in dimensions.items():
            w = float(self.weights.get(dim_id, 0.0))
            weighted += dim.score * w
            total_w += w
        if total_w <= 0.0:
            return round(sum(d.score for d in dimensions.values()) / len(dimensions), 2)
        return round(weighted / total_w, 2)
