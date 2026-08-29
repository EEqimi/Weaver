# tests/test_revision_rewriter.py
"""RevisionRewriter 的 revision max_tokens 契约测试（零 LLM / 零 token / 确定性）。

真实验收第二轮（post-freeze hotfix）根因：改写器默认沿用 provider 的 2048 completion
tokens，导致约 1000–1500 words 的 revised_text + JSON wrapper + change_descriptions 被
截断（raw 以 '{\n "revised_text": "..."' 开头却缺结尾 '}'），parse_json_response 正确
fail-closed 抛 LLMResponseError。

本文件锁定三条契约：
    A. RevisionRewriter 显式传递更大的 revision max_tokens（不沿用默认 2048）。
    B. 该 max_tokens 足够高于 2048，且进入 revision cache key（避免旧截断缓存误命中）。
    C. 约 1500 words 的长 revised_text + change_descriptions 的完整 JSON 可正常 parse
       并生成 RevisionResult（不被截断）。

绝不调用真实模型；绝不读 DEEPSEEK_API_KEY。
"""
import json
from types import SimpleNamespace

from knowledge.analysis.base import parse_json_response
from knowledge.evaluation.revision import REVISION_MAX_TOKENS, RevisionRewriter
from knowledge.evaluation.schema import RevisionItem, RevisionResult
from knowledge.providers.llm_provider import cache_key


def _plan():
    return SimpleNamespace(
        author_id="austen", passage_id="g1", style_plan_id="sp1",
        revision_items=[RevisionItem(
            priority="P3", category="language", target="f",
            instruction="Use dialogue.", reason="r")])


class _RecordingProvider:
    """记录 complete 收到的 kwargs / cache_hint，返回一段合法 revision JSON。"""

    provider_id = "deepseek"
    model = "deepseek-chat"

    def __init__(self):
        self.kwargs = None
        self.cache_hint = None

    def is_configured(self):
        return True

    def complete(self, messages, *, cache_hint=None, **kwargs):
        self.kwargs = kwargs
        self.cache_hint = cache_hint
        return json.dumps({"revised_text": "Revised text.",
                           "change_descriptions": ["changed a phrase"]})


# --------------------------------------------------------------------------- #
# A. RevisionRewriter 显式传递 revision max_tokens
# --------------------------------------------------------------------------- #
def test_rewriter_passes_revision_max_tokens():
    provider = _RecordingProvider()
    rw = RevisionRewriter(provider, blind=True)

    result = rw.rewrite("Original text.", _plan(), author_names=[])

    assert isinstance(result, RevisionResult)
    assert result.revised_text == "Revised text."
    # 必须显式传递 revision max_tokens（而非沿用 provider 默认 2048）。
    assert provider.kwargs["max_tokens"] == REVISION_MAX_TOKENS
    # cache_hint 已生成（进入缓存），供 B 验证 max_tokens 纳入 cache key。
    assert provider.cache_hint


# --------------------------------------------------------------------------- #
# B. max_tokens 足够高于默认 2048 且进入 revision cache key
# --------------------------------------------------------------------------- #
def test_revision_max_tokens_exceeds_default_and_enters_cache_key():
    assert REVISION_MAX_TOKENS > 2048

    base = dict(text="original text", analyzer_id="RevisionRewriter",
                analyzer_version="0.1.0", schema_version="0.2.0",
                model="deepseek-chat", provider_id="deepseek",
                prompt_name="revision:blind=True:n_items=1")
    k_without = cache_key(**base)
    k_with = cache_key(**base, extra={"max_tokens": REVISION_MAX_TOKENS})
    # max_tokens 进入 cache key → 同一 prompt 不同 max_tokens 不会命中旧（截断）缓存。
    assert k_without != k_with


# --------------------------------------------------------------------------- #
# C. 长 revised_text（约 1500 words）+ change_descriptions 完整 JSON 可 parse
# --------------------------------------------------------------------------- #
def test_rewriter_parses_long_revision_payload():
    long_text = " ".join(f"token{i}" for i in range(1500))
    payload = json.dumps({
        "revised_text": long_text,
        "change_descriptions": ["tightened wording", "smoothed rhythm"],
    })

    data = parse_json_response(payload)
    result = RevisionRewriter(None)._to_result(_plan(), "Original.", data)

    assert isinstance(result, RevisionResult)
    assert result.revised_text == long_text
    assert len(result.revised_text.split()) == 1500
    assert result.claimed_change_descriptions == ["tightened wording",
                                                  "smoothed rhythm"]
