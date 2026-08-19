# knowledge/corpus/chunker.py
"""确定性分块：章节感知 + 段落优先，不调用 LLM。

规则：
    1. 以空行为界识别段落，重建被硬换行切断的连续行（含去连字符）；
    2. 章节标题（CHAPTER ...）开启新章节，分块绝不跨章节；
    3. 优先在段落边界切分（贪心填充到 target 字符附近）；
    4. 超长段落（> target * MAX_CHUNK_RATIO）才在句末切分；
    5. 减少小余块：允许"轻微超 target 合并"（仍 <= MAX_CHUNK_RATIO）以避免
       小段落被孤立成微型块；句末切分的最后一片并入缓冲区与后续段落合并，
       而非单独成块（章节结尾的天然小尾巴除外）；
    6. 完全确定性：同一输入 → 同一输出。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .cleaner import is_chapter_heading
from ..config import MAX_CHUNK_RATIO

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    work_id: str
    target_chars: int
    chapter: int          # 1-based；文本在首个章节标题之前则为 0
    seq: int              # 工作内序号
    text: str
    char_count: int
    word_count: int


def _join_wrapped_lines(lines: list[str]) -> str:
    """重建被硬换行切断的连续行，并处理 "word-\nword" 连字符。"""
    parts: list[str] = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if parts and parts[-1].endswith("-") and s and s[0].islower():
            parts[-1] = parts[-1][:-1] + s   # 去连字符
        else:
            parts.append(s)
    return " ".join(parts)


def _paragraphs_by_chapter(text: str) -> list[tuple[int, str]]:
    """返回 [(chapter_index, paragraph_text)]，按行检测章节边界。"""
    result: list[tuple[int, str]] = []
    chapter = 0
    para_lines: list[str] = []

    def flush() -> None:
        nonlocal para_lines
        if para_lines:
            result.append((chapter, _join_wrapped_lines(para_lines)))
            para_lines = []

    for line in text.split("\n"):
        stripped = line.strip()
        if stripped == "":
            flush()
            continue
        if is_chapter_heading(stripped):
            flush()
            chapter += 1
            # 章节标题仅作为边界信号，不进入 chunk 正文（避免仅含标题的微型 chunk
            # 与跨卷重排导致的标题文本重复）
            continue
        para_lines.append(stripped)
    flush()
    return result


def _split_long_paragraph(para: str, target: int) -> list[str]:
    """超长段落在句末切分，尽量接近 target 字符。

    除最后一片（余片）外，各片大小接近 target；余片由调用方并入缓冲区。
    """
    sentences = _SENTENCE_SPLIT_RE.split(para)
    pieces: list[str] = []
    buf = ""
    for s in sentences:
        if buf and len(buf) + len(s) + 1 > target:
            pieces.append(buf)
            buf = s
        else:
            buf = f"{buf} {s}".strip() if buf else s
    if buf:
        pieces.append(buf)
    return pieces


def _make_chunk(work_id: str, target: int, chapter: int, seq: int,
                text: str) -> Chunk:
    return Chunk(
        chunk_id=f"{work_id}__{target}__{seq:04d}",
        work_id=work_id,
        target_chars=target,
        chapter=chapter,
        seq=seq,
        text=text,
        char_count=len(text),
        word_count=len(text.split()),
    )


def chunk_text(text: str, work_id: str, target_chars: int) -> list[Chunk]:
    """将清洗后的文本切成约 target_chars 字符的 chunk（章节内、段落优先）。

    小余块控制（规则 5）：
        - 段落层面：当"再塞一段会超过 target"时，若合并后仍在硬上限内
          （target * MAX_CHUNK_RATIO）且比"孤立当前缓冲区"更接近 target，
          则合并（允许轻微超 target），避免小段落被孤立；
        - 超长段落：句末切分后的最后一片不单独成块，而是并入缓冲区，
          与后续段落合并，杜绝句切分产生的小余块；
        - 绝不跨章节；完全确定性。
    """
    paras = _paragraphs_by_chapter(text)
    hard = target_chars * MAX_CHUNK_RATIO
    chunks: list[Chunk] = []
    buf_chapter: int | None = None
    buf_paras: list[str] = []
    buf_chars = 0
    seq = 0

    def flush(chapter: int) -> None:
        nonlocal seq
        if buf_paras:
            chunks.append(_make_chunk(work_id, target_chars, chapter or 0,
                                      seq, "\n\n".join(buf_paras)))
            seq += 1

    def emit_text(chapter: int, txt: str) -> None:
        nonlocal seq
        chunks.append(_make_chunk(work_id, target_chars, chapter, seq, txt))
        seq += 1

    for chapter, para in paras:
        plen = len(para)

        # 章节切换：先结掉上一章缓冲区
        if buf_chapter is not None and chapter != buf_chapter:
            flush(buf_chapter)
            buf_paras, buf_chars, buf_chapter = [], 0, chapter
        if buf_chapter is None:
            buf_chapter = chapter

        # 超长段落：句末切分；最后一片并入缓冲区以便与后续段落合并
        if plen > hard:
            pieces = _split_long_paragraph(para, target_chars)
            if buf_paras:
                if buf_chars + len(pieces[0]) <= hard:
                    # 前导缓冲区并入首片，避免缓冲区被孤立成小块
                    pieces[0] = "\n\n".join(buf_paras) + "\n\n" + pieces[0]
                    buf_paras, buf_chars = [], 0
                else:
                    flush(chapter)
                    buf_paras, buf_chars = [], 0
            for piece in pieces[:-1]:
                emit_text(chapter, piece)
            buf_paras = [pieces[-1]]
            buf_chars = len(pieces[-1])
            continue

        # 段落优先：贪心填充
        if not buf_paras:
            buf_paras, buf_chars = [para], plen
        elif buf_chars + plen <= target_chars:
            buf_paras.append(para)
            buf_chars += plen
        elif (buf_chars + plen <= hard
              and (buf_chars + plen - target_chars) <= (target_chars - buf_chars)):
            # 合并比孤立当前缓冲区更接近 target → 轻微超 target 后结块
            buf_paras.append(para)
            buf_chars += plen
            flush(chapter)
            buf_paras, buf_chars = [], 0
        else:
            flush(chapter)
            buf_paras, buf_chars = [para], plen

    flush(buf_chapter or 0)
    return chunks
