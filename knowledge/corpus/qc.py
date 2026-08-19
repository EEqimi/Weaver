# knowledge/corpus/qc.py
"""质量检查（QC）：空块 / 异常尺寸 / 重复块 / Gutenberg 残留 / 编码异常。

全部确定性；返回可 JSON 序列化的结构。
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter

from .chunker import Chunk

# Gutenberg 残留与编码异常模式
_RESIDUE_PATTERNS: dict[str, re.Pattern] = {
    "project_gutenberg": re.compile(r"project gutenberg", re.IGNORECASE),
    "gutenberg_url": re.compile(r"www\.gutenberg\.org", re.IGNORECASE),
    "start_marker": re.compile(r"\*\*\*\s*START OF", re.IGNORECASE),
    "end_marker": re.compile(r"\*\*\*\s*END OF", re.IGNORECASE),
    "illustration": re.compile(r"\[Illustration", re.IGNORECASE),
    "contents_header": re.compile(r"^\s*contents\.?\s*$", re.IGNORECASE | re.MULTILINE),
    "preface_header": re.compile(r"^\s*preface\.?\s*$", re.IGNORECASE | re.MULTILINE),
    "replacement_char": re.compile("�"),
    "mojibake": re.compile(r"Ã|â€|Â "),
}


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def check_cleaned(work_id: str, text: str) -> dict:
    """对清洗后文本做残留/编码检查。"""
    hits: dict[str, int] = {}
    for name, pat in _RESIDUE_PATTERNS.items():
        n = len(pat.findall(text))
        if n:
            hits[name] = n
    return {
        "work_id": work_id,
        "residue": hits,
        "clean": not hits,
        "char_count": len(text),
        "word_count": len(text.split()),
    }


def check_chunks(work_id: str, chunks: list[Chunk], target_chars: int) -> dict:
    """对单档 chunk 做空块/异常尺寸/重复块检查。

    尺寸类别（区分"真正异常"与"预期的小尾巴"）：
        empty      词数为 0
        tiny       非空但 < 20 字符（章节副标题/极短对话片段）
        small_tail 20 ~ 0.5×target（章节结尾的自然尾巴，预期存在）
        oversized  > 2×target（真正异常，应为 0）
    """
    empty: list[str] = []
    tiny: list[dict] = []
    small_tail: list[dict] = []
    oversized: list[dict] = []
    seen: dict[str, str] = {}
    duplicates: list[dict] = []

    for c in chunks:
        if c.word_count == 0:
            empty.append(c.chunk_id)
        elif c.char_count < 20:
            tiny.append({"chunk_id": c.chunk_id, "char_count": c.char_count})
        elif c.char_count < target_chars * 0.5:
            small_tail.append({"chunk_id": c.chunk_id, "char_count": c.char_count})
        if c.char_count > target_chars * 2.0:
            oversized.append({"chunk_id": c.chunk_id, "char_count": c.char_count})
        h = _sha256(c.text)
        if h in seen:
            duplicates.append({"chunk_id": c.chunk_id, "dup_of": seen[h]})
        else:
            seen[h] = c.chunk_id

    sizes = [c.char_count for c in chunks]
    return {
        "work_id": work_id,
        "target_chars": target_chars,
        "total": len(chunks),
        "empty_count": len(empty),
        "empty": empty,
        "tiny_count": len(tiny),
        "tiny": tiny,
        "small_tail_count": len(small_tail),
        "oversized_count": len(oversized),
        "oversized": oversized,
        "duplicate_count": len(duplicates),
        "duplicates": duplicates,
        "size_min": min(sizes) if sizes else None,
        "size_max": max(sizes) if sizes else None,
        "size_mean": round(sum(sizes) / len(sizes), 1) if sizes else None,
    }


def run_work_qc(work_id: str, cleaned_text: str,
                chunks_by_size: dict[int, list[Chunk]]) -> dict:
    """汇总单部作品的 QC。"""
    return {
        "work_id": work_id,
        "cleaned": check_cleaned(work_id, cleaned_text),
        "chunks": {
            str(target): check_chunks(work_id, chunks_by_size[target], target)
            for target in sorted(chunks_by_size)
        },
    }
