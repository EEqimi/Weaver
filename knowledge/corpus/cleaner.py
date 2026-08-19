# knowledge/corpus/cleaner.py
"""Project Gutenberg 文本清洗：确定性、可复现、不调用 LLM。

V0.1 范围：本 cleaner 面向 Project Gutenberg 的"章节体"英文小说（以
`CHAPTER ...` 为章节标题、含 `*** START/END OF ... GUTENBERG EBOOK ***`
头尾标记与 TOC/序言/图注前置物）。它**不是**通用文本清洗器：对无章节
标记、无 Gutenberg 头尾、或非英文的语料不保证正确。

扩展点：当前 `clean(raw_text, work_id)` 是唯一入口，由 pipeline 直接调用。
在扩展到更多来源/格式前，应把 cleaner 抽象为按 work 元数据（language/format/
source）选择的注册表（与 FeatureRegistry 同构），本模块的 `clean` 仅作为
"gutenberg_chaptered_v0.1" 实现之一。

处理步骤（对每个文件顺序执行）：
    1. 换行归一化（CRLF/LF → LF）与 Unicode NFC 归一化；
    2. 截取 START/END 标记之间（去除 Gutenberg 头尾 license 文本）；
    3. 去除前置物（标题页/序言/目录 TOC/插图页）：定位"第一个后接正文的章节标题"；
    4. 去除 [Illustration ... ] 图注块（单行与多行）；
    5. 压缩多余空行、去掉行尾空格，保留段落边界与标点。

正文起点判定（step 3）依赖一条确定性启发：从 START 之后向后扫描，
第一个"后接正文段落（>=40 字符的非标题/非图注/非分隔行）"的章节标题即正文起点。
该规则已对六部 pilot 小说验证（含 P&P 无 TOC、有 Saintsbury 序言的特殊情况）。
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Gutenberg 起止标记（容错大小写与 THE/THIS 前缀）
_START_RE = re.compile(
    r"^[ \t]*\*{3}\s*START OF (?:THE |THIS )?PROJECT GUTENBERG EBOOK",
    re.IGNORECASE | re.MULTILINE,
)
_END_RE = re.compile(
    r"^[ \t]*\*{3}\s*END OF (?:THE |THIS )?PROJECT GUTENBERG EBOOK",
    re.IGNORECASE | re.MULTILINE,
)

# 章节标题（CHAPTER/Chapter + 罗马或阿拉伯数字，容错句点/方括号/后接标题）
_CHAPTER_RE = re.compile(r"^\s*CHAPTER\s+([IVXLCDM]+|\d+)\b", re.IGNORECASE)

# 卷/册分隔行（Book the First... / VOLUME I. / PART ...）
_DIVIDER_RE = re.compile(
    r"^\s*(BOOK\s+(THE\s+)?(FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH)\b"
    r"|VOLUME\s+([IVXLCDM]+|\d+)\b"
    r"|PART\s+([IVXLCDM]+|\d+)\b)",
    re.IGNORECASE,
)

# 图注块起始（单行 [Illustration] 或多行 [Illustration: ... ]）
_ILLUSTRATION_START_RE = re.compile(r"^\s*\[Illustration\b", re.IGNORECASE)

# 正文段落最小字符数（用于区分章节标题/卷分隔与真实正文）
_PROSE_MIN_CHARS = 40


@dataclass
class CleanedText:
    work_id: str
    text: str
    char_count: int
    word_count: int
    header_removed: bool
    footer_removed: bool
    front_matter_removed: bool


def _is_chapter(line: str) -> bool:
    return bool(_CHAPTER_RE.match(line))


def is_chapter_heading(line: str) -> bool:
    """公开接口：判断一行是否为章节标题（供 chunker 复用）。"""
    return _is_chapter(line)


def _is_divider(line: str) -> bool:
    return bool(_DIVIDER_RE.match(line))


def _is_illustration(line: str) -> bool:
    return bool(_ILLUSTRATION_START_RE.match(line))


def _find_body_start(lines: list[str]) -> int:
    """返回正文第一段章节标题所在行索引（找不到则返回 0）。

    判定规则：从某章节标题向后扫描直到下一个章节标题；若其间出现正文段落
    （>= _PROSE_MIN_CHARS 的非标题/非卷分隔/非图注行），则该标题为正文起点。
    TOC 中的章节标题是连续序列，标题之间没有正文，因此不会被误判；该规则
    不依赖 TOC 长度。
    """
    for i, line in enumerate(lines):
        if not _is_chapter(line):
            continue
        j = i + 1
        while j < len(lines):
            nxt = lines[j].strip()
            if nxt == "":
                j += 1
                continue
            if _is_chapter(nxt):
                break            # 仍在标题序列中（TOC），非正文起点
            if _is_divider(nxt) or _is_illustration(nxt):
                j += 1
                continue
            if len(nxt) >= _PROSE_MIN_CHARS:
                return i           # 该标题后紧跟正文 → 正文起点
            j += 1                # 短副标题（如 "The Period"），继续
    return 0


def _strip_illustrations(text: str) -> str:
    """去除 [Illustration ... ] 图注块（单行与多行）。"""
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        if _is_illustration(lines[i]):
            # 单行完整闭合（含 "]"）则跳过该行
            if "]" in lines[i]:
                i += 1
                continue
            # 多行块：跳过直到含 "]" 的行（含该行）
            while i < len(lines) and "]" not in lines[i]:
                i += 1
            i += 1
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)


def clean(raw_text: str, work_id: str) -> CleanedText:
    # 1. 换行与 Unicode 归一化
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFC", text)

    # 2. 截取 START/END 标记之间
    start_m = _START_RE.search(text)
    header_removed = start_m is not None
    if not start_m:
        raise ValueError(f"{work_id}: 未找到 Gutenberg START 标记")
    start_pos = text.find("\n", start_m.end())
    body = text[start_pos + 1:] if start_pos != -1 else ""

    end_m = _END_RE.search(body)
    footer_removed = end_m is not None
    if end_m:
        body = body[:end_m.start()]

    # 3. 去除前置物（序言/目录/标题页）
    lines = body.split("\n")
    body_start = _find_body_start(lines)
    front_matter_removed = body_start > 0
    lines = lines[body_start:]
    body = "\n".join(lines)

    # 4. 去除图注块
    body = _strip_illustrations(body)

    # 5. 压缩空行、去行尾空格，保留段落边界与标点
    body = re.sub(r"[ \t]+$", "", body, flags=re.MULTILINE)   # 行尾空白
    body = re.sub(r"\n{3,}", "\n\n", body)                    # 多余空行 → 单个空行
    body = body.strip()

    char_count = len(body)
    word_count = len(body.split())
    return CleanedText(
        work_id=work_id,
        text=body,
        char_count=char_count,
        word_count=word_count,
        header_removed=header_removed,
        footer_removed=footer_removed,
        front_matter_removed=front_matter_removed,
    )
