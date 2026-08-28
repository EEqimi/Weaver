# knowledge/corpus/metadata.py
"""语料清单（manifest）。

作者/作品/语言/体裁/年份/角色/文件名的显式映射。之所以不靠文件名解析，
是因为原始文件名含笔误（David_Copperfield,txt.txt、pride_and_prejudice..txt），
且出版年份、体裁、训练/held-out 角色无法从文件名获得。
"""
from __future__ import annotations

from dataclasses import dataclass

TRAIN = "train"
HELD_OUT = "held_out"


@dataclass(frozen=True)
class WorkMetadata:
    work_id: str      # 稳定 slug
    author_id: str
    author: str       # 显示名
    work: str         # 作品名
    language: str
    genre: str
    year: int         # 出版年份
    role: str         # train / held_out
    filename: str     # 语料目录内的实际文件名


# 六部 pilot 小说（spec §19.2 的六作者/两作者 pilot 收窄为 Austen + Dickens）
#
# V0.1 用硬编码 CORPUS 清单即可；扩展规模前应迁移为数据驱动的 manifest 文件
# （优先 JSON，需注释/手写则 YAML），由 discover 按文件名解析 + 该文件补
# 出版年份/体裁/角色等无法从文件名得到的字段，避免每次加语料改代码。
CORPUS: tuple[WorkMetadata, ...] = (
    WorkMetadata("pride_and_prejudice", "austen", "Jane Austen",
                 "Pride and Prejudice", "en", "novel", 1813, TRAIN,
                 "pride_and_prejudice..txt"),
    WorkMetadata("emma", "austen", "Jane Austen",
                 "Emma", "en", "novel", 1815, TRAIN, "emma.txt"),
    WorkMetadata("persuasion", "austen", "Jane Austen",
                 "Persuasion", "en", "novel", 1817, HELD_OUT, "Persuasion.txt"),
    WorkMetadata("great_expectations", "dickens", "Charles Dickens",
                 "Great Expectations", "en", "bildungsroman", 1861, TRAIN,
                 "Great_Expectations.txt"),
    WorkMetadata("david_copperfield", "dickens", "Charles Dickens",
                 "David Copperfield", "en", "bildungsroman", 1850, TRAIN,
                 "David_Copperfield,txt.txt"),
    WorkMetadata("tale_of_two_cities", "dickens", "Charles Dickens",
                 "A Tale of Two Cities", "en", "historical_novel", 1859, HELD_OUT,
                 "A_Tale_of_Two_Cities.txt"),
)


def by_work_id() -> dict[str, WorkMetadata]:
    return {m.work_id: m for m in CORPUS}


def author_display_names() -> dict[str, str]:
    """author_id → 显示名（如 "austen" → "Jane Austen"）。

    数据驱动，来自 CORPUS 清单（非硬编码在泄露守卫里）。Phase 7.1 用它为
    `assert_no_author_identity` 提供"当前作者身份"名单，新增作者时无需改守卫代码。
    """
    return {m.author_id: m.author for m in CORPUS}


def by_author_id() -> dict[str, list[WorkMetadata]]:
    out: dict[str, list[WorkMetadata]] = {}
    for m in CORPUS:
        out.setdefault(m.author_id, []).append(m)
    return out


def train_works() -> list[WorkMetadata]:
    return [m for m in CORPUS if m.role == TRAIN]


def held_out_works() -> list[WorkMetadata]:
    return [m for m in CORPUS if m.role == HELD_OUT]
