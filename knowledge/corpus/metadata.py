# knowledge/corpus/metadata.py
"""语料清单（corpus registry）：由 committed manifest 文件数据驱动。

作者/作品/语言/体裁/年份/角色/文件名的显式映射。之所以不靠文件名解析，
是因为原始文件名含笔误（David_Copperfield,txt.txt、pride_and_prejudice..txt），
且出版年份、体裁、训练/held-out 角色无法从文件名获得。

数据来源是 `manifests/*.json`（见 `knowledge/corpus/manifest.py` 的 schema）。
新增作者只需新增一份 manifest 文件 + 对应语料，无需改动本模块或任何核心分析代码
（V0.1 验收：author manifest 驱动，而非硬编码 CORPUS）。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from ..config import REPO_ROOT
from .manifest import (
    HELD_OUT,
    TRAIN,
    AuthorManifest,
    ManifestError,
    load_manifest_file,
)

# 兼容重导出：旧代码 `from knowledge.corpus.metadata import TRAIN, HELD_OUT` 仍可用。
__all__ = [
    "TRAIN", "HELD_OUT", "WorkMetadata", "CORPUS",
    "manifest_dir", "load_corpus", "author_ids", "works_from_manifest",
    "by_work_id", "author_display_names", "by_author_id",
    "train_works", "held_out_works",
]


@dataclass(frozen=True)
class WorkMetadata:
    work_id: str      # 稳定 slug
    author_id: str
    author: str       # 显示名
    work: str         # 作品名
    language: str
    genre: str
    year: int | None  # 出版年份（manifest 缺失时为 None）
    role: str         # train / held_out
    filename: str     # 语料目录内的实际文件名


def manifest_dir() -> Path:
    """committed manifest 目录（默认 <repo>/manifests，可用 WEAVER_MANIFEST_DIR 覆盖）。"""
    default = REPO_ROOT / "manifests"
    return Path(os.environ.get("WEAVER_MANIFEST_DIR", str(default)))


def _work_metadata(author: AuthorManifest, work) -> WorkMetadata:
    """AuthorManifest + WorkManifest → WorkMetadata（registry 的运行时视图）。"""
    return WorkMetadata(
        work_id=work.work_id,
        author_id=author.author_id,
        author=author.display_name,
        work=work.title,
        language=author.language,
        genre=work.genre,
        year=work.year,
        role=work.role,
        filename=work.filename,
    )


def works_from_manifest(author: AuthorManifest) -> list[WorkMetadata]:
    """把单个 AuthorManifest 展开为 WorkMetadata 列表（onboarding 复用，与 CORPUS 同构）。"""
    return [_work_metadata(author, w) for w in author.works]


def load_corpus(manifest_root: str | Path | None = None) -> tuple[WorkMetadata, ...]:
    """扫描 manifest 目录下全部 `*.json`，解析校验并合成 CORPUS。

    确定性：文件按文件名排序读取；作者按 author_id 升序；作品保持 manifest 内声明顺序。
    cross-file 的 author_id / work_id 重复会 fail-closed（ManifestError）。
    """
    root = Path(manifest_root) if manifest_root is not None else manifest_dir()
    if not root.is_dir():
        raise FileNotFoundError(f"manifest 目录不存在: {root}")

    authors: list[AuthorManifest] = []
    for p in sorted(root.glob("*.json")):
        authors.extend(load_manifest_file(p))

    seen_author: set[str] = set()
    seen_work: set[str] = set()
    out: list[WorkMetadata] = []
    for author in authors:
        if author.author_id in seen_author:
            raise ManifestError(f"cross-file author_id 重复: {author.author_id!r}")
        seen_author.add(author.author_id)
        for work in author.works:
            if work.work_id in seen_work:
                raise ManifestError(f"cross-file work_id 重复: {work.work_id!r}")
            seen_work.add(work.work_id)
            out.append(_work_metadata(author, work))
    return tuple(out)


# 六部 pilot 小说（spec §19.2 的六作者/两作者 pilot 收窄为 Austen + Dickens）。
# 现在完全由 manifests/austen_dickens.json 数据驱动（见上文 load_corpus）。
CORPUS: tuple[WorkMetadata, ...] = load_corpus()


def by_work_id() -> dict[str, WorkMetadata]:
    return {m.work_id: m for m in CORPUS}


def author_ids() -> tuple[str, ...]:
    """注册的 author_id 升序元组（registry 派生的"作者全集"，非硬编码）。"""
    return tuple(sorted({m.author_id for m in CORPUS}))


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
