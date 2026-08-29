# knowledge/corpus/discover.py
"""语料文件发现：按 manifest 从语料根目录定位原始文件。"""
from __future__ import annotations

from pathlib import Path

from .metadata import CORPUS, WorkMetadata


def discover(corpus_root: str | Path,
             works: list[WorkMetadata] | tuple[WorkMetadata, ...] | None = None
             ) -> dict[str, Path]:
    """返回 {work_id: 原始文件路径}；缺失文件时抛错。

    `works` 缺省为 registry 全量（CORPUS）；传子集则只发现指定作品（用于
    单作者 onboarding / 第三作者测试，避免动到已有作者的语料）。
    """
    root = Path(corpus_root)
    items = list(works) if works is not None else list(CORPUS)
    found: dict[str, Path] = {}
    missing: list[str] = []
    for m in items:
        p = root / m.filename
        if p.is_file():
            found[m.work_id] = p
        else:
            missing.append(m.filename)
    if missing:
        raise FileNotFoundError(
            f"语料缺失 {len(missing)} 个文件（root={root}）: {', '.join(missing)}"
        )
    return found
