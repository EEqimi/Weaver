# knowledge/corpus/discover.py
"""语料文件发现：按 manifest 从语料根目录定位原始文件。"""
from __future__ import annotations

from pathlib import Path

from .metadata import CORPUS


def discover(corpus_root: str | Path) -> dict[str, Path]:
    """返回 {work_id: 原始文件路径}；缺失文件时抛错。"""
    root = Path(corpus_root)
    found: dict[str, Path] = {}
    missing: list[str] = []
    for m in CORPUS:
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
