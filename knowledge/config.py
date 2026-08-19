# knowledge/config.py
"""knowledge 包的全局配置：路径推导与 chunk 规模。

只包含确定性的路径/规模配置，不含任何密钥。
路径可通过环境变量覆盖，便于测试与部署：
    WEAVER_CORPUS_ROOT  原始语料根目录（默认 <repo 上级>/text）
    WEAVER_DATA_ROOT    生成物根目录（默认 <repo>/data）
"""
from __future__ import annotations

import os
from pathlib import Path

# ---- 路径推导 ----
PACKAGE_ROOT = Path(__file__).resolve().parent          # knowledge/
REPO_ROOT = PACKAGE_ROOT.parent                          # Weaver/


def corpus_root() -> Path:
    """原始语料根目录（只读，绝不修改）。"""
    default = REPO_ROOT.parent / "text"
    return Path(os.environ.get("WEAVER_CORPUS_ROOT", str(default)))


def data_root() -> Path:
    """生成物根目录（clean/chunks/metadata/qc，全部 gitignore）。"""
    default = REPO_ROOT / "data"
    return Path(os.environ.get("WEAVER_DATA_ROOT", str(default)))


def data_layout(root: Path | None = None) -> dict[str, Path]:
    """返回各生成物子目录的映射。"""
    base = Path(root) if root is not None else data_root()
    return {
        "clean": base / "clean",
        "chunks": base / "chunks",
        "metadata": base / "metadata",
        "qc": base / "qc",
    }


# ---- chunk 目标字符数（Phase 1 三档） ----
CHUNK_SIZES: tuple[int, ...] = (1000, 2000, 4000)

# 单段超长时的最大 chunk 倍数（用于判断"异常尺寸"与句子级切分）
MAX_CHUNK_RATIO = 1.5
