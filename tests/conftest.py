# tests/conftest.py
"""共享 fixtures：把 Weaver 根加入 sys.path，并暴露语料路径。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]   # Weaver/
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from knowledge.config import corpus_root  # noqa: E402
from knowledge.corpus.metadata import CORPUS  # noqa: E402
from knowledge.corpus.pipeline import build_corpus  # noqa: E402


@pytest.fixture(scope="session")
def real_corpus_root() -> Path:
    return corpus_root()


@pytest.fixture(scope="session")
def built(tmp_path_factory):
    """会话级：完整构建一次语料（写入临时目录），供多个集成测试复用。"""
    data_dir = tmp_path_factory.mktemp("built_data")
    result = build_corpus(data_root_=data_dir)
    return {"result": result, "data_dir": data_dir}


@pytest.fixture(scope="session")
def raw_sha256_snapshot(real_corpus_root):
    """构建前对全部原始文件做 SHA256 快照。"""
    import hashlib
    snap = {}
    for m in CORPUS:
        p = real_corpus_root / m.filename
        snap[m.work_id] = hashlib.sha256(p.read_bytes()).hexdigest()
    return snap
