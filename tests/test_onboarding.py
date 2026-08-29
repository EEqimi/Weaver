# tests/test_onboarding.py
"""Generic Author Onboarding / Corpus Registry 的第三作者确定性 smoke test。

验证 V0.1 核心验收：新增第三位作者**不修改任何 Style Engine 核心分析代码**，
只需 author manifest + 语料。全链路（manifest load → registry → validation →
corpus discovery → deterministic onboarding → metadata）零 LLM、零 token、
零 provider 实例化、绝不读 DEEPSEEK_API_KEY。

作者用 synthetic fixture（"bronte"/Test Author），非 Austen/Dickens。
"""
import json
import shutil
from pathlib import Path

import pytest

from knowledge.corpus.manifest import MANIFEST_SCHEMA_VERSION
from knowledge.corpus.metadata import load_corpus, manifest_dir
from knowledge.ingestion.onboarding import (
    STATUS_INVALID,
    STATUS_READY_FOR_NEXT_STEP,
    STATUS_REQUIRES_LLM_APPROVAL,
    build_author,
    onboard_author,
    register_author,
    validate_author,
)

# 最小合法 Gutenberg 章节体文本（cleaner 需要 START/END 标记 + CHAPTER 标题）。
GUTENBERG_TEXT = (
    "The Project Gutenberg eBook of Sample\n\n"
    "*** START OF THE PROJECT GUTENBERG EBOOK SAMPLE ***\n\n"
    "CHAPTER I. A Beginning\n\n"
    "It was the best of times, it was the worst of times. " * 30 + "\n\n"
    "It was the age of wisdom, it was the age of foolishness. " * 30 + "\n\n"
    "CHAPTER II. The Middle\n\n"
    "This is the second chapter with plenty of prose to be detected as real "
    "text by the cleaner, enough words to fill a chunk comfortably. " * 20 + "\n\n"
    "*** END OF THE PROJECT GUTENBERG EBOOK SAMPLE ***\n\n"
    "www.gutenberg.org license text.\n"
)


def _manifest_payload(author_id="bronte", work_id="sample_novel",
                      filename="sample_novel.txt", role="train"):
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "authors": [{
            "author_id": author_id,
            "display_name": "Test Author",
            "language": "en",
            "works": [{
                "work_id": work_id,
                "title": "Sample Novel",
                "year": 1850,
                "genre": "novel",
                "filename": filename,
                "role": role,
            }],
        }],
    }


def _make_fixture(tmp_path, *, author_id="bronte", work_id="sample_novel",
                  filename="sample_novel.txt", role="train", write_corpus=True):
    """构造独立临时 corpus + registry + data + manifest。"""
    corpus = tmp_path / "text"
    corpus.mkdir()
    if write_corpus:
        (corpus / filename).write_text(GUTENBERG_TEXT, encoding="utf-8")
    reg = tmp_path / "manifests"
    reg.mkdir()
    data = tmp_path / "data"
    manifest = tmp_path / "author_manifest.json"
    manifest.write_text(
        json.dumps(_manifest_payload(author_id, work_id, filename, role)),
        encoding="utf-8")
    return corpus, reg, data, manifest


# --------------------------------------------------------------------------- #
# validate_author
# --------------------------------------------------------------------------- #
def test_validate_author_ok(tmp_path):
    corpus, reg, _, manifest = _make_fixture(tmp_path)
    r = validate_author(manifest, corpus_root=corpus, registry_dir=reg)
    assert r["status"] == STATUS_READY_FOR_NEXT_STEP
    assert r["author_id"] == "bronte"
    assert r["details"]["authors"] == ["bronte"]
    assert r["details"]["works"] == ["sample_novel"]


def test_validate_author_invalid_schema(tmp_path):
    corpus, reg, _, manifest = _make_fixture(tmp_path)
    # 移除必填 role → schema 校验失败 → INVALID。
    payload = _manifest_payload()
    del payload["authors"][0]["works"][0]["role"]
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    r = validate_author(manifest, corpus_root=corpus, registry_dir=reg)
    assert r["status"] == STATUS_INVALID
    assert r["errors"]


def test_validate_author_missing_corpus(tmp_path):
    corpus, reg, _, manifest = _make_fixture(tmp_path, filename="absent.txt",
                                             write_corpus=False)
    r = validate_author(manifest, corpus_root=corpus, registry_dir=reg)
    assert r["status"] == STATUS_INVALID
    assert any("absent.txt" in e for e in r["errors"])


# --------------------------------------------------------------------------- #
# register_author
# --------------------------------------------------------------------------- #
def test_register_author_writes_manifest_and_conflict(tmp_path):
    corpus, reg, _, manifest = _make_fixture(tmp_path)
    r = register_author(manifest, registry_dir=reg)
    assert r["status"] == STATUS_READY_FOR_NEXT_STEP
    assert (reg / "bronte.json").is_file()
    # 二次注册同 author_id → INVALID（冲突）。
    r2 = register_author(manifest, registry_dir=reg)
    assert r2["status"] == STATUS_INVALID
    assert any("author_id 已注册" in e for e in r2["errors"])


# --------------------------------------------------------------------------- #
# build_author（确定性，零 LLM）
# --------------------------------------------------------------------------- #
def test_build_author_deterministic(tmp_path):
    corpus, reg, data, manifest = _make_fixture(tmp_path)
    r = build_author(manifest, corpus_root=corpus, data_root_=data)
    assert r["status"] == STATUS_READY_FOR_NEXT_STEP
    built = r["details"]["built_works"]["sample_novel"]
    assert built["work_id"] == "sample_novel"
    assert built["author_id"] == "bronte"
    assert built["role"] == "train"
    assert built["word_count"] > 0
    # 确定性产物落盘：clean / chunks / metadata / qc。
    assert (data / "clean" / "sample_novel.txt").is_file()
    assert (data / "metadata" / "sample_novel.json").is_file()
    assert (data / "qc" / "sample_novel.json").is_file()
    assert any((data / "chunks").glob("sample_novel__*.jsonl"))


# --------------------------------------------------------------------------- #
# onboard_author（编排 + LLM 阻塞）
# --------------------------------------------------------------------------- #
def test_onboard_author_requires_llm_approval(tmp_path):
    corpus, reg, data, manifest = _make_fixture(tmp_path)
    r = onboard_author(manifest, corpus_root=corpus, data_root_=data, registry_dir=reg)
    assert r["status"] == STATUS_REQUIRES_LLM_APPROVAL
    assert r["author_id"] == "bronte"
    # 确定性部分已完成；需要 LLM 的步骤只被描述、绝不执行。
    assert "built_works" in r["details"]["build"]
    assert "pending_llm_steps" in r["details"]
    assert (reg / "bronte.json").is_file()


def test_onboard_author_invalid_stops(tmp_path):
    corpus, reg, data, manifest = _make_fixture(tmp_path, filename="absent.txt",
                                                write_corpus=False)
    r = onboard_author(manifest, corpus_root=corpus, data_root_=data, registry_dir=reg)
    assert r["status"] == STATUS_INVALID


# --------------------------------------------------------------------------- #
# registry 驱动：第三作者进入 registry 后，无需改核心代码即可被发现
# --------------------------------------------------------------------------- #
def test_registry_discovers_third_author(tmp_path):
    _, reg, _, manifest = _make_fixture(tmp_path)
    # 把内置 Austen+Dickens manifest 拷入测试 registry，再注册第三作者。
    shutil.copy(manifest_dir() / "austen_dickens.json", reg / "austen_dickens.json")
    register_author(manifest, registry_dir=reg)

    works = load_corpus(reg)
    authors = sorted({m.author_id for m in works})
    assert authors == ["austen", "bronte", "dickens"]  # registry 派生，非硬编码
    by_id = {m.work_id: m for m in works}
    assert by_id["sample_novel"].author_id == "bronte"
    assert by_id["pride_and_prejudice"].author_id == "austen"


# --------------------------------------------------------------------------- #
# CLI（python -m knowledge.ingestion.add_author）
# --------------------------------------------------------------------------- #
def test_cli_end_to_end_via_env(tmp_path, monkeypatch, capsys):
    corpus, reg, data, manifest = _make_fixture(tmp_path)
    monkeypatch.setenv("WEAVER_CORPUS_ROOT", str(corpus))
    monkeypatch.setenv("WEAVER_MANIFEST_DIR", str(reg))
    monkeypatch.setenv("WEAVER_DATA_ROOT", str(data))

    from knowledge.ingestion import add_author
    rc = add_author.main([str(manifest)])
    assert rc == 0  # REQUIRES_LLM_APPROVAL 仍视为成功（确定性部分完成）
    out = capsys.readouterr()
    assert "REQUIRES_LLM_APPROVAL" in out.out
    assert (reg / "bronte.json").is_file()


def test_cli_invalid_returns_nonzero(tmp_path, capsys):
    from knowledge.ingestion import add_author
    rc = add_author.main([str(tmp_path / "does_not_exist.json")])
    assert rc == 1
    assert "INVALID" in capsys.readouterr().out


def test_cli_usage_error(tmp_path):
    from knowledge.ingestion import add_author
    assert add_author.main([]) == 2
