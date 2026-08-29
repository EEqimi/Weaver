# knowledge/corpus/pipeline.py
"""语料管线编排：RAW → CLEAN → CHUNKS → METADATA/QC。

原始文件只读、绝不修改；生成物写入 data/（gitignore）。全程确定性、可复现。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..config import (CHUNK_SIZES, corpus_root as default_corpus_root,
                      data_layout, data_root as default_data_root)
from ..schema.versions import (CHUNKER_VERSION, CLEANER_VERSION, SCHEMA_VERSION)
from .cleaner import clean
from .chunker import Chunk, chunk_text
from .discover import discover
from .metadata import CORPUS, WorkMetadata
from .qc import run_work_qc


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, chunks: list[Chunk]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps({
                "chunk_id": c.chunk_id,
                "work_id": c.work_id,
                "target_chars": c.target_chars,
                "chapter": c.chapter,
                "seq": c.seq,
                "char_count": c.char_count,
                "word_count": c.word_count,
                "text": c.text,
            }, ensure_ascii=False) + "\n")


def _chapter_count(text: str) -> int:
    """检测到的章节数（复用 cleaner 的标题判定）。"""
    from .cleaner import is_chapter_heading
    return sum(1 for line in text.split("\n") if is_chapter_heading(line.strip()))


def build_work(meta: WorkMetadata, src_path: Path, layout: dict[str, Path]) -> dict:
    """处理单部作品：清洗、分块、写盘、QC，返回元数据。"""
    raw_bytes = src_path.read_bytes()
    raw_text = raw_bytes.decode("utf-8")

    cleaned = clean(raw_text, meta.work_id)
    clean_path = layout["clean"] / f"{meta.work_id}.txt"
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    clean_path.write_text(cleaned.text, encoding="utf-8")

    chunks_by_size: dict[int, list[Chunk]] = {}
    for target in CHUNK_SIZES:
        chunks = chunk_text(cleaned.text, meta.work_id, target)
        chunks_by_size[target] = chunks
        _write_jsonl(layout["chunks"] / f"{meta.work_id}__{target}.jsonl", chunks)

    metadata = {
        "work_id": meta.work_id,
        "author_id": meta.author_id,
        "author": meta.author,
        "work": meta.work,
        "language": meta.language,
        "genre": meta.genre,
        "year": meta.year,
        "role": meta.role,
        "source_path": meta.filename,  # 语料根目录内的相对路径（可移植，非机器绝对路径）
        "raw_sha256": _sha256_bytes(raw_bytes),
        "cleaned_sha256": _sha256_bytes(cleaned.text.encode("utf-8")),
        "cleaner_version": CLEANER_VERSION,
        "chunker_version": CHUNKER_VERSION,
        "schema_version": SCHEMA_VERSION,
        "char_count": cleaned.char_count,
        "word_count": cleaned.word_count,
        "chapter_count": _chapter_count(cleaned.text),
        "chunk_counts": {str(t): len(chunks_by_size[t]) for t in CHUNK_SIZES},
    }
    _write_json(layout["metadata"] / f"{meta.work_id}.json", metadata)

    qc = run_work_qc(meta.work_id, cleaned.text, chunks_by_size)
    _write_json(layout["qc"] / f"{meta.work_id}.json", qc)

    return metadata


def build_works(works: list[WorkMetadata] | tuple[WorkMetadata, ...],
                corpus_root: str | Path | None = None,
                data_root_: str | Path | None = None) -> dict[str, dict]:
    """只构建给定作品子集（单作者 onboarding 用），写各自 metadata/qc，返回
    {work_id: metadata}。确定性、无 LLM、只写 data/（gitignore）。
    """
    root = Path(corpus_root) if corpus_root is not None else default_corpus_root()
    droot = Path(data_root_) if data_root_ is not None else default_data_root()
    layout = data_layout(droot)

    found = discover(root, works=works)
    all_metadata: dict[str, dict] = {}
    for meta in works:
        all_metadata[meta.work_id] = build_work(meta, found[meta.work_id], layout)
    return all_metadata


def build_corpus(corpus_root: str | Path | None = None,
                 data_root_: str | Path | None = None) -> dict:
    """处理全部语料，写 manifest 与 QC 汇总，返回汇总结构。"""
    root = Path(corpus_root) if corpus_root is not None else default_corpus_root()
    droot = Path(data_root_) if data_root_ is not None else default_data_root()
    layout = data_layout(droot)

    found = discover(root)
    all_metadata: dict[str, dict] = {}
    all_qc: dict[str, dict] = {}
    for meta in CORPUS:
        m = build_work(meta, found[meta.work_id], layout)
        all_metadata[meta.work_id] = m
        qc_path = layout["qc"] / f"{meta.work_id}.json"
        all_qc[meta.work_id] = json.loads(qc_path.read_text(encoding="utf-8"))

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "cleaner_version": CLEANER_VERSION,
        "chunker_version": CHUNKER_VERSION,
        # corpus_root/data_root 是本次运行的机器相关路径（调试/复现用，非可移植）；
        # 可移植的源文件标识在各 work 的 source_path（语料根目录内相对路径）。
        "corpus_root": str(root),
        "data_root": str(droot),
        "chunk_sizes": list(CHUNK_SIZES),
        "works": all_metadata,
    }
    _write_json(layout["metadata"] / "manifest.json", manifest)

    qc_summary = {
        "works": all_qc,
        "aggregate": _aggregate_qc(all_qc),
    }
    _write_json(layout["qc"] / "summary.json", qc_summary)

    return {"manifest": manifest, "qc_summary": qc_summary}


def _aggregate_qc(all_qc: dict[str, dict]) -> dict:
    """跨作品汇总 QC 计数（empty/tiny/oversized/duplicate 为硬问题，small_tail 为预期）。"""
    agg = {"residue_works": [], "empty_chunks": 0, "tiny_chunks": 0,
           "oversized_chunks": 0, "duplicate_chunks": 0, "small_tail_chunks": 0}
    for work_id, qc in all_qc.items():
        if not qc["cleaned"]["clean"]:
            agg["residue_works"].append(work_id)
        for _, cqc in qc["chunks"].items():
            agg["empty_chunks"] += cqc["empty_count"]
            agg["tiny_chunks"] += cqc["tiny_count"]
            agg["oversized_chunks"] += cqc["oversized_count"]
            agg["duplicate_chunks"] += cqc["duplicate_count"]
            agg["small_tail_chunks"] += cqc["small_tail_count"]
    return agg
