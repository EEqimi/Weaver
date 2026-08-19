# tests/test_no_raw_mutation.py
"""原始语料绝不改动：完整构建前后原始文件 SHA256 一致。"""
import hashlib

from knowledge.corpus.metadata import CORPUS


def test_raw_files_unchanged_after_build(real_corpus_root, built, raw_sha256_snapshot):
    # built fixture 已触发完整管线；此处重新计算原始文件哈希并与快照比对
    for m in CORPUS:
        p = real_corpus_root / m.filename
        after = hashlib.sha256(p.read_bytes()).hexdigest()
        assert after == raw_sha256_snapshot[m.work_id], m.work_id


def test_manifest_has_required_hashes(built):
    manifest = built["result"]["manifest"]
    for wid, w in manifest["works"].items():
        assert len(w["raw_sha256"]) == 64
        assert len(w["cleaned_sha256"]) == 64
        assert w["raw_sha256"] != w["cleaned_sha256"]  # 清洗必然改变内容
        assert w["cleaner_version"] and w["chunker_version"]


def test_qc_clean(built):
    agg = built["result"]["qc_summary"]["aggregate"]
    assert agg["residue_works"] == []
    assert agg["empty_chunks"] == 0
    assert agg["oversized_chunks"] == 0
    assert agg["duplicate_chunks"] == 0
