# tests/test_sampling.py
"""确定性分层采样测试（spec §12）：可复现、分层覆盖、held-out 排除。"""
import pytest

from knowledge.sampling.calibration import (
    DIALOGUE_BANDS, POSITION_BANDS, SampleChunk, build_calibration_manifest,
    dialogue_band, enrich_chunks, position_band, select_stratified,
)


def _chunk_dict(cid, chapter, seq, text):
    return {"chunk_id": cid, "work_id": "pride_and_prejudice",
            "chapter": chapter, "seq": seq, "char_count": len(text), "text": text}


def _make_chunks(n=24):
    """构造 n 个分层多样的合成 chunk（对话 / 叙述 / 混合）。"""
    dialogue = '"I am going to town," she said. "Will you come with me?" he asked. "Yes," she replied.'
    narration = ("The house stood upon a hill, and the fields below it stretched "
                 "far into the mist. Nobody came near the gate for many years.")
    chunks = []
    for i in range(n):
        text = dialogue if i % 3 == 0 else (narration if i % 3 == 1
                                             else narration + " " + dialogue)
        chunks.append(_chunk_dict(f"c{i:03d}", f"ch{i % 10}", i + 1, text))
    return chunks


# ---- 阈值 ----
def test_dialogue_band_thresholds():
    assert dialogue_band(0.9) == "dialogue"
    assert dialogue_band(0.1) == "mixed"
    assert dialogue_band(0.01) == "narration"


def test_position_band_thresholds():
    assert position_band(1, 100) == "early"
    assert position_band(50, 100) == "middle"
    assert position_band(99, 100) == "late"


# ---- 确定性 ----
def test_select_stratified_deterministic():
    enriched = enrich_chunks(_make_chunks(), "pride_and_prejudice")
    s1 = [c.chunk_id for c in select_stratified(enriched, 10)]
    s2 = [c.chunk_id for c in select_stratified(enriched, 10)]
    assert s1 == s2


def test_select_stratified_target_in_range():
    enriched = enrich_chunks(_make_chunks(), "pride_and_prejudice")
    selected = select_stratified(enriched, 10)
    assert 8 <= len(selected) <= 12
    # 分层覆盖：位置档与对话档都应覆盖到
    assert len({c.position for c in selected}) >= 2
    assert len({c.dialogue for c in selected}) >= 2


# ---- held-out 排除 ----
def test_manifest_rejects_held_out():
    with pytest.raises(ValueError):
        build_calibration_manifest({"persuasion": _make_chunks()}, target_per_work=10)


def test_manifest_rejects_fewer_than_eight():
    with pytest.raises(ValueError):
        build_calibration_manifest(
            {"pride_and_prejudice": _make_chunks(n=4)}, target_per_work=10)


def test_manifest_deterministic_and_excludes_heldout_metadata():
    chunks = _make_chunks()
    m1 = build_calibration_manifest({"pride_and_prejudice": chunks}, target_per_work=10)
    m2 = build_calibration_manifest({"pride_and_prejudice": chunks}, target_per_work=10)
    ids1 = [c["chunk_id"] for c in m1["works"]["pride_and_prejudice"]["selected"]]
    ids2 = [c["chunk_id"] for c in m2["works"]["pride_and_prejudice"]["selected"]]
    assert ids1 == ids2
    assert "persuasion" in m1["held_out_excluded"]
    assert "tale_of_two_cities" in m1["held_out_excluded"]
    assert m1["works"]["pride_and_prejudice"]["n_selected"] == 10


def test_enrich_chunks_orders_by_seq_not_chapter_string():
    # regression（task item 3）：chapter 字典序 "10" < "2" 曾令 position 档错乱。
    # 排序必须只用 seq（作品内全局顺序）。
    chunks = [
        _chunk_dict("c1", "Chapter 10", 1, "one."),
        _chunk_dict("c2", "Chapter 2", 2, "two."),
        _chunk_dict("c3", "Chapter 1", 3, "three."),
    ]
    enriched = enrich_chunks(chunks, "pride_and_prejudice")
    # 保持 seq 顺序（而非 chapter 字典序）
    assert [c.chapter for c in enriched] == ["Chapter 10", "Chapter 2", "Chapter 1"]
    assert [c.seq for c in enriched] == [1, 2, 3]
    assert enriched[0].position == "early"
    assert enriched[-1].position == "late"


def test_select_stratified_sorted_by_seq():
    # select_stratified 的最终排序也必须按 seq，而非 (chapter, seq)
    chunks = [
        _chunk_dict("c1", "Chapter 10", 1, "aaaa."),
        _chunk_dict("c2", "Chapter 2", 2, "bbbb."),
        _chunk_dict("c3", "Chapter 1", 3, "cccc."),
    ]
    enriched = enrich_chunks(chunks, "pride_and_prejudice")
    selected = select_stratified(enriched, target=3)
    assert [c.seq for c in selected] == sorted(c.seq for c in selected)
