# knowledge/ingestion/onboarding.py
"""Generic Author Onboarding 的确定性部分。

三步走（各自可单独调用，也可由 `onboard_author` 编排）：
    1. validate_author —— manifest schema + 语料存在 + id 冲突（registry 内）。
    2. register_author —— 把作者 manifest 写为 `manifests/{author_id}.json`。
    3. build_author   —— 复用 corpus 管线（discover→clean→chunk→QC→metadata），
                         零 LLM、确定性、只写 data/（gitignore）。

需要 LLM 的后续步骤（采样 + Layer A/B/C 特征分析 → 聚合 → 策略合并 → 画像合成）
**绝不自动执行、绝不自费调用**：确定性部分完成后返回 `REQUIRES_LLM_APPROVAL`。

状态协议（单一来源）：
    - INVALID               清单/语料/冲突错误，onboarding 中止。
    - READY_FOR_NEXT_STEP   某确定性步骤完成，可进入下一步。
    - REQUIRES_LLM_APPROVAL 下一步需要 LLM（已阻塞，等待显式批准）。

铁律：本模块绝不读 DEEPSEEK_API_KEY、绝不实例化 provider、绝不产生真实 LLM 请求。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import corpus_root as default_corpus_root, data_root as default_data_root
from ..corpus.manifest import (
    MANIFEST_SCHEMA_VERSION,
    AuthorManifest,
    ManifestError,
    load_manifest_file,
)
from ..corpus.metadata import (
    WorkMetadata,
    load_corpus,
    manifest_dir,
    works_from_manifest,
)
from ..corpus.pipeline import build_works

STATUS_INVALID = "INVALID"
STATUS_READY_FOR_NEXT_STEP = "READY_FOR_NEXT_STEP"
STATUS_REQUIRES_LLM_APPROVAL = "REQUIRES_LLM_APPROVAL"

# 确定性 onboarding 之后、需要真实 LLM 的下一步（只描述、不执行、不收费）。
_LLM_NEXT_STEPS = (
    "sampling (build_calibration_manifest) → Layer A/B/C 特征分析（LLM）"
    " → 聚合 → 策略合并 → AuthorStyleProfile 合成"
)


def _result(status: str, stage: str, author_id: str | None, message: str,
            details: dict[str, Any] | None = None,
            errors: list[str] | None = None) -> dict[str, Any]:
    return {
        "status": status,
        "stage": stage,
        "author_id": author_id,
        "message": message,
        "details": details or {},
        "errors": errors or [],
    }


def _load_authors(manifest_path: str | Path) -> list[AuthorManifest]:
    """解析 manifest（JSON/YAML）；失败抛 ManifestError（由调用方转 INVALID）。"""
    return load_manifest_file(Path(manifest_path))


def _existing_ids(registry_dir: str | Path) -> tuple[set[str], set[str]]:
    """返回 (author_ids, work_ids) 已注册集合；registry 目录不存在视为空。"""
    root = Path(registry_dir)
    if not root.is_dir():
        return set(), set()
    works: list[WorkMetadata] = list(load_corpus(root))
    return ({m.author_id for m in works}, {m.work_id for m in works})


def _conflicts(authors: list[AuthorManifest],
               existing_authors: set[str], existing_works: set[str]) -> list[str]:
    """检测待注册作者与已注册 registry 的 author_id / work_id 冲突。"""
    errors: list[str] = []
    for a in authors:
        if a.author_id in existing_authors:
            errors.append(f"author_id 已注册: {a.author_id!r}")
        for w in a.works:
            if w.work_id in existing_works:
                errors.append(f"work_id 已注册（跨作者冲突）: {w.work_id!r}")
    return errors


def _safe_filename(author_id: str) -> str:
    if any(c in author_id for c in ("/", "\\", "\x00")) or author_id in (".", ".."):
        raise ManifestError(f"author_id 含非法文件名字符: {author_id!r}")
    return f"{author_id}.json"


def validate_author(manifest_path: str | Path,
                    corpus_root: str | Path | None = None,
                    registry_dir: str | Path | None = None) -> dict[str, Any]:
    """校验 manifest：schema + 语料存在 + 无 registry id 冲突。

    失败返回 INVALID（errors 列原因）；通过返回 READY_FOR_NEXT_STEP。
    """
    try:
        authors = _load_authors(manifest_path)
    except ManifestError as e:
        return _result(STATUS_INVALID, "validate", None, str(e), errors=[str(e)])

    reg = Path(registry_dir) if registry_dir is not None else manifest_dir()
    try:
        ea, ew = _existing_ids(reg)
    except (ManifestError, FileNotFoundError) as e:
        return _result(STATUS_INVALID, "validate", None,
                       f"读取 registry 失败: {e}", errors=[str(e)])

    errors = _conflicts(authors, ea, ew)

    # 语料存在性（对每位作者的每部作品做 discover；缺失即 INVALID）。
    root = Path(corpus_root) if corpus_root is not None else default_corpus_root()
    missing: list[str] = []
    for a in authors:
        for w in works_from_manifest(a):
            if not (root / w.filename).is_file():
                missing.append(f"{a.author_id}/{w.filename}")
    if missing:
        errors.append(f"语料缺失 {len(missing)} 个文件（root={root}）: {', '.join(missing)}")

    if errors:
        return _result(STATUS_INVALID, "validate", None,
                       f"校验失败（{len(errors)} 项）", errors=errors)

    aid = authors[0].author_id if len(authors) == 1 else None
    details = {
        "authors": [a.author_id for a in authors],
        "works": [w.work_id for a in authors for w in a.works],
        "corpus_root": str(root),
        "registry_dir": str(reg),
    }
    return _result(STATUS_READY_FOR_NEXT_STEP, "validate", aid,
                   f"manifest 校验通过：{len(authors)} 作者 / "
                   f"{sum(len(a.works) for a in authors)} 作品",
                   details=details)


def register_author(manifest_path: str | Path,
                    registry_dir: str | Path | None = None) -> dict[str, Any]:
    """把作者 manifest 注册进 corpus registry（写 `{author_id}.json`）。

    每作者写一个单作者 manifest 文件（保持 registry 文件粒度简单、便于 diff）。
    冲突 / 坏清单返回 INVALID；成功返回 READY_FOR_NEXT_STEP。
    """
    try:
        authors = _load_authors(manifest_path)
    except ManifestError as e:
        return _result(STATUS_INVALID, "register", None, str(e), errors=[str(e)])

    reg = Path(registry_dir) if registry_dir is not None else manifest_dir()
    try:
        ea, ew = _existing_ids(reg)
    except (ManifestError, FileNotFoundError) as e:
        return _result(STATUS_INVALID, "register", None,
                       f"读取 registry 失败: {e}", errors=[str(e)])

    errors = _conflicts(authors, ea, ew)
    if errors:
        return _result(STATUS_INVALID, "register", None,
                       f"注册冲突（{len(errors)} 项）", errors=errors)

    written: list[str] = []
    try:
        reg.mkdir(parents=True, exist_ok=True)
        for a in authors:
            payload = {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "authors": [
                    {
                        "author_id": a.author_id,
                        "display_name": a.display_name,
                        "language": a.language,
                        "works": [
                            {
                                "work_id": w.work_id,
                                "title": w.title,
                                "filename": w.filename,
                                "role": w.role,
                                **({"year": w.year} if w.year is not None else {}),
                                **({"genre": w.genre} if w.genre != "unknown" else {}),
                            }
                            for w in a.works
                        ],
                    }
                ],
            }
            out = reg / _safe_filename(a.author_id)
            out.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            written.append(str(out))
    except (OSError, ManifestError) as e:
        return _result(STATUS_INVALID, "register", None,
                       f"写入 registry 失败: {e}", errors=[str(e)])

    aid = authors[0].author_id if len(authors) == 1 else None
    return _result(STATUS_READY_FOR_NEXT_STEP, "register", aid,
                   f"已注册 {len(written)} 个作者 manifest",
                   details={"files": written})


def build_author(manifest_path: str | Path,
                 corpus_root: str | Path | None = None,
                 data_root_: str | Path | None = None) -> dict[str, Any]:
    """确定性处理作者语料：discover→clean→chunk→QC→metadata（零 LLM）。

    成功返回 READY_FOR_NEXT_STEP（details 含各作品 metadata 摘要）。
    """
    try:
        authors = _load_authors(manifest_path)
    except ManifestError as e:
        return _result(STATUS_INVALID, "build", None, str(e), errors=[str(e)])

    root = Path(corpus_root) if corpus_root is not None else default_corpus_root()
    droot = Path(data_root_) if data_root_ is not None else default_data_root()

    works: list[WorkMetadata] = [w for a in authors for w in works_from_manifest(a)]
    try:
        built = build_works(works, corpus_root=root, data_root_=droot)
    except FileNotFoundError as e:
        return _result(STATUS_INVALID, "build", None,
                       f"语料发现失败: {e}", errors=[str(e)])
    except Exception as e:  # clean/chunk/qc 的任何确定性失败
        return _result(STATUS_INVALID, "build", None,
                       f"确定性处理失败: {e}", errors=[str(e)])

    aid = authors[0].author_id if len(authors) == 1 else None
    details = {
        "built_works": {
            wid: {
                "work_id": m["work_id"],
                "author_id": m["author_id"],
                "role": m["role"],
                "char_count": m["char_count"],
                "word_count": m["word_count"],
                "chapter_count": m["chapter_count"],
                "raw_sha256": m["raw_sha256"],
            }
            for wid, m in built.items()
        },
        "data_root": str(droot),
    }
    return _result(STATUS_READY_FOR_NEXT_STEP, "build", aid,
                   f"确定性处理完成：{len(built)} 部作品", details=details)


def onboard_author(manifest_path: str | Path,
                   corpus_root: str | Path | None = None,
                   data_root_: str | Path | None = None,
                   registry_dir: str | Path | None = None) -> dict[str, Any]:
    """编排 validate → register → build；随后需要 LLM 的步骤**不执行**。

    返回：
        - INVALID              任何确定性步骤失败（校验/注册/构建）。
        - REQUIRES_LLM_APPROVAL  确定性部分完成；下一步 = 采样 + LLM 特征分析 →
                                  聚合 → 策略合并 → 画像合成（阻塞，等待批准）。
    """
    v = validate_author(manifest_path, corpus_root=corpus_root, registry_dir=registry_dir)
    if v["status"] == STATUS_INVALID:
        return v

    r = register_author(manifest_path, registry_dir=registry_dir)
    if r["status"] == STATUS_INVALID:
        return r

    b = build_author(manifest_path, corpus_root=corpus_root, data_root_=data_root_)
    if b["status"] == STATUS_INVALID:
        return b

    aid = v.get("author_id") or r.get("author_id") or b.get("author_id")
    details = {
        "validate": v["details"],
        "register": r["details"],
        "build": b["details"],
        "pending_llm_steps": _LLM_NEXT_STEPS,
    }
    return _result(
        STATUS_REQUIRES_LLM_APPROVAL, "onboard", aid,
        "确定性 onboarding 完成；后续需真实 LLM 的步骤已阻塞、未执行（不产生任何 "
        "LLM 请求/费用）。请显式批准后再运行特征分析与画像合成。",
        details=details,
    )
