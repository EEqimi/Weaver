# knowledge/corpus/manifest.py
"""作者 / 作品 manifest schema + 解析校验（corpus registry 的**数据**输入）。

manifest 是 JSON（stdlib、确定性、无第三方依赖、便于 diff）。schema（V0.1，最小可扩展）：

    {
      "schema_version": "0.1.0",
      "authors": [
        {
          "author_id": "austen",
          "display_name": "Jane Austen",
          "language": "en",
          "works": [
            {
              "work_id": "pride_and_prejudice",
              "title": "Pride and Prejudice",
              "year": 1813,
              "genre": "novel",
              "filename": "pride_and_prejudice..txt",
              "role": "train"
            }
          ]
        }
      ]
    }

字段按现有 pipeline 的真实需要取舍（不为了"通用"过度设计）：
    - `filename` 是语料根目录内的相对文件名（`discover` / `build_work` 直接
      `root / filename` 定位），非绝对路径、非机器相关。
    - `role` 仅 `train` | `held_out`（spec §9.3：held-out 绝不进入训练/画像）。
    - `year` / `genre` 仅用于 metadata 报告，缺失时不阻碍注册（year→None、genre→"unknown"）。

绝不含任何密钥 / 机器相关路径；一份 manifest 可声明多位作者（如内置 Austen+Dickens）。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA_VERSION = "0.1.0"
TRAIN = "train"
HELD_OUT = "held_out"
_ROLES = (TRAIN, HELD_OUT)


class ManifestError(ValueError):
    """manifest 解析 / 校验失败（fail-closed，绝不静默接受坏清单）。"""


@dataclass(frozen=True)
class WorkManifest:
    work_id: str
    title: str
    filename: str
    role: str
    year: int | None = None
    genre: str = "unknown"

    @classmethod
    def from_dict(cls, d: dict[str, Any], *, author_id: str) -> "WorkManifest":
        work_id = _require_str(d, "work_id", author_id)
        title = _require_str(d, "title", author_id, work_id)
        filename = _require_str(d, "filename", author_id, work_id)
        role = d.get("role")
        if role not in _ROLES:
            raise ManifestError(
                f"work {work_id!r}: role 必须为 {list(_ROLES)} 之一，得到 {role!r}")
        year = d.get("year")
        if year is not None and not isinstance(year, int):
            raise ManifestError(f"work {work_id!r}: year 必须为 int 或省略，得到 {year!r}")
        genre = d.get("genre")
        if genre is None:
            genre = "unknown"
        elif not isinstance(genre, str) or not genre.strip():
            raise ManifestError(f"work {work_id!r}: genre 必须为非空字符串")
        return cls(work_id=work_id, title=title, filename=filename, role=role,
                   year=year, genre=genre)


@dataclass(frozen=True)
class AuthorManifest:
    author_id: str
    display_name: str
    language: str
    works: tuple[WorkManifest, ...] = ()

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AuthorManifest":
        author_id = _require_str(d, "author_id")
        display_name = _require_str(d, "display_name", author_id)
        language = d.get("language")
        if language is None:
            language = "en"
        elif not isinstance(language, str) or not language.strip():
            raise ManifestError(f"author {author_id!r}: language 必须为非空字符串")
        works_raw = d.get("works")
        if not isinstance(works_raw, list) or not works_raw:
            raise ManifestError(f"author {author_id!r}: works 必须为非空列表")
        works = tuple(WorkManifest.from_dict(w, author_id=author_id) for w in works_raw)
        return cls(author_id=author_id, display_name=display_name,
                   language=language, works=works)


def _require_str(d: dict[str, Any], key: str, *ctx: str) -> str:
    v = d.get(key)
    if not isinstance(v, str) or not v.strip():
        where = " / ".join(ctx) or key
        raise ManifestError(f"{where}: 缺少必填非空字段 {key!r}")
    return v.strip()


def parse_manifest(obj: Any) -> list[AuthorManifest]:
    """解析并校验 manifest 顶层结构，返回作者清单（按 author_id 升序，确定性）。

    校验（fail-closed）：
        - schema_version 必须 == MANIFEST_SCHEMA_VERSION；
        - authors 为非空列表；author_id 全局唯一且非空；display_name / language 非空；
        - 每作者 works 非空；work_id 全局唯一；title / filename 非空；role ∈ 二值。
    """
    if not isinstance(obj, dict):
        raise ManifestError(f"manifest 顶层必须为对象，得到 {type(obj).__name__}")
    if obj.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ManifestError(
            f"schema_version 不匹配: 期望 {MANIFEST_SCHEMA_VERSION!r}, "
            f"得到 {obj.get('schema_version')!r}")
    authors_raw = obj.get("authors")
    if not isinstance(authors_raw, list) or not authors_raw:
        raise ManifestError("manifest 必须含非空 authors 列表")

    authors = [AuthorManifest.from_dict(a) for a in authors_raw]
    seen_authors: set[str] = set()
    seen_works: set[str] = set()
    for a in authors:
        if a.author_id in seen_authors:
            raise ManifestError(f"author_id 重复: {a.author_id!r}")
        seen_authors.add(a.author_id)
        for w in a.works:
            if w.work_id in seen_works:
                raise ManifestError(f"work_id 重复（跨作者）: {w.work_id!r}")
            seen_works.add(w.work_id)
    return sorted(authors, key=lambda a: a.author_id)


def load_manifest_file(path: str | Path) -> list[AuthorManifest]:
    """读取单个 manifest 文件并解析校验，返回作者清单（author_id 升序）。

    序列化格式按扩展名自动识别：
        - `.json`：内置 registry 与 onboarding 的**规范格式**（stdlib、确定性、无依赖）；
        - `.yaml` / `.yml`：可选（仅当安装了 PyYAML 时可用，否则给出明确错误并建议改用 JSON）。

    失败一律抛 `ManifestError`（fail-closed），绝不静默接受坏清单。
    """
    p = Path(path)
    if not p.is_file():
        raise ManifestError(f"manifest 文件不存在: {p}")
    suffix = p.suffix.lower()
    if suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ImportError as e:
            raise ManifestError(
                f"解析 YAML manifest 需要 PyYAML（当前未安装）。"
                f"请安装 PyYAML，或改用 JSON manifest（推荐，无额外依赖）: {p}"
            ) from e
        try:
            obj = yaml.safe_load(p.read_text(encoding="utf-8"))
        except Exception as e:  # YAML 解析错误类型随 PyYAML 版本而变
            raise ManifestError(f"YAML 解析失败 {p.name}: {e}") from e
    else:  # 默认按 JSON
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ManifestError(f"JSON 解析失败 {p.name}: {e}") from e
    return parse_manifest(obj)
