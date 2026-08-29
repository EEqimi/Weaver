# knowledge/ingestion/add_author.py
"""CLI：python -m knowledge.ingestion.add_author path/to/author_manifest.{json,yaml}

新增一位作者：校验 manifest → 注册进 corpus registry → 确定性处理语料
（discover→clean→chunk→QC→metadata）。需要真实 LLM 的后续步骤（特征分析 →
聚合 → 策略合并 → 画像合成）绝不自动执行，只打印 REQUIRES_LLM_APPROVAL。

零 LLM、零费用、只读 DEEPSEEK_API_KEY 之外的本地文件；只写 data/（gitignore）
与 manifests/{author_id}.json（committed registry）。
"""
from __future__ import annotations

import json
import sys

from .onboarding import (
    STATUS_INVALID,
    STATUS_REQUIRES_LLM_APPROVAL,
    onboard_author,
)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print("用法: python -m knowledge.ingestion.add_author "
              "path/to/author_manifest.{json,yaml}", file=sys.stderr)
        return 2

    manifest_path = argv[0]
    result = onboard_author(manifest_path)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] == STATUS_INVALID:
        return 1
    if result["status"] == STATUS_REQUIRES_LLM_APPROVAL:
        print("\n[阻塞] 后续需真实 LLM 的步骤未执行；请显式批准后再继续。",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
