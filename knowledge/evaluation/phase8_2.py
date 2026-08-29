# knowledge/evaluation/phase8_2.py
"""Phase 8.2 真实验证入口（fresh generation + 四阶 gate 反馈闭环）。

严格按 spec §三–§二十四：
    - 全新生成 austen_02 / dickens_02（experiment_id = phase8_2-generation-v0.1，fresh
      request，绝不读 Phase 7 生成缓存；绝不覆盖 data/analysis/generation/ 旧产物）；
    - 同中性 brief（复用 NEUTRAL_REQUEST，两位作者一致，唯一变量是画像导出风格控制）；
    - 基线评估 → RevisionPlan（空 → no_action，停）→ 一次 RevisionRewriter
      （max_iterations=1，不自动第二轮）→ RevisionEffectAnalyzer（non-substantive →
      no_effect，停，记录省下的 provider 调用/token）→ 仅 substantive 才进
      Content Integrity / Literary Quality / Style Fidelity 三阶 gate；
    - 产物写 evaluation_v3/{author}_02/ + phase8_2_real_validation_summary.json /
      phase8_2_real_validation_report.md（机器可读 + 人类可读）。

绝不调用真实模型之外的任何东西；密钥只读（DEEPSEEK_API_KEY）；绝不合并 main / 开 PR。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import data_root as default_data_root
from ..generation.run import EXPERIMENT_ID_82, run_generation
from .run import build_provider, run_evaluation


def run_phase8_2_real_validation(data_root_: Path | None = None) -> dict[str, Any]:
    base = Path(data_root_) if data_root_ is not None else default_data_root()

    # 1. 全新生成（fresh request，独立 experiment_id，落盘 generation/{experiment_id}/）。
    gen_summary = run_generation(data_root_=base, experiment_id=EXPERIMENT_ID_82)

    # 2. 基线评估 + 一次改写 + 四阶 gate（fresh 文本 → 缓存全 miss，真实调用）。
    provider = build_provider(base)
    eval_summary = run_evaluation(
        data_root_=base, provider=provider,
        generation_experiment_id=EXPERIMENT_ID_82,
        run_tag="02", summary_prefix="phase8_2_real_validation",
        max_iterations=1)

    return {"generation": gen_summary, "evaluation": eval_summary}


def main() -> None:
    result = run_phase8_2_real_validation()
    gen = result["generation"]
    ev = result["evaluation"]
    print(f"generation experiment_id: {gen['experiment_id']}")
    print(f"generation total_tokens: {gen['total_tokens']}")
    for aid, a in ev["authors"].items():
        d = a["decision"]
        eff = a.get("revision_effect")
        print(f"{aid}: outcome={d['outcome']} rev_items={a['n_revision_items']} "
              f"effect={eff['effect_status'] if eff else None} "
              f"substantive={eff['substantive_edit'] if eff else None}")
    print(f"eval token_usage: {ev['token_usage']}")
    print(f"eval cache_hits/misses: {ev['cache_hits']} / {ev['cache_misses']}")
    print("artifacts: data/analysis/generation/phase8_2-generation-v0.1/ + "
          "data/analysis/evaluation_v3/{author}_02/ + "
          "phase8_2_real_validation_summary.json + phase8_2_real_validation_report.md")


if __name__ == "__main__":
    main()
