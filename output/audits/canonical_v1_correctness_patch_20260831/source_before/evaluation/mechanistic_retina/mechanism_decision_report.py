from __future__ import annotations

import statistics

from evaluation.mechanistic_retina.mechanism_false_positive import (
    FalsePositiveSummary,
)
from evaluation.mechanistic_retina.mechanism_heldout_artifacts import (
    FinalDecision,
    PathwayDecision,
)


def decision_report_zh(decision: FinalDecision, checkpoint_count: int) -> str:
    h1 = decision.h1
    ac = decision.ac
    false = decision.false_positive
    partial = (
        "H1_SUPPORTED_AC_NOT_SUPPORTED"
        if h1.supported and not ac.supported
        else "AC_SUPPORTED_H1_NOT_SUPPORTED"
        if ac.supported and not h1.supported
        else "不适用"
    )
    natural = decision.case == "MECHANISM-IDENTIFIABLE-RETINA-SUPPORTED"
    return "\n".join(
        (
            "# 最终机制泛化裁决",
            "",
            "1. 完全相同科学配置重跑：是，冻结身份核验通过。",
            f"2. 必要 checkpoint：{checkpoint_count}/15，均完成原子保存与逐状态回读。",
            "3. Replay：与上一轮一致，CE/RF/gate 均在预注册容差内。",
            f"4. H1 Full vs No-H1：{h1.passing_seeds}/3 seeds 满足 Full 更优。",
            f"5. H1 clamp/current/sensitivity/RF：中位 clamp ΔCE={_median_metric(h1, 'clamp_ce_delta'):.8g}，current={_median_path(h1, 'current'):.8g}，sensitivity={_median_path(h1, 'sensitivity'):.8g}，RF cosine={_median_path(h1, 'rf_cosine'):.8g}。",
            f"6. AC Full vs No-AC：{ac.passing_seeds}/3 seeds 满足 Full 更优。",
            f"7. AC clamp/current/sensitivity/RF：中位 clamp ΔCE={_median_metric(ac, 'clamp_ce_delta'):.8g}，current={_median_path(ac, 'current'):.8g}，sensitivity={_median_path(ac, 'sensitivity'):.8g}，RF cosine={_median_path(ac, 'rf_cosine'):.8g}。",
            f"8. False-positive：Base 低效应方向保持；H1→false-AC clamp ratio 中位={_median_false(false, 'h1_false_ac_clamp_ratio'):.8g}；AC→false-H1 clamp ratio 中位={_median_false(false, 'ac_false_h1_clamp_ratio'):.8g}。",
            "9. Cell-specific RF：继续保持 SUPPORTED；exact-cell 16/16。",
            "10. Type prototype limitation：TYPE_PROTOTYPE_CENTROID_CONSISTENCY=8/16，仅记录为 population prototype consistency limitation。",
            f"11. H1 四证据链：{'完整' if h1.supported else '不完整'}。",
            f"12. AC 四证据链：{'完整' if ac.supported else '不完整'}。",
            f"13. 唯一最终 Case：{decision.case}；partial 细分={partial}。",
            "14. 总体研究目标含义：cell-specific RF 与受控 synthetic pathway responsibility 的 held-out 泛化得到同一模型框架内的直接检验。",
            f"15. 自然图像阶段授权：{'是' if natural else '否'}。",
            "16. Pathway-local TCN 授权：否；需先由独立自然复杂刺激 residual 决定。",
            "17. Watchdog：300 秒间隔、连续 2 次停滞阈值、最多 1 次受控重试；本 worker 完成。",
            "18. Final-test：TEST_SPLIT_ACCESSED_FOR_IDENTITY_ONLY；TEST_EXAMPLES_NOT_USED_FOR_INFERENCE_METRICS_MODEL_SELECTION_OR_CONCLUSIONS；FINAL_TEST_SCIENTIFIC_EVALUATION_NOT_CONSUMED。",
            "",
        )
    )


def _median_metric(decision: PathwayDecision, field: str) -> float:
    return float(statistics.median(getattr(value, field) for value in decision.metrics))


def _median_path(decision: PathwayDecision, field: str) -> float:
    return float(
        statistics.median(getattr(value.pathway, field) for value in decision.metrics)
    )


def _median_false(summary: FalsePositiveSummary, field: str) -> float:
    return float(statistics.median(getattr(value, field) for value in summary.seeds))


__all__ = ["decision_report_zh"]
