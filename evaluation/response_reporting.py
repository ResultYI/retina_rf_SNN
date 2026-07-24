from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from baselines.point_process_glm import GLMFitResult
from evaluation.response_metrics import ResponseMetrics
from evaluation.rf_dynamic import DynamicRFResult
from evaluation.rf_static import StaticRFResult


def write_response_report(
    output_dir: str | Path,
    *,
    conditional: ResponseMetrics,
    free_running: ResponseMetrics,
    glm: GLMFitResult,
    static_rf: StaticRFResult,
    static_reference: dict[str, float] | None,
    dynamic_rf: DynamicRFResult,
    initialized_dynamic_rf: DynamicRFResult,
    synthetic: bool,
    checkpoint: str,
) -> None:
    output = Path(output_dir)
    metrics = {
        "evidence_kind": "synthetic_method_validation" if synthetic else "real_recording",
        "checkpoint": checkpoint,
        "response_prediction": {
            "conditional": asdict(conditional),
            "free_running": asdict(free_running),
            "glm": asdict(glm.metrics),
        },
        "static_rf": {
            "identifiable": static_rf.identifiable,
            "finite_difference_relative_error": static_rf.finite_difference_relative_error,
            "kernel_shape": list(static_rf.kernels.shape),
            "reference_comparison": static_reference,
        },
        "dynamic_rf": {
            "trained": asdict(dynamic_rf),
            "initialized": asdict(initialized_dynamic_rf),
        },
    }
    (output / "final_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output / "run_status.json").write_text(
        json.dumps({"status": "COMPLETED", "synthetic": synthetic}, indent=2),
        encoding="utf-8",
    )
    title = "合成方法验证" if synthetic else "真实 RGC 响应拟合"
    report = (
        f"# RGC 响应拟合报告\n\n"
        f"证据类型：{title}\n\n"
        f"- 条件测试 NLL：{conditional.nll:.6f}\n"
        f"- 自由运行测试 NLL：{free_running.nll:.6f}\n"
        f"- 静态 GLM 测试 NLL：{glm.metrics.nll:.6f}\n"
        f"- Static RF：{'可辨识' if static_rf.identifiable else '不可辨识'}\n"
        f"- Dynamic RF：{dynamic_rf.status}\n"
        f"- Dynamic RF context pairs：{dynamic_rf.pair_count}\n\n"
        "RF 由训练后 spike logit 的局部输入导数提取，不作为训练标签。"
    )
    if static_reference is not None:
        report += (
            "\n- Synthetic static RF correlation："
            f"{static_reference['mean_kernel_correlation']:.6f}"
        )
    if synthetic:
        report += "\n\n本报告仅验证方法链路，不构成真实视网膜生理结论。"
    (output / "final_report_zh.md").write_text(report + "\n", encoding="utf-8")


__all__ = ["write_response_report"]
