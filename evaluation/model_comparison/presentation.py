from __future__ import annotations

import math
from pathlib import Path
from collections.abc import Mapping, Sequence

from evaluation.mechanistic_retina.metrics import JsonValue


def write_pareto(path: Path, rows: Sequence[Mapping[str, JsonValue]]) -> None:
    import matplotlib.pyplot as plt

    selected = tuple(row for row in rows if row["model"] != "Bias")
    figure, axis = plt.subplots(figsize=(7.2, 5.0))
    for row in selected:
        params = float(row["params"])
        axis.errorbar(
            float(row["val_ce_mean"]),
            float(row["global_rf_mean"]),
            xerr=float(row["val_ce_sd"]),
            yerr=float(row["global_rf_sd"]),
            fmt="o",
            markersize=5.0 + 2.0 * math.log10(max(1.0, params)),
            capsize=3,
            label=str(row["model"]),
        )
    axis.set_xlabel("Teacher-expected validation CE (lower is better)")
    axis.set_ylabel("Global RF cosine (higher is better)")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def decision_report(
    rows: Sequence[Mapping[str, JsonValue]], decision: Mapping[str, JsonValue]
) -> str:
    row_lines = [
        "| Model | Params | Val CE ↓ | Bits/spike ↑ | Global RF ↑ | Spatial ↑ | Temporal ↑ | Exact cell ↑ | Stability ↑ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        row_lines.append(
            "| {model} | {params} | {ce:.6f} | {bits:.4f} | {rf} | {spatial} | {temporal} | {exact} | {stability} |".format(
                model=row["model"],
                params=row["params"],
                ce=float(row["val_ce_mean"]),
                bits=float(row["bits_per_spike_mean"]),
                rf=_fmt(row["global_rf_mean"]),
                spatial=_fmt(row["spatial_mean"]),
                temporal=_fmt(row["temporal_mean"]),
                exact=_fmt(row["exact_cell_mean"]),
                stability=_fmt(row["stability"]),
            )
        )
    interpretation = (
        "| Model | Explicit RF | Structured subunits | Named H1/BC/AC | Structural ablation | Held-out mechanism recovery |\n"
        "|---|---|---|---|---|---|\n"
        "| GLM | Yes | No | No | No | No |\n"
        "| LN-LN | Yes/subunit | Yes | BC-like only | Subunit | Limited |\n"
        "| Graph-TCN | Jacobian | No | No | Post-hoc | No named mechanism |\n"
        "| Mechanistic | Base+effective | Yes | Yes | Yes | H1/AC supported |"
    )
    return (
        "# Canonical Candidate0 T=2 模型比较\n\n"
        f"唯一科学 Case：`{decision['case']}`。完整首轮五模型比较已完成，未修改当前架构。\n\n"
        + "\n".join(row_lines)
        + "\n\n## 解释能力\n\n"
        + interpretation
        + "\n\n## 决策\n\n"
        f"主模型相对最佳 baseline 的 CE 差为 {float(decision['main_ce_gap_to_best_baseline']):.6f}，"
        f"RF cosine 差为 {float(decision['main_rf_gap_to_best_baseline']):.6f}。"
        f"Failure mode={decision['failure_mode']}；后续唯一允许的最小改动为："
        f"{decision['authorized_minimal_future_change'] or '无，冻结当前架构'}。本轮未执行架构优化。\n\n"
        "既有 H1/AC teacher 与 held-out mechanism evidence 仅引用，未重跑。"
        "T=2 不足以形成可靠 repeated-validation PSTH，状态为 `PSTH_CORRELATION_NOT_RELIABLE`。\n\n"
        "Final-test 状态：`TEST_SPLIT_ACCESSED_FOR_IDENTITY_ONLY`；"
        "`TEST_EXAMPLES_NOT_USED_FOR_INFERENCE_METRICS_MODEL_SELECTION_OR_CONCLUSIONS`；"
        "`FINAL_TEST_SCIENTIFIC_EVALUATION_NOT_CONSUMED`。\n"
    )


def _fmt(value: JsonValue) -> str:
    return "—" if value is None else f"{float(value):.4f}"


__all__ = ["decision_report", "write_pareto"]
