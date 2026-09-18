from __future__ import annotations

import csv
from datetime import datetime
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from adapter import AUTHOR_SHA
from data_adapter import DEST, FINAL, ROOT, digest, exclusive_json, read_json
from run_assessment import write_csv
from training_engine import SEEDS, MAX_UPDATES


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def complexity(completion: dict, initial: dict, recovery: dict | None) -> list[dict]:
    prior = read_rows(ROOT / "output/experiments/retipath_spatial_ei_phase2_population/training_per_seed.csv")
    recovered = {r["seed"]: r for r in recovery["resume_records"]} if recovery else {}
    execution = read_json(DEST / "checkpoints/provenance/execution_lock.json")
    wall_seconds = (datetime.fromisoformat(completion["completed_utc"]) - datetime.fromisoformat(execution["started_utc"])).total_seconds()
    rows = []
    for seed in SEEDS:
        phase2 = [r for r in prior if r["condition"] == "C" and int(r["seed"]) == seed]
        end_times = {}
        for r in phase2:
            key = (r["cell_id"], r["phase"])
            end_times[key] = max(end_times.get(key, 0.0), float(r["elapsed_seconds"]))
        refit_steps = sum(refs[str(seed)]["selected_updates"] for refs in initial["retipath_identity"].values())
        selection = next(r for r in completion["selections"] if r["seed"] == seed)
        refit = next(r for r in completion["refits"] if r["seed"] == seed)
        replayed = recovered.get(seed, {}).get("recomputed_already_completed_updates", 0)
        failed_seconds = recovered.get(seed, {}).get("discarded_uncheckpointed_wall_seconds_estimate", 0.0)
        rows.append({"model": "RetiPath", "seed": seed, "cells": 22, "trainable_parameters_per_fitted_model": 37,
            "fitted_models_per_seed": 22, "total_trainable_parameters_per_seed": 37 * 22,
            "fixed_parameter_entries_per_cell": 96, "shared_core_trainable_parameters": 0,
            "sharing": "independent per-cell models; fixed geometry and mechanistic priors",
            "stimulus_causal_window": "full available independent-sequence prefix, at most 150 bins",
            "history": "strictly past exponential occupancy, tau=30ms, reset each 150-bin sequence",
            "selection_opportunities": "historical Phase 2 3000 updates/cell/seed maximum; 25-step inner-validation checks",
            "new_updates_in_this_assessment": 0, "historical_refit_updates_sum_over_cells": refit_steps,
            "historical_sum_of_fit_elapsed_seconds": sum(end_times.values()),
            "architecture_and_geometry_historically_selected": True,
            "training_cost_scope": "historical CPU fits; sum of overlapping per-fit durations, not elapsed wall time"})
        rows.append({"model": "OpenRetina作者实现的任务适配版本", "seed": seed, "cells": 22,
            "trainable_parameters_per_fitted_model": initial["model_parameters"], "fitted_models_per_seed": 1,
            "total_trainable_parameters_per_seed": initial["model_parameters"],
            "fixed_parameter_entries_per_cell": 0, "shared_core_trainable_parameters": initial["shared_core_parameters"],
            "sharing": "one shared author Core; independent cell-specific author readout/history; real recordings retained",
            "stimulus_causal_window": "31 frames inclusive; lag 0..30 = 0..200ms at 150Hz",
            "history": "same strictly-past 30ms occupancy feature, independent trainable inhibitory coefficient per cell",
            "selection_opportunities": "one architecture, one training/regularization config; 3 fixed seeds; step0 and 15 checks",
            "selection_updates": selection["completed_steps"], "refit_updates": refit["completed_steps"],
            "new_updates_in_this_assessment": selection["completed_steps"] + refit["completed_steps"] + replayed,
            "unique_trajectory_updates": selection["completed_steps"] + refit["completed_steps"],
            "engineering_recomputed_updates": replayed,
            "additional_uncheckpointed_failed_work_seconds_estimate": failed_seconds,
            "assessment_total_wall_seconds_including_recovery_and_wait": wall_seconds,
            "selection_elapsed_seconds": selection["elapsed_seconds"], "refit_elapsed_seconds": refit["elapsed_seconds"],
            "logical_batch": "4 movie windows; 4 real recording/trial fragments per cell, 88 independent fragments/update",
            "optimizer": "AdamW + official OneCycleLR + clip1 + official core/readout regularizers",
            "architecture_and_geometry_historically_selected": False,
            "training_cost_scope": "GPU Core + deterministic CPU readout; concurrent fit durations overlap; failed unsaved work reported separately; total wall covers original launch through final evaluation"})
    return rows


def figures(comparisons: list[dict[str, str]], completion: dict) -> None:
    dest = DEST / "checkpoints/figures"
    dest.mkdir(exist_ok=False)
    labels = ("[16,20)", "[20,60)", "[240,300)")
    fig, ax = plt.subplots(figsize=(9, 4.6), layout="constrained")
    for i, label in enumerate(labels):
        agg = next(r for r in comparisons if r["range"] == label and r["seed"] == "aggregate")
        mean, low, high = (float(agg[k]) for k in ("paired_mean", "ci_low", "ci_high"))
        ax.errorbar(mean, i, xerr=[[mean-low], [high-mean]], color="#126b91", fmt="o", capsize=5, lw=2)
        for j, seed in enumerate(SEEDS):
            r = next(r for r in comparisons if r["range"] == label and r["seed"] == str(seed))
            ax.plot(float(r["paired_mean"]), i + .11 * (j-1), ".", color="#777777", markersize=6)
    ax.axvline(0, color="#333333", lw=1, ls="--")
    ax.set_yticks(range(3), [f"{r} s" for r in labels]); ax.invert_yaxis()
    ax.set_xlabel("RetiPath NLL - OpenRetina-adapted NLL (nats / scored bin)")
    ax.set_title("Frozen descriptive temporal comparison\nBlue: seed aggregate with paired-cell 95% CI; grey: seed means")
    ax.grid(axis="x", alpha=.2)
    fig.savefig(dest / "prediction.png", dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout="constrained")
    colors = ("#0072b2", "#d55e00", "#009e73")
    for color, seed in zip(colors, SEEDS, strict=True):
        select = next(r for r in completion["selections"] if r["seed"] == seed)
        refit = next(r for r in completion["refits"] if r["seed"] == seed)
        rs = select["curve_records"]
        steps = [r["step"] for r in rs]
        axes[0].plot(steps, [r["train_nll"] for r in rs], "--", color=color, label=str(seed)+" fit")
        axes[0].plot(steps, [r["inner_validation_nll"] for r in rs], "-o", markersize=3, color=color, label=str(seed)+" val")
        axes[0].axvline(select["selected_steps"], color=color, alpha=.25)
        rr = refit["curve_records"]
        axes[1].plot([r["step"] for r in rr], [r["train_nll"] for r in rr], "-o", color=color, markersize=3, label=str(seed))
        axes[2].plot(steps, [r["lr_next_update"] for r in rs], "-o", markersize=3, color=color, label=str(seed))
    for ax in axes:
        ax.set_xlabel("Optimizer updates"); ax.grid(alpha=.2)
    axes[0].set_title("Train-only selection"); axes[0].set_ylabel("Equal-cell Bernoulli NLL")
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].set_title("Fresh full-training refit\nFixed previously selected length"); axes[1].set_ylabel("Full-training NLL")
    axes[1].legend(fontsize=8)
    axes[2].set_title("Official OneCycle schedule\nShown: LR for next update"); axes[2].set_ylabel("Learning rate"); axes[2].set_yscale("log")
    fig.savefig(dest / "training.png", dpi=170); plt.close(fig)


def classification(row: dict[str, str]) -> str:
    low, high = float(row["ci_low"]), float(row["ci_high"])
    return "RetiPath预测优势" if high < 0 else "RetiPath预测劣势" if low > 0 else "比较未分辨（不表示等效）"


def report(comparisons: list[dict[str, str]], completion: dict, initial: dict, recovery: dict | None) -> str:
    aggregate = [r for r in comparisons if r["seed"] == "aggregate"]
    primary = next(r for r in aggregate if r["range"] == "[20,60)")
    table = ["| 区间(s) | N | RetiPath NLL | OpenRetina-adapted NLL | Mean ΔNLL | Median ΔNLL | RetiPath胜/负/平 | Paired-cell 95% CI |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in aggregate:
        table.append(f"| {r['range']} | {r['n_cells']} | {float(r['retipath_absolute_nll']):.6f} | {float(r['openretina_adapted_absolute_nll']):.6f} | {float(r['paired_mean']):+.6f} | {float(r['paired_median']):+.6f} | {r['wins_retipath']}/{r['losses_retipath']}/{r['ties']} | [{float(r['ci_low']):+.6f}, {float(r['ci_high']):+.6f}] |")
    seed_table = ["| 区间 | Seed | Mean ΔNLL | RetiPath胜/负/平 | 95% CI |", "| --- | --- | --- | --- | --- |"]
    for r in comparisons:
        if r["seed"] == "aggregate":
            continue
        seed_table.append(f"| {r['range']} | {r['seed']} | {float(r['paired_mean']):+.6f} | {r['wins_retipath']}/{r['losses_retipath']}/{r['ties']} | [{float(r['ci_low']):+.6f}, {float(r['ci_high']):+.6f}] |")
    training = ["| Seed | Selection完成步数 | 选定/Refit步数 | 停止原因 | 最低inner-val NLL | Selection/refit分钟 |",
                "| --- | --- | --- | --- | --- | --- |"]
    limits = []
    for s in completion["selections"]:
        r = next(r for r in completion["refits"] if r["seed"] == s["seed"])
        training.append(f"| {s['seed']} | {s['completed_steps']} | {s['selected_steps']} | {s['stop_reason']} | {s['best_inner_validation_nll']:.6f} | {s['elapsed_seconds']/60:.1f}/{r['elapsed_seconds']/60:.1f} |")
        curve = s["curve_records"]
        last = curve[-1]["inner_validation_nll"]
        earlier = curve[max(0, len(curve)-4)]["inner_validation_nll"]
        limits.append(f"seed {s['seed']}：最后最多三个验证间隔的NLL变化为 {last-earlier:+.6f}，best位于step {s['selected_steps']}/{s['completed_steps']}")
    cap_count = sum(r["stop_reason"] == "budget_limit" for r in completion["selections"])
    endpoint_count = sum(r["selected_steps"] == r["completed_steps"] for r in completion["selections"])
    consistency = []
    for a in aggregate:
        values = [float(r["paired_mean"]) for r in comparisons if r["range"] == a["range"] and r["seed"] != "aggregate"]
        directions = "三个seed的mean方向一致" if all(v > 0 for v in values) or all(v < 0 for v in values) else "seed mean方向不一致"
        consistency.append(f"{a['range']}：{directions}，mean范围[{min(values):+.6f},{max(values):+.6f}]")
    evidence = "；".join(f"{r['range']}为{classification(r)}" for r in aggregate)
    primary_seeds = [r for r in comparisons if r["range"] == "[20,60)" and r["seed"] != "aggregate"]
    stable_disadvantage = float(primary["ci_low"]) > 0 and all(float(r["paired_mean"]) > 0 for r in primary_seeds)
    next_step = ("有理由针对已经观察到的预测差距提出下一轮优化，但本次比较尚不能指定应增加哪一层或哪种生理机制；特别不能据此认定H1→BC中间层必要。是否实施和采用何种方案需另行决定。"
                 if stable_disadvantage else
                 "本次没有建立跨seed稳定的主要区间预测劣势，因此没有仅凭这一比较自动启动RetiPath结构优化的充分理由。保留当前模型，不能把比较未分辨解释为等效，也不能据此排除更充分训练的其他结果。")
    recovery_note = ""
    if recovery:
        total_replayed = sum(r["recomputed_already_completed_updates"] for r in recovery["resume_records"])
        recovery_note = f"发生一次CUDA launch failure事件，两个refit从step46的完整model/optimizer/scheduler/RNG保存状态恢复，各重放21个未checkpoint的已完成更新，共{total_replayed}次额外计算。全部重复段记录的损失、正则和梯度数值一致，原299/207/253终点保持；step67的完整参数state未保存，因此不宣称该点的全state逐位比对。同期Windows记录nvlddmkm事件13/153，底层故障原因为UNVERIFIED，作者和adapter代码均无需修改。完成的fit参数及输出有限。额外计算及约12.2分钟的未保存失败工作估计单列于complexity.csv，不隐去，也不计为新的科学选择机会。"
    content = f"""# 当前 RetiPath 能力评估

## 1. 实际采用哪个作者模型，做了哪些适配？

采用 **OpenRetina作者实现的任务适配版本**：官方 [vystrcilova_2024_nm_cnn.yaml](https://github.com/open-retina/open-retina/blob/{AUTHOR_SHA}/configs/vystrcilova_2024_nm_cnn.yaml)，固定commit `{AUTHOR_SHA}`，marmoset自然电影Core+Readout配置。直接导入作者三层64/64/64-channel Core和Gaussian Readout，保留原核宽、时间basis、BatchNorm/ELU/dropout和主要正则。没有用其他细胞的预训练权重，不称SOTA或原论文复现。

必要适配为17×17空间zero padding、30-bin时间左padding、官方readout的softplus之前取drive并接Bernoulli link、项目相同的严格过去30ms occupancy history及抑制方向系数、独立recording/trial的mask和联合训练接口。唯一output bias保留在作者readout。Core在GPU、作者Readout在CPU保证确定性；没有复制自制CNN。[协议](PROTOCOL.md)和[适配说明](adapter_notes.md)记录全部差别及来源。

正式RetiPath的66个冻结fresh-refit checkpoints均经manifest核对：conductance、K=2、37个原可学习参数/cell，另有96个固定算子参数条目。没有训练或改动RetiPath。

## 2. 训练是否正常，是否有明显预算限制？

工程检查和正式训练的参数/输出均有限；全部预期参数进入optimizer并在真实小batch获得梯度、发生更新，同seed两步训练完整state可逐位重复。梯度使用作者norm clip=1，较大的裁剪前梯度不构成无效基线的判据。只有一个来源训练/正则配置，三个固定seeds；预算由作者15 epochs换算为最多345次联合updates，训练内NLL选择长度后fresh refit，未用三个评价区间调参或选步。

{chr(10).join(training)}

{cap_count}/3 selections达到预算上限，{endpoint_count}/3的最优点位于实际停止点。{"；".join(limits)}。后期训练NLL继续下降，验证NLL则在附近波动，三个最优点均早于上限；没有出现所有seed在预算末端持续改善的明显迹象。可报告这个配置下的验证平台，但达到预算上限不等于充分收敛，也不能证明更长训练或其他合法训练配置没有收益。

{recovery_note}

![训练曲线](checkpoints/figures/training.png)

作者模型每seed一个共享Core，合计{initial['model_parameters']:,}个trainable parameters；RetiPath是22个逐cell模型，每个37个原可学习参数。作者模型每update含4个movie窗口、各cell各4条独立真实响应片段；相同输入的Core计算共享，响应没有拼成同时记录。RetiPath具有冻结geometry和机制先验，作者readout空间位置从其原初始化学习。学习数据、训练机会、窗口和成本的差别见[complexity.csv](complexity.csv)，不能由参数比宣传效率。GPU/CPU混合实现的耗时也不代表优化过的生产吞吐。

## 3. 三个区间上RetiPath与它相差多少？

以下为seed aggregate，单位nats/scored bin，ΔNLL=RetiPath−OpenRetina-adapted，负有利RetiPath。先逐cell平均三个seed的loss，再计算equal-cell统计和paired-cell percentile95% bootstrap CI（100000次）。绝不平均权重或概率；不将cell×seed或不同时间段作为额外生物重复。

{chr(10).join(table)}

![预测差值](checkpoints/figures/prediction.png)

[20,60)是本次主要能力比较，[16,20)是development描述，[240,300)是较远时间secondary；后者17cells由真实连续记录覆盖决定。三段均已消费，因此全部是descriptive temporal evaluation，不是新独立confirmation。相同17×17 Weber输入、150Hz occupancy、150-bin独立sequence、30-bin warmup/120-bin评分、source IDs、targets、mask和严格过去history已逐cell匹配。RetiPath复用合法冻结logits，作者模型全部checkpoint冻结后才评价；未读取[300,360)或新时间段。

## 4. 结果在cells和seeds之间是否稳定？

{"；".join(consistency)}。Cell层面的胜/负/平与CI分别报告；population mean方向不能写成每个cell都受益。

{chr(10).join(seed_table)}

逐cell原始absolute NLL及配对差见[per_cell_scores.csv](per_cell_scores.csv)，完整每seed及aggregate统计见[comparison_summary.csv](comparison_summary.csv)。

## 5. 支持预测优势、预测劣势，还是未分辨？

在本次固定实现、预算和已消费区间内：{evidence}。主要区间结论是**{classification(primary)}**。CI跨零只表示未分辨，不宣称等效；方向优势也不扩展为对所有现代模型的普遍排序或生物机制验证。

## 6. 当前最需要解决的是表示能力、训练充分性，还是尚无法定位？

**尚无法定位为单一原因。** 本次比较同时涉及共享训练与逐cell训练、自由Gaussian位置与固定geometry、有限31-frame刺激窗口与完整sequence状态、作者正则和不同训练预算。验证曲线可以指出是否仍有预算限制，却不能把两模型差异独立归因于表示能力或优化充分性。未进行本轮禁止的机制修改或额外训练实验，不能由NLL直接推断H1→BC需要加层。

既有normalized RF跨seed变化、未统一的RF-size方向，以及Mach/SBC模型响应和pathway ranking限制，均按原结果保留在[功能证据引用表](existing_functional_evidence.md)，没有重跑，也没有随本次预测胜负改写。错视响应稳定不是生物机制验证。

## 7. 是否已有充分理由启动下一轮RetiPath优化？

{next_step}本轮能力评估到此停止，没有自动修改RetiPath、增加机制、重跑RF/错视或消费新时间块。
"""
    for relative in ("PROTOCOL.md", "adapter_notes.md", "complexity.csv", "per_cell_scores.csv",
                     "comparison_summary.csv", "existing_functional_evidence.md",
                     "checkpoints/figures/prediction.png", "checkpoints/figures/training.png"):
        content = content.replace(f"]({relative})", f"]({(DEST / relative).as_posix()})")
    return content


def main() -> None:
    completion = read_json(DEST / "checkpoints/provenance/run_complete.json")
    initial = read_json(DEST / "checkpoints/provenance/pretraining_lock.json")
    verified = read_json(DEST / "checkpoints/provenance/verification_complete.json")
    assert verified["status"] == "VERIFIED"
    recovery = verified.get("runtime_recovery")
    comparisons = read_rows(DEST / "comparison_summary.csv")
    write_csv(DEST / "complexity.csv", complexity(completion, initial, recovery))
    figures(comparisons, completion)
    with (DEST / "REPORT.md").open("x", encoding="utf-8") as stream:
        stream.write(report(comparisons, completion, initial, recovery))
    exclusive_json(DEST / "checkpoints/provenance/report_generation.json",
                   {"script_sha256": digest(Path(__file__)), "report_sha256": digest(DEST / "REPORT.md"),
                    "functional_evidence_referenced_not_rerun": True})
    print("REPORT AND FIGURES GENERATED")


if __name__ == "__main__":
    main()
