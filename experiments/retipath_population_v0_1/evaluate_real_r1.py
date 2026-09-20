from __future__ import annotations

import json
from pathlib import Path
import statistics
import sys
import traceback

import torch

from data.schottdorf_lee_catalog import public_recordings
from data.schottdorf_lee_multirecording import load_schottdorf_cell, load_schottdorf_movie_drive
from experiments.retipath_population_v0_1.circuit import BCOutput, PopulationRetipath
from experiments.retipath_population_v0_1.real_data import recording_mapping
from experiments.retipath_population_v0_1.real_r1 import (
    BASELINES, CONFIG, DTYPE, MOVIE, OUT, PROTOCOL, RECORDINGS, ROOT,
    TrainingBundle, forward, manifest, nll, protocol_contract, require,
    save_tensor_file, sha256, split_evidence, state_hash, tensor_hash, utc,
    verify_sources, write_csv, write_json,
)


MODELS = ("Population", "Constant", "LN", "CNN", "Canonical")


def verify_checkpoint_lock():
    require((OUT / "CHECKPOINT_LOCK.json").is_file(), "outer evaluation requires CHECKPOINT_LOCK")
    require(not (OUT / "EVALUATION_STARTED.json").exists(), "outer evaluation already started; no automatic repeat")
    require(not (OUT / "failure.json").exists(), "failed training cannot be evaluated")
    lock = json.loads((OUT / "CHECKPOINT_LOCK.json").read_text(encoding="utf-8"))
    cells, frozen = protocol_contract()
    verify_sources(frozen)
    require(lock["cohort_complete"] and lock["cell_ids"] == [c.cell_id for c in cells], "incomplete cohort")
    require(lock["recording_ids"] == [r for c in cells for r in c.recording_ids], "locked recordings differ")
    require(lock["protocol_sha256"] == sha256(PROTOCOL), "protocol changed")
    require(lock["source_lock_sha256"] == sha256(OUT / "source_lock.json"), "source lock changed")
    require(lock["preflight_sha256"] == sha256(OUT / "preflight.json"), "preflight changed")
    source = json.loads((OUT / "source_lock.json").read_text(encoding="utf-8"))
    verify_sources(source["source_sha256"])
    require(len(lock["checkpoints"]) == 9 and len(lock["training_summaries"]) == 9, "lock count differs")
    for path, digest in (lock["checkpoints"] | lock["training_summaries"]).items():
        require(sha256(OUT / path) == digest, f"locked artifact changed: {path}")
    for c in cells:
        summary = json.loads((OUT / "cells" / c.slug / "training-summary.json").read_text(encoding="utf-8"))
        require(summary["status"] == "COMPLETE" and summary["cell_id"] == c.cell_id, "incomplete cell")
        require(summary["inner"]["best_step"] == summary["refit"]["stop_step"], "K/refit mismatch")
    return cells, lock, source


def verify_saved_predictions(saved, split):
    require(torch.equal(saved["target"], split.spike_events), "baseline occupancy mismatch")
    require(torch.equal(saved["valid_mask"], split.valid_mask), "baseline mask mismatch")
    require(tuple(saved["source_image_ids"]) == split.source_image_ids, "baseline frame/recording/order mismatch")
    require(tuple(saved["trial_indices"]) == split.trial_indices, "baseline trial order mismatch")
    require(saved["logits_trained"].shape == split.spike_events.shape, "baseline logit shape mismatch")
    require(bool(torch.isfinite(saved["logits_trained"]).all()), "nonfinite baseline logits")


def make_report(per_cell, aggregates, audit):
    lines = ["# Population RetiPath real-data R1 results", "",
             f"完成时间：{utc()}。状态：**COMPLETE / VERIFIED**。", "",
             "严格执行 [冻结 R1 protocol](RETIPATH_REAL_R1_PROTOCOL.md)：MC ON 5 / MC OFF 4 biological cells，"
             "16 recordings；每 cell 独立参数与 optimizer，同 cell recordings 共用参数，每个 sequence baseline reset。"
             "LegacyPReLU，359 raw scalars 全部列入 joint optimizer；仅 RGC conditional Bernoulli NLL + 原 hierarchy penalty。", "",
             "9 个唯一 final checkpoints 全部写入并锁定后，独立 evaluator 才开始读取 outer-validation 响应。"
             "未进行 validation-driven 选模或重跑；没有改 loss/prior/architecture、缩减 cohort 或加入其他实验。", "",
             "## 每 cell validation NLL", "",
             "单位：nats / scored bin。负的 Population−baseline 差表示 Population NLL 较低。"
             "所有模型使用相同 recording/trial/frame order、target occupancy 与 mask；"
             "旧模型 logits 以 float64 依同一 stable Bernoulli 公式归约，没有重训 baseline。", "",
             "| Cell | Type | Scored bins | Population | Constant | LN | CNN | Canonical |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in per_cell:
        lines.append(f"| {row['cell_id']} | {row['type']} | {row['scored_bins']} | " +
                     " | ".join(f"{row[m]:.9f}" for m in MODELS) + " |")
    lines += ["", "## 等 cell 权重汇总", "",
              "| Group | Cells | Population | Constant | LN | CNN | Canonical |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for row in aggregates:
        lines.append(f"| {row['group']} | {row['cells']} | " + " | ".join(f"{row[m]:.9f}" for m in MODELS) + " |")
    lines += ["", "## Population − baseline", "",
              "| Cell / Group | −Constant | −LN | −CNN | −Canonical |", "|---|---:|---:|---:|---:|"]
    for row in [*per_cell, *aggregates]:
        lines.append(f"| {row.get('cell_id', row.get('group'))} | " +
                     " | ".join(f"{row['Population'] - row[m]:+.9f}" for m in MODELS[1:]) + " |")
    lines += ["", "## 训练审计", "",
              "Inner-dev 含 step 0；每步按旧 DevelopmentStop 选 K，max 1000 / patience 200 / min_delta 1e−7。"
              "丢弃 inner 权重后，从相同 constructor 初值、重置的原 sampling seed，在 full train 恰好更新 K 次。"
              "Adam lr=0.03，batch=4，CPU float64 / 2 threads，无 gradient clip。", "",
              "| Cell | Seed | K | Inner stop | Full-train updates | Trainable | Optimizer-listed | Actually-updated scalars |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in audit:
        lines.append("| " + " | ".join(str(row[k]) for k in
                     ("cell_id", "seed", "K", "inner_stop", "refit_updates", "trainable", "optimizer_listed", "actually_updated")) + " |")
    lines += ["", f"总 optimizer updates：{sum(r['inner_stop'] + r['refit_updates'] for r in audit)}；"
              "nonfinite / failed cells：0 / 0。Actually-updated 按与 constructor 初值精确不相等的 raw scalar 计数，"
              "不等同于生理参数已辨识。每步 data NLL / penalty / objective、inner-dev NLL、抽样 indices、"
              "initial/final state hashes 和 per-tensor optimizer step 均保存在 raw artifacts。", "",
              "## 工件与核验范围", "",
              "根目录：`output/real_data/retipath_population_r1/`。", "",
              "- `protocol.md` / `protocol.json` / `source_lock.json` / `source_snapshot/` / `data_lock.json`。",
              "- `preflight.json` 与各 cell 的 `training_data_contract.json`。",
              "- `cells/<cell>/final.pt`、`inner-trajectory.csv`、`refit-trajectory.csv`、`training-summary.json`。",
              "- `CHECKPOINT_LOCK.json`、`EVALUATION_STARTED.json`，记录先锁定后评价的时间与哈希。",
              "- 各 cell 的 `validation-predictions.pt` / `validation-metrics.json`：五模型 logits、真实 targets、counts、masks 与身份。",
              "- `per-cell-metrics.csv/json`、`aggregate-metrics.csv/json`、`comparison.csv/json`、`training-audit.csv`。",
              "- `verification.json`、`training-manifest.json`、`manifest.json`。", "",
              "合同检查覆盖冻结哈希、9-cell/16-recording 映射、train-only 返回对象、精确 masks/bin counts、"
              "physical Q support、recorded port、strict-past history、prefix causality、reset/batch 隔离、"
              "完整无重复 optimizer 参数。原 spike parser 读取整份原始文本，但训练适配器只返回 [0,2400) bins；"
              "完整 parser 对象不返回 trainer、不缓存为外层 targets。训练前没有物化 validation targets 或读取旧 validation predictions。", "",
              "## 结果边界", "",
              "本报告仅是已消费旧 benchmark validation 上的 matched-cohort conditional prediction 比较；"
              "独立新 real test = **NONE**。保留 R0 的 750/751 绝对帧零点未决、未知绝对光强、有限 FOV 与 baseline-reset 限制。"
              "不同 baseline 的历史容量、正则和计算预算不同。本轮没有新增显著性检验、成功阈值、机制分析、RF、人工刺激或 pathway ablation，"
              "不作下一轮研究决策。", ""]
    path = ROOT / "docs/RETIPATH_REAL_R1_RESULTS.md"
    with path.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    return path


@torch.no_grad()
def evaluate():
    cells, lock, source = verify_checkpoint_lock()
    torch.set_num_threads(2)
    write_json(OUT / "EVALUATION_STARTED.json", {"started_utc": utc(),
               "checkpoint_lock_sha256": sha256(OUT / "CHECKPOINT_LOCK.json"),
               "checkpoint_locked_utc": lock["locked_utc"], "single_attempt": True,
               "validation_window": [2400, 3000], "independent_test": "NONE"})
    catalog = {r.recording_id: r for r in public_recordings(RECORDINGS)}
    movie = load_schottdorf_movie_drive(MOVIE, CONFIG)
    rows, audit, validation_checks = [], [], []
    for c in cells:
        cell_dir = OUT / "cells" / c.slug
        data = load_schottdorf_cell(tuple(catalog[r] for r in c.recording_ids), movie, CONFIG)
        split = data.validation
        require(split.cone_drive.shape == (c.trials * 4, 150, 289), "validation input shape differs")
        require(int(split.valid_mask.sum()) == c.validation_bins, "validation bin count differs")
        require(torch.equal(split.valid_mask, (torch.arange(150) >= 30)[None, :, None].expand_as(split.valid_mask)),
                "validation warmup mask differs")
        require(torch.equal(split.spike_events, (split.spike_counts > 0).to(split.spike_events)), "occupancy differs")
        expected_train = json.loads((cell_dir / "training_data_contract.json").read_text(encoding="utf-8"))["full_train"]
        require(split_evidence(data.train)["tensor_sha256"] == expected_train["tensor_sha256"],
                "train-only versus full loader tensor identity differs")
        checkpoint = torch.load(cell_dir / "final.pt", map_location="cpu", weights_only=True)
        require(checkpoint["cell"]["cell_id"] == c.cell_id and tuple(checkpoint["cell"]["recording_ids"]) == c.recording_ids,
                "checkpoint instance differs")
        model = PopulationRetipath(bc_output=BCOutput.LEGACY_PRELU, dtype=DTYPE).eval().requires_grad_(False)
        model.load_state_dict(checkpoint["model"], strict=True)
        require(state_hash(model.state_dict()) == checkpoint["final_state_sha256"], "checkpoint reconstruction differs")
        bundle = TrainingBundle(c, data.train, data.cone_positions_degs, recording_mapping(catalog[c.recording_ids[0]]), ())
        logits = {"Population": forward(model, split.cone_drive, split.spike_events, bundle)}
        p = torch.tensor(checkpoint["constant_probability"], dtype=DTYPE)
        logits["Constant"] = torch.logit(p).expand_as(logits["Population"])
        baseline_hashes = {}
        for name, folder in BASELINES.items():
            path = folder / "cells" / c.slug / "validation-predictions.pt"
            baseline_hashes[name] = {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)}
            saved = torch.load(path, map_location="cpu", weights_only=True)
            verify_saved_predictions(saved, split)
            logits[name] = saved["logits_trained"].to(DTYPE)
        metrics = {m: float(nll(z, split.spike_events, split.valid_mask)) for m, z in logits.items()}
        row = {"cell_id": c.cell_id, "type": c.type, "recordings": ";".join(c.recording_ids),
               "scored_bins": c.validation_bins, **metrics}
        rows.append(row)
        save_tensor_file(cell_dir / "validation-predictions.pt", {
            "cell_id": c.cell_id, "recording_ids": c.recording_ids, "logits": logits,
            "target": split.spike_events, "spike_counts": split.spike_counts, "valid_mask": split.valid_mask,
            "source_image_ids": split.source_image_ids, "trial_indices": split.trial_indices,
            "stimulus_sha256": tensor_hash(split.cone_drive), "checkpoint_sha256": sha256(cell_dir / "final.pt")})
        write_json(cell_dir / "validation-metrics.json", row | {"baseline_sources": baseline_hashes})
        require(state_hash(model.state_dict()) == checkpoint["final_state_sha256"], "evaluation mutated model")
        training = json.loads((cell_dir / "training-summary.json").read_text(encoding="utf-8"))
        a = training["refit"]["parameter_audit"]
        audit.append({"cell_id": c.cell_id, "seed": c.seed, "K": training["inner"]["best_step"],
                      "inner_stop": training["inner"]["stop_step"], "refit_updates": training["refit"]["stop_step"],
                      "trainable": a["trainable"], "optimizer_listed": a["optimizer_listed"],
                      "actually_updated": a["actually_updated"]})
        validation_checks.append({"cell_id": c.cell_id, "saved_baselines_exact_targets_masks_ids_trials": True,
                                  "train_only_full_loader_identity": True, "final_state_unchanged": True,
                                  "validation_scored_bins": c.validation_bins})
        print(f"EVALUATED {c.cell_id}: {metrics}", flush=True)
    aggregates = []
    for group in ("ALL_9", "MC ON", "MC OFF"):
        subset = [r for r in rows if group == "ALL_9" or r["type"] == group]
        aggregates.append({"group": group, "cells": len(subset),
                           **{m: statistics.mean(r[m] for r in subset) for m in MODELS}})
    comparisons = []
    for row in [*rows, *aggregates]:
        for baseline in MODELS[1:]:
            comparisons.append({"cell_or_group": row.get("cell_id", row.get("group")),
                                "baseline": baseline, "Population_NLL": row["Population"],
                                "baseline_NLL": row[baseline], "Population_minus_baseline": row["Population"] - row[baseline]})
    for name, value in (("per-cell-metrics", rows), ("aggregate-metrics", aggregates), ("comparison", comparisons)):
        write_json(OUT / f"{name}.json", {"rows": value})
        write_csv(OUT / f"{name}.csv", value)
    write_csv(OUT / "training-audit.csv", audit)
    verify_sources(source["source_sha256"])
    for path, digest in lock["checkpoints"].items():
        require(sha256(OUT / path) == digest, "final checkpoint mutated during evaluation")
    write_json(OUT / "verification.json", {"status": "VERIFIED", "completed_utc": utc(),
               "frozen_sources_unchanged": True, "final_checkpoints_unchanged": True,
               "all_nine_cells_complete": True, "recordings": 16, "validation_scored_bins": sum(c.validation_bins for c in cells),
               "validation_attempts": 1, "evaluation_after_checkpoint_lock": True, "checks": validation_checks,
               "failed_cells": [], "nonfinite_cells": [], "optimizer_configuration_changed": False,
               "new_baselines": False, "mechanism_RF_artificial_stimulus_ablation_evaluations": 0})
    report = make_report(rows, aggregates, audit)
    write_json(OUT / "report.json", {"path": report.relative_to(ROOT).as_posix(), "sha256": sha256(report)})
    manifest("COMPLETE")
    print("COMPLETE: R1 matched-cohort evaluation and all artifacts written.", flush=True)


if __name__ == "__main__":
    try:
        evaluate()
    except Exception as exc:
        if OUT.exists() and not (OUT / "evaluation_failure.json").exists():
            write_json(OUT / "evaluation_failure.json", {"utc": utc(), "error": str(exc),
                       "traceback": traceback.format_exc(), "automatic_retry": False})
        traceback.print_exc()
        sys.exit(1)
