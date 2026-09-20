from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime
import itertools
import math
from pathlib import Path
import statistics

import numpy as np
import torch

from . import stage_b1 as x


def csv_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def diagnose(run: Path) -> dict:
    trajectory, norms, cosines, clips = [], [], [], []
    for w in x.WORLDS:
        for c in x.ALL:
            for seed in x.b.SEEDS:
                root = x.PARENT if c in x.b.CONDITIONS else run
                dest = root / f"worlds/{w}/fits/{c}_{seed}"
                rows = csv_rows(dest / "trajectory.csv")
                for label, low, high in (("all", 1, 400), ("1-50", 1, 50), ("51-100", 51, 100),
                                         ("101-200", 101, 200), ("201-300", 201, 300), ("301-400", 301, 400), ("351-400", 351, 400)):
                    selected = [r for r in rows if low <= int(r["step"]) <= high]
                    trajectory.append({"world": w, "condition": c, "seed": seed, "window": label,
                                       "base_R_BCE": statistics.mean(float(r["loss_base_R"]) for r in selected),
                                       "hierarchy": statistics.mean(float(r["hierarchy"]) for r in selected),
                                       "preclip_norm": statistics.mean(float(r["preclip_norm"]) for r in selected)})
                clip_steps = [int(r["step"]) for r in rows if float(r["preclip_norm"]) > 1]
                clips.append({"world": w, "condition": c, "seed": seed, "updates": 400, "triggered": len(clip_steps),
                              "last_trigger": max(clip_steps, default=0), "max_preclip_norm": max(float(r["preclip_norm"]) for r in rows)})
                buckets = defaultdict(list)
                for r in csv_rows(dest / "gradient_norms.csv"):
                    buckets[r["dataset"], r["group"]].append(r)
                for (dataset, group), rr in buckets.items():
                    norms.append({"world": w, "condition": c, "seed": seed, "dataset": dataset, "group": group,
                                  "raw_mean": statistics.mean(float(r["raw_norm"]) for r in rr),
                                  "weighted_mean": statistics.mean(float(r["weighted_norm"]) for r in rr), "updates": len(rr)})
                buckets = defaultdict(list)
                for r in csv_rows(dest / "gradient_cosines.csv"):
                    buckets[r["left"], r["right"], r["group"]].append(r)
                for (left, right, group), rr in buckets.items():
                    values = [float(r["cosine"]) for r in rr if r["cosine"]]
                    cosines.append({"world": w, "condition": c, "seed": seed, "left": left, "right": right, "group": group,
                                    "mean": statistics.mean(values) if values else None,
                                    "mean_abs": statistics.mean(abs(v) for v in values) if values else None,
                                    "negative": sum(v < 0 for v in values), "defined": len(values), "total": len(rr)})
    return {"training_trajectory_summary": trajectory, "gradient_norm_summary": norms,
            "gradient_cosine_summary": cosines, "clip_summary": clips}


def verify(run: Path) -> None:
    x.check(run)
    x.parent_inputs(run, verify=True)
    assert x.io.read_json(run / "EVALUATION_COMPLETED.json")["checkpoints"] == 75
    get = lambda name: x.io.read_json(run / (name + ".json"))
    stamps = [get(name)["utc"] for name in ("protocol", "FRESH_TEST_LOCK", "PREFLIGHT", "TRAINING_STARTED", "CHECKPOINT_LOCK", "TEST_CONSUMED", "EVALUATION_COMPLETED")]
    assert all(datetime.fromisoformat(a) < datetime.fromisoformat(b) for a, b in zip(stamps, stamps[1:]))
    lock = get("CHECKPOINT_LOCK")
    assert len(lock["files"]) == 150
    for path, expected in lock["files"].items():
        assert x.io.digest(Path(path)) == expected
    fresh = get("FRESH_TEST_LOCK")
    for name, expected in fresh["files"].items():
        assert x.io.digest(run / name) == expected
    old_seeds, new_seeds, old_phases, new_phases = set(), [], set(), []
    for w in x.WORLDS:
        old = x.io.read_json(x.PARENT / f"worlds/{w}/STIMULUS_MANIFEST.json")
        new = x.io.read_json(run / f"worlds/{w}/FRESH_TEST_MANIFEST.json")
        for record in itertools.chain.from_iterable(old.values()):
            old_phases.add(tuple(record["phases"]))
            old_seeds.update(x.io.seeded_key(record["family"] + suffix) for suffix in (":waveform", ":events"))
        for record, seeds in zip(new["records"], new["numeric_seeds"]):
            assert seeds["family"] == record["family"]
            assert seeds["waveform"] == x.io.seeded_key(record["family"] + ":waveform")
            assert seeds["events"] == x.io.seeded_key(record["family"] + ":events")
            new_seeds.extend((seeds["waveform"], seeds["events"]))
            new_phases.append(tuple(record["phases"]))
    assert len(set(new_seeds)) == 160 and not set(new_seeds).intersection(old_seeds)
    assert len(set(new_phases)) == 80 and not set(new_phases).intersection(old_phases)
    processes = get("PROCESS_COMPLETIONS")
    assert len(processes) == 30 and all(p["returncode"] == 0 for p in processes)
    exposure, initial_gradient_errors = [], []
    for w in x.WORLDS:
        schedules = x.io.read_json(x.PARENT / f"worlds/{w}/SCHEDULES.json")
        for seed in x.b.SEEDS:
            initial_path = x.PARENT / f"worlds/{w}/initial_students/{seed}.pt"
            initial = x.io.load_tensor(initial_path)["model_state"]
            old_first = {}
            for c in ("A", "C"):
                old_first[c] = {(r["dataset"], r["group"]): float(r["raw_norm"]) for r in
                                csv_rows(x.PARENT / f"worlds/{w}/fits/{c}_{seed}/gradient_norms.csv") if r["step"] == "1"}
            for c, streams in x.STREAMS.items():
                dest = run / f"worlds/{w}/fits/{c}_{seed}"
                done = x.io.read_json(dest / "completed.json")
                assert done["steps"] == 400 and done["dataset_batches"] == {n: 400 for n in streams}
                assert done["sequence_exposures"] == len(streams) * 1600
                assert done["RGC_sequence_exposures"] == (1600 if c == "CW" else 4800)
                assert done["initial_sha256"] == x.io.digest(initial_path)
                assert done["fresh_lock_sha256"] == x.io.digest(run / "FRESH_TEST_LOCK.json")
                assert done["schedule_sha256"] == x.io.digest(x.PARENT / f"worlds/{w}/SCHEDULES.json")
                allowed = {str(p.resolve()).lower() for p in [initial_path, *[x.PARENT / f"worlds/{w}/train_data/{n}.pt" for n in ("base_R", "H", "BC")]]}
                assert set(done["loaded_tensor_files"]) == allowed and done["evaluator_reads"] == 0
                for n in streams:
                    assert len(schedules[n]) == 400 and all(len(indices) == 4 for indices in schedules[n])
                cp = x.io.load_tensor(dest / "final.pt")
                assert cp["steps"] == 400 and cp["protocol_sha256"] == x.io.digest(run / "protocol.json")
                assert cp["initial_sha256"] == x.io.digest(initial_path)
                group = cp["optimizer_state"]["param_groups"][0]
                assert group["lr"] == .01 and group["betas"] == (.9, .999) and group["eps"] == 1e-8 and group["weight_decay"] == 0
                assert all(int(s["step"]) == 400 for s in cp["optimizer_state"]["state"].values())
                for name, value in cp["model_state"].items():
                    assert bool(torch.isfinite(value).all())
                    if not name.endswith((".center", ".contrast")):
                        assert torch.equal(value, initial[name]), name
                tr = csv_rows(dest / "trajectory.csv")
                assert [int(r["step"]) for r in tr] == list(range(1, 401))
                for row in tr:
                    objective = sum(x.WEIGHTS[c][n] * float(row["loss_" + n]) for n in streams)
                    assert abs(objective - float(row["weighted_data_loss"])) < 1e-12
                    assert (row["clip_triggered"] == "True") == (float(row["preclip_norm"]) > 1)
                gr = csv_rows(dest / "gradient_norms.csv")
                assert len(gr) == 400 * (len(streams) + 1) * 4
                for r in gr:
                    weight = 1 if r["dataset"] == "hierarchy" else x.WEIGHTS[c][r["dataset"]]
                    assert math.isclose(float(r["weighted_norm"]), weight * float(r["raw_norm"]), rel_tol=2e-7, abs_tol=1e-12)
                errors = []
                for r in gr:
                    if r["step"] == "1":
                        ref = "A" if r["dataset"].startswith("base_R") else "C"
                        errors.append(abs(float(r["raw_norm"]) - old_first[ref][r["dataset"], r["group"]]))
                assert max(errors) == 0
                initial_gradient_errors.append({"world": w, "condition": c, "seed": seed, "max_raw_norm_error": max(errors)})
                co = csv_rows(dest / "gradient_cosines.csv")
                assert len(co) == 400 * math.comb(len(streams), 2) * 4
                assert all(not r["cosine"] or math.isfinite(float(r["cosine"])) for r in co)
                exposure.append({"world": w, "condition": c, "seed": seed, "steps": 400,
                                 "microbatches": sum(done["dataset_batches"].values()), "sequence_exposures": done["sequence_exposures"],
                                 "RGC_sequence_exposures": done["RGC_sequence_exposures"], "evaluator_reads": 0})
    results = get("results")
    observed = {(r["world"], r["condition"], str(r["seed"]), r["sequence"], r["metric"]): r["value"] for r in results["per_sequence"]}
    ambiguous = {(r["world"], r["condition"], str(r["seed"]), r["sequence"], r["metric"]): r["value"] for r in results["ambiguity_per_sequence"]}
    max_error, max_ambiguity = 0., 0.
    for w in x.WORLDS:
        truth = x.io.load_tensor(run / f"worlds/{w}/evaluator_only/fresh_test.pt")
        for c in x.ALL:
            predictions = {seed: x.io.load_tensor(run / f"worlds/{w}/evaluation/{c}_{seed}_raw.pt") for seed in x.b.SEEDS}
            for seed, pred in predictions.items():
                for i, record in enumerate(truth["records"]):
                    for metric, (port, nodes) in x.METRICS.items():
                        a = pred[port][i, 60:].numpy().astype(np.float64)
                        z = truth[port][i, 60:].numpy().astype(np.float64)
                        delta = a - z
                        if nodes is not None:
                            delta = delta[..., nodes]
                        value = float(np.sqrt(np.mean(delta * delta)))
                        max_error = max(max_error, abs(value - observed[w, c, str(seed), record["id"], metric]))
                    ell = pred["logit"][i, 60:].numpy().astype(np.float64)
                    target = truth["logit"][i, 60:].numpy().astype(np.float64)
                    p = truth["probability"][i, 60:].numpy().astype(np.float64)
                    value = float(np.mean(np.logaddexp(0, ell) - p * ell - (np.logaddexp(0, target) - p * target)))
                    max_error = max(max_error, abs(value - observed[w, c, str(seed), record["id"], "rgc_excess_ce"]))
            for left, right in itertools.combinations(x.b.SEEDS, 2):
                for i, record in enumerate(truth["records"]):
                    for metric, (port, nodes) in x.METRICS.items():
                        delta = predictions[left][port][i, 60:].numpy().astype(np.float64) - predictions[right][port][i, 60:].numpy().astype(np.float64)
                        if nodes is not None:
                            delta = delta[..., nodes]
                        value = float(np.sqrt(np.mean(delta * delta)))
                        max_ambiguity = max(max_ambiguity, abs(value - ambiguous[w, c, f"{left}-{right}", record["id"], metric]))
    assert max_error < 1e-12 and max_ambiguity < 1e-12
    assert len(observed) == 75 * 16 * 12 and len(ambiguous) == 75 * 16 * 11
    for prefix in ("", "ambiguity_"):
        sequence = results[prefix + "per_sequence"]
        expected = x.b.aggregate(sequence)
        actual = results["ambiguity_per_seed_pair" if prefix else "per_fit"]
        assert expected == actual
        a, w, o = x.comparisons(actual)
        assert a == results[prefix + "paired_differences"]
        assert w == results[prefix + "teacher_summary"] and o == results[prefix + "descriptive_summary"]
    for name, rows in results.items():
        if isinstance(rows, list):
            saved = csv_rows(run / (name + ".csv"))
            assert len(saved) == len(rows)
            for a, z in zip(rows, saved):
                assert all(str(value) == z[key] for key, value in a.items())
    diagnostics = diagnose(run)
    for name, rows in {"verified_exposures": exposure, "initial_gradient_pairing": initial_gradient_errors, **diagnostics}.items():
        x.io.write_csv(run / (name + ".csv"), rows)
    x.io.write_json(run / "diagnostics.json", diagnostics)
    x.parent_inputs(run, verify=True)
    x.check(run)
    x.io.write_json(run / "VERIFICATION.json", {"utc": x.io.utc(), "status": "VERIFIED", "new_fits": 30, "evaluated_checkpoints": 75,
                    "new_optimizer_updates": 12000, "fresh_sequences": 80, "fresh_lock_before_training": True, "all_checkpoints_before_test_consumption": True,
                    "trainer_evaluator_reads": 0, "initial_raw_gradient_norm_pairing_max_error": 0.,
                    "fresh_waveform_event_seeds_disjoint_from_original": True,
                    "numpy_metric_max_abs_error": max_error, "numpy_ambiguity_max_abs_error": max_ambiguity,
                    "checkpoint_optimizer_schedule_exposure_verified": True, "old_inputs_and_checkpoints_unchanged": True,
                    "raw_metric_rows": len(observed), "ambiguity_rows": len(ambiguous), "csv_json_and_aggregation_verified": True,
                    "verification_source_sha256": x.io.digest(Path(__file__)), "additional_student_forward": 0,
                    "scope": "saved raw-array arithmetic verification only; numeric tolerance is correctness tolerance, not scientific success threshold"})
    print("B1_VERIFIED", max_error, max_ambiguity, flush=True)


def report(run: Path) -> None:
    x.check(run)
    verified = x.io.read_json(run / "VERIFICATION.json")
    assert verified["status"] == "VERIFIED"
    results = x.io.read_json(run / "results.json")
    diagnostic = x.io.read_json(run / "diagnostics.json")
    overall = {(r["metric"], r["comparison"]): r for r in results["descriptive_summary"]}
    lookup = {(r["world"], r["condition"], r["seed"], r["metric"]): r["value"] for r in results["per_fit"]}
    mean = statistics.mean

    def fmt(value):
        return f"{value:.9g}"

    def table(headers, rows):
        return "| " + " | ".join(headers) + " |\n| " + " | ".join("---" for _ in headers) + " |\n" + "".join(
            "| " + " | ".join(str(v) for v in row) + " |\n" for row in rows)

    def averages(metric):
        return [mean(lookup[w, c, seed, metric] for w in x.WORLDS for seed in x.b.SEEDS) for c in x.ALL]

    cw, cx = overall["rgc_excess_ce", "CW-C"], overall["rgc_excess_ce", "CX-A"]
    ca = overall["rgc_excess_ce", "C-A"]
    direct = overall["direct_delta_logit_rmse", "CX-A"]
    cw_text = ("CW 的平均 RGC excess CE 低于 C，与原 C penalty 含有 loss-weighting 因素相容；CW 未增加 RGC exposure，不能把改善归于增加曝光。"
               if cw["difference"] < 0 else "CW 的平均 RGC excess CE 未低于 C；本次结果未提供平均 prediction 改善的 loss-weighting 证据。")
    if cx["difference"] > 0:
        cx_text = "CX 在匹配 A 的 RGC exposure、数据来源和总 loss 系数后，平均 excess CE 仍高于 A，保留 residual tradeoff。不能自动归因于 gradient conflict。"
    elif direct["difference"] < 0:
        cx_text = "CX 的平均 RGC excess CE 不高于 A，且平均 direct-intervention error 低于 A；与匹配 RGC supervision 后多层观测可改善机制预测而不必然牺牲 RGC prediction 相容。方向一致性见逐 teacher/seed 表，不升级为生物结论。"
    else:
        cx_text = "CX 的平均 RGC excess CE 不高于 A，但没有同时保留平均 direct-intervention advantage，不能将两项解释合同合并为支持。"
    p = []
    p.append("# Population RetiPath Stage B.1：RGC supervision targeted follow-up\n\n"
             "状态：**30 个新 fits 与 75 个冻结 checkpoint 的 fresh-test 统一评价已完成；必要核对 VERIFIED；停止于 B.1。**\n\n"
             "本轮是 **post-hoc targeted follow-up，不是 confirmatory**。只检验原 C 的 RGC prediction penalty 与 RGC loss weighting / exposure 的关系。\n\n"
             f"{cw_text}\n\n{cx_text}\n\n"
             f"Fresh test 的 C−A excess CE={fmt(ca['difference'])}；CW−C={fmt(cw['difference'])}；CX−A={fmt(cx['difference'])} nats/RGC-bin。"
             "没有预定义等价 margin，不把小正差值自动写成‘相同’。本实验不能估计各因素的因果贡献比例，不能仅由均值宣布‘主要来自’某一项。\n")
    p.append("## 1. 冻结合同与 fresh test\n\n"
             "Population v0.1、LegacyPReLU、所有 hierarchy/prior、Q、AC routing、conductance、interventions 不变。"
             "复用 Stage B 的五个 teacher、每 teacher 三个 paired initializations、原始训练数据和逐步 schedules；H1 5/25、BC 10/50 固定 sites 不变。"
             "未新增 E/I observation，未修改 γBR/γAR prior，未训练或改写原 A/C/E。\n\n"
             "CW：L_R + 0.4 L_H + 0.4 L_BC + P；CX：(L_R+L_R1+L_R2)/3 + 0.4 L_H + 0.4 L_BC + P。"
             "H/BC 和 P 均为正项。RGC mean BCE、H/BC Gaussian mean loss(sigma0.03)、P 系数1均复用原定义。"
             "400 updates，batch4，Adam lr0.01/betas(0.9,0.999)/eps1e-8/weight_decay0，global clip1；只保留 step400，无选择、early stop、延期或结果后调整。\n\n")
    p.append(table(["条件", "训练状态", "每步 streams", "RGC batches/fit", "全部 batches/fit", "全部 sequence exposures", "RGC loss 系数总和"], [
        ["A", "原 checkpoint", "3 base-RGC", 1200, 1200, 4800, 1], ["C", "原 checkpoint", "R+H+BC", 400, 1200, 4800, "1/3"],
        ["E", "原 checkpoint", "R+2 extra-RGC", 1200, 1200, 4800, 1], ["CW", "新增15 fits", "R+H+BC", 400, 1200, 4800, 1],
        ["CX", "新增15 fits", "R+R1+R2+H+BC", 1200, 2000, 8000, 1]]))
    p.append("\nCX 的三个 RGC streams 逐步复用 A 的 base-pool schedules，没有新增 extra-RGC information。"
             "CW/CX 的 H/BC 使用 C 的原 schedules。CX 与 A 的 RGC exposure/来源/总系数相同，但额外消耗 H/BC microbatches，故不称 total-exposure/compute/information-matched。"
             "新训练合计 12,000 optimizer updates、48,000 microbatches、192,000 sequence exposures。\n\n"
             "Fresh test 在任何新训练前生成并锁定：每 teacher 16 条，尺度0.225/0.45deg各8条，共80条；"
             "仍为32×32、完整2×2deg、300 bins@150Hz、60-bin warmup，刺激频谱和物理合同不变。"
             "随机流 namespace 为 PopulationStageB1Fresh20260920:Wxx:v1，显式 waveform/event seeds 与 records 已保存。"
             "同五个已冻结 teacher，仅 phases/centers/event draws 为新固定随机流；不筛选难度、不重抽，不复用 Stage B 已消费 test。\n\n"
             "Fresh-test truth 位于 evaluator_only；训练 worker 有文件读取白名单与 evaluator-only 拒绝规则。全部30新 checkpoint 与原45 checkpoint 哈希锁定后，"
             "才登记 TEST_CONSUMED 并统一评价75个模型。测试没有参与选择或诊断调参。原 Stage B 数字不与 fresh-test 数字混合。\n")
    p.append("## 2. Primary：RGC excess CE\n\n单位 nats/RGC-bin。先按序列评分，再在每 scale 内等权、最后两 scales 等权；teacher/seed等权。差值为左条件减右条件，负值表示误差更低。\n\n")
    p.append(table(["A", "C", "E", "CW", "CX"], [[fmt(v) for v in averages("rgc_excess_ce")]]))
    p.append(table(["比较", "定位", "均值差", "负差 teacher", "负差 paired seeds"], [[comp, "primary" if comp in ("CW-C", "CX-A") else "事前登记的解释参照",
        fmt(overall["rgc_excess_ce", comp]["difference"]), f"{overall['rgc_excess_ce', comp]['negative_worlds']}/5",
        f"{overall['rgc_excess_ce', comp]['negative_paired_seeds']}/15"] for comp in ("CW-C", "CX-A", "CX-CW", "C-A")]))
    p.append("\n逐 teacher（每行三个 paired seeds 均值）：\n\n")
    p.append(table(["teacher", *x.ALL, "CW−C", "CX−A"], [[w, *[fmt(mean(lookup[w, c, seed, "rgc_excess_ce"] for seed in x.b.SEEDS)) for c in x.ALL],
        *[fmt(mean(lookup[w, left, seed, "rgc_excess_ce"] - lookup[w, right, seed, "rgc_excess_ce"] for seed in x.b.SEEDS)) for left, right in x.PAIRS[:2]]] for w in x.WORLDS]))
    p.append("\n全部15个配对初值：\n\n")
    p.append(table(["teacher / seed", *x.ALL, "CW−C", "CX−A"], [[f"{w}/{seed}", *[fmt(lookup[w, c, seed, "rgc_excess_ce"]) for c in x.ALL],
        *[fmt(lookup[w, left, seed, "rgc_excess_ce"] - lookup[w, right, seed, "rgc_excess_ce"]) for left, right in x.PAIRS[:2]]] for w in x.WORLDS for seed in x.b.SEEDS]))
    p.append("\n## 3. Secondary：各接口分别报告\n\n以下全部来自同一个 fresh test。state/output、drive、干预和 prediction 分开，不合成 mechanism score。"
             "干预 delta-logit=blocked−normal；相同 teacher-normal events 条件化、相同 baseline reset，属于计算通路干预，不等价药理阻断。\n\n")
    p.append(table(["RMSE", *x.ALL, "CW−C", "CX−A", "CX−A负差 teacher / seeds"], [[metric, *[fmt(v) for v in averages(metric)],
        fmt(overall[metric, "CW-C"]["difference"]), fmt(overall[metric, "CX-A"]["difference"]),
        f"{overall[metric, 'CX-A']['negative_worlds']}/5; {overall[metric, 'CX-A']['negative_paired_seeds']}/15"] for metric in x.METRICS]))
    p.append("\nCX−A direct-intervention 与 prediction 的共同事实：\n\n")
    p.append(table(["teacher / seed", "CX−A excess CE", "CX−A direct delta-logit RMSE", "CX−A d_E RMSE"], [[f"{w}/{seed}", *[
        fmt(lookup[w, "CX", seed, metric] - lookup[w, "A", seed, metric]) for metric in ("rgc_excess_ce", "direct_delta_logit_rmse", "d_E_rmse")]] for w in x.WORLDS for seed in x.b.SEEDS]))
    p.append("\n全部 secondary 的逐 teacher/paired seed 原始差值均保存在 teacher_summary.csv / paired_differences.csv；没有省略负方向结果。\n")
    p.append("## 4. Cross-seed ambiguity\n\n每 teacher 三对 seeds 的 trajectory RMSE，沿用同一观察节点分区与 scale 归约。下表是 seed-pair距离，不是真值误差；低分散不证明唯一恢复。\n\n")
    amb = results["ambiguity_per_seed_pair"]
    p.append(table(["seed-pair RMSE", *x.ALL], [[metric, *[fmt(mean(r["value"] for r in amb if r["condition"] == c and r["metric"] == metric)) for c in x.ALL]] for metric in x.METRICS]))
    p.append("\n## 5. 训练诊断：记录，不改变优化器\n\nbase_R 是五条件共享的第一个 base stream。表中为更新前实际 batch BCE 的均值，不是重新评分固定全训练集；横向逐 update 配对成立，窗口之间 batch 改变。\n\n")
    trajectory = diagnostic["training_trajectory_summary"]
    p.append(table(["update 窗口", *x.ALL], [[window, *[fmt(mean(r["base_R_BCE"] for r in trajectory if r["condition"] == c and r["window"] == window)) for c in x.ALL]] for window in ("1-50", "51-100", "101-200", "201-300", "301-400", "351-400")]))
    ng = diagnostic["gradient_norm_summary"]
    p.append("\n全程逐 dataset gradient L2 均值；raw 为原 loss 梯度，weighted 为乘冻结系数后的梯度。state/output/coupling 列均 weighted。norm 之和不是合计梯度 norm。\n\n")
    grad_rows = []
    for c in x.ALL:
        for dataset in sorted({r["dataset"] for r in ng if r["condition"] == c}):
            subset = [r for r in ng if r["condition"] == c and r["dataset"] == dataset]
            grad_rows.append([c + "/" + dataset, fmt(mean(r["raw_mean"] for r in subset if r["group"] == "all")),
                              *[fmt(mean(r["weighted_mean"] for r in subset if r["group"] == g)) for g in ("all", "state", "output", "coupling")]])
    p.append(table(["条件/dataset", "all raw", "all weighted", "state", "output", "coupling"], grad_rows))
    p.append("\nCosine 仅在双方共同 autograd-supported 坐标上计算，缺失共同坐标为N/A。全组均值接近零不排除局部冲突；不据此自动解释 residual tradeoff。下面列新条件，完整组别与A/C/E参照见日志和 gradient_cosine_summary.csv。\n\n")
    cg = diagnostic["gradient_cosine_summary"]
    cosine_rows = []
    for c in x.NEW:
        for left, right in itertools.combinations(x.STREAMS[c], 2):
            rr = [r for r in cg if r["condition"] == c and r["left"] == left and r["right"] == right and r["group"] == "all"]
            cosine_rows.append([c, left + " / " + right, fmt(mean(r["mean"] for r in rr)), fmt(mean(r["mean_abs"] for r in rr)),
                                f"{sum(r['negative'] for r in rr)}/{sum(r['defined'] for r in rr)}"])
    p.append(table(["条件", "pair", "mean cosine", "mean abs cosine", "negative/defined"], cosine_rows))
    clip = diagnostic["clip_summary"]
    p.append("\nGlobal clip 按原 norm > 1 规则统计：\n\n")
    p.append(table(["条件", "触发 / 6000 updates", "最晚触发 step", "最大 preclip norm"], [[c,
        f"{sum(r['triggered'] for r in clip if r['condition'] == c)}/6000", max(r["last_trigger"] for r in clip if r["condition"] == c),
        fmt(max(r["max_preclip_norm"] for r in clip if r["condition"] == c))] for c in x.ALL]))
    p.append("\n## 6. 验证、工件与停止边界\n\n"
             f"必要核对 VERIFIED：30新fits均400步；75 checkpoint统一锁定；所有原始数据/initial/schedules及A/C/E checkpoint哈希未变；"
             f"新worker首步各dataset raw gradient norms与原配对A/C对应来源逐项完全相等，最大差0。trainer evaluator-only读取0。"
             f"已保存raw arrays的独立NumPy复算最大指标差={fmt(verified['numpy_metric_max_abs_error'])}，ambiguity差={fmt(verified['numpy_ambiguity_max_abs_error'])}；"
             "这些是数值正确性核对，不是科学成功阈值。核对没有新增 student forward、训练或checkpoint选择。CSV/JSON、逐序列→scale→teacher聚合一致。\n\n"
             f"独立工件目录：[{run.name}]({run.as_posix()})。\n\n"
             "- protocol.json / PROTOCOL.md、SOURCE_LOCK.json、PARENT_INPUT_LOCK.json：训练前冻结合同与来源。\n"
             "- worlds/*/FRESH_TEST_MANIFEST.json、evaluator_only/fresh_test.pt、FRESH_TEST_LOCK.json：新随机流与封存truth。\n"
             "- worlds/*/fits/{CW,CX}_*/：30个final checkpoints、完整trajectory/gradient/cosine/completed记录。\n"
             "- CHECKPOINT_LOCK.json、TEST_CONSUMED.json：75 checkpoint gate及fresh test消费记录。\n"
             "- worlds/*/evaluation/*_raw.pt：75组fresh预测/干预raw arrays。\n"
             "- per_sequence.csv、per_fit.csv、paired_differences.csv、teacher_summary.csv、descriptive_summary.csv、results.json：全部误差与比较。\n"
             "- ambiguity_*.csv、diagnostics.json、training_trajectory_summary.csv、gradient_*_summary.csv、clip_summary.csv：分散度与训练诊断。\n"
             "- VERIFICATION.json、verified_exposures.csv、initial_gradient_pairing.csv、FILE_MANIFEST.json：验收证据。\n\n"
             "本轮未执行Git、修改旧模型/旧结果、增加condition、调整prior/H/BC权重、延长训练、增加RF或E/I observation。"
             "仅报告冻结条件下的事实与解释合同，不作下一阶段研究决策，不进入Stage C。\n")
    target = x.ROOT / "docs/RETIPATH_POPULATION_STAGE_B1.md"
    with target.open("x", encoding="utf-8") as f:
        f.write("\n".join(p))
    source_copy = run / "source/experiments/retipath_population_v0_1/stage_b1_delivery.py"
    with source_copy.open("xb") as f:
        f.write(Path(__file__).read_bytes())
    files = {p.relative_to(run).as_posix(): {"sha256": x.io.digest(p), "bytes": p.stat().st_size}
             for p in sorted(run.rglob("*")) if p.is_file()}
    x.io.write_json(run / "FILE_MANIFEST.json", {"utc": x.io.utc(), "files": files,
                    "external_report": {"path": str(target), "sha256": x.io.digest(target), "bytes": target.stat().st_size}})
    print("B1_REPORT_WRITTEN", target, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("verify", "report"))
    parser.add_argument("--run", type=Path, default=x.DEFAULT_RUN)
    args = parser.parse_args()
    torch.set_num_threads(1)
    {"verify": verify, "report": report}[args.action](args.run)


if __name__ == "__main__":
    main()
