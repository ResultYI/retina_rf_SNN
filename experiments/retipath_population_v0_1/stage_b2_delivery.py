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

from . import stage_b2 as x


def csv_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def parameters(run: Path) -> None:
    x.check(run)
    assert x.io.read_json(run / "EVALUATION_COMPLETED.json")["checkpoints"] == 60
    for path, expected in x.io.read_json(run / "CHECKPOINT_LOCK.json")["files"].items():
        assert x.io.digest(Path(path)) == expected
    fields = (*x.REMOVED, "a_H")
    template = x.b.model()
    def decode(state, name):
        f = template.fields[name]
        prefix = "fields." + name + "."
        center = state[prefix + "center"].numpy().astype(np.float64)
        family = state[prefix + "family"].numpy()
        raw = center[family].copy()
        if prefix + "contrast" in state:
            raw += state[prefix + "basis"].numpy().astype(np.float64) @ state[prefix + "contrast"].numpy().astype(np.float64)
        physical = f.lo + (f.hi - f.lo) / (1 + np.exp(-raw))
        anchor = state[prefix + "initial_center"].numpy().astype(np.float64)
        reference = f.lo + (f.hi - f.lo) / (1 + np.exp(-anchor[family]))
        return physical, center, anchor, reference
    rms = lambda a: float(np.sqrt(np.mean(a * a)))
    fits, coords, raw_coords, ambiguity, priors = [], [], [], [], []
    for w in x.WORLDS:
        teacher = x.io.load_tensor(x.PARENT / f"worlds/{w}/evaluator_only/teacher.pt")["model_state"]
        for c in x.ALL:
            saved = {}
            for seed in x.b.SEEDS:
                initial = x.io.load_tensor(x.PARENT / f"worlds/{w}/initial_students/{seed}.pt")["model_state"]
                final = x.io.load_tensor(x.condition_root(run, c) / f"worlds/{w}/fits/{c}_{seed}/final.pt")["model_state"]
                saved[seed] = {}
                for name in fields:
                    tv, tc, ta, _ = decode(teacher, name)
                    iv, ic, ia, ref = decode(initial, name)
                    fv, fc, fa, _ = decode(final, name)
                    assert np.array_equal(ia, fa) and np.array_equal(ta, fa)
                    f = template.fields[name]
                    assert np.all(fv > f.lo) and np.all(fv < f.hi)
                    prefix = "fields." + name + "."
                    raw_move = [fc - ic]
                    if prefix + "contrast" in final:
                        raw_move.append((final[prefix + "contrast"].double() - initial[prefix + "contrast"].double()).numpy())
                    values = {"teacher_rmse": rms(fv - tv), "physical_movement_rms": rms(fv - iv),
                              "raw_parameter_movement_rms": rms(np.concatenate(raw_move)),
                              "raw_center_distance_from_constructor": rms(fc - fa),
                              "physical_distance_from_constructor": rms(fv - ref)}
                    for metric, value in values.items():
                        fits.append({"world": w, "condition": c, "seed": seed, "metric": name + "_" + metric, "value": value})
                    for j in range(len(fv)):
                        coords.append({"world": w, "condition": c, "seed": seed, "field": name, "coordinate": j,
                                       "teacher": float(tv[j]), "initial": float(iv[j]), "final": float(fv[j]), "constructor_reference": float(ref[j]),
                                       "fixed_center_prior_in_original": name in x.REMOVED})
                    for j in range(len(fc)):
                        raw_coords.append({"world": w, "condition": c, "seed": seed, "field": name, "center_coordinate": j,
                                           "teacher": float(tc[j]), "initial": float(ic[j]), "final": float(fc[j]), "old_anchor": float(fa[j])})
                    if name in x.REMOVED:
                        priors.append({"world": w, "condition": c, "seed": seed, "field": name,
                                       "counterfactual_old_center_penalty_at_final": .5 * float(np.sum(((fc - fa) / .3) ** 2)),
                                       "center_penalty_active_in_training": c in ("A", "CX")})
                    saved[seed][name] = fv
            for left, right in itertools.combinations(x.b.SEEDS, 2):
                for name in fields:
                    ambiguity.append({"world": w, "condition": c, "seed": f"{left}-{right}", "metric": name + "_parameter_seed_pair_rmse",
                                      "value": rms(saved[left][name] - saved[right][name])})
    paired, world, overall = x.comparisons(fits)
    ap, aw, ao = x.comparisons(ambiguity)
    result = {"parameter_per_fit": fits, "parameter_coordinates": coords, "parameter_raw_centers": raw_coords,
              "parameter_paired_differences": paired, "parameter_teacher_summary": world, "parameter_descriptive_summary": overall,
              "parameter_ambiguity_per_seed_pair": ambiguity, "parameter_ambiguity_paired_differences": ap,
              "parameter_ambiguity_teacher_summary": aw, "parameter_ambiguity_descriptive_summary": ao, "counterfactual_old_priors": priors}
    data = x.io.read_json(run / "results.json")
    lookup = {(r["world"], r["seed"], r["comparison"]): r["difference"] for r in data["paired_differences"] if r["metric"] == "rgc_excess_ce"}
    gap = [{"world": w, "seed": seed, "old_gap_CX_minus_A": lookup[w, seed, "CX-A"],
            "free_gap_CXF_minus_AF": lookup[w, seed, "CXF-AF"],
            "gap_change": lookup[w, seed, "CXF-AF"] - lookup[w, seed, "CX-A"]} for w in x.WORLDS for seed in x.b.SEEDS]
    result["prediction_gap_change"] = gap
    for name, rows in result.items():
        x.io.write_csv(run / (name + ".csv"), rows)
    x.io.write_json(run / "parameter_results.json", {"utc": x.io.utc(), "parameter_decode": "float64 bounded-sigmoid decode of frozen raw checkpoint coordinates; no circuit forward",
                                                    "a_H_reference": "constructor reference, no fixed center prior", **result})
    print("B2_PARAMETER_RESULTS", len(fits), len(coords), flush=True)


def diagnose(run: Path) -> dict:
    trajectory, norms, cosines, clips = [], [], [], []
    for w in x.WORLDS:
        for c in x.ALL:
            for seed in x.b.SEEDS:
                root = x.condition_root(run, c)
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
    assert x.io.read_json(run / "EVALUATION_COMPLETED.json")["checkpoints"] == 60
    get = lambda name: x.io.read_json(run / (name + ".json"))
    stamps = [get(name)["utc"] for name in ("protocol", "FRESH_TEST_LOCK", "PREFLIGHT", "TRAINING_STARTED", "CHECKPOINT_LOCK", "TEST_CONSUMED", "EVALUATION_COMPLETED")]
    assert all(datetime.fromisoformat(a) < datetime.fromisoformat(b) for a, b in zip(stamps, stamps[1:]))
    lock = get("CHECKPOINT_LOCK")
    assert len(lock["files"]) == 120
    for path, expected in lock["files"].items():
        assert x.io.digest(Path(path)) == expected
    fresh = get("FRESH_TEST_LOCK")
    for name, expected in fresh["files"].items():
        assert x.io.digest(run / name) == expected
    old_seeds, new_seeds, old_phases, new_phases = set(), [], set(), []
    for w in x.WORLDS:
        old = x.io.read_json(x.PARENT / f"worlds/{w}/STIMULUS_MANIFEST.json")
        new = x.io.read_json(run / f"worlds/{w}/FRESH_TEST_MANIFEST.json")
        previous = x.io.read_json(x.B1 / f"worlds/{w}/FRESH_TEST_MANIFEST.json")["records"]
        for record in itertools.chain(itertools.chain.from_iterable(old.values()), previous):
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
            for c in ("A", "CX"):
                old_first[c] = {(r["dataset"], r["group"]): float(r["raw_norm"]) for r in
                                csv_rows(x.condition_root(run, c) / f"worlds/{w}/fits/{c}_{seed}/gradient_norms.csv") if r["step"] == "1"}
            for c, streams in x.STREAMS.items():
                dest = run / f"worlds/{w}/fits/{c}_{seed}"
                done = x.io.read_json(dest / "completed.json")
                assert done["steps"] == 400 and done["dataset_batches"] == {n: 400 for n in streams}
                assert done["sequence_exposures"] == len(streams) * 1600
                assert done["RGC_sequence_exposures"] == 4800
                assert done["initial_sha256"] == x.io.digest(initial_path)
                assert done["fresh_lock_sha256"] == x.io.digest(run / "FRESH_TEST_LOCK.json")
                assert done["schedule_sha256"] == x.io.digest(x.PARENT / f"worlds/{w}/SCHEDULES.json")
                allowed = {str(p.resolve()).lower() for p in [initial_path, *[x.PARENT / f"worlds/{w}/train_data/{n}.pt" for n in (("base_R",) if c == "AF" else ("base_R", "H", "BC"))]]}
                assert set(done["loaded_tensor_files"]) == allowed and done["evaluator_reads"] == 0
                for n in streams:
                    assert len(schedules[n]) == 400 and all(len(indices) == 4 for indices in schedules[n])
                cp = x.io.load_tensor(dest / "final.pt")
                assert cp["steps"] == 400 and cp["protocol_sha256"] == x.io.digest(run / "protocol.json")
                assert cp["initial_sha256"] == x.io.digest(initial_path)
                group = cp["optimizer_state"]["param_groups"][0]
                assert group["lr"] == .01 and group["betas"] == (.9, .999) and group["eps"] == 1e-8 and group["weight_decay"] == 0
                expected_params = dict(x.b.model().named_parameters())
                assert len(group["params"]) == len(expected_params)
                assert set(group["params"]) == set(cp["optimizer_state"]["state"])
                assert sum(cp["model_state"][name].numel() for name in expected_params) == 359
                assert all(param.requires_grad for param in expected_params.values())
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
                    if r["step"] == "1" and r["dataset"] != "hierarchy":
                        ref = "A" if c == "AF" else "CX"
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
    assert len(observed) == 60 * 16 * 12 and len(ambiguous) == 60 * 16 * 11
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
    x.parent_inputs(run, verify=True)
    x.check(run)
    parameter_results = get("parameter_results")
    assert len(parameter_results["parameter_per_fit"]) == 60 * 4 * 5
    assert len(parameter_results["parameter_ambiguity_per_seed_pair"]) == 60 * 4
    for name, rows in parameter_results.items():
        if isinstance(rows, list):
            saved = csv_rows(run / (name + ".csv"))
            assert len(saved) == len(rows)
            for a, z in zip(rows, saved):
                assert all(str(value) == z[key] for key, value in a.items())
    for prefix, fit_name in (("parameter_", "parameter_per_fit"), ("parameter_ambiguity_", "parameter_ambiguity_per_seed_pair")):
        pp, pw, po = x.comparisons(parameter_results[fit_name])
        assert pp == parameter_results[prefix + "paired_differences"]
        assert pw == parameter_results[prefix + "teacher_summary"] and po == parameter_results[prefix + "descriptive_summary"]
    coord_errors = []
    buckets = defaultdict(list)
    for r in parameter_results["parameter_coordinates"]:
        buckets[r["world"], r["condition"], r["seed"], r["field"]].append(r)
    p_lookup = {(r["world"], r["condition"], r["seed"], r["metric"]): r["value"] for r in parameter_results["parameter_per_fit"]}
    for (w, c, seed, field), rows in buckets.items():
        for metric, reference in (("teacher_rmse", "teacher"), ("physical_movement_rms", "initial"), ("physical_distance_from_constructor", "constructor_reference")):
            value = math.sqrt(statistics.mean((r["final"] - r[reference]) ** 2 for r in rows))
            coord_errors.append(abs(value - p_lookup[w, c, seed, field + "_" + metric]))
    assert max(coord_errors) < 1e-12
    diagnostics = diagnose(run)
    for name, rows in {"verified_exposures": exposure, "initial_gradient_pairing": initial_gradient_errors, **diagnostics}.items():
        x.io.write_csv(run / (name + ".csv"), rows)
    x.io.write_json(run / "diagnostics.json", diagnostics)
    x.io.write_json(run / "VERIFICATION.json", {"utc": x.io.utc(), "status": "VERIFIED", "new_fits": 30, "evaluated_checkpoints": 60,
                    "new_optimizer_updates": 12000, "fresh_sequences": 80, "fresh_lock_before_training": True, "all_checkpoints_before_test_consumption": True,
                    "trainer_evaluator_reads": 0, "initial_raw_gradient_norm_pairing_max_error": 0.,
                    "fresh_waveform_event_seeds_disjoint_from_B_and_B1": True,
                    "numpy_metric_max_abs_error": max_error, "numpy_ambiguity_max_abs_error": max_ambiguity,
                    "checkpoint_optimizer_schedule_exposure_verified": True, "old_inputs_and_checkpoints_unchanged": True,
                    "raw_metric_rows": len(observed), "ambiguity_rows": len(ambiguous), "csv_json_and_aggregation_verified": True,
                    "parameter_coordinate_rmse_max_abs_error": max(coord_errors), "parameter_csv_json_aggregation_verified": True,
                    "verification_source_sha256": x.io.digest(Path(__file__)), "additional_student_forward": 0,
                    "scope": "saved raw-array arithmetic verification only; numeric tolerance is correctness tolerance, not scientific success threshold"})
    print("B2_VERIFIED", max_error, max_ambiguity, flush=True)


def report(run: Path) -> None:
    x.check(run)
    verified = x.io.read_json(run / "VERIFICATION.json")
    assert verified["status"] == "VERIFIED"
    results = x.io.read_json(run / "results.json")
    pr = x.io.read_json(run / "parameter_results.json")
    diagnostics = x.io.read_json(run / "diagnostics.json")
    mean = statistics.mean
    fmt = lambda v: f"{v:.10g}"
    lookup = {(r["world"], r["condition"], r["seed"], r["metric"]): r["value"] for r in results["per_fit"]}
    overall = {(r["metric"], r["comparison"]): r for r in results["descriptive_summary"]}
    parameter_lookup = {(r["world"], r["condition"], r["seed"], r["metric"]): r["value"] for r in pr["parameter_per_fit"]}
    parameter_overall = {(r["metric"], r["comparison"]): r for r in pr["parameter_descriptive_summary"]}
    pairs = [left + "-" + right for left, right in x.PAIRS]
    def table(headers, rows):
        return "| " + " | ".join(headers) + " |\n| " + " | ".join("---" for _ in headers) + " |\n" + "".join(
            "| " + " | ".join(str(v) for v in row) + " |\n" for row in rows)
    def averages(metric, values=lookup):
        return [mean(values[w, c, seed, metric] for w in x.WORLDS for seed in x.b.SEEDS) for c in x.ALL]
    old_gap = overall["rgc_excess_ce", "CX-A"]["difference"]
    free_gap = overall["rgc_excess_ce", "CXF-AF"]["difference"]
    direct = overall["direct_delta_logit_rmse", "CXF-AF"]
    p = ["# Population RetiPath Stage B.2：downstream center-anchor targeted follow-up\n\n"
         "状态：**30 个新 fits、60 个冻结 checkpoint 的 fresh-test 统一评价及必要核对已完成，VERIFIED；停止于 B.2。**\n\n"
         "本轮是 **post-hoc targeted follow-up，不是 confirmatory**。只检验 CX 的 residual RGC prediction tradeoff 与指定 downstream fixed center anchors 的关系。\n\n"
         f"在同一 B.2 fresh test 上，原 CX−A 的 RGC excess CE gap 为 {fmt(old_gap)}，移除指定锚点后的 CXF−AF 为 {fmt(free_gap)}，"
         f"差中差为 {fmt(free_gap - old_gap)} nats/RGC-bin。CXF−AF direct-block delta-logit RMSE 差为 {fmt(direct['difference'])}，"
         f"负差方向覆盖 {direct['negative_worlds']}/5 teachers、{direct['negative_paired_seeds']}/15 paired seeds。\n\n"
         "解释只限五个 synthetic teacher instances 与冻结训练合同；不是生物 population 推断。不把三项 prior 的联合移除拆解成某一字段的独立因果贡献，不宣称唯一原因。\n"]
    p.append("**本轮结论：fixed downstream center anchors contributed to the previous residual tradeoff.**\n\n"
             "gap在5/5 teachers、15/15 paired seeds中缩小。CXF−AF的excess CE在4/5 teachers、12/15 paired seeds中为负；"
             "W02三个seed仍为正，保留该teacher的residual tradeoff。这里没有等价margin，不能把剩余正差写成已消失。"
             "结论仅说明这些固定锚点有贡献，不是唯一原因，也不分解γBR、γAR、bias各自的贡献。\n\n"
             "prediction改善没有带来所有机制指标的同步改善：相对原CX，CXF的direct-block RMSE由"
             f"{fmt(overall['direct_delta_logit_rmse', 'CXF-CX']['reference'])}升至{fmt(overall['direct_delta_logit_rmse', 'CXF-CX']['left'])}，"
             "但仍低于AF。其d_I与AC-block RMSE均值也高于CX；所有方向均在后表保留。\n\n"
             "参数结果分别为：相对CX，CXF的γBR teacher RMSE从"
             f"{fmt(parameter_overall['gamma_BR_teacher_rmse', 'CXF-CX']['reference'])}变为{fmt(parameter_overall['gamma_BR_teacher_rmse', 'CXF-CX']['left'])}；"
             "γAR从"
             f"{fmt(parameter_overall['gamma_AR_teacher_rmse', 'CXF-CX']['reference'])}变为{fmt(parameter_overall['gamma_AR_teacher_rmse', 'CXF-CX']['left'])}；"
             "RGC bias从"
             f"{fmt(parameter_overall['bias_teacher_rmse', 'CXF-CX']['reference'])}降至{fmt(parameter_overall['bias_teacher_rmse', 'CXF-CX']['left'])}。"
             "γBR/γAR平均恢复未改善，bias恢复改善；不合成coupling score。a_H prior保持不变，其恢复变化另列。\n\n"
             "Anchor-free条件的seed分散增大：CX→CXF的γBR seed-pair RMSE为2.465736796e-5→0.1043891966，"
             "γAR为9.442496904e-6→0.08146344428；direct-block effect的seed-pair RMSE为8.475080769e-5→0.001565697888。"
             "原样报告该分散度，不追加正则或延长训练。\n")
    p.append("## 1. 唯一改动及冻结合同\n\n"
             "只对 AF/CXF 的训练目标移除 gamma_BR、gamma_AR、RGC bias 的 fixed constructor-center quadratic penalty：\n\n"
             "`P_free = sum(field.penalty() for all fields except gamma_BR, gamma_AR, bias)`。\n\n"
             "被排除的三个字段均无 unit contrasts；其原惩罚为 `0.5*sum(((raw_center-initial_center)/0.3)^2)`。"
             "initial_center 是固定 constructor anchor，不是扰动后的 paired student initialization。原 physical anchors 为 γBR=1.5、γAR=0.375、bias=−2.4；"
             "边界分别为 (0.2,8)、(0,8)、(−4,−0.5)，保持不变。三个字段仍可训练，没有冻结或重初始化。\n\n"
             "a_H contrast SD=0.1、H1/BC partial pooling、AC family sharing 以及其他所有 hierarchy 项完全保留。"
             "Population v0.1、LegacyPReLU、Q、routing、conductance、interventions 的源码与 forward 不变。"
             "全部359个 raw parameters仍进入同一Adam。原A/C/E/CW/CX checkpoints与旧结果均保持原字节。\n\n"
             "AF：`(L_R+L_R1+L_R2)/3 + P_free`；CXF：`(L_R+L_R1+L_R2)/3 + 0.4 L_H + 0.4 L_BC + P_free`。"
             "H/BC是正项。RGC mean Bernoulli BCE；H/BC原sigma0.03 Gaussian mean loss；BC state/output一并平均。\n\n"
             "复用Stage B同5 teachers、3 paired initial states、所有train pools和逐步schedules。H1为center/cardinal外侧5/25 nodes，BC为相同位置ON5+OFF5，"
             "只监督h_H、s_B、delta_r_B；其余latent为evaluator-only。未加入AC或E/I observations。\n\n"
             "每fit固定400 updates、batch4、Adam lr0.01/betas(0.9,0.999)/eps1e-8/weight_decay0、global clip1、CPU float32、每process1线程。"
             "每步prior只加一次。无early stop、selection、延长、重跑或结果后调参，只用step400。\n")
    p.append(table(["条件", "checkpoint来源", "每步streams", "RGC batches/fit", "全部batches/fit", "全部sequence exposures"], [
        ["A", "Stage B冻结", "base_R+base_R_1+base_R_2", 1200, 1200, 4800],
        ["CX", "B.1冻结", "3 base_R + H + BC", 1200, 2000, 8000],
        ["AF", "本轮15 fits", "与A完全相同", 1200, 1200, 4800],
        ["CXF", "本轮15 fits", "与CX完全相同", 1200, 2000, 8000]]))
    p.append("\n四组RGC exposure、来源、总loss系数均匹配；CX/CXF额外消费H/BC，不能称总compute或information-matched。"
             "本轮新训练共12,000 updates、48,000 microbatches、192,000 sequence exposures。\n\n"
             "## 2. Fresh-test合同与指标\n\n"
             "训练前生成并锁定80条新序列：每teacher16条，held-out scales 0.225/0.45 deg各8条；32×32 grid、完整2×2deg、300bins@150Hz、60bin warmup。"
             "沿用刺激分布与评价定义，namespace=`PopulationStageB2Fresh20260920:Wxx:v1`；waveform/event numeric seeds、records全部保存，"
             "与Stage B/B.1已消费test及旧训练流无重复，不按响应筛选或重抽。没有读取旧test payload来评分。\n\n"
             "trainer有tensor白名单并拒绝evaluator_only访问。全部30新final与原A/CX共30个final锁定后才登记TEST_CONSUMED，统一评价60个checkpoint。"
             "此处trainer指训练worker；协调程序对旧teacher checkpoint只做SHA256完整性读取，不反序列化或向worker提供truth。"
             "参数truth只在评价阶段解码；fresh-test payload在生成后保持封存至统一评价，新test不参与优化或选择。\n\n"
             "误差先逐序列算，scale内等权，再两scales等权；teacher和seed等权。RGC excess CE单位nats/RGC-bin，以teacher概率计算。"
             "intervention delta-logit=blocked−normal，沿用相同teacher-normal events、baseline reset、strictly-past history；是计算通路干预。"
             "不新增RF、显著性检验、等价margin、成功阈值或生物population推断。负差表示误差更低。\n\n"
             "## 3. Primary：RGC excess CE\n")
    p.append(table(list(x.ALL), [[fmt(v) for v in averages("rgc_excess_ce")]]))
    p.append(table(["比较", "定位", "均值差", "负差teacher", "负差paired seeds"], [[comp, "primary" if i < 3 else "冻结的解释参照",
        fmt(overall["rgc_excess_ce", comp]["difference"]), f"{overall['rgc_excess_ce', comp]['negative_worlds']}/5",
        f"{overall['rgc_excess_ce', comp]['negative_paired_seeds']}/15"] for i, comp in enumerate(pairs)]))
    p.append("\n每teacher三seed均值：\n")
    p.append(table(["teacher", *x.ALL, *pairs, "gap change"], [[w,
        *[fmt(mean(lookup[w, c, s, "rgc_excess_ce"] for s in x.b.SEEDS)) for c in x.ALL],
        *[fmt(mean(lookup[w, a, s, "rgc_excess_ce"] - lookup[w, z, s, "rgc_excess_ce"] for s in x.b.SEEDS)) for a, z in x.PAIRS],
        fmt(mean(r["gap_change"] for r in pr["prediction_gap_change"] if r["world"] == w))] for w in x.WORLDS]))
    p.append("\n全部15 paired initializations：\n")
    p.append(table(["teacher/seed", *x.ALL, *pairs, "gap change"], [[f"{w}/{s}",
        *[fmt(lookup[w, c, s, "rgc_excess_ce"]) for c in x.ALL],
        *[fmt(lookup[w, a, s, "rgc_excess_ce"] - lookup[w, z, s, "rgc_excess_ce"]) for a, z in x.PAIRS],
        fmt(next(r["gap_change"] for r in pr["prediction_gap_change"] if r["world"] == w and r["seed"] == s))] for w in x.WORLDS for s in x.b.SEEDS]))
    p.append("\n## 4. Secondary：接口分别报告\n\nstate、output、drive、intervention与prediction分别统计，不合成mechanism score。全部来自同一新fresh test。\n")
    p.append(table(["RMSE", *x.ALL, *pairs[:3]], [[m, *[fmt(v) for v in averages(m)],
        *[fmt(overall[m, comp]["difference"]) for comp in pairs[:3]]] for m in x.METRICS]))
    p.append(table(["RMSE", "比较", "负差teacher", "负差paired seeds"], [[m, comp,
        f"{overall[m, comp]['negative_worlds']}/5", f"{overall[m, comp]['negative_paired_seeds']}/15"] for m in x.METRICS for comp in pairs[:3]]))
    p.append("\nDirect advantage与prediction gap的逐teacher事实：\n")
    p.append(table(["teacher", "CXF−AF excess CE", "CXF−AF direct-block RMSE", "CXF−AF d_E RMSE", "CXF−AF AC-block RMSE", "CXF−AF d_I RMSE"],
        [[w, *[fmt(mean(lookup[w, "CXF", s, m] - lookup[w, "AF", s, m] for s in x.b.SEEDS)) for m in
         ("rgc_excess_ce", "direct_delta_logit_rmse", "d_E_rmse", "AC_delta_logit_rmse", "d_I_rmse")]] for w in x.WORLDS]))
    p.append("\n全部secondary逐teacher/seed误差及差值均在per_fit.csv、paired_differences.csv、teacher_summary.csv，无筛选。\n\n"
             "## 5. 参数恢复：各字段独立\n\n"
             "由原始checkpoint的raw center/contrast在float64中按冻结有界sigmoid解码；不运行额外回路forward。"
             "teacher RMSE先在字段的physical unit coordinates内计算，再对seed/teacher等权。γBR为4坐标，γAR为16坐标，bias为2坐标，a_H为25坐标。"
             "这些是模型归一化参数，不能当生理单位或合成coupling score。\n")
    fields = (*x.REMOVED, "a_H")
    p.append(table(["field teacher RMSE", *x.ALL, *pairs[:3]], [[f, *[fmt(v) for v in averages(f + "_teacher_rmse", parameter_lookup)],
        *[fmt(parameter_overall[f + "_teacher_rmse", comp]["difference"]) for comp in pairs[:3]]] for f in fields]))
    p.append(table(["field", "比较", "负差teacher", "负差paired seeds"], [[f, comp,
        f"{parameter_overall[f + '_teacher_rmse', comp]['negative_worlds']}/5",
        f"{parameter_overall[f + '_teacher_rmse', comp]['negative_paired_seeds']}/15"] for f in fields for comp in pairs[:3]]))
    p.append("\n逐teacher参数恢复（每格三seed均值）：\n")
    p.append(table(["teacher/field", *x.ALL], [[w + "/" + f,
        *[fmt(mean(parameter_lookup[w, c, s, f + "_teacher_rmse"] for s in x.b.SEEDS)) for c in x.ALL]] for w in x.WORLDS for f in fields]))
    p.append("\nInitial→final movement和final距旧anchor分别报告。raw movement含该字段所有trainable center/contrast坐标；"
             "raw center distance只含center。physical distance以constructor reference为基准；**a_H的constructor reference没有fixed center penalty**，其contrast partial pooling从未移除。\n")
    p.append(table(["field/统计", *x.ALL], [[f + "/" + metric,
        *[fmt(v) for v in averages(f + "_" + metric, parameter_lookup)]] for f in fields for metric in
        ("physical_movement_rms", "raw_parameter_movement_rms", "raw_center_distance_from_constructor", "physical_distance_from_constructor")]))
    p.append("\nparameter_coordinates.csv保存每个teacher/condition/seed/coordinate的teacher、initial、final、constructor reference；"
             "parameter_raw_centers.csv保存raw center。counterfactual_old_priors.csv仅说明在新final点若重新施加旧prior会有多少惩罚，未加入训练或重评目标。\n\n"
             "## 6. Cross-seed ambiguity\n\n同teacher的3对seed间RMSE，低分散不等于真值恢复或唯一可辨识。保持任何增加，不追加正则。\n")
    p.append(table(["trajectory seed-pair RMSE", *x.ALL], [[m, *[fmt(mean(r["value"] for r in results["ambiguity_per_seed_pair"] if r["condition"] == c and r["metric"] == m)) for c in x.ALL]] for m in x.METRICS]))
    p.append(table(["parameter seed-pair RMSE", *x.ALL], [[f, *[fmt(mean(r["value"] for r in pr["parameter_ambiguity_per_seed_pair"] if r["condition"] == c and r["metric"] == f + "_parameter_seed_pair_rmse")) for c in x.ALL]] for f in fields]))
    p.append("\n两类ambiguity的逐teacher、逐seed-pair数据和配对比较分别保存在ambiguity_*与parameter_ambiguity_* CSV/JSON。\n\n"
             "## 7. 训练诊断与完整性\n\n诊断仅记录，不改变optimizer。base_R BCE来自实际更新前batch；窗口内取平均，横向逐update配对，窗口间batch变化。\n")
    p.append(table(["update窗口", *x.ALL], [[window, *[fmt(mean(r["base_R_BCE"] for r in diagnostics["training_trajectory_summary"] if r["condition"] == c and r["window"] == window)) for c in x.ALL]]
        for window in ("1-50", "51-100", "101-200", "201-300", "301-400", "351-400")]))
    clip = diagnostics["clip_summary"]
    p.append(table(["条件", "clip触发/6000", "最后触发step", "最大preclip norm"], [[c,
        f"{sum(r['triggered'] for r in clip if r['condition'] == c)}/6000", max(r["last_trigger"] for r in clip if r["condition"] == c),
        fmt(max(r["max_preclip_norm"] for r in clip if r["condition"] == c))] for c in x.ALL]))
    p.append("\n每update的raw/weighted gradient norm、state/output/coupling组norm、共同autograd-supported坐标cosine与global clip均完整保存。"
             "零均值cosine不能排除局部冲突，本轮不自动用gradient conflict解释结果。\n\n"
             f"必要验证VERIFIED：30fits均400步、359参数保持trainable/optimizer-listed；新旧配对首步dataset raw gradient norms最大差0；"
             f"预检中除22个指定center坐标外prior梯度差0，a_H contract不变。所有新训练worker evaluator-only读取0。"
             f"80条fresh序列先锁定，60个checkpoint锁定后才消费。独立NumPy复算已保存raw arrays的最大指标差={fmt(verified['numpy_metric_max_abs_error'])}，"
             f"ambiguity差={fmt(verified['numpy_ambiguity_max_abs_error'])}；参数坐标复算差={fmt(verified['parameter_coordinate_rmse_max_abs_error'])}。"
             "数值容差只用于实现核对，不构成科学成功阈值。验证没有额外student forward或optimizer update。\n\n"
             "## 8. 工件、源码入口与停止边界\n\n"
             f"工件目录：[{run.name}]({run.as_posix()})。\n\n"
             "- protocol.json / PROTOCOL.md、SOURCE_LOCK.json、PARENT_INPUT_LOCK.json、B1_INPUT_LOCK.json：合同和旧工件哈希。\n"
             "- FRESH_TEST_LOCK.json、worlds/*/FRESH_TEST_MANIFEST.json、evaluator_only/fresh_test.pt：新测试随机流与truth。\n"
             "- worlds/*/fits/{AF,CXF}_*/：30个final.pt及全部训练轨迹/梯度/完成记录。\n"
             "- CHECKPOINT_LOCK.json、TEST_CONSUMED.json、worlds/*/evaluation/*_raw.pt：60个checkpoint统一评价证据。\n"
             "- results.json、per_sequence/per_fit/paired_differences/teacher_summary/descriptive_summary.csv：主次指标完整数据。\n"
             "- parameter_results.json、parameter_*.csv、prediction_gap_change.csv：参数恢复、movement、anchor距离、分散度及gap变化。\n"
             "- ambiguity_*.csv、diagnostics.json、gradient_*_summary.csv、clip_summary.csv：分散度及诊断。\n"
             "- PREFLIGHT.json、VERIFICATION.json、verified_exposures.csv、initial_gradient_pairing.csv、FILE_MANIFEST.json：合同验收。\n\n"
             f"新执行入口：[stage_b2.py]({(x.ROOT / 'experiments/retipath_population_v0_1/stage_b2.py').as_posix()})，"
             f"参数/数值核对与报告入口：[stage_b2_delivery.py]({Path(__file__).as_posix()})。"
             "训练worker源码与B.1仅prior调用不同；原circuit.py、stage_b.py、stage_b1.py哈希保持。\n\n"
             "本轮未执行Git，未覆盖旧checkpoint/结果，未更改a_H prior、H/BC权重、训练预算或架构。"
             "未新增E/I观测、RF、机制、Softplus、BC coupling、AC→BC或outer nonlinearity。"
             "完成后停止，不追加条件/正则/重训，不进入Stage C，不作下一阶段研究决策。\n")
    target = x.ROOT / "docs/RETIPATH_POPULATION_STAGE_B2.md"
    with target.open("x", encoding="utf-8") as f:
        f.write("\n\n".join(p))
    source_copy = run / "source/experiments/retipath_population_v0_1/stage_b2_delivery.py"
    with source_copy.open("xb") as f:
        f.write(Path(__file__).read_bytes())
    files = {path.relative_to(run).as_posix(): {"sha256": x.io.digest(path), "bytes": path.stat().st_size}
             for path in sorted(run.rglob("*")) if path.is_file()}
    x.io.write_json(run / "FILE_MANIFEST.json", {"utc": x.io.utc(), "files": files,
                    "external_report": {"path": str(target), "sha256": x.io.digest(target), "bytes": target.stat().st_size}})
    print("B2_REPORT_WRITTEN", target, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("parameters", "verify", "report"))
    parser.add_argument("--run", type=Path, default=x.DEFAULT_RUN)
    args = parser.parse_args()
    torch.set_num_threads(1)
    {"parameters": parameters, "verify": verify, "report": report}[args.action](args.run)


if __name__ == "__main__":
    main()

