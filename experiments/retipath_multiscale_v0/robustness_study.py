from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import csv
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys

import numpy as np
import torch

from . import synthetic_study as g2


BASE_RUN = g2.ROOT / "output/experiments/retipath_g2_multiscale_20260918"
RUNNER_SHA256 = "3993fa9f0bfe53bb3c295ce7e8ac5b818f4ed0daa063defefb1d9b37ab2bb304"
COMPARISONS = ("C-B", "D-C")
PRIMARY = "primary_logit_intervention_rmse"
METRICS = (PRIMARY, "excess_ce", "h_rmse", "s_B_rmse", "o_B_rmse", "d_E_rmse", "relative_logit_intervention_error")
TABLES = ("per_sequence_metrics", "per_fit_metrics", "development_metrics", "paired_comparisons", "ambiguity", "rf_metrics")


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def describe(values: list[float]) -> dict:
    return {"mean": statistics.mean(values), "min": min(values), "max": max(values), "values": values}


def summarize_world(per_fit: list[dict], paired: list[dict]) -> dict:
    result = {"status": "COMPLETED_FIXED_BUDGET", "scientific_verdict": "NOT_ASSIGNED", "conditions": {}, "paired": {}}
    for condition in "ABCDE":
        selected = [r for r in per_fit if r["condition"] == condition and r["stratum"] == "heldout_scales" and r["split"] == "test"]
        result["conditions"][condition] = {metric: describe([r[metric] for r in selected]) for metric in METRICS}
    for name in COMPARISONS:
        values = [r["difference"] for r in paired if r["comparison"] == name and r["metric"] == PRIMARY]
        result["paired"][name] = {PRIMARY: {**describe(values), "negative_count": sum(v < 0 for v in values)}}
    return result


def make_world(world_id: str, base: dict) -> dict:
    draw = torch.randn(len(g2.PARAMETERS), dtype=torch.float64, generator=g2.generator(f"G3:{world_id}:teacher"))
    physical, raw, perturbation = {}, {}, {}
    for index, (name, (lo, hi, _, _)) in enumerate(g2.PARAMETERS.items()):
        center = base["teacher"]["physical_parameters"][name]
        initial = math.atanh(center / 4) if name.startswith("delta_") else math.log((center - lo) / (hi - center))
        perturbation[name] = 0.5 * float(draw[index])
        raw[name] = initial + perturbation[name]
        physical[name] = 4 * math.tanh(raw[name]) if name.startswith("delta_") else lo + (hi - lo) / (1 + math.exp(-raw[name]))
        assert lo < physical[name] < hi
    return {"id": world_id, "rng_namespace": f"G3:{world_id}", "teacher_raw_coordinates": raw,
            "teacher_raw_perturbation": perturbation, "teacher_physical_parameters": physical,
            "teacher_draw_seed": g2.seeded_key(f"G3:{world_id}:teacher")}


def check(run: Path) -> dict:
    protocol = g2.read_json(run / "protocol.json")
    lock = g2.read_json(run / "SOURCE_LOCK.json")
    assert g2.digest(run / "protocol.json") == lock["protocol_sha256"]
    for rel, expected in lock["sources"].items():
        assert g2.digest(g2.ROOT / rel) == expected, rel
    for world in protocol["worlds"]:
        directory = run / "worlds" / world["id"]
        g2.check_sources(directory)
        current = g2.read_json(directory / "SOURCE_LOCK.json")
        assert g2.digest(directory / "SOURCE_LOCK.json") == lock["world_source_locks"][world["id"]]
        for filename, key in (("STIMULUS_MANIFEST.json", "manifest_sha256"), ("SCHEDULES.json", "schedules_sha256")):
            assert g2.digest(directory / filename) == current[key]
    return protocol


def freeze(run: Path, count: int) -> None:
    assert count in (3, 5)
    base = g2.read_json(BASE_RUN / "protocol.json")
    runner = Path(g2.__file__).resolve()
    assert g2.digest(runner) == RUNNER_SHA256
    sources = {**base["source_hashes"], str(runner.relative_to(g2.ROOT)).replace("\\", "/"): RUNNER_SHA256,
               str(Path(__file__).resolve().relative_to(g2.ROOT)).replace("\\", "/"): g2.digest(Path(__file__)),
               str((BASE_RUN / "protocol.json").relative_to(g2.ROOT)).replace("\\", "/"): g2.digest(BASE_RUN / "protocol.json"),
               str((BASE_RUN / "FILE_MANIFEST.json").relative_to(g2.ROOT)).replace("\\", "/"): g2.digest(BASE_RUN / "FILE_MANIFEST.json")}
    for rel, expected in sources.items():
        assert g2.digest(g2.ROOT / rel) == expected
    worlds = [make_world(f"W{i:02}", base) for i in range(1, count + 1)]
    protocol = {
        "schema": "retipath_g3_synthetic_robustness_v1", "date": "2026-09-18", "worlds": worlds,
        "frozen_g2_protocol": base, "student_seeds": base["student_seeds"], "fits_per_world": 15,
        "total_fits": count * 15, "total_optimizer_steps": count * 2700, "total_microbatches": count * 7200,
        "teacher_generation": "Independently add N(0,0.5^2) to every G2 teacher raw coordinate, then use unchanged G1 sigmoid/tanh bounds. Fixed engineering sampling distribution, not a biological prior. No rejection, ranking, stratification, retuning or redraw.",
        "independence": "Each world has a unique SHA-derived teacher RNG and G3:world family namespace. Stimulus phases/centers, Gaussian noise and event uniforms are independent streams. Paired narrow/multi common random numbers remain within world. No G2 world is included in the G3 aggregate.",
        "initialization": "Reuse G2's same three student seeds and initial tensors in every world; pair A-E within world. Teacher/data worlds are independent draws conditional on these shared starts, not independent biological samples.",
        "schedule": "G2 ordered batch indices, layer slots, coefficients and 180-step parameter-group schedule are exactly unchanged. Only waveform family namespace and teacher parameters vary.",
        "primary_comparisons": list(COMPARISONS), "primary_metric": base["evaluation"]["primary"],
        "primary_aggregation": "For each world, calculate paired-seed C-B and D-C differences, then equal mean over its three seeds. Report every world, world-mean signs and within-world seed signs. Equal-weight descriptive world summaries only; no p-values, confidence intervals, biological population inference or seed pooling as independent worlds.",
        "direction": "Negative means lower primary error in C vs B or D vs C; exact zero is tie. No result-dependent tolerance or success threshold.",
        "D_E_reporting": list(METRICS), "RF": "Secondary only; unmodified G2 rf_matrix/rf_profiles/rf_metrics and evaluation inputs rule. Anchor is the first common-scale test family in each independently drawn world.",
        "evaluation_order": "All final checkpoints in all worlds must be locked before any student heldout evaluation. One final evaluation per world, followed only by labelled verification replay. No result-dependent rerun.",
        "runtime": {"python": sys.version, "torch": torch.__version__, "platform": platform.platform(),
                    "device": "cpu", "dtype": "float32", "reduction_dtype": "float64", "threads_per_process": 1,
                    "deterministic_algorithms": True, "maximum_training_processes": 6},
        "allowed_G2_differences": ["teacher parameter sets", "world-specific stimulus/noise/event draws", "two primary comparison summaries and world aggregation"],
        "unchanged": ["G1/G2 architecture", "conditions A-E", "physical scale split", "loss", "optimizer and initialization distribution", "180-step schedule and exposure", "intervention metric", "RF evaluator"],
        "source_hashes": sources,
    }
    run.mkdir(parents=True, exist_ok=False)
    g2.write_json(run / "protocol.json", protocol)
    all_families, all_waveforms = set(), set()
    world_locks, preflight = {}, {}
    base_schedule = g2.read_json(BASE_RUN / "SCHEDULES.json")
    for world in worlds:
        directory = run / "worlds" / world["id"]
        namespace = world["rng_namespace"]
        manifest = {
            "multi": g2.make_records(namespace + ":base", [0.15, 0.3, 0.6], 24, "train"),
            "extra": g2.make_records(namespace + ":extra", [0.15, 0.3, 0.6], 40, "train"),
            "development": g2.make_records(namespace + ":base", [0.15, 0.3, 0.6], 8, "development"),
            "test": g2.make_records(namespace + ":base", [0.15, 0.225, 0.3, 0.45, 0.6], 8, "test"),
        }
        manifest["narrow"] = [{**r, "id": "narrow:" + r["id"], "scale": 0.3} for r in manifest["multi"]]
        families = {r["family"] for records in manifest.values() for r in records}
        waveforms = {(r["scale"], tuple(r["center"]), tuple(r["phases"])) for records in manifest.values() for r in records}
        assert not families.intersection(all_families) and not waveforms.intersection(all_waveforms)
        all_families.update(families)
        all_waveforms.update(waveforms)
        schedule = g2.build_schedules(manifest)
        assert schedule == base_schedule
        preflight[world["id"]] = g2.validate_schedules(manifest, schedule)
        local = copy.deepcopy(base)
        local["teacher"]["physical_parameters"] = world["teacher_physical_parameters"]
        local["evaluation"]["paired_comparisons"] = list(COMPARISONS)
        local["g3_world"] = world
        for key in ("conditions", "student_seeds", "stimulus", "data", "optimization", "rf"):
            assert local[key] == base[key]
        g2.write_json(directory / "protocol.json", local)
        g2.write_json(directory / "STIMULUS_MANIFEST.json", manifest)
        g2.write_json(directory / "SCHEDULES.json", schedule)
        g2.write_json(directory / "SOURCE_LOCK.json", {"utc": g2.utc(), "sources": sources,
                      "protocol_sha256": g2.digest(directory / "protocol.json"),
                      "manifest_sha256": g2.digest(directory / "STIMULUS_MANIFEST.json"),
                      "schedules_sha256": g2.digest(directory / "SCHEDULES.json")})
        world_locks[world["id"]] = g2.digest(directory / "SOURCE_LOCK.json")
    g2.write_json(run / "SOURCE_LOCK.json", {"utc": g2.utc(), "sources": sources,
                  "protocol_sha256": g2.digest(run / "protocol.json"), "world_source_locks": world_locks})
    for rel in sources:
        destination = run / "source" / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write((g2.ROOT / rel).read_bytes())
    g2.write_json(run / "PREFLIGHT.json", {"status": "VERIFIED", "utc": g2.utc(), "targets_created": False,
                  "cross_world_family_and_waveform_disjoint": True, "G2_schedules_exact": True, "world_budgets": preflight})
    print("ALL_WORLDS_FROZEN", count, "fits", count * 15, flush=True)


def generate_all(run: Path) -> None:
    protocol = check(run)
    g2.write_json(run / "GENERATION_STARTED.json", {"utc": g2.utc(), "protocol_sha256": g2.digest(run / "protocol.json")})
    initial_reference = None
    data_hashes = []
    for world in protocol["worlds"]:
        directory = run / "worlds" / world["id"]
        g2.generate(directory)
        created = g2.read_json(directory / "DATA_CREATED.json")
        if initial_reference is None:
            initial_reference = created["initial_student_sha256"]
        assert created["initial_student_sha256"] == initial_reference
        expected_initial = g2.read_json(BASE_RUN / "DATA_CREATED.json")["initial_student_sha256"]
        assert created["initial_student_sha256"] == expected_initial
        assert max(created["sampler_forward_max_abs_errors"].values()) < 2e-7
        data_hashes.append(created["files"])
        print("WORLD_DATA_COMPLETE", world["id"], flush=True)
    for key in data_hashes[0]:
        assert len({files[key] for files in data_hashes}) == len(data_hashes)
    g2.write_json(run / "GENERATION_COMPLETED.json", {"utc": g2.utc(), "worlds": len(protocol["worlds"]),
                  "G2_initial_student_hashes_exact": initial_reference, "distinct_world_data_hashes": True})


def train_worker(run: Path, world_id: str, seed: int) -> None:
    protocol = check(run)
    assert world_id in {w["id"] for w in protocol["worlds"]} and seed in protocol["student_seeds"]
    directory = run / "worlds" / world_id
    for condition in "ABCDE":
        g2.train_fit(directory, condition, seed)


def train_all(run: Path) -> None:
    protocol = check(run)
    assert (run / "GENERATION_COMPLETED.json").exists()
    tasks = [(w["id"], seed) for w in protocol["worlds"] for seed in protocol["student_seeds"]]
    g2.write_json(run / "TRAINING_STARTED.json", {"utc": g2.utc(), "tasks": tasks, "attempts_per_task": 1})
    logs = run / "logs"
    logs.mkdir(exist_ok=False)

    def launch(world_id: str, seed: int) -> dict:
        with (logs / f"{world_id}_{seed}.log").open("x", encoding="utf-8") as output:
            process = subprocess.run([sys.executable, "-B", "-m", "experiments.retipath_multiscale_v0.robustness_study",
                                      "train-worker", "--run", str(run), "--world", world_id, "--seed", str(seed)],
                                     cwd=g2.ROOT, stdout=output, stderr=subprocess.STDOUT, check=False)
        return {"world": world_id, "seed": seed, "exit_code": process.returncode}

    outcomes = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(launch, world_id, seed) for world_id, seed in tasks]
        for future in as_completed(futures):
            outcome = future.result()
            outcomes.append(outcome)
            print("TRAINING_TASK_COMPLETE", outcome, flush=True)
    g2.write_json(run / "TRAINING_OUTCOMES.json", {"utc": g2.utc(), "tasks": outcomes})
    assert all(r["exit_code"] == 0 for r in outcomes), "Failed task retained; no automatic retry"
    checkpoints = {}
    for world in protocol["worlds"]:
        for condition in "ABCDE":
            for seed in protocol["student_seeds"]:
                path = run / "worlds" / world["id"] / f"fits/{condition}_{seed}/final.pt"
                complete = g2.read_json(path.parent / "completed.json")
                assert complete["steps"] == 180 and sum(complete["microbatches"].values()) == 480
                assert complete["checkpoint_sha256"] == g2.digest(path)
                checkpoints[str(path.relative_to(run)).replace("\\", "/")] = g2.digest(path)
    assert len(checkpoints) == protocol["total_fits"]
    g2.write_json(run / "ALL_FINAL_CHECKPOINTS_LOCK.json", {"utc": g2.utc(), "checkpoints": checkpoints})


def independent_verify(directory: Path) -> None:
    created = g2.read_json(directory / "DATA_CREATED.json")
    for rel, expected in created["files"].items():
        assert g2.digest(directory / rel) == expected
    assert g2.digest(directory / "evaluator_only/teacher.pt") == created["teacher_sha256"]
    truth = g2.load_tensor(directory / "evaluator_only/test.pt")
    expected = {(r["condition"], int(r["seed"]), r["sequence_id"]): r for r in read_rows(directory / "per_sequence_metrics.csv")}
    errors = {name: 0.0 for name in (PRIMARY, "rgc_nll", "excess_ce", "h_rmse", "s_B_rmse", "o_B_rmse")}
    for checkpoint in sorted((directory / "fits").glob("*/final.pt")):
        saved = g2.load_tensor(checkpoint)
        condition, seed = saved["condition"], saved["seed"]
        steps = sorted(int(v["step"]) for v in saved["optimizer_state"]["state"].values())
        assert steps == ([120] * 7 + [150] * 5 if condition == "D" else [180] * 12)
        group = saved["optimizer_state"]["param_groups"][0]
        assert (group["lr"], group["betas"], group["eps"], group["weight_decay"]) == (0.01, (0.9, 0.999), 1e-8, 0)
        complete = g2.read_json(checkpoint.parent / "completed.json")
        assert all(value > 0 for value in complete["raw_parameter_absolute_changes"].values())
        pred = g2.load_tensor(directory / f"evaluation/{condition}_{seed}_test.pt")
        for i, record in enumerate(truth["records"]):
            p = truth["p"][i, 60:].numpy().astype(np.float64)
            ell = pred["ell"][i, 60:].numpy().astype(np.float64)
            event = truth["events"][i, 60:].numpy().astype(np.float64)
            delta = ((pred["block_ell"][i, 60:].numpy() - pred["ell"][i, 60:].numpy()) -
                     (truth["block_ell"][i, 60:].numpy() - truth["ell"][i, 60:].numpy())).astype(np.float64)
            values = {PRIMARY: float(np.sqrt(np.mean(delta ** 2))),
                      "rgc_nll": float(np.mean(np.logaddexp(0, ell) - event * ell)),
                      "excess_ce": float(np.mean(np.logaddexp(0, ell) - p * ell + p * np.log(p) + (1 - p) * np.log1p(-p)))}
            for name in ("h", "s_B", "o_B"):
                diff = (pred[name][i, 60:].numpy() - truth[name][i, 60:].numpy()).astype(np.float64)
                values[name + "_rmse"] = float(np.sqrt(np.mean(diff ** 2)))
            row = expected[(condition, seed, record["id"])]
            for name, value in values.items():
                error = abs(value - float(row[name]))
                errors[name] = max(errors[name], error)
                assert error < 1e-12
    g2.write_json(directory / "INDEPENDENT_VERIFICATION.json", {"utc": g2.utc(), "status": "VERIFIED", "rows": len(expected),
                  "method": "Independent NumPy reductions of all saved test traces, Adam-state and generated-file-hash checks",
                  "max_absolute_metric_difference": errors, "test_access": "verification of already consumed test; no fitting"})


def evaluate_all(run: Path) -> None:
    protocol = check(run)
    lock = g2.read_json(run / "ALL_FINAL_CHECKPOINTS_LOCK.json")
    assert len(lock["checkpoints"]) == protocol["total_fits"]
    for rel, expected in lock["checkpoints"].items():
        assert g2.digest(run / rel) == expected
    g2.write_json(run / "EVALUATION_STARTED.json", {"utc": g2.utc(), "all_checkpoint_lock_sha256": g2.digest(run / "ALL_FINAL_CHECKPOINTS_LOCK.json"),
                  "primary_comparisons": list(COMPARISONS), "reruns_allowed": False})
    g2.summarize = summarize_world
    for world in protocol["worlds"]:
        directory = run / "worlds" / world["id"]
        g2.evaluate(directory)
        independent_verify(directory)
        g2.verify(directory)
        print("WORLD_EVALUATION_VERIFIED", world["id"], flush=True)
    g2.write_json(run / "EVALUATION_COMPLETED.json", {"utc": g2.utc(), "worlds": len(protocol["worlds"]), "fits": protocol["total_fits"]})


def aggregate(run: Path) -> None:
    protocol = check(run)
    assert (run / "EVALUATION_COMPLETED.json").exists()
    collected = {name: [] for name in TABLES}
    world_primary, world_metrics, de_rows, rf_rows = [], [], [], []
    for world in protocol["worlds"]:
        world_id = world["id"]
        directory = run / "worlds" / world_id
        assert g2.read_json(directory / "verification.json")["status"] == "VERIFIED"
        summary = g2.read_json(directory / "summary.json")
        for name in TABLES:
            rows = read_rows(directory / (name + ".csv"))
            collected[name].extend({"world": world_id, **row} for row in rows)
        for condition in "ABCDE":
            row = {"world": world_id, "condition": condition}
            row.update({metric: summary["conditions"][condition][metric]["mean"] for metric in METRICS})
            world_metrics.append(row)
            if condition in "DE":
                de_rows.append(row.copy())
        for comparison in COMPARISONS:
            result = summary["paired"][comparison][PRIMARY]
            world_primary.append({"world": world_id, "comparison": comparison, "mean_difference": result["mean"],
                                  "min_seed_difference": result["min"], "max_seed_difference": result["max"],
                                  "negative_seed_pairs": result["negative_count"], "positive_seed_pairs": sum(v > 0 for v in result["values"]),
                                  "tied_seed_pairs": sum(v == 0 for v in result["values"]), "seed_pairs": len(result["values"])})
        for condition in "ABCDE":
            selected = [r for r in collected["rf_metrics"] if r["world"] == world_id and r["condition"] == condition]
            row = {"world": world_id, "condition": condition}
            for metric in ("teacher_gain", "student_gain", "gain_error", "relative_gain_error", "spatial_profile_l2", "temporal_profile_l2", "rf_rmse"):
                values = [float(r[metric]) for r in selected if r[metric] != ""]
                row[metric] = statistics.mean(values) if values else None
                row[metric + "_defined"] = len(values)
            rf_rows.append(row)
    for name, rows in collected.items():
        g2.write_csv(run / (name + ".csv"), rows)
        g2.write_json(run / (name + ".json"), rows)
    primary_pairs = [r for r in collected["paired_comparisons"] if r["metric"] == PRIMARY]
    for name, rows in (("primary_paired_seeds", primary_pairs), ("world_primary", world_primary),
                       ("world_metrics", world_metrics), ("D_E_metrics", de_rows), ("world_rf_secondary", rf_rows)):
        g2.write_csv(run / (name + ".csv"), rows)
        g2.write_json(run / (name + ".json"), rows)
    directions = {}
    for comparison in COMPARISONS:
        rows = [r for r in world_primary if r["comparison"] == comparison]
        values = [r["mean_difference"] for r in rows]
        directions[comparison] = {**describe(values), "negative_worlds": sum(v < 0 for v in values),
                                  "positive_worlds": sum(v > 0 for v in values), "tied_worlds": sum(v == 0 for v in values),
                                  "all_three_seeds_negative_worlds": sum(r["negative_seed_pairs"] == 3 for r in rows), "world_count": len(rows)}
    g2.write_json(run / "summary.json", {"status": "COMPLETED_FIXED_BUDGET", "technical_verification": "VERIFIED",
                  "worlds": len(protocol["worlds"]), "fits": protocol["total_fits"], "primary_directions": directions,
                  "D_E_world_metrics": de_rows, "inference": "descriptive directions only; no biological population or significance inference",
                  "research_decision": "NOT_ASSIGNED"})
    by_key = {(r["world"], r["condition"], int(r["seed"])): r for r in collected["per_fit_metrics"] if r["stratum"] == "heldout_scales"}
    for row in primary_pairs:
        left, right = row["comparison"].split("-")
        a, b = by_key[(row["world"], left, int(row["seed"]))], by_key[(row["world"], right, int(row["seed"]))]
        assert float(a[PRIMARY]) - float(b[PRIMARY]) == float(row["difference"])
    for row in world_primary:
        values = [float(p["difference"]) for p in primary_pairs if p["world"] == row["world"] and p["comparison"] == row["comparison"]]
        assert statistics.mean(values) == row["mean_difference"] and len(values) == 3
    g2.write_json(run / "verification.json", {"status": "VERIFIED", "utc": g2.utc(), "worlds": len(protocol["worlds"]),
                  "fits": protocol["total_fits"], "optimizer_steps": protocol["total_optimizer_steps"],
                  "microbatches": protocol["total_microbatches"], "G1_G2_sources_unchanged": True,
                  "all_worlds_locked_before_targets": True, "all_fits_locked_before_student_test": True,
                  "world_primary_recomputed_from_per_fit_CSV": True, "primary_comparisons": list(COMPARISONS),
                  "all_world_independent_formula_and_replay_checks_passed": True})
    files = [p for p in run.rglob("*") if p.is_file()]
    g2.write_json(run / "FILE_MANIFEST.json", {str(p.relative_to(run)).replace("\\", "/"): {"bytes": p.stat().st_size, "sha256": g2.digest(p)} for p in files})
    print("G3_ALL_WORLDS_VERIFIED", directions, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "generate", "train", "train-worker", "evaluate", "aggregate"))
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--world-count", type=int, default=5)
    parser.add_argument("--world")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    run = args.run.resolve()
    if args.stage == "freeze":
        freeze(run, args.world_count)
    elif args.stage == "train-worker":
        train_worker(run, args.world, args.seed)
    else:
        {"generate": generate_all, "train": train_all, "evaluate": evaluate_all, "aggregate": aggregate}[args.stage](run)


if __name__ == "__main__":
    main()
