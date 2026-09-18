from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import csv
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch

from . import synthetic_study as g2
from . import robustness_study as g3


CONDITIONS = {"A": "RGC_ONLY_BASE", "C": "MULTILEVEL_JOINT", "E": "EXTRA_RGC"}
PRIMARY = ("h_rmse", "s_B_rmse", "o_B_rmse", "primary_logit_intervention_rmse")
METRICS = PRIMARY + ("excess_ce",)
COMPARISONS = ("C-A", "C-E")
TABLES = ("per_sequence_metrics", "per_fit_metrics", "paired_comparisons", "ambiguity")
BASE = g3.BASE_RUN
G3_RUN = g2.ROOT / "output/experiments/retipath_g3_robustness_20260918"


def table_files(path: Path, rows: list[dict]) -> None:
    g2.write_csv(path.with_suffix(".csv"), rows)
    g2.write_json(path.with_suffix(".json"), rows)


def signs(values: list[float]) -> dict:
    return {"negative": sum(v < 0 for v in values), "positive": sum(v > 0 for v in values),
            "zero": sum(v == 0 for v in values)}


def make_teacher(world: str, base: dict) -> dict:
    key = f"G4a:{world}:teacher"
    draw = torch.randn(len(g2.PARAMETERS), dtype=torch.float64, generator=g2.generator(key))
    physical, raw, delta = {}, {}, {}
    for i, (name, (lo, hi, _, _)) in enumerate(g2.PARAMETERS.items()):
        value = base["teacher"]["physical_parameters"][name]
        center = math.atanh(value / 4) if name.startswith("delta_") else math.log((value - lo) / (hi - value))
        delta[name] = 0.5 * float(draw[i])
        raw[name] = center + delta[name]
        physical[name] = 4 * math.tanh(raw[name]) if name.startswith("delta_") else lo + (hi - lo) / (1 + math.exp(-raw[name]))
        assert lo < physical[name] < hi
    return {"id": world, "rng_namespace": f"G4a:{world}", "teacher_draw_seed": g2.seeded_key(key),
            "teacher_raw_coordinates": raw, "teacher_raw_perturbation": delta, "teacher_physical_parameters": physical}


def schedules(manifest: dict) -> dict:
    streams = {layer: g2.make_stream(g2.eligible(manifest["multi"], layer), count, "base:" + layer)
               for layer, count in (("H", 150), ("BC", 150), ("R", 180))}
    extras = g2.make_stream(manifest["extra"], 300, "extra")
    counts = Counter()
    result = {c: [] for c in CONDITIONS}
    for step in range(1, 181):
        joint, extra = [], []
        for layer in g2.schedule_slots("C", step):
            batch = {"dataset": f"multi/{layer}", "layer": layer, "indices": streams[layer][counts[layer]],
                     "coefficient": 1 / 3 if layer == "R" else 0.4}
            counts[layer] += 1
            joint.append(batch)
            if layer == "R":
                extra.append(copy.deepcopy(batch))
            else:
                extra.append({"dataset": "extra/R", "layer": "R", "indices": extras[counts["extra"]], "coefficient": 1 / 3})
                counts["extra"] += 1
        result["A"].append([copy.deepcopy(b) for b in joint if b["layer"] == "R"])
        result["C"].append(joint)
        result["E"].append(extra)
    return result


def validate(manifest: dict, schedule: dict) -> dict:
    assert set(schedule) == set(CONDITIONS)
    assert {k: len(v) for k, v in manifest.items()} == {"multi": 72, "extra": 120, "test": 40}
    families = [r["family"] for records in manifest.values() for r in records]
    assert len(families) == len(set(families))
    result = {}
    for c, steps in schedule.items():
        assert len(steps) == 180
        counts = Counter(b["dataset"] for step in steps for b in step)
        expected = {"A": Counter({"multi/R": 180}), "C": Counter({"multi/H": 150, "multi/BC": 150, "multi/R": 180}),
                    "E": Counter({"multi/R": 180, "extra/R": 300})}[c]
        assert counts == expected
        for batches in steps:
            for batch in batches:
                pool, layer = batch["dataset"].split("/")
                records = g2.eligible(manifest[pool], layer)
                assert len(batch["indices"]) == len(set(batch["indices"])) == 4
                assert all(0 <= i < len(records) for i in batch["indices"])
                assert len({records[i]["scale"] for i in batch["indices"]}) == 1
                assert batch["coefficient"] == (1 / 3 if layer == "R" else 0.4)
        n = sum(counts.values())
        result[c] = {"steps": 180, "dataset_batches": dict(counts), "microbatches": n, "sequence_exposures": 4 * n,
                     "base_RGC_batches": 180, "base_RGC_sequence_exposures": 720, "base_RGC_coefficient_total": 60.0,
                     "scalar_target_exposures": sum(n * 4 * 240 * {"H": 5, "BC": 10, "R": 1}[k.split("/")[1]] for k, n in counts.items())}
    for a, c, e in zip(schedule["A"], schedule["C"], schedule["E"]):
        assert a == [b for b in c if b["dataset"] == "multi/R"] == [b for b in e if b["dataset"] == "multi/R"]
    original = g2.read_json(BASE / "SCHEDULES.json")
    assert schedule["C"] == original["C"] and schedule["E"] == original["E"]
    return result


def freeze(run: Path) -> None:
    base = g2.check_sources()
    assert g2.digest(Path(g2.__file__)) == g3.RUNNER_SHA256
    sources = {**base["source_hashes"]}
    for path in (Path(g2.__file__), Path(g3.__file__), Path(__file__), BASE / "protocol.json", G3_RUN / "protocol.json"):
        sources[path.resolve().relative_to(g2.ROOT).as_posix()] = g2.digest(path)
    worlds = [make_teacher(f"W{i:02}", base) for i in range(1, 6)]
    prior = [base["teacher"]["physical_parameters"]] + [w["teacher_physical_parameters"] for w in g2.read_json(G3_RUN / "protocol.json")["worlds"]]
    assert all(w["teacher_physical_parameters"] not in prior for w in worlds)
    assert len({tuple(w["teacher_raw_coordinates"].values()) for w in worlds}) == 5
    protocol = {
        "schema": "retipath_g4a_multilevel_confirmatory_v1", "date": "2026-09-18", "conditions": CONDITIONS,
        "worlds": worlds, "student_seeds": base["student_seeds"], "source_hashes": sources,
        "teacher_generation": "Five new realizable teachers: G2 teacher raw center plus independent N(0,0.5^2), original G1 bounds; no rejection, screening, redraw or parameter tuning. New G4a:Wxx namespaces.",
        "random_streams": "G2 SHA256 seed derivation with new G4a world/family keys separates teacher, waveform, H noise, BC noise and event uniforms; no old teacher or data instance reused.",
        "initialization": "G1 center + N(0,0.15^2) raw perturbation from the same three existing student seeds; tensors paired A/C/E and across worlds. No teacher-derived initialization.",
        "stimulus": base["stimulus"],
        "data": {"base_train_sequences": 72, "base_train_per_scale": 24, "extra_train_sequences": 120, "extra_train_per_scale": 40,
                 "test_sequences": 40, "test_per_scale": 8, "development_sequences": 0,
                 "H_visible_scales": [0.3, 0.6], "BC_visible_scales": [0.15, 0.3], "R_visible_scales": [0.15, 0.3, 0.6],
                 "H_nodes": base["data"]["H_nodes"], "BC_nodes": base["data"]["BC_nodes"], "continuous_noise_sigma": 0.03,
                 "H_observation": "noisy h, 48 base sequences", "BC_observation": "noisy o_B, two branches, 48 base sequences; s_B not directly supervised",
                 "R_observation": "causal teacher-event Bernoulli; no probability targets", "extra_independence": "120 families distinct from 72 base and all test families; only RGC labels saved for training"},
        "optimization": {"steps": 180, "microbatch_size": 4, "optimizer": "Adam", "lr": 0.01, "betas": [0.9, 0.999], "eps": 1e-8,
                         "weight_decay": 0.0, "clip_grad_norm": 1.0, "trainable": "all12 raw parameters joint from step1 in every condition; no progressive",
                         "loss": base["optimization"]["loss"], "loss_coefficients": base["optimization"]["loss_coefficients"],
                         "schedule": "Same base R batch every step in A/C/E; C adds H at cycle positions1..5 and BC at2..6; E replaces each of those 300 added slots with independent extra R. No division by total batches.",
                         "microbatches_per_condition": {"A": 180, "C": 480, "E": 480}, "base_RGC_batches_all_conditions": 180,
                         "base_RGC_coefficient_sum_all_conditions": 60.0, "added_coefficient_sums": {"C_H": 60.0, "C_BC": 60.0, "E_extra_R": 100.0},
                         "selection": "step180 only; no validation selection, restart, tuning, budget/data adjustment, condition addition or result-dependent rerun",
                         "nonfinite": "stop affected fit, retain failure, no rescue retry"},
        "fairness": "A/C/E have identical base R data, events, per-step indices, masks, reduction and coefficients. C/E match added 300 batches, not scalar targets or information. A intentionally has fewer total batches. Shared clipping acts on total gradient so optimizer updates are not equal base-gradient effects.",
        "primary_metrics": list(PRIMARY), "secondary_metrics": ["excess_ce", "cross_seed_ambiguity"], "comparisons": list(COMPARISONS),
        "evaluation": {"stratum": "heldout_scales", "heldout_scales": [0.225, 0.45], "warmup_bins": 60, "scored_bins": 240,
                       "reduction": "existing per-sequence RMSE or CE, equal sequences within scale, equal heldout scales; full clean nodes and both BC branches. 16 sequences per fit.",
                       "intervention": base["evaluation"]["intervention"], "history": base["evaluation"]["history"],
                       "comparisons": "four separate primary metrics x C-A/C-E; each paired seed and each teacher reported; no composite score or new success threshold",
                       "summary": "within-teacher mean over3 paired seeds; equal mean and range over5 new teachers, exact signs and per-seed directions. No p-values or biological-population inference; old G2/G3 excluded.",
                       "ambiguity": base["evaluation"]["ambiguity"], "ambiguity_pairs": "(1801,1802),(1801,1803),(1802,1803); compare matched seed-pair distances across conditions",
                       "auxiliary_saved_columns": "unchanged G2 evaluator also saves NLL, local d_E, teacher RMS, relative intervention error and node subgroups; descriptive only, not additional primary",
                       "all_scale_tables": "unchanged per-fit strata retained as descriptive detail; all registered comparisons use heldout_scales",
                       "test_lock": "all45 final checkpoints locked before any student test access; per-world TEST_CONSUMED marker before target load; replay only for numerical verification"},
        "RF": "NOT_RUN: no RF generation, evaluation or metrics", "total_fits": 45, "total_optimizer_steps": 8100, "total_microbatches": 17100,
        "runtime": {"device": "cpu", "dtype": "float32", "reduction_dtype": "float64", "threads_per_process": 1,
                    "deterministic_algorithms": True, "maximum_training_processes": 3, "python": sys.version, "torch": torch.__version__},
        "stop_boundary": "deliver fixed G4a results and stop; no research decision or model-mismatch experiment",
    }
    old_families, old_waves = set(), set()
    for path in [BASE / "STIMULUS_MANIFEST.json"] + list((G3_RUN / "worlds").glob("*/STIMULUS_MANIFEST.json")):
        for records in g2.read_json(path).values():
            for r in records:
                old_families.add(r["family"])
                old_waves.add((r["scale"], tuple(r["center"]), tuple(r["phases"])))
    prepared = []
    for world in worlds:
        key = world["rng_namespace"]
        manifest = {"multi": g2.make_records(key + ":base", [0.15, 0.3, 0.6], 24, "train"),
                    "extra": g2.make_records(key + ":extra", [0.15, 0.3, 0.6], 40, "train"),
                    "test": g2.make_records(key + ":base", [0.15, 0.225, 0.3, 0.45, 0.6], 8, "test")}
        families = {r["family"] for records in manifest.values() for r in records}
        waves = {(r["scale"], tuple(r["center"]), tuple(r["phases"])) for records in manifest.values() for r in records}
        assert len(families) == len(waves) == 232
        assert not families.intersection(old_families) and not waves.intersection(old_waves)
        old_families.update(families)
        old_waves.update(waves)
        schedule = schedules(manifest)
        prepared.append((world, manifest, schedule, validate(manifest, schedule)))
    run.mkdir(parents=True, exist_ok=False)
    g2.write_json(run / "protocol.json", protocol)
    locks, budgets = {}, {}
    for world, manifest, schedule, budget in prepared:
        directory = run / "worlds" / world["id"]
        local = {"schema": protocol["schema"], "source_hashes": sources, "student_seeds": protocol["student_seeds"],
                 "teacher": {"physical_parameters": world["teacher_physical_parameters"]}, "world": world,
                 "parent_protocol_sha256": g2.digest(run / "protocol.json")}
        for filename, obj in (("protocol.json", local), ("STIMULUS_MANIFEST.json", manifest), ("SCHEDULES.json", schedule)):
            g2.write_json(directory / filename, obj)
        g2.write_json(directory / "SOURCE_LOCK.json", {"utc": g2.utc(), "sources": sources,
                      "protocol_sha256": g2.digest(directory / "protocol.json"), "manifest_sha256": g2.digest(directory / "STIMULUS_MANIFEST.json"),
                      "schedules_sha256": g2.digest(directory / "SCHEDULES.json")})
        locks[world["id"]] = g2.digest(directory / "SOURCE_LOCK.json")
        budgets[world["id"]] = budget
    g2.write_json(run / "SOURCE_LOCK.json", {"utc": g2.utc(), "sources": sources, "protocol_sha256": g2.digest(run / "protocol.json"), "world_source_locks": locks})
    for rel in sources:
        dest = run / "source" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("xb") as stream:
            stream.write((g2.ROOT / rel).read_bytes())
    g2.write_json(run / "PREFLIGHT.json", {"status": "VERIFIED", "utc": g2.utc(), "targets_created": False, "world_budgets": budgets,
                  "new_teachers": True, "no_old_or_cross_world_family_or_waveform_overlap": True, "base_RGC_per_step_identity": True,
                  "C_E_schedules_equal_G2": True, "no_narrow_progressive_or_RF": True})
    print("G4A_PREREGISTERED", run, "45 fits", flush=True)


def generate(run: Path) -> None:
    protocol = g3.check(run)
    g2.write_json(run / "GENERATION_STARTED.json", {"utc": g2.utc(), "protocol_sha256": g2.digest(run / "protocol.json")})
    reference = g2.read_json(BASE / "DATA_CREATED.json")["initial_student_sha256"]
    for world in protocol["worlds"]:
        directory = run / "worlds" / world["id"]
        local = g2.check_sources(directory)
        teacher = g2.build_teacher(local)
        g2.save_tensor(directory / "evaluator_only/teacher.pt", {"model_state": teacher.state_dict()})
        for seed in protocol["student_seeds"]:
            g2.save_tensor(directory / f"initial_students/{seed}.pt", {"model_state": g2.LocalRetipathV0(seed=seed).state_dict(), "seed": seed})
        files, errors = {}, {}
        manifest = g2.read_json(directory / "STIMULUS_MANIFEST.json")
        for pool_name, records in manifest.items():
            pool, errors[pool_name] = g2.generate_pool(teacher, records, evaluation=pool_name == "test")
            outputs = {"evaluator_only/test.pt": pool} if pool_name == "test" else {
                f"train_data/{pool_name}/{layer}.pt": g2.layer_dataset(pool, layer)
                for layer in (("R",) if pool_name == "extra" else ("H", "BC", "R"))}
            for rel, data in outputs.items():
                g2.save_tensor(directory / rel, data)
                files[rel] = g2.digest(directory / rel)
        initial = {str(s): g2.digest(directory / f"initial_students/{s}.pt") for s in protocol["student_seeds"]}
        assert initial == reference
        g2.write_json(directory / "DATA_CREATED.json", {"utc": g2.utc(), "files": files, "sampler_forward_max_abs_errors": errors,
                      "event_decisions_match_frozen_forward": True, "initial_student_sha256": initial,
                      "teacher_sha256": g2.digest(directory / "evaluator_only/teacher.pt")})
        print("G4A_DATA_COMPLETE", world["id"], flush=True)
    g3.check(run)
    g2.write_json(run / "GENERATION_COMPLETED.json", {"utc": g2.utc(), "worlds": 5, "paired_initial_hashes": reference})


def train_fit(directory: Path, condition: str, seed: int) -> None:
    g2.check_sources(directory)
    schedule = g2.read_json(directory / "SCHEDULES.json")[condition]
    creation = g2.read_json(directory / "DATA_CREATED.json")
    datasets = {}
    for name in sorted({b["dataset"] for step in schedule for b in step}):
        rel = f"train_data/{name}.pt"
        assert g2.digest(directory / rel) == creation["files"][rel]
        datasets[name] = g2.load_tensor(directory / rel)
        allowed = {"records", "spatial", "temporal", "layer", "targets"} | ({"events"} if name.endswith("/R") else set())
        assert set(datasets[name]) == allowed
    initial_path = directory / f"initial_students/{seed}.pt"
    assert g2.digest(initial_path) == creation["initial_student_sha256"][str(seed)]
    initial = g2.load_tensor(initial_path)
    model = g2.LocalRetipathV0()
    model.load_state_dict(initial["model_state"], strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    dest = directory / f"fits/{condition}_{seed}"
    dest.mkdir(parents=True, exist_ok=False)
    counts, dataset_counts = Counter(), Counter()
    started = time.perf_counter()
    with (dest / "trajectory.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["step", "active_parameters", "microbatches", "loss_H", "loss_BC", "loss_base_R", "loss_extra_R", "weighted_loss", "grad_norm", "elapsed_seconds"])
        writer.writeheader()
        for step, batches in enumerate(schedule, 1):
            active = model.configure_trainable("joint", step)
            optimizer.zero_grad(set_to_none=True)
            losses = defaultdict(float)
            total = 0.0
            for batch in batches:
                dataset, indices = datasets[batch["dataset"]], batch["indices"]
                stimulus = g2.inputs_from(dataset, indices)
                events = dataset["events"][torch.tensor(indices)] if batch["layer"] == "R" else torch.zeros((4, 300, 1))
                trace = model(stimulus, observed_events=events)
                loss = g2.observation_loss(trace, g2.observed_batch(dataset, indices, stimulus))
                assert bool(torch.isfinite(loss)), (condition, seed, step)
                key = ("extra_R" if batch["dataset"] == "extra/R" else "base_R") if batch["layer"] == "R" else batch["layer"]
                losses[key] += float(loss.detach())
                total += batch["coefficient"] * float(loss.detach())
                (batch["coefficient"] * loss).backward()
                counts[batch["layer"]] += 1
                dataset_counts[batch["dataset"]] += 1
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            assert all(bool(torch.isfinite(p).all()) for p in model.parameters())
            writer.writerow({"step": step, "active_parameters": len(active), "microbatches": len(batches),
                             **{"loss_" + k: losses[k] for k in ("H", "BC", "base_R", "extra_R")},
                             "weighted_loss": total, "grad_norm": float(norm), "elapsed_seconds": time.perf_counter() - started})
            stream.flush()
            if step % 30 == 0:
                print("TRAIN", directory.name, condition, seed, step, dict(dataset_counts), flush=True)
    expected_batches = 180 if condition == "A" else 480
    assert sum(counts.values()) == expected_batches and dataset_counts["multi/R"] == 180
    changes = {n: float((p.detach() - initial["model_state"]["raw." + n]).abs()) for n, p in model.raw.items()}
    checkpoint = {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "condition": condition, "seed": seed,
                  "steps": 180, "microbatches": dict(counts), "dataset_batches": dict(dataset_counts), "initial_sha256": g2.digest(initial_path),
                  "protocol_sha256": g2.digest(directory / "protocol.json")}
    g2.save_tensor(dest / "final.pt", checkpoint)
    g2.write_json(dest / "completed.json", {"status": "VERIFIED", "utc": g2.utc(), "condition": condition, "seed": seed, "steps": 180,
                  "microbatches": dict(counts), "dataset_batches": dict(dataset_counts), "sequence_exposures": expected_batches * 4,
                  "forward_calls": expected_batches, "backward_calls": expected_batches, "elapsed_seconds": time.perf_counter() - started,
                  "raw_parameter_absolute_changes": changes, "physical_parameters": {n: float(v.detach()) for n, v in model.physical_parameters().items()},
                  "loaded_training_files": sorted(datasets), "initial_sha256": g2.digest(initial_path), "checkpoint_sha256": g2.digest(dest / "final.pt")})
    g2.check_sources(directory)


def train(run: Path) -> None:
    protocol = g3.check(run)
    assert (run / "GENERATION_COMPLETED.json").exists()
    tasks = [(w["id"], seed) for w in protocol["worlds"] for seed in protocol["student_seeds"]]
    g2.write_json(run / "TRAINING_STARTED.json", {"utc": g2.utc(), "tasks": tasks, "attempts_per_task": 1})
    (run / "logs").mkdir(exist_ok=False)
    def launch(world: str, seed: int) -> dict:
        with (run / "logs" / f"{world}_{seed}.log").open("x", encoding="utf-8") as out:
            result = subprocess.run([sys.executable, "-B", "-m", "experiments.retipath_multiscale_v0.confirmatory_study", "train-worker",
                                     "--run", str(run), "--world", world, "--seed", str(seed)], cwd=g2.ROOT, stdout=out, stderr=subprocess.STDOUT, check=False)
        return {"world": world, "seed": seed, "exit_code": result.returncode}
    outcomes = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(launch, w, s) for w, s in tasks]
        for future in as_completed(futures):
            outcomes.append(future.result())
            print("WORKER_COMPLETE", outcomes[-1], flush=True)
    g2.write_json(run / "TRAINING_OUTCOMES.json", {"utc": g2.utc(), "tasks": outcomes})
    assert all(row["exit_code"] == 0 for row in outcomes), "Failed runs retained; no automatic retries"
    lock = {}
    for w, seed in tasks:
        for c in CONDITIONS:
            path = run / "worlds" / w / f"fits/{c}_{seed}/final.pt"
            done = g2.read_json(path.parent / "completed.json")
            assert done["steps"] == 180 and done["dataset_batches"]["multi/R"] == 180
            assert g2.digest(path) == done["checkpoint_sha256"]
            lock[path.relative_to(run).as_posix()] = g2.digest(path)
    assert len(lock) == 45
    g2.write_json(run / "ALL_FINAL_CHECKPOINTS_LOCK.json", {"utc": g2.utc(), "checkpoints": lock})
    g3.check(run)


def ambiguity_rows(predictions: dict, truth: dict, seeds: list[int]) -> list[dict]:
    rows = []
    for c in CONDITIONS:
        for i, a in enumerate(seeds):
            for b in seeds[i + 1:]:
                first, second = predictions[f"{c}_{a}"], predictions[f"{c}_{b}"]
                diffs = {m: first[m] - second[m] for m in ("p", "h", "s_B", "o_B")}
                diffs["logit_intervention"] = (first["block_ell"] - first["ell"]) - (second["block_ell"] - second["ell"])
                for stratum, scales in g2.strata().items():
                    indices = [j for j, r in enumerate(truth["records"]) if r["scale"] in scales]
                    for metric, diff in diffs.items():
                        rows.append({"condition": c, "seed_a": a, "seed_b": b, "stratum": stratum, "metric": metric,
                                     "rms_distance": statistics.mean(g2.rms(diff[j, 60:]) for j in indices)})
    return rows


def verify_world(directory: Path, truth: dict, predictions: dict, per_fit: list[dict], paired: list[dict]) -> None:
    g3.independent_verify(directory)
    schedule = g2.read_json(directory / "SCHEDULES.json")
    budget = validate(g2.read_json(directory / "STIMULUS_MANIFEST.json"), schedule)
    rows = []
    max_error = 0.0
    for identity, prediction in predictions.items():
        condition, seed_str = identity.split("_")
        seed = int(seed_str)
        fit = directory / "fits" / identity
        ckpt = g2.load_tensor(fit / "final.pt")
        assert ckpt["dataset_batches"] == budget[condition]["dataset_batches"]
        assert ckpt["initial_sha256"] == g2.digest(directory / f"initial_students/{seed}.pt")
        log = g3.read_rows(fit / "trajectory.csv")
        assert [int(r["step"]) for r in log] == list(range(1, 181))
        assert all(int(r["active_parameters"]) == 12 for r in log)
        assert [int(r["microbatches"]) for r in log] == [len(b) for b in schedule[condition]]
        saved = g2.load_tensor(directory / f"evaluation/{identity}_test.pt")
        rows.extend(g2.sequence_metrics(saved, truth, condition, seed, "test"))
        model = g2.LocalRetipathV0().requires_grad_(False).eval()
        model.load_state_dict(ckpt["model_state"], strict=True)
        indices = list(range(8, 12))
        stimulus = g2.inputs_from(truth, indices)
        with torch.no_grad():
            normal = model(stimulus, observed_events=truth["events"][indices])
            blocked = model(stimulus, observed_events=truth["events"][indices], intervention=g2.BLOCK)
        for key, value in {"p": normal.outputs["p"], "ell": normal.outputs["ell"], "block_ell": blocked.outputs["ell"]}.items():
            error = float((value - saved[key][indices]).abs().max())
            max_error = max(max_error, error)
            assert error == 0
    assert g2.aggregate_rows(rows) == per_fit
    index = {(r["condition"], r["seed"]): r for r in per_fit if r["stratum"] == "heldout_scales"}
    for row in paired:
        left, right = row["comparison"].split("-")
        assert row["difference"] == index[left, row["seed"]][row["metric"]] - index[right, row["seed"]][row["metric"]]
    g2.write_json(directory / "verification.json", {"status": "VERIFIED", "utc": g2.utc(), "fits": 9, "optimizer_steps": 1620,
                  "microbatches": 3420, "budget": budget, "student_replay_max_abs_error": max_error,
                  "teacher_replay": "bitwise exact", "metrics_recomputed_from_saved_traces": True,
                  "paired_differences_recomputed": True, "all12_Adam_state_steps_180": True,
                  "test_access": "verification of already consumed test, no selection or fitting"})


def evaluate(run: Path) -> None:
    protocol = g3.check(run)
    lock = g2.read_json(run / "ALL_FINAL_CHECKPOINTS_LOCK.json")
    assert len(lock["checkpoints"]) == 45
    for rel, expected in lock["checkpoints"].items():
        assert g2.digest(run / rel) == expected
    g2.write_json(run / "EVALUATION_STARTED.json", {"utc": g2.utc(), "checkpoint_lock_sha256": g2.digest(run / "ALL_FINAL_CHECKPOINTS_LOCK.json")})
    for world in protocol["worlds"]:
        directory = run / "worlds" / world["id"]
        g2.write_json(directory / "TEST_CONSUMED.json", {"utc": g2.utc(), "purpose": "single final G4a evaluation of9 fixed fits, no RF",
                      "global_checkpoint_lock_sha256": g2.digest(run / "ALL_FINAL_CHECKPOINTS_LOCK.json"), "further_training_allowed": False})
        creation = g2.read_json(directory / "DATA_CREATED.json")
        assert g2.digest(directory / "evaluator_only/test.pt") == creation["files"]["evaluator_only/test.pt"]
        truth = g2.load_tensor(directory / "evaluator_only/test.pt")
        teacher = g2.LocalRetipathV0().requires_grad_(False).eval()
        assert g2.digest(directory / "evaluator_only/teacher.pt") == creation["teacher_sha256"]
        teacher.load_state_dict(g2.load_tensor(directory / "evaluator_only/teacher.pt")["model_state"], strict=True)
        for key, value in g2.trace_for(teacher, truth).items():
            torch.testing.assert_close(value, truth[key], rtol=0, atol=0)
        rows, cached = [], {}
        for c in CONDITIONS:
            for seed in protocol["student_seeds"]:
                identity = f"{c}_{seed}"
                model = g2.LocalRetipathV0().requires_grad_(False).eval()
                model.load_state_dict(g2.load_tensor(directory / f"fits/{identity}/final.pt")["model_state"], strict=True)
                prediction = g2.trace_for(model, truth)
                g2.save_tensor(directory / f"evaluation/{identity}_test.pt", prediction)
                cached[identity] = prediction
                rows.extend(g2.sequence_metrics(prediction, truth, c, seed, "test"))
                print("EVALUATED", world["id"], identity, flush=True)
        per_fit = g2.aggregate_rows(rows)
        index = {(r["condition"], r["seed"]): r for r in per_fit if r["stratum"] == "heldout_scales"}
        paired = []
        for comparison in COMPARISONS:
            left, right = comparison.split("-")
            for seed in protocol["student_seeds"]:
                for metric in METRICS:
                    paired.append({"comparison": comparison, "seed": seed, "metric": metric, "role": "primary" if metric in PRIMARY else "secondary",
                                   "difference": index[left, seed][metric] - index[right, seed][metric]})
        ambiguity = ambiguity_rows(cached, truth, protocol["student_seeds"])
        for name, values in zip(TABLES, (rows, per_fit, paired, ambiguity)):
            table_files(directory / name, values)
        verify_world(directory, truth, cached, per_fit, paired)
        print("WORLD_VERIFIED", world["id"], flush=True)
    g3.check(run)
    g2.write_json(run / "EVALUATION_COMPLETED.json", {"utc": g2.utc(), "worlds": 5, "fits": 45, "RF": "NOT_RUN"})


def aggregate(run: Path) -> None:
    protocol = g3.check(run)
    assert (run / "EVALUATION_COMPLETED.json").exists()
    combined = {name: [] for name in TABLES}
    for world in protocol["worlds"]:
        directory = run / "worlds" / world["id"]
        assert g2.read_json(directory / "verification.json")["status"] == "VERIFIED"
        for name in TABLES:
            combined[name].extend({"world": world["id"], **r} for r in g2.read_json(directory / (name + ".json")))
    for name, rows in combined.items():
        table_files(run / name, rows)
    world_metrics, contrasts, ambiguity_means, ambiguity_pairs = [], [], [], []
    for world in protocol["worlds"]:
        w = world["id"]
        for c in CONDITIONS:
            rows = [r for r in combined["per_fit_metrics"] if r["world"] == w and r["condition"] == c and r["stratum"] == "heldout_scales"]
            assert len(rows) == 3
            world_metrics.append({"world": w, "condition": c, **{m: statistics.mean(r[m] for r in rows) for m in METRICS + ("d_E_rmse", "relative_logit_intervention_error")}})
        for comparison in COMPARISONS:
            for metric in METRICS:
                rows = [r for r in combined["paired_comparisons"] if (r["world"], r["comparison"], r["metric"]) == (w, comparison, metric)]
                assert len(rows) == 3
                values = [r["difference"] for r in rows]
                contrasts.append({"world": w, "comparison": comparison, "metric": metric, "role": rows[0]["role"],
                                  "mean_difference": statistics.mean(values), **signs(values), **{f"seed_{r['seed']}": r["difference"] for r in rows}})
        for metric in ("p", "h", "s_B", "o_B", "logit_intervention"):
            rows = [r for r in combined["ambiguity"] if r["world"] == w and r["stratum"] == "heldout_scales" and r["metric"] == metric]
            by_key = {(r["condition"], r["seed_a"], r["seed_b"]): r["rms_distance"] for r in rows}
            means = {c: statistics.mean(r["rms_distance"] for r in rows if r["condition"] == c) for c in CONDITIONS}
            ambiguity_means.append({"world": w, "metric": metric, **means, "C-A": means["C"] - means["A"], "C-E": means["C"] - means["E"]})
            for comparison in COMPARISONS:
                left, right = comparison.split("-")
                for i, a in enumerate(protocol["student_seeds"]):
                    for b in protocol["student_seeds"][i + 1:]:
                        ambiguity_pairs.append({"world": w, "comparison": comparison, "metric": metric, "seed_a": a, "seed_b": b,
                                                "difference": by_key[left, a, b] - by_key[right, a, b]})
    for name, rows in (("world_metrics", world_metrics), ("world_comparisons", contrasts), ("world_ambiguity", ambiguity_means), ("ambiguity_paired_differences", ambiguity_pairs)):
        table_files(run / name, rows)
    directions = {}
    for comparison in COMPARISONS:
        directions[comparison] = {}
        for metric in METRICS:
            rows = [r for r in contrasts if r["comparison"] == comparison and r["metric"] == metric]
            values = [r["mean_difference"] for r in rows]
            directions[comparison][metric] = {**g3.describe(values), **signs(values), "all_three_seeds_negative_worlds": sum(r["negative"] == 3 for r in rows)}
    g2.write_json(run / "summary.json", {"status": "COMPLETED_FIXED_BUDGET", "primary_metrics": list(PRIMARY), "directions": directions,
                  "worlds": 5, "fits": 45, "research_decision": "NOT_ASSIGNED", "RF": "NOT_RUN", "inference": "paired values and descriptive world directions only"})
    index = {(r["world"], r["condition"], r["seed"]): r for r in combined["per_fit_metrics"] if r["stratum"] == "heldout_scales"}
    for row in combined["paired_comparisons"]:
        left, right = row["comparison"].split("-")
        assert row["difference"] == index[row["world"], left, row["seed"]][row["metric"]] - index[row["world"], right, row["seed"]][row["metric"]]
    lock = g2.read_json(run / "ALL_FINAL_CHECKPOINTS_LOCK.json")
    for rel, expected in lock["checkpoints"].items():
        assert g2.digest(run / rel) == expected
    g3.check(run)
    g2.write_json(run / "verification.json", {"status": "VERIFIED", "utc": g2.utc(), "worlds": 5, "fits": 45, "optimizer_steps": 8100,
                  "microbatches": 17100, "sequence_exposures": 68400, "base_RGC_batches_per_fit_all_conditions": 180,
                  "all_worlds_passed_independent_formula_and_replay_checks": True, "frozen_sources_and_checkpoints_unchanged": True,
                  "paired_comparisons_recomputed": True, "RF": "NOT_RUN"})
    files = [p for p in run.rglob("*") if p.is_file()]
    g2.write_json(run / "FILE_MANIFEST.json", {p.relative_to(run).as_posix(): {"bytes": p.stat().st_size, "sha256": g2.digest(p)} for p in files})
    print("G4A_ALL45_VERIFIED", directions, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "generate", "train", "train-worker", "evaluate", "aggregate"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--world")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    run = args.run.resolve()
    if args.stage == "train-worker":
        protocol = g3.check(run)
        assert args.world in {w["id"] for w in protocol["worlds"]} and args.seed in protocol["student_seeds"]
        for condition in CONDITIONS:
            train_fit(run / "worlds" / args.world, condition, args.seed)
    else:
        {"freeze": freeze, "generate": generate, "train": train, "evaluate": evaluate, "aggregate": aggregate}[args.stage](run)


if __name__ == "__main__":
    main()
