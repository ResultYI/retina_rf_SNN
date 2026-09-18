from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import statistics
import time

import torch
from torch.nn import functional as F

from .circuit import LocalRetipathV0, PARAMETERS
from .contracts import DT_MS, Intervention, InterventionSpec, ObservationBatch, StimulusBatch, observation_loss, regular_pixel_bounds


ROOT = Path(__file__).resolve().parents[2]
SPEC = Path(__file__).with_name("protocol.json")
HC = torch.tensor((12, 10, 14, 2, 22))
BC = torch.tensor((12, 6, 8, 16, 18))
BOUNDS = regular_pixel_bounds(32)
AREA = (BOUNDS[..., 1] - BOUNDS[..., 0]).prod(-1)
TIME = (torch.arange(300) + 0.5) * DT_MS
BLOCK = InterventionSpec(Intervention.BLOCK_DIRECT_BC_DRIVE)


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def save_tensor(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        torch.save(value, stream)


def load_tensor(path: Path):
    return torch.load(path, map_location="cpu", weights_only=True)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def seeded_key(text: str) -> int:
    return int.from_bytes(sha256(("2026091810:" + text).encode()).digest()[:8], "little") % (2**63 - 1)


def generator(key: str) -> torch.Generator:
    return torch.Generator().manual_seed(seeded_key(key))


def check_sources(run: Path | None = None) -> dict:
    protocol = read_json(SPEC if run is None else run / "protocol.json")
    for rel, expected in protocol["source_hashes"].items():
        if digest(ROOT / rel) != expected:
            raise RuntimeError("Frozen G1/design source changed: " + rel)
    if run is not None:
        corrections_path = run / "RUNTIME_CORRECTIONS.json"
        corrections = read_json(corrections_path)["sources"] if corrections_path.exists() else {}
        for rel, expected in read_json(run / "SOURCE_LOCK.json")["sources"].items():
            if rel in corrections:
                assert rel == "experiments/retipath_multiscale_v0/synthetic_study.py"
                assert corrections[rel]["before_sha256"] == expected
                expected = corrections[rel]["after_sha256"]
            if digest(ROOT / rel) != expected:
                raise RuntimeError("Frozen producer changed: " + rel)
        if digest(run / "protocol.json") != read_json(run / "SOURCE_LOCK.json")["protocol_sha256"]:
            raise RuntimeError("Run protocol changed")
    return protocol


def make_records(pool: str, scales: list[float], count: int, split: str) -> list[dict]:
    records = []
    for scale in scales:
        for row in range(count):
            family = f"{split}:{pool}:{scale:.3f}:{row:02d}"
            rng = random.Random(seeded_key(family + ":waveform"))
            records.append({"id": family, "family": family, "split": split, "scale": scale,
                            "logical_scale": scale, "center": [rng.choice((-0.1, 0.0, 0.1)) for _ in range(2)],
                            "phases": [rng.uniform(0, 2 * math.pi) for _ in range(4)]})
    return records


def eligible(records: list[dict], layer: str) -> list[dict]:
    allowed = {"H": (0.3, 0.6), "BC": (0.15, 0.3), "R": (0.15, 0.3, 0.6)}[layer]
    return [r for r in records if r["logical_scale"] in allowed]


def make_stream(records: list[dict], count: int, key: str) -> list[list[int]]:
    scales = sorted({r["logical_scale"] for r in records})
    buckets = {s: [i for i, r in enumerate(records) if r["logical_scale"] == s] for s in scales}
    rng = {s: random.Random(seeded_key(key + str(s))) for s in scales}
    queues = {s: [] for s in scales}
    result = []
    for number in range(count):
        scale = scales[number % len(scales)]
        if len(queues[scale]) < 4:
            order = list(buckets[scale])
            rng[scale].shuffle(order)
            queues[scale].extend(order)
        result.append(queues[scale][:4])
        queues[scale] = queues[scale][4:]
    return result


def schedule_slots(condition: str, step: int) -> list[str]:
    if condition == "D":
        if step <= 30:
            return ["H"]
        if step <= 60:
            return ["BC"]
        return ["H", "BC", "R"] + (["R"] if (step - 60) % 2 == 0 else [])
    position = (step - 1) % 6 + 1
    return (["H"] if position <= 5 else []) + (["BC"] if position >= 2 else []) + ["R"]


def build_schedules(manifest: dict) -> dict:
    streams = {layer: make_stream(eligible(manifest["multi"], layer), count, "base:" + layer)
               for layer, count in (("H", 150), ("BC", 150), ("R", 180))}
    repeats = make_stream(manifest["multi"], 300, "repeat")
    extras = make_stream(manifest["extra"], 300, "extra")
    schedules = {}
    for condition in "ABCDE":
        counts = Counter()
        steps = []
        for step in range(1, 181):
            batches = []
            for slot in schedule_slots(condition, step):
                if condition in "AE" and slot != "R":
                    layer, source = "R", "multi" if condition == "A" else "extra"
                    indices = (repeats if condition == "A" else extras)[counts["supplement"]]
                    counts["supplement"] += 1
                else:
                    layer = slot
                    source = "narrow" if condition == "B" else "multi"
                    indices = streams[layer][counts[layer]]
                    counts[layer] += 1
                batches.append({"dataset": f"{source}/{layer}", "layer": layer, "indices": indices,
                                "coefficient": 1 / 3 if layer == "R" else 0.4})
            steps.append(batches)
        schedules[condition] = steps
    return schedules


def validate_schedules(manifest: dict, schedules: dict) -> dict:
    result = {}
    for condition, steps in schedules.items():
        assert len(steps) == 180
        counts = Counter(b["layer"] for step in steps for b in step)
        assert sum(counts.values()) == 480
        assert counts == (Counter(R=480) if condition in "AE" else Counter(H=150, BC=150, R=180))
        assert all(len(b["indices"]) == 4 for step in steps for b in step)
        result[condition] = {"steps": len(steps), "microbatches": dict(counts), "sequence_exposures": 1920,
                             "scalar_target_exposures": sum(counts[k] * 4 * 240 * {"H": 5, "BC": 10, "R": 1}[k] for k in counts),
                             "coefficient_sum": math.fsum(b["coefficient"] for s in steps for b in s)}
    for layer in ("H", "BC", "R"):
        c = [b for s in schedules["C"] for b in s if b["layer"] == layer]
        d = [b for s in schedules["D"] for b in s if b["layer"] == layer]
        assert c == d
        b = [v for s in schedules["B"] for v in s if v["layer"] == layer]
        assert [v["indices"] for v in b] == [v["indices"] for v in c]
        assert len(eligible(manifest["multi"], layer)) == len(eligible(manifest["narrow"], layer))
    families = defaultdict(set)
    for records in manifest.values():
        for record in records:
            families[record["family"]].add(record["split"])
    assert all(len(splits) == 1 for splits in families.values())
    for records in manifest.values():
        waveforms = {(r["scale"], tuple(r["center"]), tuple(r["phases"])) for r in records}
        assert len(waveforms) == len(records)
    assert len(manifest["multi"]) == len(manifest["narrow"]) == 72
    return result


def freeze(run: Path) -> None:
    protocol = check_sources()
    run.mkdir(parents=True, exist_ok=False)
    manifest = {
        "multi": make_records("base", [0.15, 0.3, 0.6], 24, "train"),
        "extra": make_records("extra", [0.15, 0.3, 0.6], 40, "train"),
        "development": make_records("base", [0.15, 0.3, 0.6], 8, "development"),
        "test": make_records("base", [0.15, 0.225, 0.3, 0.45, 0.6], 8, "test"),
    }
    manifest["narrow"] = [{**r, "id": "narrow:" + r["id"], "scale": 0.3} for r in manifest["multi"]]
    schedules = build_schedules(manifest)
    fairness = validate_schedules(manifest, schedules)
    write_json(run / "protocol.json", protocol)
    write_json(run / "STIMULUS_MANIFEST.json", manifest)
    write_json(run / "SCHEDULES.json", schedules)
    write_json(run / "PREFLIGHT.json", {"status": "VERIFIED", "utc": utc(), "targets_created": False,
                                       "fairness": fairness, "families_disjoint_across_splits": True})
    sources = {**protocol["source_hashes"], str(Path(__file__).relative_to(ROOT)).replace("\\", "/"): digest(Path(__file__)),
               str(SPEC.relative_to(ROOT)).replace("\\", "/"): digest(SPEC)}
    write_json(run / "SOURCE_LOCK.json", {"utc": utc(), "sources": sources, "protocol_sha256": digest(run / "protocol.json"),
                                          "manifest_sha256": digest(run / "STIMULUS_MANIFEST.json"),
                                          "schedules_sha256": digest(run / "SCHEDULES.json")})
    for rel in sources:
        destination = run / "source" / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write((ROOT / rel).read_bytes())
    print("PROTOCOL_FROZEN", run, flush=True)


def physical_arrays(records: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    spatial, temporal = [], []
    frequencies = torch.tensor((0.5, 1.5, 3.0, 6.0))
    for r in records:
        sigma = r["scale"]
        lower = BOUNDS[..., 0] - torch.tensor(r["center"])
        upper = BOUNDS[..., 1] - torch.tensor(r["center"])
        integral = sigma * math.sqrt(math.pi / 2) * (torch.erf(upper / (math.sqrt(2) * sigma)) - torch.erf(lower / (math.sqrt(2) * sigma)))
        spatial.append(integral.prod(-1) / AREA)
        temporal.append(0.4 * torch.sin(2 * math.pi * TIME[:, None] / 1000 * frequencies + torch.tensor(r["phases"])).mean(-1))
    return torch.stack(spatial), torch.stack(temporal)


def make_stimulus(values: torch.Tensor, records: list[dict]) -> StimulusBatch:
    return StimulusBatch(values, BOUNDS, AREA, TIME, torch.ones_like(values, dtype=torch.bool),
                         tuple(r["id"] for r in records), tuple(r["family"] for r in records))


def inputs_from(data: dict, indices: list[int]) -> StimulusBatch:
    idx = torch.tensor(indices)
    records = [data["records"][i] for i in indices]
    values = (data["temporal"][idx, :, None] * data["spatial"][idx, None, :])[..., None]
    return make_stimulus(values, records)


def build_teacher(protocol: dict) -> LocalRetipathV0:
    teacher = LocalRetipathV0()
    with torch.no_grad():
        for name, value in protocol["teacher"]["physical_parameters"].items():
            lo, hi, _, _ = PARAMETERS[name]
            fraction = (value - lo) / (hi - lo)
            raw = math.atanh(value / 4) if name.startswith("delta_") else math.log(fraction / (1 - fraction))
            teacher.raw[name].fill_(raw)
    return teacher.requires_grad_(False).eval()


def causal_sample(base_logits: torch.Tensor, uniforms: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rho = base_logits.new_tensor(math.exp(-DT_MS / 40))
    q = torch.zeros_like(base_logits[:, 0])
    previous = torch.zeros_like(q)
    sampled, probabilities = [], []
    for t in range(base_logits.shape[1]):
        q = rho * q + (1 - rho) * previous
        p = (base_logits[:, t] - q).sigmoid()
        previous = (uniforms[:, t] < p).to(p.dtype)
        sampled.append(previous)
        probabilities.append(p)
    return torch.stack(sampled, 1), torch.stack(probabilities, 1)


def generate_pool(teacher: LocalRetipathV0, records: list[dict], *, evaluation: bool) -> tuple[dict, float]:
    spatial, temporal = physical_arrays(records)
    data = {"records": records, "spatial": spatial, "temporal": temporal}
    collected = defaultdict(list)
    largest_error = 0.0
    for start in range(0, len(records), 4):
        indices = list(range(start, min(start + 4, len(records))))
        stimulus = inputs_from(data, indices)
        with torch.no_grad():
            zeros = torch.zeros((len(indices), 300, 1))
            base = teacher(stimulus, observed_events=zeros)
            uniforms = torch.stack([torch.rand((300, 1), generator=generator(records[i]["family"] + ":events")) for i in indices])
            events, sampled_p = causal_sample(base.outputs["ell"], uniforms)
            normal = teacher(stimulus, observed_events=events)
            error = float((sampled_p - normal.outputs["p"]).abs().max())
            largest_error = max(largest_error, error)
            assert error < 2e-7
            assert torch.equal(events, (uniforms < normal.outputs["p"]).float())
            collected["events"].append(events)
            h = normal.states["h"].index_select(2, HC)
            bc = normal.outputs["o_B"].index_select(2, BC)
            h_noise = torch.stack([torch.randn((300, 5, 1), generator=generator(records[i]["family"] + ":Hnoise")) for i in indices])
            bc_noise = torch.stack([torch.randn((300, 5, 2), generator=generator(records[i]["family"] + ":BCnoise")) for i in indices])
            collected["H"].append(h + 0.03 * h_noise)
            collected["BC"].append(bc + 0.03 * bc_noise)
            if evaluation:
                blocked = teacher(stimulus, observed_events=events, intervention=BLOCK)
                for name, value in {"p": normal.outputs["p"], "ell": normal.outputs["ell"],
                                    "h": normal.states["h"], "s_B": normal.states["s_B"],
                                    "o_B": normal.outputs["o_B"], "d_E": normal.outputs["d_E"],
                                    "block_ell": blocked.outputs["ell"]}.items():
                    collected[name].append(value)
        print("GENERATE", records[start]["split"], records[start]["id"], start + len(indices), "/", len(records), flush=True)
    data.update({key: torch.cat(values) for key, values in collected.items()})
    return data, largest_error


def layer_dataset(pool: dict, layer: str) -> dict:
    visible = {r["id"] for r in eligible(pool["records"], layer)}
    indices = [i for i, r in enumerate(pool["records"]) if r["id"] in visible]
    idx = torch.tensor(indices)
    result = {"records": [pool["records"][i] for i in indices], "spatial": pool["spatial"][idx],
              "temporal": pool["temporal"][idx], "layer": layer,
              "targets": pool["events"][idx, ..., None] if layer == "R" else pool[layer][idx]}
    if layer == "R":
        result["events"] = pool["events"][idx]
    return result


def generate(run: Path) -> None:
    protocol = check_sources(run)
    if (run / "DATA_CREATED.json").exists():
        raise RuntimeError("Data already exist; no regeneration")
    teacher = build_teacher(protocol)
    save_tensor(run / "evaluator_only/teacher.pt", {"model_state": teacher.state_dict()})
    for seed in protocol["student_seeds"]:
        save_tensor(run / f"initial_students/{seed}.pt", {"model_state": LocalRetipathV0(seed=seed).state_dict(), "seed": seed})
    manifest = read_json(run / "STIMULUS_MANIFEST.json")
    errors, files = {}, {}
    for pool_name in ("multi", "narrow", "extra", "development", "test"):
        evaluation = pool_name == "test"
        pool, errors[pool_name] = generate_pool(teacher, manifest[pool_name], evaluation=evaluation)
        if pool_name == "development":
            for i, record in enumerate(pool["records"]):
                if record["scale"] == 0.15:
                    pool["H"][i] = float("nan")
                if record["scale"] == 0.6:
                    pool["BC"][i] = float("nan")
        if pool_name in ("development", "test"):
            path = run / f"evaluator_only/{pool_name}.pt"
            save_tensor(path, pool)
            files[str(path.relative_to(run))] = digest(path)
        else:
            for layer in (("R",) if pool_name == "extra" else ("H", "BC", "R")):
                path = run / f"train_data/{pool_name}/{layer}.pt"
                save_tensor(path, layer_dataset(pool, layer))
                files[str(path.relative_to(run))] = digest(path)
    write_json(run / "DATA_CREATED.json", {"utc": utc(), "files": files, "sampler_forward_max_abs_errors": errors,
                                            "event_decisions_match_frozen_forward": True,
                                            "initial_student_sha256": {str(s): digest(run / f"initial_students/{s}.pt") for s in protocol["student_seeds"]},
                                            "teacher_sha256": digest(run / "evaluator_only/teacher.pt")})


def observed_batch(data: dict, indices: list[int], stimulus: StimulusBatch) -> ObservationBatch:
    layer = data["layer"]
    target = data["targets"][torch.tensor(indices)]
    mask = torch.ones_like(target, dtype=torch.bool)
    mask[:, :60] = False
    return ObservationBatch(
        {"H": "HC", "BC": "BC", "R": "RGC"}[layer], {"H": "h", "BC": "o_B", "R": "events"}[layer],
        {"H": HC, "BC": BC, "R": torch.tensor((0,))}[layer],
        torch.tensor((0, 1)) if layer == "BC" else torch.tensor((0,)),
        target, mask, torch.arange(300), "binary_occupancy" if layer == "R" else "synthetic_effective",
        "bernoulli" if layer == "R" else "gaussian", None if layer == "R" else 0.03,
        stimulus.sequence_id, stimulus.split_group_id,
    )


def train_fit(run: Path, condition: str, seed: int) -> None:
    protocol = check_sources(run)
    directory = run / f"fits/{condition}_{seed}"
    directory.mkdir(parents=True, exist_ok=False)
    schedule = read_json(run / "SCHEDULES.json")[condition]
    creation = read_json(run / "DATA_CREATED.json")
    datasets = {}
    for name in sorted({batch["dataset"] for step in schedule for batch in step}):
        rel = f"train_data/{name}.pt"
        assert digest(run / rel) == {k.replace("\\", "/"): v for k, v in creation["files"].items()}[rel]
        datasets[name] = load_tensor(run / rel)
        permitted = {"records", "spatial", "temporal", "layer", "targets"} | ({"events"} if name.endswith("/R") else set())
        assert set(datasets[name]) == permitted
    model = LocalRetipathV0()
    initial_path = run / f"initial_students/{seed}.pt"
    assert digest(initial_path) == creation["initial_student_sha256"][str(seed)]
    initial = load_tensor(initial_path)
    model.load_state_dict(initial["model_state"], strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    counts = Counter()
    elapsed_forward = elapsed_backward = 0.0
    started = time.perf_counter()
    with (directory / "trajectory.csv").open("x", encoding="utf-8", newline="") as stream:
        columns = ["step", "active_parameters", "microbatches", "loss_H", "loss_BC", "loss_R", "weighted_loss", "grad_norm", "elapsed_seconds"]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for step_number, batches in enumerate(schedule, 1):
            active = model.configure_trainable("progressive" if condition == "D" else "joint", step_number)
            optimizer.zero_grad(set_to_none=True)
            losses = defaultdict(float)
            weighted_loss = 0.0
            for batch in batches:
                dataset = datasets[batch["dataset"]]
                indices = batch["indices"]
                stimulus = inputs_from(dataset, indices)
                history = dataset["events"][torch.tensor(indices)] if batch["layer"] == "R" else torch.zeros((4, 300, 1))
                t0 = time.perf_counter()
                trace = model(stimulus, observed_events=history)
                loss = observation_loss(trace, observed_batch(dataset, indices, stimulus))
                elapsed_forward += time.perf_counter() - t0
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError(f"Nonfinite loss {condition}/{seed}/{step_number}")
                losses[batch["layer"]] += float(loss.detach())
                weighted_loss += batch["coefficient"] * float(loss.detach())
                t0 = time.perf_counter()
                (batch["coefficient"] * loss).backward()
                elapsed_backward += time.perf_counter() - t0
                counts[batch["layer"]] += 1
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
                raise RuntimeError("Nonfinite parameter after update")
            writer.writerow({"step": step_number, "active_parameters": len(active), "microbatches": len(batches),
                             "loss_H": losses["H"], "loss_BC": losses["BC"], "loss_R": losses["R"],
                             "weighted_loss": weighted_loss, "grad_norm": float(norm), "elapsed_seconds": time.perf_counter() - started})
            stream.flush()
            if step_number % 30 == 0:
                print("TRAIN", condition, seed, step_number, "batches", sum(counts.values()), "elapsed", round(time.perf_counter() - started, 2), flush=True)
    assert sum(counts.values()) == 480
    changes = {n: float((p.detach() - initial["model_state"]["raw." + n]).abs()) for n, p in model.raw.items()}
    physical = {n: float(v.detach()) for n, v in model.physical_parameters().items()}
    checkpoint = {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "condition": condition,
                  "seed": seed, "steps": 180, "microbatches": dict(counts), "initial_sha256": digest(initial_path),
                  "protocol_sha256": digest(run / "protocol.json")}
    save_tensor(directory / "final.pt", checkpoint)
    write_json(directory / "completed.json", {"status": "VERIFIED", "utc": utc(), "condition": condition, "seed": seed,
                                               "steps": 180, "microbatches": dict(counts), "sequence_exposures": 1920,
                                               "forward_calls": 480, "backward_calls": 480,
                                               "forward_seconds": elapsed_forward, "backward_seconds": elapsed_backward,
                                               "elapsed_seconds": time.perf_counter() - started, "raw_parameter_absolute_changes": changes,
                                               "physical_parameters": physical, "loaded_training_files": sorted(datasets),
                                               "initial_sha256": digest(initial_path), "checkpoint_sha256": digest(directory / "final.pt")})
    check_sources(run)


def trace_for(model: LocalRetipathV0, data: dict) -> dict[str, torch.Tensor]:
    result = defaultdict(list)
    with torch.no_grad():
        for start in range(0, len(data["records"]), 4):
            indices = list(range(start, min(start + 4, len(data["records"]))))
            stimulus = inputs_from(data, indices)
            events = data["events"][torch.tensor(indices)]
            normal = model(stimulus, observed_events=events)
            blocked = model(stimulus, observed_events=events, intervention=BLOCK)
            for name, value in {"p": normal.outputs["p"], "ell": normal.outputs["ell"], "block_ell": blocked.outputs["ell"],
                                "h": normal.states["h"], "s_B": normal.states["s_B"], "o_B": normal.outputs["o_B"], "d_E": normal.outputs["d_E"]}.items():
                result[name].append(value)
    return {name: torch.cat(values) for name, values in result.items()}


def rms(value: torch.Tensor) -> float:
    return float(value.double().square().mean().sqrt())


def sequence_metrics(pred: dict, truth: dict, condition: str, seed: int, split: str) -> list[dict]:
    rows = []
    unobserved_h = torch.tensor([i for i in range(25) if i not in HC.tolist()])
    unobserved_bc = torch.tensor([i for i in range(25) if i not in BC.tolist()])
    for i, record in enumerate(truth["records"]):
        ell = pred["ell"][i, 60:].double()
        p_true = truth["p"][i, 60:].double()
        entropy = -p_true * p_true.log() - (1 - p_true) * torch.log1p(-p_true)
        delta = pred["block_ell"][i, 60:] - pred["ell"][i, 60:]
        delta_true = truth["block_ell"][i, 60:] - truth["ell"][i, 60:]
        error, effect = rms(delta - delta_true), rms(delta_true)
        row = {"condition": condition, "seed": seed, "split": split, "sequence_id": record["id"], "scale": record["scale"],
               "primary_logit_intervention_rmse": error, "teacher_logit_effect_rms": effect,
               "relative_logit_intervention_error": error / effect if effect else None,
               "rgc_nll": float((F.softplus(ell) - truth["events"][i, 60:].double() * ell).mean()),
               "excess_ce": float((F.softplus(ell) - p_true * ell - entropy).mean())}
        for key in ("h", "s_B", "o_B", "d_E"):
            diff = pred[key][i, 60:] - truth[key][i, 60:]
            row[key + "_rmse"] = rms(diff)
            row[key + "_teacher_rms"] = rms(truth[key][i, 60:])
            if key in ("h", "s_B", "o_B"):
                obs, unobs = (HC, unobserved_h) if key == "h" else (BC, unobserved_bc)
                row[key + "_observed_nodes_rmse"] = rms(diff.index_select(1, obs))
                row[key + "_unobserved_nodes_rmse"] = rms(diff.index_select(1, unobs))
        rows.append(row)
    return rows


def development_metrics(pred: dict, data: dict, condition: str, seed: int) -> list[dict]:
    rows = []
    for i, record in enumerate(data["records"]):
        ell = pred["ell"][i, 60:].double()
        h_error = (pred["h"][i, 60:].index_select(1, HC) - data["H"][i, 60:]).double()
        bc_error = (pred["o_B"][i, 60:].index_select(1, BC) - data["BC"][i, 60:]).double()
        rows.append({"condition": condition, "seed": seed, "sequence_id": record["id"], "scale": record["scale"],
                     "rgc_nll": float((F.softplus(ell) - data["events"][i, 60:].double() * ell).mean()),
                     "H_gaussian_nll_without_constant": float(0.5 * (h_error / 0.03).square().mean()) if record["scale"] in (0.3, 0.6) else None,
                     "BC_gaussian_nll_without_constant": float(0.5 * (bc_error / 0.03).square().mean()) if record["scale"] in (0.15, 0.3) else None})
    return rows


def strata() -> dict[str, tuple[float, ...]]:
    return {"all": (0.15, 0.225, 0.3, 0.45, 0.6), "heldout_scales": (0.225, 0.45),
            "seen_scales": (0.15, 0.3, 0.6), **{f"scale_{s}": (s,) for s in (0.15, 0.225, 0.3, 0.45, 0.6)}}


def aggregate_rows(rows: list[dict]) -> list[dict]:
    metrics = [k for k in rows[0] if k not in ("condition", "seed", "split", "sequence_id", "scale")]
    result = []
    groups = defaultdict(list)
    for row in rows:
        groups[(row["condition"], row["seed"], row["split"])].append(row)
    for (condition, seed, split), records in groups.items():
        for name, scales in strata().items():
            selected = [r for r in records if r["scale"] in scales]
            if not selected:
                continue
            out = {"condition": condition, "seed": seed, "split": split, "stratum": name, "sequences": len(selected)}
            for metric in metrics:
                by_scale = [[r[metric] for r in selected if r["scale"] == s and r[metric] is not None] for s in scales]
                means = [statistics.mean(v) for v in by_scale if v]
                out[metric] = statistics.mean(means) if means else None
            result.append(out)
    return result


def rf_matrix(model: LocalRetipathV0, anchor: dict, scale: float) -> torch.Tensor:
    record = {**anchor, "scale": scale}
    spatial, temporal = physical_arrays([record])
    temporal[:, 60:] = 0
    base = (temporal[:, :, None] * spatial[:, None, :])[..., None]
    node_xy = model.input_xy
    lower = torch.maximum(BOUNDS[None, :, :, 0], node_xy[:, None, :] - 0.05)
    upper = torch.minimum(BOUNDS[None, :, :, 1], node_xy[:, None, :] + 0.05)
    probe = (upper - lower).clamp_min(0).prod(-1) / AREA[None]
    responses = []
    with torch.no_grad():
        for sign in (1, -1):
            parts = []
            for start in range(0, 25, 5):
                values = base.expand(5, -1, -1, -1).clone()
                values[:, 60, :, 0] += sign * 0.01 * probe[start:start + 5]
                records = [{**record, "id": f"rf:{scale}:{sign}:{i}"} for i in range(start, start + 5)]
                trace = model(make_stimulus(values, records), observed_events=torch.zeros((5, 300, 1)))
                parts.append(trace.outputs["ell"][:, 61:91, 0])
            responses.append(torch.cat(parts))
    return (responses[0] - responses[1]).double() / 0.02


def rf_profiles(matrix: torch.Tensor) -> dict:
    result = {"gain": rms(matrix)}
    for name, vector in (("spatial", matrix.mean(1)), ("temporal", matrix.mean(0))):
        norm = float(vector.norm())
        result[name] = (vector / norm).tolist() if norm >= 1e-12 else None
        result[name + "_projection_norm"] = norm
    return result


def rf_metrics(pred: torch.Tensor, truth: torch.Tensor, condition: str, seed: int, scale: float) -> dict:
    p, q = rf_profiles(pred), rf_profiles(truth)
    row = {"condition": condition, "seed": seed, "history_scale": scale, "teacher_gain": q["gain"], "student_gain": p["gain"],
           "gain_error": abs(p["gain"] - q["gain"]), "relative_gain_error": abs(p["gain"] - q["gain"]) / q["gain"] if q["gain"] else None,
           "rf_rmse": rms(pred - truth)}
    for name in ("spatial", "temporal"):
        if p[name] is None or q[name] is None:
            row[name + "_profile_l2"] = row[name + "_profile_cosine"] = None
        else:
            a, b = torch.tensor(p[name], dtype=torch.float64), torch.tensor(q[name], dtype=torch.float64)
            row[name + "_profile_l2"] = float((a - b).norm())
            row[name + "_profile_cosine"] = float(a @ b)
    return row


def summarize(per_fit: list[dict], paired: list[dict]) -> dict:
    result = {"status": "COMPLETED_FIXED_BUDGET", "scientific_verdict": "NOT_ASSIGNED", "conditions": {}, "paired": {}}
    for condition in "ABCDE":
        selected = [r for r in per_fit if r["condition"] == condition and r["stratum"] == "heldout_scales" and r["split"] == "test"]
        result["conditions"][condition] = {}
        for metric in ("primary_logit_intervention_rmse", "rgc_nll", "excess_ce", "h_rmse", "s_B_rmse", "o_B_rmse"):
            values = [r[metric] for r in selected]
            result["conditions"][condition][metric] = {"mean": statistics.mean(values), "min": min(values), "max": max(values), "values": values}
    for name in ("B-A", "C-B", "D-C", "E-A", "C-E"):
        result["paired"][name] = {}
        for metric in ("primary_logit_intervention_rmse", "excess_ce"):
            values = [r["difference"] for r in paired if r["comparison"] == name and r["metric"] == metric]
            result["paired"][name][metric] = {"mean": statistics.mean(values), "values": values, "negative_count": sum(v < 0 for v in values)}
    return result


def evaluate(run: Path) -> None:
    protocol = check_sources(run)
    checkpoints = {}
    for condition in "ABCDE":
        for seed in protocol["student_seeds"]:
            directory = run / f"fits/{condition}_{seed}"
            complete = read_json(directory / "completed.json")
            assert complete["steps"] == 180 and sum(complete["microbatches"].values()) == 480
            checkpoints[f"{condition}_{seed}"] = digest(directory / "final.pt")
            assert checkpoints[f"{condition}_{seed}"] == complete["checkpoint_sha256"]
    write_json(run / "FINAL_CHECKPOINTS_LOCK.json", {"utc": utc(), "checkpoints": checkpoints})
    write_json(run / "TEST_CONSUMED.json", {"utc": utc(), "purpose": "single fixed final evaluation of all15 frozen fits and RF secondary",
                                            "protocol_sha256": digest(run / "protocol.json"), "checkpoint_lock_sha256": digest(run / "FINAL_CHECKPOINTS_LOCK.json"),
                                            "selection_or_further_training_allowed": False})
    data_record = read_json(run / "DATA_CREATED.json")
    assert digest(run / "evaluator_only/teacher.pt") == data_record["teacher_sha256"]
    truth = {}
    for split in ("test", "development"):
        rel = f"evaluator_only/{split}.pt"
        assert digest(run / rel) == {k.replace("\\", "/"): v for k, v in data_record["files"].items()}[rel]
        truth[split] = load_tensor(run / rel)
    teacher = LocalRetipathV0().requires_grad_(False).eval()
    teacher.load_state_dict(load_tensor(run / "evaluator_only/teacher.pt")["model_state"], strict=True)
    teacher_replay = trace_for(teacher, truth["test"])
    for name, value in teacher_replay.items():
        torch.testing.assert_close(value, truth["test"][name], rtol=0, atol=0)
    anchor = next(r for r in truth["test"]["records"] if r["scale"] == 0.3)
    teacher_rf = {str(scale): rf_matrix(teacher, anchor, scale) for scale in protocol["rf"]["history_scales_deg"]}
    save_tensor(run / "evaluation/teacher_rf.pt", teacher_rf)
    profile_records = {"teacher": {s: rf_profiles(rf) for s, rf in teacher_rf.items()}}
    rows, rf_rows, dev_rows = [], [], []
    cached_test = {}
    for condition in "ABCDE":
        for seed in protocol["student_seeds"]:
            identity = f"{condition}_{seed}"
            model = LocalRetipathV0().requires_grad_(False).eval()
            model.load_state_dict(load_tensor(run / f"fits/{identity}/final.pt")["model_state"], strict=True)
            for split in ("test", "development"):
                prediction = trace_for(model, truth[split])
                save_tensor(run / f"evaluation/{identity}_{split}.pt", prediction)
                if split == "test":
                    rows.extend(sequence_metrics(prediction, truth[split], condition, seed, split))
                    cached_test[identity] = prediction
                else:
                    dev_rows.extend(development_metrics(prediction, truth[split], condition, seed))
            rf = {str(scale): rf_matrix(model, anchor, scale) for scale in protocol["rf"]["history_scales_deg"]}
            save_tensor(run / f"evaluation/{identity}_rf.pt", rf)
            profile_records[identity] = {s: rf_profiles(value) for s, value in rf.items()}
            for scale in protocol["rf"]["history_scales_deg"]:
                rf_rows.append(rf_metrics(rf[str(scale)], teacher_rf[str(scale)], condition, seed, scale))
            print("EVALUATED", identity, "test/development/RF", flush=True)
    for row in rf_rows:
        key = f"{row['condition']}_{row['seed']}"
        common = profile_records[key]["0.3"]
        teacher_common = profile_records["teacher"]["0.3"]
        row["student_gain_change_from_common"] = row["student_gain"] - common["gain"]
        row["teacher_gain_change_from_common"] = row["teacher_gain"] - teacher_common["gain"]
        for name in ("spatial", "temporal"):
            here, ref = profile_records[key][str(row["history_scale"])][name], common[name]
            true_here, true_ref = profile_records["teacher"][str(row["history_scale"])][name], teacher_common[name]
            row[name + "_change_error"] = (rms((torch.tensor(here) - torch.tensor(ref)) - (torch.tensor(true_here) - torch.tensor(true_ref)))
                                            if all(v is not None for v in (here, ref, true_here, true_ref)) else None)
    per_fit = aggregate_rows(rows)
    write_csv(run / "per_sequence_metrics.csv", rows)
    write_csv(run / "per_fit_metrics.csv", per_fit)
    write_csv(run / "development_metrics.csv", dev_rows)
    write_csv(run / "rf_metrics.csv", rf_rows)
    write_json(run / "rf_profiles.json", profile_records)
    paired = []
    for comparison in protocol["evaluation"]["paired_comparisons"]:
        left, right = comparison.split("-")
        for seed in protocol["student_seeds"]:
            a = next(r for r in per_fit if r["condition"] == left and r["seed"] == seed and r["split"] == "test" and r["stratum"] == "heldout_scales")
            b = next(r for r in per_fit if r["condition"] == right and r["seed"] == seed and r["split"] == "test" and r["stratum"] == "heldout_scales")
            for metric in ("primary_logit_intervention_rmse", "excess_ce", "h_rmse", "s_B_rmse", "o_B_rmse"):
                paired.append({"comparison": comparison, "seed": seed, "metric": metric, "difference": a[metric] - b[metric]})
    write_csv(run / "paired_comparisons.csv", paired)
    ambiguity = []
    seeds = protocol["student_seeds"]
    for condition in "ABCDE":
        for i, first in enumerate(seeds):
            for second in seeds[i + 1:]:
                a, b = cached_test[f"{condition}_{first}"], cached_test[f"{condition}_{second}"]
                differences = {name: a[name] - b[name] for name in ("p", "h", "s_B", "o_B")}
                differences["logit_intervention"] = (a["block_ell"] - a["ell"]) - (b["block_ell"] - b["ell"])
                for name, scales in strata().items():
                    indices = [j for j, r in enumerate(truth["test"]["records"]) if r["scale"] in scales]
                    for metric, diff in differences.items():
                        value = statistics.mean(rms(diff[j, 60:]) for j in indices)
                        ambiguity.append({"condition": condition, "seed_a": first, "seed_b": second, "stratum": name, "metric": metric, "rms_distance": value})
    write_csv(run / "ambiguity.csv", ambiguity)
    write_json(run / "summary.json", summarize(per_fit, paired))
    write_json(run / "EVALUATION_COMPLETED.json", {"utc": utc(), "fits": 15, "per_sequence_rows": len(rows), "per_fit_rows": len(per_fit),
                                                   "rf_rows": len(rf_rows), "teacher_replay": "bitwise exact", "test_status": "CONSUMED"})


def verify(run: Path) -> None:
    protocol = check_sources(run)
    completed = read_json(run / "EVALUATION_COMPLETED.json")
    assert completed["fits"] == 15
    actual_schedule = read_json(run / "SCHEDULES.json")
    fairness = validate_schedules(read_json(run / "STIMULUS_MANIFEST.json"), actual_schedule)
    lock = read_json(run / "SOURCE_LOCK.json")
    assert digest(run / "STIMULUS_MANIFEST.json") == lock["manifest_sha256"]
    assert digest(run / "SCHEDULES.json") == lock["schedules_sha256"]
    test = load_tensor(run / "evaluator_only/test.pt")
    reconstructed = []
    replay_error = 0.0
    for condition in "ABCDE":
        initial_hashes = []
        for seed in protocol["student_seeds"]:
            identity = f"{condition}_{seed}"
            directory = run / f"fits/{identity}"
            ckpt = load_tensor(directory / "final.pt")
            assert ckpt["steps"] == 180 and sum(ckpt["microbatches"].values()) == 480
            assert ckpt["initial_sha256"] == digest(run / f"initial_students/{seed}.pt")
            assert digest(directory / "final.pt") == read_json(run / "FINAL_CHECKPOINTS_LOCK.json")["checkpoints"][identity]
            initial_hashes.append(ckpt["initial_sha256"])
            with (directory / "trajectory.csv").open(encoding="utf-8") as f:
                log = list(csv.DictReader(f))
            assert [int(r["step"]) for r in log] == list(range(1, 181))
            assert sum(int(r["microbatches"]) for r in log) == 480
            saved = load_tensor(run / f"evaluation/{identity}_test.pt")
            reconstructed.extend(sequence_metrics(saved, test, condition, seed, "test"))
            model = LocalRetipathV0().requires_grad_(False).eval()
            model.load_state_dict(ckpt["model_state"], strict=True)
            indices = list(range(8, 12))
            stimulus = inputs_from(test, indices)
            with torch.no_grad():
                normal = model(stimulus, observed_events=test["events"][indices])
                block = model(stimulus, observed_events=test["events"][indices], intervention=BLOCK)
            for name, value in {"p": normal.outputs["p"], "ell": normal.outputs["ell"], "block_ell": block.outputs["ell"]}.items():
                error = float((value - saved[name][indices]).abs().max())
                replay_error = max(replay_error, error)
                assert error == 0
            for name, parameter in model.state_dict().items():
                assert bool(torch.isfinite(parameter).all()), name
        assert len(set(initial_hashes)) == 3
    with (run / "per_fit_metrics.csv").open(encoding="utf-8") as f:
        original = {(r["condition"], int(r["seed"]), r["stratum"]): r for r in csv.DictReader(f) if r["split"] == "test"}
    for row in aggregate_rows(reconstructed):
        previous = original[(row["condition"], row["seed"], row["stratum"])]
        for key, value in row.items():
            if key not in ("condition", "seed", "stratum", "split", "sequences") and value is not None:
                assert math.isclose(value, float(previous[key]), rel_tol=1e-12, abs_tol=1e-14), (key, value, previous[key])
    with (run / "paired_comparisons.csv").open(encoding="utf-8") as f:
        pairs = list(csv.DictReader(f))
    recomputed_pairs = []
    for row in pairs:
        left, right = row["comparison"].split("-")
        a = original[(left, int(row["seed"]), "heldout_scales")]
        b = original[(right, int(row["seed"]), "heldout_scales")]
        difference = float(a[row["metric"]]) - float(b[row["metric"]])
        assert math.isclose(difference, float(row["difference"]), rel_tol=1e-12, abs_tol=1e-14)
        recomputed_pairs.append({**row, "difference": difference})
    assert summarize(aggregate_rows(reconstructed), recomputed_pairs) == read_json(run / "summary.json")
    anchor = next(r for r in test["records"] if r["scale"] == 0.3)
    teacher = LocalRetipathV0().requires_grad_(False).eval()
    teacher.load_state_dict(load_tensor(run / "evaluator_only/teacher.pt")["model_state"], strict=True)
    repeated_rf = rf_matrix(teacher, anchor, 0.3)
    torch.testing.assert_close(repeated_rf, load_tensor(run / "evaluation/teacher_rf.pt")["0.3"], rtol=0, atol=0)
    with (run / "rf_metrics.csv").open(encoding="utf-8") as f:
        rf_rows = list(csv.DictReader(f))
    assert len(rf_rows) == 75
    for row in rf_rows:
        identity = f"{row['condition']}_{row['seed']}"
        pred = load_tensor(run / f"evaluation/{identity}_rf.pt")[row["history_scale"]]
        true = load_tensor(run / "evaluation/teacher_rf.pt")[row["history_scale"]]
        values = rf_metrics(pred, true, row["condition"], int(row["seed"]), float(row["history_scale"]))
        for name, value in values.items():
            if isinstance(value, float):
                assert math.isclose(value, float(row[name]), rel_tol=1e-12, abs_tol=1e-14)
    write_json(run / "verification.json", {"status": "VERIFIED", "utc": utc(), "completed_fits": 15, "optimizer_steps": 2700,
                                          "microbatches": 7200, "fairness": fairness, "source_hashes_unchanged": True,
                                          "frozen_schedule_manifest_unchanged": True, "paired_initialization": True,
                                          "checkpoint_replay_max_abs_error": replay_error,
                                          "CSV_recomputed_from_saved_traces": True, "RF_CSV_recomputed_from_raw_RF": True,
                                          "teacher_RF_common_history_replay": "bitwise exact",
                                          "verification_test_access": "explicit replay of already consumed test; no selection or training"})
    paths = [f for f in run.rglob("*") if f.is_file()]
    write_json(run / "FILE_MANIFEST.json", {str(f.relative_to(run)).replace("\\", "/"): {"bytes": f.stat().st_size, "sha256": digest(f)} for f in paths})
    print("ALL_15_FITS_VERIFIED", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "generate", "train", "evaluate", "verify"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    run = args.run.resolve()
    if args.stage == "train":
        protocol = check_sources(run)
        if args.seed not in protocol["student_seeds"]:
            raise ValueError("training requires one frozen student seed")
        for condition in "ABCDE":
            train_fit(run, condition, args.seed)
    else:
        {"freeze": freeze, "generate": generate, "evaluate": evaluate, "verify": verify}[args.stage](run)


if __name__ == "__main__":
    main()
