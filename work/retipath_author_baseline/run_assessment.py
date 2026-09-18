from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from adapter import OpenRetinaAdapted
from data_adapter import DEST, FINAL, ROOT, digest, exclusive_json, make_bank, read_json, training_banks
from training_engine import SEEDS, evaluate, fit, write_training_summary

RANGES = ((16, 20), (20, 60), (240, 300))


def verify_frozen() -> dict:
    lock = read_json(DEST / "checkpoints/provenance/pretraining_lock.json")
    for path, expected in lock["frozen_sha256"].items():
        if digest(ROOT / path) != expected:
            raise ValueError(f"Frozen source changed: {path}")
    for cell, records in lock["retipath_identity"].items():
        for ref in records.values():
            if digest(ROOT / ref["path"]) != ref["sha256"]:
                raise ValueError(f"Frozen RetiPath changed: {cell}")
    for entry in lock["data"].values():
        if digest(ROOT / entry["input_path"]) != entry["input_sha256"]:
            raise ValueError("Frozen training input archive changed")
    return lock


def train_worker(seed: int, phase: str, selected: int | None) -> dict:
    banks, _ = training_banks()
    return fit(seed, phase, banks["inner_fit" if phase == "selection" else "refit"],
               banks["inner_validation"] if phase == "selection" else None, selected)


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def temporal_evaluation() -> tuple[list[dict], list[dict], dict]:
    lock_path = DEST / "checkpoints/provenance/evaluation_lock.json"
    lock = read_json(lock_path)
    models = {}
    for seed in SEEDS:
        path = DEST / "checkpoints" / str(seed) / "refit.pt"
        assert digest(path) == lock["refit_sha256"][str(seed)]
        cp = torch.load(path, weights_only=True, map_location="cpu")
        model = OpenRetinaAdapted(tuple(cp["cells"]), cp["training_occupancy_initialization"], seed).place()
        model.load_state_dict(cp["model"], strict=True)
        model.eval().requires_grad_(False)
        models[seed] = model
    sys.path.insert(0, str(ROOT / "work"))
    from retipath_final_prediction import load_split
    from retipath_final_common import identity
    registry = read_json(FINAL / "model_registry.json")
    prior_lock = read_json(FINAL / "source_lock.json")
    eligible = {r["cell_id"] for r in prior_lock["continuous_cells"]}
    rows, summaries, contracts = [], [], []
    saved_logits = DEST / "checkpoints/evaluation_logits"
    saved_logits.mkdir(exist_ok=False)
    maximum_cache_error = 0.0
    for start, stop in RANGES:
        label = f"[{start},{stop})"
        cells = sorted(c for c in registry["cells"] if start < 240 or c in eligible)
        splits = {cell: load_split(cell, start, stop) for cell in cells}
        bank = make_bank(splits, label)
        adapted_scores, adapted_logits = {}, {}
        for seed, model in models.items():
            before = model.cpu_state()
            _, adapted_scores[seed], adapted_logits[seed] = evaluate(model, bank)
            assert all(torch.equal(v, model.state_dict()[k].cpu()) for k, v in before.items())
        for cell in cells:
            safe = cell.replace("#", "_")
            cached = read_json(FINAL / "prediction/cells" / f"{safe}.json")
            shared = identity(splits[cell])
            expected_records = [r for r in cached["rows"] if r["range"] == label and r["model"] == "RetiPath"]
            assert len(expected_records) == 3
            archive = np.load(FINAL / "prediction/cells" / f"{safe}.npz", allow_pickle=False)
            saved = {}
            for seed in SEEDS:
                original = next(r for r in expected_records if int(r["seed"]) == seed)
                assert all(shared[k] == original[k] for k in shared), (cell, label, "DATA MISMATCH")
                assert original["checkpoint_sha256"] == registry["cells"][cell]["RetiPath"][str(seed)]["sha256"]
                retipath_logits = torch.from_numpy(archive[original["logits_key"]])
                mask, target = splits[cell].valid_mask, splits[cell].spike_events
                replay = float(F.binary_cross_entropy_with_logits(retipath_logits.double()[mask], target.double()[mask]))
                maximum_cache_error = max(maximum_cache_error, abs(replay - original["nll"]))
                assert abs(replay - original["nll"]) < 1e-12
                r_score, o_score = original["nll"], adapted_scores[seed][cell]
                rows.append({"range": label, "cell_id": cell, "group": registry["cells"][cell]["group"],
                    "seed": seed, "retipath_nll": r_score, "openretina_adapted_nll": o_score,
                    "delta_nll": r_score - o_score, "scored_bins": int(mask.sum()),
                    "sequences": len(mask), **shared, "retipath_checkpoint_sha256": original["checkpoint_sha256"],
                    "openretina_checkpoint_sha256": lock["refit_sha256"][str(seed)],
                    "evidence_status": "descriptive temporal evaluation of previously consumed data"})
                saved[str(seed)] = adapted_logits[seed][cell].numpy()
            archive.close()
            with (saved_logits / f"{safe}_{start}.npz").open("xb") as stream:
                np.savez_compressed(stream, **saved)
            selected_rows = [r for r in rows if r["range"] == label and r["cell_id"] == cell]
            rows.append({"range": label, "cell_id": cell, "group": registry["cells"][cell]["group"],
                "seed": "aggregate", "retipath_nll": np.mean([r["retipath_nll"] for r in selected_rows]),
                "openretina_adapted_nll": np.mean([r["openretina_adapted_nll"] for r in selected_rows]),
                "delta_nll": np.mean([r["delta_nll"] for r in selected_rows]), "scored_bins": int(mask.sum()),
                "sequences": len(mask), **shared, "evidence_status": "cell-wise mean of three seed losses; no probability ensemble"})
            contracts.append({"range": label, "cell_id": cell, **shared,
                              "source_ids": list(splits[cell].source_image_ids),
                              "trial_indices": list(splits[cell].trial_indices)})
        draws = np.random.default_rng(20260908).integers(0, len(cells), (100000, len(cells)))
        for seed in (*SEEDS, "aggregate"):
            per_cell = sorted((r for r in rows if r["range"] == label and r["seed"] == seed), key=lambda r: r["cell_id"])
            values = np.array([r["delta_nll"] for r in per_cell])
            low, high = np.quantile(values[draws].mean(1), (0.025, 0.975))
            summaries.append({"range": label, "seed": seed, "n_cells": len(cells),
                "retipath_absolute_nll": np.mean([r["retipath_nll"] for r in per_cell]),
                "openretina_adapted_absolute_nll": np.mean([r["openretina_adapted_nll"] for r in per_cell]),
                "paired_mean": values.mean(), "paired_median": np.median(values),
                "wins_retipath": int((values < -1e-7).sum()), "losses_retipath": int((values > 1e-7).sum()),
                "ties": int((np.abs(values) <= 1e-7).sum()), "ci_low": low, "ci_high": high,
                "bootstrap_samples": 100000, "bootstrap_unit": "cell after averaging seeds within cell for aggregate",
                "evidence_status": "descriptive temporal evaluation"})
        print(f"EVALUATION {label}: {len(cells)} cells, all 3 frozen seeds scored", flush=True)
    write_csv(DEST / "per_cell_scores.csv", rows)
    write_csv(DEST / "comparison_summary.csv", summaries)
    evidence = {"range_contracts": contracts, "retipath_cache_manual_nll_max_abs_error": maximum_cache_error,
                "all_refit_weights_unchanged_by_evaluation": True, "new_target_blocks": 0,
                "scoring_after_all_refit_checkpoints_frozen": True, "per_cell_rows": len(rows),
                "comparison_summary_rows": len(summaries)}
    exclusive_json(DEST / "checkpoints/provenance/evaluation_complete.json", evidence)
    return rows, summaries, evidence


def main() -> None:
    verify_frozen()
    provenance = DEST / "checkpoints/provenance"
    execution = provenance / "execution_lock.json"
    if not execution.exists():
        exclusive_json(execution, {"started_utc": datetime.now(timezone.utc).isoformat(),
            "run_script_sha256": digest(Path(__file__)), "parallel_seed_processes": 3,
            "protocol_sha256": digest(DEST / "PROTOCOL.md")})
    else:
        assert read_json(execution)["run_script_sha256"] == digest(Path(__file__))
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(train_worker, seed, "selection", None) for seed in SEEDS]
        selections = [future.result() for future in futures]
    selection_path = provenance / "selection_lock.json"
    if not selection_path.exists():
        exclusive_json(selection_path, {"locked_utc": datetime.now(timezone.utc).isoformat(),
            "selected_steps": {str(r["seed"]): r["selected_steps"] for r in selections},
            "selected_checkpoint_sha256": {str(r["seed"]): r["checkpoint_sha256"] for r in selections},
            "selection_source": "inner validation only; one architecture and one training/regularization config"})
    write_training_summary(selections)
    print("ALL SELECTIONS LOCKED; starting fresh full-training refits", flush=True)
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(train_worker, r["seed"], "refit", r["selected_steps"]) for r in selections]
        refits = [future.result() for future in futures]
    write_training_summary(selections + refits)
    verify_frozen()
    evaluation_path = provenance / "evaluation_lock.json"
    if not evaluation_path.exists():
        exclusive_json(evaluation_path, {"locked_utc": datetime.now(timezone.utc).isoformat(),
            "refit_sha256": {str(r["seed"]): r["checkpoint_sha256"] for r in refits},
            "selected_steps": {str(r["seed"]): r["selected_steps"] for r in refits},
            "protocol_sha256": digest(DEST / "PROTOCOL.md"), "ranges": [list(r) for r in RANGES],
            "all_selection_and_training_complete_before_evaluation": True})
    print("ALL REFITS FROZEN; beginning the three consumed-range evaluations", flush=True)
    _, _, evaluation = temporal_evaluation()
    verify_frozen()
    exclusive_json(provenance / "run_complete.json", {"completed_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.perf_counter() - start, "selections": selections, "refits": refits,
        "new_retipath_updates": 0, "new_target_blocks": 0,
        "status": "TRAINING_AND_FROZEN_EVALUATION_COMPLETE"})
    print("TRAINING AND THREE-RANGE EVALUATION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
