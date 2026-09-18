from __future__ import annotations

import csv
from datetime import datetime
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

from adapter import OpenRetinaAdapted
from data_adapter import DEST, FINAL, ROOT, digest, exclusive_json, read_json
from run_assessment import verify_frozen
from training_engine import SEEDS, MAX_UPDATES, EVAL_EVERY, MIN_DELTA, PATIENCE_CHECKS


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    initial = verify_frozen()
    provenance = DEST / "checkpoints/provenance"
    execution = read_json(provenance / "execution_lock.json")
    protocol_sha = digest(DEST / "PROTOCOL.md")
    assert protocol_sha == execution["protocol_sha256"]
    assert digest(Path(__file__).with_name("run_assessment.py")) == execution["run_script_sha256"]
    completed = read_json(provenance / "run_complete.json")
    selection_lock = read_json(provenance / "selection_lock.json")
    evaluation_lock = read_json(provenance / "evaluation_lock.json")
    selection_time = datetime.fromisoformat(selection_lock["locked_utc"])
    evaluation_time = datetime.fromisoformat(evaluation_lock["locked_utc"])
    for result in completed["selections"]:
        assert result["protocol_sha256"] == protocol_sha
        assert digest(DEST / "checkpoints" / str(result["seed"]) / "selected.pt") == result["checkpoint_sha256"]
        assert datetime.fromisoformat(result["completed_utc"]) <= selection_time
        assert 0 <= result["selected_steps"] <= result["completed_steps"] <= MAX_UPDATES
        records = result["curve_records"]
        actual_best = min(records, key=lambda r: (r["inner_validation_nll"], r["step"]))
        assert actual_best["step"] == result["selected_steps"]
        assert result["all_finite"]
        plateau, last = math.inf, 0
        for record in records:
            assert record["step"] % EVAL_EVERY == 0
            if record["inner_validation_nll"] < plateau - MIN_DELTA:
                plateau, last = record["inner_validation_nll"], record["step"]
        if result["stop_reason"] == "inner_validation_patience":
            assert result["completed_steps"] - last >= PATIENCE_CHECKS * EVAL_EVERY
        else:
            assert result["completed_steps"] == MAX_UPDATES
    for result in completed["refits"]:
        assert result["protocol_sha256"] == protocol_sha
        assert digest(DEST / "checkpoints" / str(result["seed"]) / "refit.pt") == result["checkpoint_sha256"]
        assert datetime.fromisoformat(result["completed_utc"]) <= evaluation_time
        assert result["completed_steps"] == selection_lock["selected_steps"][str(result["seed"])]
        assert result["all_finite"]
        assert all(r["inner_validation_nll"] is None for r in result["curve_records"])
        selected = torch.load(DEST / "checkpoints" / str(result["seed"]) / "selected.pt", weights_only=True, map_location="cpu")
        refit = torch.load(DEST / "checkpoints" / str(result["seed"]) / "refit.pt", weights_only=True, map_location="cpu")
        assert selected["scheduler"]["total_steps"] == refit["scheduler"]["total_steps"] == MAX_UPDATES
        assert selected["scheduler"]["last_epoch"] == refit["scheduler"]["last_epoch"] == result["selected_steps"]
        assert selected["phase"] == "selection" and refit["phase"] == "refit"
        assert selected["protocol_sha256"] == refit["protocol_sha256"] == protocol_sha
    scores = csv_rows(DEST / "per_cell_scores.csv")
    summaries = csv_rows(DEST / "comparison_summary.csv")
    assert len(scores) == (22 + 22 + 17) * 4 and len(summaries) == 12
    sys.path.insert(0, str(ROOT / "work"))
    from retipath_final_prediction import load_split
    from retipath_final_common import identity
    nll_error, summary_error = 0.0, 0.0
    for start, stop, n in ((16, 20, 22), (20, 60, 22), (240, 300, 17)):
        label = f"[{start},{stop})"
        paired = [r for r in scores if r["range"] == label]
        cells = sorted({r["cell_id"] for r in paired})
        assert len(cells) == n
        for cell in cells:
            split = load_split(cell, start, stop)
            shared = identity(split)
            saved = np.load(DEST / "checkpoints/evaluation_logits" / f"{cell.replace('#', '_')}_{start}.npz", allow_pickle=False)
            cell_rows = [r for r in paired if r["cell_id"] == cell]
            for seed in SEEDS:
                row = next(r for r in cell_rows if r["seed"] == str(seed))
                assert all(row[k] == v for k, v in shared.items())
                logits = saved[str(seed)].astype(np.float64)
                targets = split.spike_events.numpy().astype(np.float64)
                mask = split.valid_mask.numpy()
                manual = np.logaddexp(0.0, logits) - logits * targets
                value = float(manual[mask].mean())
                nll_error = max(nll_error, abs(value - float(row["openretina_adapted_nll"])))
                assert abs(value - float(row["openretina_adapted_nll"])) < 1e-12
                assert abs(float(row["delta_nll"]) - (float(row["retipath_nll"]) - value)) < 1e-12
                assert int(row["scored_bins"]) == int(mask.sum())
            saved.close()
            agg = next(r for r in cell_rows if r["seed"] == "aggregate")
            for field in ("retipath_nll", "openretina_adapted_nll", "delta_nll"):
                expected = np.mean([float(r[field]) for r in cell_rows if r["seed"] != "aggregate"])
                assert abs(float(agg[field]) - expected) < 1e-12
        rng = np.random.default_rng(20260908)
        indices = rng.integers(0, n, size=(100000, n))
        for seed in (*map(str, SEEDS), "aggregate"):
            records = sorted((r for r in paired if r["seed"] == seed), key=lambda r: r["cell_id"])
            delta = np.array([float(r["delta_nll"]) for r in records])
            summary = next(r for r in summaries if r["range"] == label and r["seed"] == seed)
            low, high = np.quantile(delta[indices].mean(1), (.025, .975))
            expected = {"paired_mean": delta.mean(), "paired_median": np.median(delta), "ci_low": low, "ci_high": high,
                "retipath_absolute_nll": np.mean([float(r["retipath_nll"]) for r in records]),
                "openretina_adapted_absolute_nll": np.mean([float(r["openretina_adapted_nll"]) for r in records]),
                "wins_retipath": int((delta < -1e-7).sum()), "losses_retipath": int((delta > 1e-7).sum()),
                "ties": int((np.abs(delta) <= 1e-7).sum())}
            for name, value in expected.items():
                error = abs(float(summary[name]) - value)
                summary_error = max(summary_error, error)
                assert error < 1e-12
    cell = "67#14"
    split = load_split(cell, 16, 20)
    replays = []
    for seed in SEEDS:
        cp = torch.load(DEST / "checkpoints" / str(seed) / "refit.pt", weights_only=True, map_location="cpu")
        model = OpenRetinaAdapted(tuple(cp["cells"]), cp["training_occupancy_initialization"], seed).place()
        model.load_state_dict(cp["model"], strict=True)
        model.eval().requires_grad_(False)
        before = model.cpu_state()
        with torch.no_grad():
            actual = model(split.cone_drive[:1], split.spike_events[:1], cell)
            future = split.cone_drive[:1].clone(); future[:, 80:] += .25
            changed = model(future, split.spike_events[:1], cell)
            future_error = float((actual[:, :80] - changed[:, :80]).abs().max())
            assert future_error <= 1e-6
        with np.load(DEST / "checkpoints/evaluation_logits/67_14_16.npz", allow_pickle=False) as archive:
            cached = torch.from_numpy(archive[str(seed)][:1])
        torch.testing.assert_close(actual, cached, rtol=2e-5, atol=2e-6)
        assert all(torch.equal(v, model.state_dict()[k].cpu()) for k, v in before.items())
        assert all(float(model.history_coefficient(c)) > 0 for c in model.cells)
        replays.append({"seed": seed, "batch1_vs_cached_batch4_max_abs_error": float((actual - cached).abs().max()),
                        "trained_model_future_frame_max_abs_error": future_error, "weights_unchanged": True})
    recovery = None
    recovery_folder = provenance / "runtime_recovery_001"
    if recovery_folder.exists():
        recovery = read_json(recovery_folder / "recovery_plan.json")
        for entry in recovery["resume_records"]:
            seed = entry["seed"]
            assert digest(recovery_folder / f"{seed}_refit_latest.pt") == entry["checkpoint_sha256"]
            old = {int(line.split()[0].split("=")[1]): line for line in
                   (recovery_folder / f"{seed}_refit.log").read_text().splitlines() if line.startswith("update=")}
            current = {}
            for line in (DEST / "checkpoints" / str(seed) / "refit.log").read_text().splitlines():
                if line.startswith("update="):
                    current.setdefault(int(line.split()[0].split("=")[1]), []).append(line)
            repeated = range(entry["resume_step"] + 1, entry["last_logged_completed_update"] + 1)
            assert all(len(current[step]) == 2 and current[step][1] == old[step] for step in repeated)
            assert max(current) == entry["selected_target_updates"]
        preserved = read_json(recovery_folder / "preserved_completed_refit.json")
        assert digest(DEST / "checkpoints" / str(preserved["seed"]) / "refit.pt") == preserved["checkpoint_sha256"]
        recovery["all_repeated_logged_loss_regularizer_gradient_values_identical"] = True
        recovery["all_original_selected_endpoints_preserved"] = True
        recovery["full_parameter_state_at_original_step67_not_saved"] = True
    final = read_json(DEST / "correctness.json")
    final.update({"status": "VERIFIED", "formal_training_started": True,
        "formal_training_and_evaluation_complete": True, "all_selected_steps_inner_validation_only": True,
        "all_refits_completed_before_scoring": True, "NLL_independent_numpy_max_abs_error": nll_error,
        "comparison_and_bootstrap_independent_max_abs_error": summary_error,
        "frozen_trained_model_replays": replays, "retipath_parameters_unchanged": True,
        "new_retipath_updates": 0, "new_target_blocks": 0, "new_RF_or_illusion_inference": 0,
        "runtime_recovery": recovery,
        "verification_scope": "author source fidelity, adapter causality/serialization/gradient checks, frozen training/selection rules, paired scoring and summaries"})
    (DEST / "correctness.json").write_text(json.dumps(final, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    verify_frozen()
    exclusive_json(provenance / "verification_complete.json", {"status": "VERIFIED", "script_sha256": digest(Path(__file__)),
        "NLL_error": nll_error, "summary_error": summary_error, "frozen_trained_replays": replays,
        "runtime_recovery": recovery})
    print(f"VERIFIED: {len(scores)} paired cell/seed rows, {len(summaries)} summaries; max NLL error={nll_error:.3g}")


if __name__ == "__main__":
    main()
