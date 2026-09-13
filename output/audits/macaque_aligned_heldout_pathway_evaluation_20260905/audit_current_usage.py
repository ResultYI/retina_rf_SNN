# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.6.0", "numpy", "opencv-python", "pydantic"]
# ///
# How to run: D:/anaconda/python.exe -B audit_current_usage.py
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Final

import cv2
import torch

ROOT: Final = Path(__file__).resolve().parents[3]
OUT: Final = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from data.schottdorf_lee_catalog import mc_pc_recordings
from data.schottdorf_lee_spikes import parse_recording_spike_trials


def write_csv(name: str, records: list[dict]) -> None:
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def tensor_sha(value: torch.Tensor) -> str:
    return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()


def main() -> None:
    torch.set_num_threads(2)
    manifest = json.loads((OUT / "evidence_manifest.initial.json").read_text())
    alignment_manifest = json.loads((ROOT / manifest["lineages"]["fixed_alignment_audit"] / "analysis-manifest.json").read_text())
    trained = [p.replace("\\", "/") for p in alignment_manifest["inputs_sha256"] if p.endswith("model-trained.pt")]
    aligned_roots = {str(Path(p).parents[2]).replace("\\", "/") for p in trained}
    assert len(aligned_roots) == 1 and len(trained) == 22
    aligned = ROOT / aligned_roots.pop()
    zero = ROOT / manifest["lineages"]["zero_center_reference"]
    ln = ROOT / manifest["lineages"]["LN"]
    cnn = ROOT / manifest["lineages"]["CNN"]
    populations = {label: json.loads((folder / "results.json").read_text()) for label, folder in
                   (("aligned", aligned), ("zero", zero), ("LN", ln), ("CNN", cnn))}
    cells = populations["zero"]["cells"]
    assert len(cells) == 22
    records = mc_pc_recordings(ROOT / "data/real/schottdorf_lee_2021_repository/data")
    assert len(records) == 37 and {r.recording_id for r in records} == {r for c in cells for r in c["recording_ids"]}
    observed, artifacts, checkpoints = [], [], []
    pattern = re.compile(r"(lSS\d+)-live-frames-(\d+)-(\d+)-trial-(\d+)")
    for cell in cells:
        cid, name = cell["cell_id"], cell["cell_id"].replace("#", "_")
        source = cnn / "inputs" / f"{name}.pt"
        bundle = torch.load(source, map_location="cpu", weights_only=True)
        assert bundle["cell_id"] == cid
        for split_name in ("train", "validation"):
            split = bundle[split_name]
            assert split["cone_drive"].shape[1:] == (150, 289)
            assert torch.equal(split["spike_events"], (split["spike_counts"] > 0).float())
            mask = split["valid_mask"]
            assert not mask[:, :30].any() and mask[:, 30:].all()
            assert int(mask.sum()) == cell["train_valid_bins" if split_name == "train" else "validation_valid_bins"]
            for index, (identity, trial_global) in enumerate(zip(split["source_image_ids"], split["trial_indices"], strict=True)):
                match = pattern.fullmatch(identity)
                assert match is not None
                recording, start, stop, trial_local = match.groups()
                start, stop = int(start), int(stop)
                assert stop - start == 149
                assert (0 <= start < 2400) if split_name == "train" else (2400 <= start < 3000)
                observed.append({"cell_id": cid, "recording_id": recording, "trial_local_one_based": int(trial_local),
                    "trial_cell_global_zero_based": trial_global, "split": split_name, "sequence_index": index,
                    "live_start_inclusive": start, "live_stop_exclusive": stop + 1,
                    "scored_start_inclusive": start + 30, "scored_stop_exclusive": stop + 1,
                    "scored_bins": int(mask[index].sum()), "source_image_id": identity,
                    "source_artifact": source.relative_to(ROOT).as_posix()})
        for label, folder, filename in (("zero", zero, "model-trained.pt"), ("aligned", aligned, "model-trained.pt"),
                                       ("LN", ln, "ln-trained.pt"), ("CNN", cnn, "cnn-trained.pt")):
            path = folder / "cells" / name / filename
            cp = torch.load(path, map_location="cpu", weights_only=True)
            assert cp["cell_id"] == cid
            if label in {"zero", "aligned"}:
                assert cp["model_name"] == "Canonical V1" and cp["revision"] == 4 and cp["stage"] == "trained"
                assert cp["model_config"]["causal_contract"] == "h1-shared-bc-direct-broad-ac"
                assert cp["model_config"]["spatial_contract"] == "bc-central-disk_ac-overlapping-full-disk"
                assert cp["steps"] == cp["training_contract"]["fresh_full_train_refit_steps"]
                count = next(c["parameter_counts"]["requires_grad"] for c in populations[label]["cells"] if c["cell_id"] == cid)
                assert count == 33
                center_source = torch.load(ln / "cells" / name / "ln-trained.pt", weights_only=True)["model"]["center_xy"]
                expected = torch.zeros((1, 2)) if label == "zero" else (center_source * torch.tensor([3 * 4.6 / 256, -3 * 4.6 / 256])).reshape(1, 2)
                assert torch.equal(cp["cell_positions_degs"], expected)
                steps = cp["steps"]
            else:
                assert cp["refit_steps"] == cp["best_step"]
                count = 128 if label == "LN" else 2990
                steps = cp["refit_steps"]
            checkpoints.append({"cell_id": cid, "model": label, "path": path.relative_to(ROOT).as_posix(),
                "metadata_identity_pass": True, "fresh_refit_steps": steps, "reported_trainable_scalars": count,
                "strict_model_load": "NOT_RUN_IN_USAGE_AUDIT", "center_trainable": False if label in {"zero", "aligned"} else "not applicable"})
            predicted_path = folder / "cells" / name / "validation-predictions.pt"
            saved = torch.load(predicted_path, map_location="cpu", weights_only=True)
            validation = bundle["validation"]
            assert torch.equal(saved["target"], validation["spike_events"])
            assert torch.equal(saved["valid_mask"], validation["valid_mask"])
            assert tuple(saved["source_image_ids"]) == tuple(validation["source_image_ids"])
            assert tuple(saved["trial_indices"]) == tuple(validation["trial_indices"])
            artifacts.append({"cell_id": cid, "model": label, "path": predicted_path.relative_to(ROOT).as_posix(),
                "start_frame": 2400, "stop_exclusive": 3000, "target_shape": list(saved["target"].shape),
                "target_sha256": tensor_sha(saved["target"]), "mask_sha256": tensor_sha(saved["valid_mask"]),
                "source_order_sha256": hashlib.sha256(json.dumps(saved["source_image_ids"]).encode()).hexdigest(),
                "target_mask_order_equal": True, "new_predictions_computed": False})
    recording_rows = []
    for record in records:
        parsed = parse_recording_spike_trials(record)
        duration = 60 if record.recording_kind.value == "6x1min" else 600
        trial_count = len(parsed.live_times_ms_by_trial)
        assert trial_count == (6 if duration == 60 else 1)
        saved_trials = {r["trial_local_one_based"] for r in observed if r["recording_id"] == record.recording_id}
        assert saved_trials == set(range(1, trial_count + 1))
        recording_rows.append({"recording_id": record.recording_id, "cell_id": record.cell_id,
            "recording_kind": record.recording_kind.value, "catalog_kind": record.catalog_recording_kind.value,
            "trial_count": trial_count, "live_duration_s_per_trial": duration,
            "candidate_start_s": 20, "candidate_stop_s_exclusive": 60,
            "headers_and_repeat_counts_valid": True, "duplicate_payload_removed_by_existing_parser": parsed.duplicate_payload_removed,
            "path": record.path.relative_to(ROOT).as_posix(), "only_source_validation_not_candidate_statistics": True})
    movie = ROOT / "data/real/schottdorf_lee_2021_macaque/1x10_256.mpg"
    capture = cv2.VideoCapture(str(movie))
    assert capture.isOpened()
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    assert frame_count >= 751 + 9000
    write_csv("production_source_ranges.csv", observed)
    write_csv("recording_trial_inventory.csv", recording_rows)
    write_csv("model_artifact_inventory.csv", checkpoints)
    (OUT / "saved_prediction_identity.json").write_text(json.dumps(artifacts, indent=2), encoding="utf-8")
    summary = {"cell_count": len(cells), "recording_count": len(records), "trial_count": sum(r["trial_count"] for r in recording_rows),
        "production_train_live_frames": [0, 2400], "production_validation_live_frames": [2400, 3000], "bounds": "half-open",
        "warmup_bins_per_sequence": 30, "scored_bins_per_sequence": 120,
        "saved_prediction_families_checked": 4, "saved_prediction_files_checked": len(artifacts), "all_target_mask_order_checks_pass": True,
        "checkpoint_metadata_files_checked": len(checkpoints), "aligned_root_resolved_from_analysis_manifest": aligned.relative_to(ROOT).as_posix(),
        "candidate_physical_availability": True, "common_movie_duration_s": min(r["live_duration_s_per_trial"] for r in recording_rows),
        "movie_decoded_frame_count_metadata": frame_count, "frozen_live_start_frame": 751,
        "new_inference_runs": 0, "new_metrics": 0, "data_usage_clean_gate": "NOT_YET_ADJUDICATED"}
    (OUT / "production_usage.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
