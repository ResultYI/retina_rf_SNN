# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.6.0"]
# ///
# Frozen inference replay: D:/anaconda/python.exe -B replay.py
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Final

import torch

OUT: Final = Path(__file__).resolve().parent
ROOT: Final = OUT.parents[2]
OLD_DIAGNOSTIC: Final = ROOT / "output/real_data/schottdorf_ln_r4_illusion_diagnostics_20260830"
OLD_ORIGINAL: Final = ROOT / "output/real_data/schottdorf_r4_dev_visual_illusions_20260830"
NEW_CHECKPOINTS: Final = ROOT / "output/real_data/schottdorf_canonical_v1_overlapping_support_development_22cell_20260830"
sys.path.insert(0, str(OLD_DIAGNOSTIC))
sys.path.insert(0, str(OLD_ORIGINAL))
sys.path.insert(0, str(ROOT))

from diagnostic_stimuli import VARIANTS, diagnostic_bank
from metrics import aggregate, cell_rows, mach_rows, response_metrics, write_csv
from stimuli import DT_MS
from models.mechanistic_retina.contracts import ArchitectureMode, MechanisticRetinaConfig, PathwayClamp
from models.mechanistic_retina.model import build_mechanistic_retina

CLAMPS: Final = {
    "normal": frozenset(),
    "H1_off": frozenset({PathwayClamp.H1}),
    "AC_off": frozenset({PathwayClamp.AMACRINE_LOCAL, PathwayClamp.AMACRINE_TRANSIENT}),
}
OLD_RESPONSE_KEY: Final = {"normal": "R4_normal", "H1_off": "R4_H1_off", "AC_off": "R4_AC_off"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pair_trace(response: dict[str, torch.Tensor], channel: str, left: int, right: int) -> torch.Tensor:
    return response[channel][left] - response[channel][right]


def main() -> None:
    assert not (OUT / "results.json").exists(), "fresh output directory required"
    torch.set_num_threads(2)
    bank = diagnostic_bank()
    inputs = torch.load(OLD_DIAGNOSTIC / "inputs.pt", weights_only=True)
    old = torch.load(OLD_DIAGNOSTIC / "responses.pt", weights_only=True)
    old_input = torch.load(OLD_ORIGINAL / "input-tensors.pt", weights_only=True)
    old_stimuli = torch.load(OLD_ORIGINAL / "stimuli.pt", weights_only=True)
    assert torch.equal(inputs["patches"], bank.patches)
    assert tuple(inputs["names"]) == bank.names
    assert torch.equal(inputs["original_drive"], old_input["cone_drive"])
    assert torch.equal(inputs["original_history"], old_input["history"])
    drive = torch.cat((inputs["original_drive"], inputs["diagnostic_drive"]))
    history = torch.cat((inputs["original_history"], inputs["diagnostic_history"]))
    time_ms = inputs["time_ms"]
    assert drive.shape == (len(bank.names), 150, 289)
    assert history.shape == (len(bank.names), 150, 1)
    assert tuple(old["names"]) == bank.names
    checkpoint_results = json.loads((NEW_CHECKPOINTS / "results.json").read_text(encoding="utf-8"))
    checkpoint_cells = {row["cell_id"]: row for row in checkpoint_results["cells"]}
    assert len(checkpoint_cells) == len(old["metadata"]) == 22
    source_paths = (
        tuple((ROOT / "models/mechanistic_retina").glob("*.py"))
        + tuple(OUT.glob("*.py"))
        + (OLD_DIAGNOSTIC / "inputs.pt", OLD_DIAGNOSTIC / "responses.pt", OLD_DIAGNOSTIC / "protocol.json",
           OLD_ORIGINAL / "input-tensors.pt", OLD_ORIGINAL / "stimuli.pt", OLD_ORIGINAL / "stimulus-contract.json",
           NEW_CHECKPOINTS / "results.json")
    )
    hashes = {str(path): sha256(path) for path in source_paths}
    cells, rows, comparisons, boundaries, checks = {}, [], [], [], []
    for index, (cell_id, old_metadata) in enumerate(old["metadata"].items()):
        cell_dir = NEW_CHECKPOINTS / "cells" / cell_id.replace("#", "_")
        checkpoint_path = cell_dir / "model-trained.pt"
        hashes[str(checkpoint_path)] = sha256(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        current_metadata = checkpoint_cells[cell_id]
        group = f"{current_metadata['retinal_class']}_{current_metadata['polarity']}"
        assert group == old_metadata["group"]
        config = dict(checkpoint["model_config"])
        assert config["spatial_contract"] == "bc-central-disk_ac-overlapping-full-disk"
        assert abs(float(config["dt_ms"]) - DT_MS) < 1e-10
        config["architecture_mode"] = ArchitectureMode(config["architecture_mode"])
        assert torch.allclose(checkpoint["cone_positions_degs"], old_stimuli["cone_positions_degs"], atol=1e-7, rtol=0.0)
        model = build_mechanistic_retina(
            MechanisticRetinaConfig(**config), checkpoint["cone_positions_degs"],
            checkpoint["cell_positions_degs"], checkpoint["cell_types"], checkpoint["polarities"],
        )
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        response = {}
        for mode, clamps in CLAMPS.items():
            with torch.no_grad():
                result = model.forward_sequence(drive, observed_counts=history, clamps=clamps)
            response[mode] = {
                "logit": result.logits[..., 0],
                "probability": result.spike_probability[..., 0],
                "ac_local": result.amacrine_local_current[..., 0],
                "ac_transient": result.amacrine_transient_current[..., 0],
            }
            assert all(torch.isfinite(value).all() for value in response[mode].values())
            if mode == "H1_off":
                assert torch.count_nonzero(result.h1_surround_contribution) == 0
            if mode == "AC_off":
                assert torch.count_nonzero(result.amacrine_local_current) == 0
                assert torch.count_nonzero(result.amacrine_transient_current) == 0
            for pair in bank.pairs:
                if pair.control:
                    assert torch.equal(result.logits[pair.a], result.logits[pair.b])
        with torch.no_grad():
            repeat = model.forward_sequence(drive, observed_counts=history)
        assert torch.equal(repeat.logits[..., 0], response["normal"]["logit"])
        assert all(torch.equal(value, checkpoint["model"][name]) for name, value in model.state_dict().items())
        assert all(parameter.grad is None for parameter in model.parameters())
        cells[cell_id] = response
        rows.extend(cell_rows(cell_id, group, bank, response, time_ms))
        boundaries.extend(mach_rows(cell_id, group, bank, response, time_ms))
        for mode, response_values in response.items():
            old_values = old["cells"][cell_id][OLD_RESPONSE_KEY[mode]]
            for pair in bank.pairs:
                for channel in ("logit", "probability"):
                    delta = pair_trace(response_values, channel, pair.a, pair.b) - pair_trace(old_values, channel, pair.a, pair.b)
                    comparisons.append({
                        "cell_id": cell_id, "group": group, "mode": mode, "channel": channel,
                        "family": pair.family, "name": pair.name, "kind": "overlapping_minus_exclusive_pair",
                    } | response_metrics(delta, time_ms))
            for family, variant, original_a, original_b, _, _ in VARIANTS:
                diagnostic_pair = next(pair for pair in bank.pairs if pair.name == f"{variant}_A_minus_B")
                original_pair = next(pair for pair in bank.pairs if pair.family == family and not pair.control and pair.a == original_a and pair.b == original_b)
                for channel in ("logit", "probability"):
                    new_delta = pair_trace(response_values, channel, diagnostic_pair.a, diagnostic_pair.b) - pair_trace(response_values, channel, original_pair.a, original_pair.b)
                    old_delta = pair_trace(old_values, channel, diagnostic_pair.a, diagnostic_pair.b) - pair_trace(old_values, channel, original_pair.a, original_pair.b)
                    comparisons.append({
                        "cell_id": cell_id, "group": group, "mode": mode, "channel": channel,
                        "family": family, "name": variant, "kind": "new_diagnostic_minus_original",
                    } | response_metrics(new_delta, time_ms))
                    comparisons.append({
                        "cell_id": cell_id, "group": group, "mode": mode, "channel": channel,
                        "family": family, "name": variant, "kind": "old_diagnostic_minus_original",
                    } | response_metrics(old_delta, time_ms))
        checks.append({
            "cell_id": cell_id, "group": group, "checkpoint_state_unchanged": True,
            "normal_reentry_bitwise_equal": True, "all_gradients_none": True,
            "H1_clamp_exact_zero": True, "AC_clamp_exact_zero": True, "paired_controls_bitwise_equal": True,
        })
        print(f"{index + 1}/22 {cell_id}: frozen inference checks PASS", flush=True)
    assert all(sha256(Path(path)) == digest for path, digest in hashes.items())
    torch.save({"cells": cells, "metadata": old["metadata"], "time_ms": time_ms, "names": bank.names}, OUT / "response-tensors.pt")
    write_csv(OUT / "per-cell-responses.csv", rows)
    write_csv(OUT / "group-responses.csv", aggregate(rows))
    write_csv(OUT / "per-cell-comparisons.csv", comparisons)
    write_csv(OUT / "group-comparisons.csv", aggregate(comparisons))
    write_csv(OUT / "mach-boundary-extrema.csv", boundaries)
    payload = {
        "source": {"old_exclusive_annulus": str(OLD_DIAGNOSTIC), "new_overlapping_support": str(NEW_CHECKPOINTS)},
        "stimulus_contract": str(OLD_ORIGINAL / "stimulus-contract.json"),
        "diagnostic_contract": str(OLD_DIAGNOSTIC / "protocol.json"),
        "cell_count": 22, "conditions": list(CLAMPS), "dt_ms": float(time_ms[1] - time_ms[0]),
        "groups": aggregate(rows), "comparisons": aggregate(comparisons), "checks": checks,
        "all_sources_unchanged": True, "training": False, "checkpoint_writes": 0,
    }
    (OUT / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (OUT / "verification.json").write_text(json.dumps({"checks": checks, "source_sha256": hashes,
        "all_sources_unchanged": True, "training": False, "checkpoint_writes": 0}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
