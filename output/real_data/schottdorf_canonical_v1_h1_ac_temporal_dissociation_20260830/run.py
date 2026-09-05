#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.6.0", "matplotlib", "numpy"]
# ///
# How to run: D:/anaconda/python.exe -B run.py (frozen repository environment).
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Final

import torch

OUT: Final = Path(__file__).resolve().parent
ROOT: Final = OUT.parents[2]
SOURCE: Final = ROOT / "output/real_data/schottdorf_canonical_v1_overlapping_support_development_22cell_20260830"
sys.path.insert(0, str(ROOT))

from models.mechanistic_retina.contracts import ArchitectureMode, MechanisticRetinaConfig, PathwayClamp
from models.mechanistic_retina.model import build_mechanistic_retina
from probe import ACTIVE_MS, CONTRAST, DT_MS, EVENT_MS, NAMES, ONSET_MS, REFERENCES, build_bank, verify_bank
from report import create_report

CLAMPS: Final = {"normal": frozenset(), "H1_off": frozenset({PathwayClamp.H1}),
                 "AC_off": frozenset({PathwayClamp.AMACRINE_LOCAL, PathwayClamp.AMACRINE_TRANSIENT})}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Execute the input-frozen protocol, with unchanged trained states."""
    assert not (OUT / "protocol.json").exists(), "fresh output required"
    torch.set_num_threads(2)
    metadata = json.loads((SOURCE / "results.json").read_text())["cells"]
    assert len(metadata) == 22
    paths = tuple((ROOT / "models/mechanistic_retina").glob("*.py")) + tuple(OUT.glob("*.py"))
    paths += (ROOT / "evaluation/mechanistic_retina/temporal_center_surround.py", SOURCE / "results.json")
    hashes = {str(path): sha(path) for path in paths}
    models, banks, checkpoints, groups = {}, {}, {}, {}
    for cell in metadata:
        cid = cell["cell_id"]
        path = SOURCE / "cells" / cid.replace("#", "_") / "model-trained.pt"
        hashes[str(path)] = sha(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        config = checkpoint["model_config"] | {"architecture_mode": ArchitectureMode.MECHANISM_IDENTIFIABLE}
        assert config["spatial_contract"] == "bc-central-disk_ac-overlapping-full-disk"
        assert abs(config["dt_ms"] - DT_MS) < 1e-10
        model = build_mechanistic_retina(MechanisticRetinaConfig(**config), checkpoint["cone_positions_degs"],
                                        checkpoint["cell_positions_degs"], checkpoint["cell_types"], checkpoint["polarities"])
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        bank = build_bank((model.feature_bank.bc_support[0], model.feature_bank.ac_support[0]), cell["polarity"])
        verify_bank(bank)
        group = f"{cell['retinal_class']}_{cell['polarity']}"
        if group in banks:
            assert torch.equal(banks[group].drive, bank.drive)
        banks[group] = bank
        groups[cid] = group
        models[cid], checkpoints[cid] = model, checkpoint
    inputs = {group: asdict(bank) | {"intervals": [[asdict(e) for e in events] for events in bank.intervals]}
              for group, bank in banks.items()}
    torch.save(inputs, OUT / "inputs.pt")
    input_hash = sha(OUT / "inputs.pt")
    protocol = {
        "source": str(SOURCE), "names": list(NAMES), "dt_ms": DT_MS, "time_bins": 450,
        "onset_ms": ONSET_MS, "active_ms": ACTIVE_MS, "total_ms": 3000,
        "L_plus_M": "L(t)/L0 = 1 + Weber cone_drive; common pedestal L0=1; polarity matched center",
        "weber_peak_contrast": CONTRAST, "periodic_waveform": "bipolar square wave, 50% duty; 1,2,10,20 Hz",
        "slow_step": "+contrast for 1000 ms then -contrast for 1000 ms",
        "transient": "+contrast 50 ms at 300 ms; counterbalancing -contrast 50 ms at 1300 ms",
        "large_field": "same balanced transient over all 289 input locations; requested spatial exception, reported separately",
        "center": "same preferred-polarity 2000-ms step in all six annular conditions; large-field reference uses matched center transient",
        "geometry": "center=checkpoint BC disk; stimulus annulus=AC disk minus BC disk; no model support changed",
        "mean_matching": "six annular probes have identical per-cone time mean and identical center trace; surround mean is zero",
        "contrast_matching": "same continuous-time +/-0.25 Weber extrema; native-bin averaging attenuates boundary bins",
        "rms_matching": False, "energy_note": "RMS/exposure differs for step, flicker, transient; no amplitude normalization",
        "intervals": [[asdict(e) for e in events] for events in next(iter(banks.values())).intervals],
        "history": "all-zero strictly-past observed history, identical for all conditions; new zero-state forward each call",
        "conditions": list(CLAMPS), "references": list(REFERENCES), "source_sha256": hashes,
        "inputs_sha256": input_hash, "frozen_before_inference": True,
        "metrics": {"response": "same-clamp blank-subtracted logit/probability",
                    "peak_integral_latency": "unmodified summarize_response; peak is signed maximum-absolute deviation; integral over full 3 s",
                    "suppression": "condition minus same-clamp center-only peak/integral; negative difference denotes smaller response",
                    "onset_offset": "unmodified summarize_response, 50-ms windows for each signed event; at high frequency windows may include later transitions",
                    "clamp_effect": "off minus normal raw response; mean absolute difference in common 300-2300-ms window; no baseline subtraction",
                    "no_new_mechanism_index": True},
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    responses, checks = {}, []
    for idx, (cid, model) in enumerate(models.items()):
        bank = banks[groups[cid]]
        history = torch.zeros(bank.drive.shape[:2] + (1,))
        modes = {}
        for mode, clamps in CLAMPS.items():
            with torch.no_grad():
                result = model.forward_sequence(bank.drive, observed_counts=history, clamps=clamps)
            modes[mode] = {"logit": result.logits[..., 0], "probability": result.spike_probability[..., 0],
                           "ac_local": result.amacrine_local_current[..., 0], "ac_transient": result.amacrine_transient_current[..., 0]}
            assert all(torch.isfinite(value).all() for value in modes[mode].values())
            if PathwayClamp.H1 in clamps:
                assert torch.count_nonzero(result.h1_surround_contribution) == 0
            if PathwayClamp.AMACRINE_LOCAL in clamps:
                assert torch.count_nonzero(result.amacrine_local_current) == 0
                assert torch.count_nonzero(result.amacrine_transient_current) == 0
        with torch.no_grad():
            reentry = model.forward_sequence(bank.drive, observed_counts=history)
        assert torch.equal(reentry.logits[..., 0], modes["normal"]["logit"])
        assert all(torch.equal(v, checkpoints[cid]["model"][k]) for k, v in model.state_dict().items())
        assert all(p.grad is None for p in model.parameters())
        responses[cid] = modes
        checks.append({"cell_id": cid, "state_unchanged": True, "normal_reentry_equal": True,
                       "H1_clamp_exact_zero": True, "AC_clamp_exact_zero": True, "gradients_none": True})
        print(f"{idx + 1}/22 {cid}: frozen checks PASS", flush=True)
    assert all(sha(Path(path)) == digest for path, digest in hashes.items())
    assert sha(OUT / "inputs.pt") == input_hash
    torch.save({"cells": responses, "groups": groups}, OUT / "responses.pt")
    create_report(OUT, banks, responses)
    (OUT / "verification.json").write_text(json.dumps({"checks": checks, "source_sha256": hashes,
        "inputs_sha256": input_hash, "all_sources_unchanged": True, "training": False, "optimizer_created": False,
        "finite_responses": True, "checkpoint_writes": 0}, indent=2), encoding="utf-8")
    print("COMPLETE: 22 cells, frozen input/model checks PASS", flush=True)


if __name__ == "__main__":
    main()
