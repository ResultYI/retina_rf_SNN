# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.6.0", "matplotlib"]
# ///
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / "output/real_data/schottdorf_r4_development_22cell_20260830_verified"
sys.path.insert(0, str(ROOT))

from models.mechanistic_retina.contracts import ArchitectureMode, MechanisticRetinaConfig, PathwayClamp
from models.mechanistic_retina.model import build_mechanistic_retina
from evaluation.mechanistic_retina.temporal_center_surround import _optional_box
from metrics import aggregate, cell_rows, mach_rows, write_csv
from stimuli import DT_MS, DURATION_MS, ONSET_MS, PITCH_DEG, TOTAL_MS, build_stimuli

CLAMPS = {"normal": frozenset(), "H1_off": frozenset({PathwayClamp.H1}),
          "AC_off": frozenset({PathwayClamp.AMACRINE_LOCAL, PathwayClamp.AMACRINE_TRANSIENT})}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    assert not (OUT / "results.json").exists(), "Refuse to overwrite completed inference"
    torch.set_num_threads(2)
    bank = build_stimuli()
    saved = torch.load(OUT / "stimuli.pt", weights_only=True)
    assert torch.equal(saved["patches"], bank.patches)
    assert tuple(saved["names"]) == bank.names
    time_ms = torch.arange(round(TOTAL_MS / DT_MS), dtype=torch.float32) * DT_MS
    envelope = _optional_box(time_ms, ONSET_MS, DURATION_MS, DT_MS)
    drive = bank.patches.flatten(1)[:, None, :] * envelope[None, :, None]
    history = torch.zeros(drive.shape[0], drive.shape[1], 1)
    torch.save({"cone_drive": drive, "time_ms": time_ms, "history": history, "envelope": envelope},
               OUT / "input-tensors.pt")
    source = json.loads((SOURCE / "results.json").read_text())
    assert len(source["cells"]) == 22
    provenance = {str(p): sha256(p) for p in tuple((ROOT / "models/mechanistic_retina").glob("*.py"))
                  + tuple(OUT.glob("*.py")) + (SOURCE / "results.json", OUT / "stimuli.pt", OUT / "stimulus-contract.json",
                     ROOT / "evaluation/mechanistic_retina/temporal_center_surround.py")}
    all_responses, metadata, rows, boundary_rows, checks = {}, {}, [], [], []
    for index, cell in enumerate(source["cells"]):
        cell_id, group = cell["cell_id"], cell["group"]
        path = SOURCE / "cells" / cell_id.replace("#", "_") / "model-trained.pt"
        provenance[str(path)] = sha256(path)
        checkpoint = torch.load(path, weights_only=True)
        config = dict(checkpoint["model_config"])
        config["architecture_mode"] = ArchitectureMode(config["architecture_mode"])
        assert abs(config["dt_ms"] - DT_MS) < 1e-10
        assert torch.allclose(checkpoint["cone_positions_degs"], saved["cone_positions_degs"], atol=1e-7, rtol=0)
        assert checkpoint["cone_positions_degs"].shape == (289, 2)
        model = build_mechanistic_retina(MechanisticRetinaConfig(**config), checkpoint["cone_positions_degs"],
                                         checkpoint["cell_positions_degs"], checkpoint["cell_types"], checkpoint["polarities"])
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        responses = {}
        for mode, clamps in CLAMPS.items():
            with torch.no_grad():
                result = model.forward_sequence(drive, observed_counts=history, clamps=clamps)
            responses[mode] = {"logit": result.logits[..., 0], "probability": result.spike_probability[..., 0],
                               "ac_local": result.amacrine_local_current[..., 0],
                               "ac_transient": result.amacrine_transient_current[..., 0]}
            assert all(torch.isfinite(t).all() for t in responses[mode].values())
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
        assert torch.equal(repeat.logits[..., 0], responses["normal"]["logit"])
        assert all(torch.equal(v, checkpoint["model"][k]) for k, v in model.state_dict().items())
        assert all(p.grad is None for p in model.parameters())
        all_responses[cell_id] = responses
        metadata[cell_id] = {"group": group, "type": checkpoint["cell_types"], "polarity": checkpoint["polarities"]}
        rows.extend(cell_rows(cell_id, group, bank, responses, time_ms))
        boundary_rows.extend(mach_rows(cell_id, group, bank, responses, time_ms))
        checks.append({"cell_id": cell_id, "group": group, "weights_state_unchanged": True,
                       "clamp_exact_zero": True, "normal_reentry_bitwise_equal": True,
                       "all_gradients_none": True, "paired_controls_bitwise_equal": True})
        print(f"{index + 1}/22 {cell_id} {group}: frozen/clamp/control checks PASS", flush=True)
    assert all(sha256(Path(p)) == digest for p, digest in provenance.items())
    groups = aggregate(rows)
    write_csv(OUT / "per-cell-responses.csv", rows)
    write_csv(OUT / "group-responses.csv", groups)
    write_csv(OUT / "mach-boundary-extrema.csv", boundary_rows)
    torch.save({"time_ms": time_ms, "names": bank.names, "cells": all_responses, "metadata": metadata},
               OUT / "response-tensors.pt")
    (OUT / "verification.json").write_text(json.dumps({"checks": checks, "source_sha256": provenance,
        "all_source_hashes_unchanged": True, "no_training": True, "new_model_checkpoints": 0}, indent=2), encoding="utf-8")
    (OUT / "results.json").write_text(json.dumps({"cells": metadata, "group_metrics": groups,
        "mach_boundary_extrema": boundary_rows, "dt_ms": DT_MS, "pitch_deg": PITCH_DEG,
        "stimulus_count_including_blank": len(bank.names), "conditions": list(CLAMPS),
        "pairs": [asdict(p) for p in bank.pairs]}, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
