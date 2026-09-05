from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Final

import torch


OUT: Final = Path(__file__).resolve().parent
ROOT: Final = OUT.parents[2]
APP: Final = ROOT / "output/real_data/schottdorf_canonical_v1_shared_bc_frozen_applications_20260830"
SOURCE: Final = ROOT / "output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830"
LN_ROOT: Final = ROOT / "output/real_data/schottdorf_center_surround_ln_22cell_seed61001_20260830"
CNN_ROOT: Final = ROOT / ".omo/evidence/compact_causal_cnn_baseline"
sys.path[:0] = [str(OUT), str(APP), str(ROOT)]

from baselines.center_surround_ln import CenterSurroundLN  # noqa: E402
from baselines.compact_causal_cnn import CompactCausalCNN  # noqa: E402
from common import infer, load_model  # noqa: E402
from contract import build_bank  # noqa: E402

CellResponses = dict[str, str | dict[str, torch.Tensor]]
VerificationValue = str | float | dict[str, bool | float]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def baseline_logits(model: torch.nn.Module, drive: torch.Tensor, history: torch.Tensor) -> tuple[torch.Tensor, dict[str, bool]]:
    model.eval()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        logits = model.forward_with_history(drive, history)[..., 0]
        repeat = model.forward_with_history(drive, history)[..., 0]
    checks = {
        "finite": bool(torch.isfinite(logits).all()),
        "state_unchanged": all(torch.equal(value, before[name]) for name, value in model.state_dict().items()),
        "eval_mode_unchanged": not model.training,
        "gradients_absent": all(parameter.grad is None for parameter in model.parameters()),
        "repeat_bitwise_equal": torch.equal(logits, repeat),
    }
    assert all(checks.values())
    return logits, checks


def load_ln(cell_id: str) -> tuple[CenterSurroundLN, Path]:
    path = LN_ROOT / "cells" / cell_id.replace("#", "_") / "ln-trained.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    assert checkpoint["cell_id"] == cell_id and checkpoint["context_bins"] == 60
    history = checkpoint["history"]
    model = CenterSurroundLN(dt_ms=float(history["dt_ms"]), history_tau_ms=float(history["tau_ms"]),
                             seed=int(checkpoint["seed"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    return model, path


def load_cnn(cell_id: str) -> tuple[CompactCausalCNN, Path]:
    path = CNN_ROOT / "cells" / cell_id.replace("#", "_") / "cnn-trained.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    assert checkpoint["cell_id"] == cell_id and checkpoint["schema"] == "schottdorf_compact_causal_cnn_v1"
    history = checkpoint["history"]
    model = CompactCausalCNN(dt_ms=float(history["dt_ms"]), history_tau_ms=float(history["tau_ms"]),
                             seed=int(checkpoint["contract"]["seed"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    return model, path


def main() -> None:
    torch.set_num_threads(4)
    bank = build_bank()
    metadata = json.loads((SOURCE / "results.json").read_text(encoding="utf-8"))["cells"]
    assert len(metadata) == 22
    old_inputs_path = APP / "illusion/inputs.pt"
    old_responses_path = APP / "illusion/responses.pt"
    old_responses = torch.load(old_responses_path, map_location="cpu", weights_only=True)
    cells: dict[str, CellResponses] = {}
    verification: list[dict[str, VerificationValue]] = []
    checkpoint_hashes: dict[str, dict[str, str]] = {}
    replay_pairs = ((50, "SBC", "bright_minus_dark"), (8, "Mach", "dark_ramp_minus_uniform"),
                    (16, "Mach", "bright_ramp_minus_uniform"))
    for index, identity in enumerate(metadata, start=1):
        cell_id = identity["cell_id"]
        canonical = load_model(cell_id)
        canonical_response, canonical_checks = infer(canonical, bank.drive, bank.history)
        ln, ln_path = load_ln(cell_id)
        cnn, cnn_path = load_cnn(cell_id)
        ln_logits, ln_checks = baseline_logits(ln, bank.drive, bank.history)
        cnn_logits, cnn_checks = baseline_logits(cnn, bank.drive, bank.history)
        canonical_logits = {mode: values["logit"] for mode, values in canonical_response.items()}

        replay_error = 0.0
        for old_index, family, signature in replay_pairs:
            row = next(row for row in bank.comparisons
                       if row.family == family and row.signature == signature
                       and row.contrast == 0.25 and row.extent_px == 8)
            current = canonical_logits["normal"][row.a] - canonical_logits["normal"][row.b]
            old = (old_responses["cells"][cell_id]["normal"]["logit"][old_index]
                   - old_responses["cells"][cell_id]["normal"]["logit"][old_index + (1 if family == "SBC" else 25)])
            replay_error = max(replay_error, float((current - old).abs().max()))
        assert replay_error <= 2e-6

        canonical_path = SOURCE / "cells" / cell_id.replace("#", "_") / "model-trained.pt"
        cells[cell_id] = {
            "group": f"{identity['retinal_class']}_{identity['polarity']}",
            "canonical": canonical_logits,
            "ln": {"normal": ln_logits},
            "cnn": {"normal": cnn_logits},
        }
        verification.append({
            "cell_id": cell_id,
            "canonical": canonical_checks,
            "ln": ln_checks,
            "cnn": cnn_checks,
            "original_grid_replay_max_abs_logit_error": replay_error,
        })
        checkpoint_hashes[cell_id] = {
            "canonical": sha256(canonical_path), "ln": sha256(ln_path), "cnn": sha256(cnn_path)
        }
        print(f"INFERENCE {index}/22 {cell_id}", flush=True)

    comparisons = [asdict(row) for row in bank.comparisons]
    torch.save({"cells": cells, "names": bank.names, "comparisons": comparisons,
                "time_ms": torch.arange(bank.drive.shape[1]) * (1000 / 150)}, OUT / "responses.pt")
    torch.save({"patches": bank.patches, "drive": bank.drive, "history": bank.history,
                "names": bank.names, "comparisons": comparisons}, OUT / "stimuli.pt")
    manifest = {
        "status": "FROZEN_INFERENCE_COMPLETE", "cells": len(cells), "models": ["canonical", "ln", "cnn"],
        "conditions": ["normal", "H1_off", "direct_BC_off", "AC_off"],
        "training_performed": False, "shared_zero_history": True,
        "original_input_sha256": sha256(old_inputs_path), "original_response_sha256": sha256(old_responses_path),
        "checkpoint_sha256": checkpoint_hashes, "verification": verification,
    }
    (OUT / "inference_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
