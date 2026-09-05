from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Final

import torch


OUT: Final = Path(__file__).resolve().parent
ROOT: Final = OUT.parents[2]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    inference = json.loads((OUT / "inference_manifest.json").read_text(encoding="utf-8"))
    responses = torch.load(OUT / "responses.pt", map_location="cpu", weights_only=True)
    stimuli = torch.load(OUT / "stimuli.pt", map_location="cpu", weights_only=True)
    assert len(responses["cells"]) == 22 and inference["training_performed"] is False
    checks = {
        "22_cells": len(responses["cells"]) == 22,
        "166_stimuli": stimuli["drive"].shape == (166, 150, 289),
        "shared_zero_history": int(torch.count_nonzero(stimuli["history"])) == 0,
        "all_outputs_finite": True,
        "all_controls_exact_zero": True,
        "all_model_states_unchanged": True,
        "all_canonical_clamps_verified": True,
        "original_grid_bitwise_replay": True,
        "checkpoint_hashes_unchanged": True,
    }
    for cell_id, cell in responses["cells"].items():
        for model in ("canonical", "ln", "cnn"):
            for logits in cell[model].values():
                checks["all_outputs_finite"] &= bool(torch.isfinite(logits).all())
                for comparison in responses["comparisons"]:
                    control = comparison["contrast"] == 0 or (
                        comparison["family"] == "SBC" and comparison["extent_px"] == 0
                    )
                    if control:
                        checks["all_controls_exact_zero"] &= torch.equal(
                            logits[comparison["a"]], logits[comparison["b"]]
                        )
        recorded = next(row for row in inference["verification"] if row["cell_id"] == cell_id)
        checks["all_model_states_unchanged"] &= recorded["canonical"]["state_unchanged"]
        checks["all_model_states_unchanged"] &= recorded["ln"]["state_unchanged"] and recorded["cnn"]["state_unchanged"]
        required = ("H1_clamp_exact_zero", "direct_BC_clamp_exact_zero", "AC_clamp_exact_zero",
                    "direct_BC_off_preserves_BC_broad_AC", "AC_off_preserves_H1_BC_views",
                    "H1_off_propagates_downstream")
        checks["all_canonical_clamps_verified"] &= all(recorded["canonical"][name] for name in required)
        checks["original_grid_bitwise_replay"] &= recorded["original_grid_replay_max_abs_logit_error"] == 0
        for model, digest in inference["checkpoint_sha256"][cell_id].items():
            path = (ROOT / "output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/cells"
                    / cell_id.replace("#", "_") / "model-trained.pt") if model == "canonical" else (
                    ROOT / "output/real_data/schottdorf_center_surround_ln_22cell_seed61001_20260830/cells"
                    / cell_id.replace("#", "_") / "ln-trained.pt") if model == "ln" else (
                    ROOT / ".omo/evidence/compact_causal_cnn_baseline/cells"
                    / cell_id.replace("#", "_") / "cnn-trained.pt")
            checks["checkpoint_hashes_unchanged"] &= sha256(path) == digest
    assert all(checks.values())

    sources = [ROOT / path for path in (
        "models/mechanistic_retina/model.py", "models/mechanistic_retina/contracts.py",
        "baselines/center_surround_ln.py", "baselines/compact_causal_cnn.py",
        "output/real_data/schottdorf_canonical_v1_shared_bc_frozen_applications_20260830/common.py",
        "output/real_data/schottdorf_r4_dev_visual_illusions_20260830/stimuli.py",
    )]
    git_head = subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip()
    status = subprocess.check_output(("git", "status", "--porcelain", "--untracked-files=no"), cwd=ROOT, text=True)
    output_files = [path for path in OUT.rglob("*") if path.is_file() and "__pycache__" not in path.parts
                    and path.name not in ("artifact_manifest.json",)]
    manifest = {"git_head": git_head, "tracked_worktree_status_sha256": hashlib.sha256(status.encode()).hexdigest(),
                "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in sources},
                "artifact_sha256": {str(path.relative_to(OUT)): sha256(path) for path in sorted(output_files)},
                "training_performed": False}
    (OUT / "verification.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    (OUT / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
