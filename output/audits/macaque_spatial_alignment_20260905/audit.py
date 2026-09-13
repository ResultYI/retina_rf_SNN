# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "numpy"]
# ///
# How to run: D:/anaconda/python.exe -B output/audits/macaque_spatial_alignment_20260905/audit.py
# Uses the existing scientific runtime. Reads frozen artifacts; never fits a model.
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Final

import numpy as np
import torch
from torch.nn import functional as F

ROOT: Final = Path(__file__).resolve().parents[3]
OUT: Final = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from data.schottdorf_lee_2021 import SchottdorfAdapterConfig, _cone_positions
from models.mechanistic_retina.contracts import MechanisticRetinaConfig
from models.mechanistic_retina.model import build_mechanistic_retina

CAN: Final = ROOT / "output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830"
LN: Final = ROOT / "output/real_data/schottdorf_center_surround_ln_22cell_seed61001_20260830"
OFFICIAL: Final = ROOT / "data/real/schottdorf_lee_2021_repository"
PITCH: Final = 3 * 4.6 / 256


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=ROOT, encoding="utf-8").strip()


def ranks(values: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - counts + (counts + 1) / 2)[inverse]


def main() -> None:
    torch.set_num_threads(2)
    initial_status = git("status", "--short", "--untracked-files=all")
    sources = {ROOT / "AUDIT_INDEX.md", CAN / "run.py", CAN / "results.json", CAN / "run-manifest.json",
               LN / "results.json", LN / "run-manifest.json", ROOT / "baselines/center_surround_ln.py",
               ROOT / "training/mechanistic_retina/center_surround_ln.py",
               ROOT / "evaluation/mechanistic_retina/schottdorf_center_surround_ln.py",
               ROOT / "evaluation/mechanistic_retina/schottdorf_ln_source.py",
               ROOT / "evaluation/mechanistic_retina/rf_effective.py",
               ROOT / "evaluation/mechanistic_retina/pathway_decomposition.py",
               OFFICIAL / "README.md", OFFICIAL / "datacite.yml", OFFICIAL / "retinatools/library.py"}
    sources.update((ROOT / "data").glob("schottdorf_lee*.py"))
    sources.update((ROOT / "models/mechanistic_retina").glob("*.py"))
    sources.update((OFFICIAL / "run_model").glob("*.ipynb"))
    sources.update((OFFICIAL / "data").glob("*.docx"))
    sources.update((OFFICIAL / "exampleRFs").glob("*.xlsx"))
    cells = json.loads((CAN / "results.json").read_text())["cells"]
    for cell in cells:
        name = cell["cell_id"].replace("#", "_")
        sources.update((CAN / "cells" / name / leaf) for leaf in ("model-trained.pt", "results.json", "validation-predictions.pt"))
        sources.update((LN / "cells" / name / leaf) for leaf in ("ln-trained.pt", "results.json", "validation-predictions.pt"))
        sources.update((LN / "cells" / name).glob("inner-lambda-*.pt"))
    before = {str(p.relative_to(ROOT)).replace("\\", "/"): digest(p) for p in sorted(sources)}
    rows, identities, consistency, prediction_checks = [], [], [], []
    expected_grid = torch.from_numpy(_cone_positions(SchottdorfAdapterConfig()))
    for cell in cells:
        cid = cell["cell_id"]
        cname = cid.replace("#", "_")
        cp, lp = CAN / "cells" / cname, LN / "cells" / cname
        canonical = torch.load(cp / "model-trained.pt", map_location="cpu", weights_only=True)
        ln = torch.load(lp / "ln-trained.pt", map_location="cpu", weights_only=True)
        lr = json.loads((lp / "results.json").read_text())
        cr = json.loads((cp / "results.json").read_text())
        assert canonical["cell_id"] == ln["cell_id"] == cid
        assert canonical["stage"] == "trained" and canonical["revision"] == 4
        assert canonical["schema"] == "schottdorf_canonical_v1_shared_bc_development"
        assert canonical["model_name"] == "Canonical V1"
        config = MechanisticRetinaConfig(**canonical["model_config"])
        assert config.causal_contract == "h1-shared-bc-direct-broad-ac"
        assert config.spatial_contract == "bc-central-disk_ac-overlapping-full-disk"
        assert torch.equal(canonical["cell_positions_degs"], torch.zeros(1, 2))
        assert torch.equal(canonical["cone_positions_degs"], expected_grid)
        assert lr["training_contract"]["full_train_fresh_refit"]
        assert not lr["training_contract"]["original_validation_used_for_selection"]
        assert ln["refit_steps"] == ln["best_step"] == lr["training_contract"]["full_train_refit_steps"]
        assert ln["schema"] == "schottdorf_center_surround_separable_ln_v1"
        assert lr["adapter_config"] == {"train_sequence_count": 16, "validation_sequence_count": 4,
                                      "sequence_steps": 150, "warmup_steps": 30, "crop_pixels": 51, "pool_factor": 3}
        model = build_mechanistic_retina(config, expected_grid, canonical["cell_positions_degs"],
                                        tuple(canonical["cell_types"]), tuple(canonical["polarities"]))
        geometry_keys = [key for key in model.state_dict() if any(t in key for t in ("spatial_basis", "support", "h1.graph"))]
        assert all(torch.equal(model.state_dict()[key], canonical["model"][key]) for key in geometry_keys)
        model.load_state_dict(canonical["model"], strict=True)
        parameter_names = [name for name, value in model.named_parameters() if value.requires_grad]
        assert not any(t in name for name in parameter_names for t in ("position", "center", "spatial", "translation"))
        xy = ln["model"]["center_xy"].double().numpy()
        grid = ln["model"]["grid_xy"].reshape(-1, 2)
        assert torch.allclose(grid[:, 0] * PITCH, expected_grid[:, 0], atol=1e-7, rtol=0)
        assert torch.allclose(-grid[:, 1] * PITCH, expected_grid[:, 1], atol=1e-7, rtol=0)
        ln_nll, can_nll = lr["validation_nll_trained"], cell["validation_nll_trained"]
        assert can_nll == cr["validation_nll_trained"]
        radius = math.hypot(*xy)
        bc_radius = {"PC": 0.06, "MC": 0.10}[cell["retinal_class"]]
        ac_radius = {"PC": 0.13, "MC": 0.15}[cell["retinal_class"]]
        rows.append({"cell_id": cid, "group": cell["retinal_class"] + "_" + cell["polarity"],
                     "ln_center_xy_raw": json.dumps(xy.tolist()), "x_offset_grid_pixels": xy[0],
                     "y_offset_grid_pixels_image_down": xy[1], "radial_offset_grid_pixels": radius,
                     "x_offset_degree": xy[0] * PITCH, "y_offset_degree_cartesian_up": -xy[1] * PITCH,
                     "radial_offset_degree": radius * PITCH, "pooled_pixel_pitch_degree": PITCH,
                     "ln_validation_nll": ln_nll, "canonical_validation_nll": can_nll,
                     "canonical_minus_ln_nll": can_nll - ln_nll, "bc_radius_degree": bc_radius,
                     "ac_radius_degree": ac_radius, "offset_over_bc_radius": radius * PITCH / bc_radius,
                     "ln_center_sigma_pixels": float((F.softplus(ln["model"]["raw_widths"])[0] + 1e-6)),
                     "ln_refit_steps": ln["refit_steps"], "ln_checkpoint": str((lp / "ln-trained.pt").relative_to(ROOT)).replace("\\", "/"),
                     "canonical_checkpoint": str((cp / "model-trained.pt").relative_to(ROOT)).replace("\\", "/")})
        identities.append({"cell_id": cid, "schema": canonical["schema"], "stage": canonical["stage"],
                           "revision": canonical["revision"], "model_config": canonical["model_config"],
                           "cell_positions_degs": canonical["cell_positions_degs"].tolist(), "strict_load": True,
                           "fixed_geometry_equal_current_constructor": True, "checked_geometry_keys": geometry_keys,
                           "bc_support_pixels": int(model.feature_bank.bc_support.sum()),
                           "ac_support_pixels": int(model.feature_bank.ac_support.sum()),
                           "trainable_parameter_names": parameter_names})
        for candidate_path in sorted(lp.glob("inner-lambda-*.pt")):
            candidate = torch.load(candidate_path, map_location="cpu", weights_only=True)
            cxy = candidate["model"]["center_xy"].double().numpy()
            consistency.append({"cell_id": cid, "artifact": str(candidate_path.relative_to(ROOT)).replace("\\", "/"),
                                "lambda": candidate["regularization"], "inner_x_pixels": cxy[0], "inner_y_pixels": cxy[1],
                                "inner_radial_pixels": float(np.linalg.norm(cxy)),
                                "distance_from_final_refit_pixels": float(np.linalg.norm(cxy - xy)),
                                "direction_cosine_to_final": float(np.dot(cxy, xy) / (np.linalg.norm(cxy) * radius))})
        pp, pl = (torch.load(path / "validation-predictions.pt", map_location="cpu", weights_only=True) for path in (cp, lp))
        assert torch.equal(pp["target"], pl["target"]) and torch.equal(pp["valid_mask"], pl["valid_mask"])
        assert tuple(pp["source_image_ids"]) == tuple(pl["source_image_ids"])
        assert tuple(pp["trial_indices"]) == tuple(pl["trial_indices"])
        recomputed = [float((F.softplus(p["logits_trained"]) - p["target"] * p["logits_trained"])[p["valid_mask"]].mean()) for p in (pp, pl)]
        assert abs(recomputed[0] - can_nll) < 1e-7 and abs(recomputed[1] - ln_nll) < 1e-7
        prediction_checks.append({"cell_id": cid, "identical_targets_masks_ids_trials": True,
                                  "canonical_nll_recomputed": recomputed[0], "ln_nll_recomputed": recomputed[1]})
    assert len(rows) == 22 and len({r["cell_id"] for r in rows}) == 22
    stats = []
    for label in ("all", *sorted({r["group"] for r in rows}), "all_except_68#10"):
        selected = [r for r in rows if label == "all" or r["group"] == label or (label == "all_except_68#10" and r["cell_id"] != "68#10")]
        d = np.array([r["radial_offset_grid_pixels"] for r in selected])
        gap = np.array([r["canonical_minus_ln_nll"] for r in selected])
        stats.append({"subset": label, "n": len(selected), "d_min_pixels": float(d.min()),
                      "d_q25_pixels": float(np.quantile(d, .25)), "d_median_pixels": float(np.median(d)),
                      "d_q75_pixels": float(np.quantile(d, .75)), "d_max_pixels": float(d.max()),
                      "d_mean_pixels": float(d.mean()), "d_median_degree": float(np.median(d) * PITCH),
                      "gap_mean_nats_per_bin": float(gap.mean()), "pearson_r": float(np.corrcoef(d, gap)[0, 1]),
                      "spearman_rho": float(np.corrcoef(ranks(d), ranks(gap))[0, 1]),
                      "n_offset_gt_1_pixel": int((d > 1).sum()), "n_offset_gt_bc_radius": sum(r["offset_over_bc_radius"] > 1 for r in selected),
                      "canonical_wins": int((gap < 0).sum())})
    for name, data in (("per_cell_alignment.csv", rows), ("alignment_vs_prediction.csv", stats),
                       ("frozen_inner_center_consistency.csv", consistency)):
        with (OUT / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    manifest = {"branch": git("branch", "--show-current"), "head": git("rev-parse", "HEAD"),
                "git_status_before_analysis": initial_status, "git_status_after_analysis": git("status", "--short", "--untracked-files=all"),
                "reference_head_matches": git("rev-parse", "HEAD") == "fea28de038821fadee279b93728688b34bcb3bac",
                "training_runs": 0, "production_source_model_data_modified": False,
                "checkpoint_reads": {"canonical_final": 22, "ln_final_refit": 22, "ln_inner_supplement_only": 88, "frozen_validation_predictions": 44},
                "torch_version": torch.__version__, "numerical_method": "CPU checkpoint inspection and frozen-logit Bernoulli NLL recomputation; no new model forward, fit, or RF estimation",
                "pixel_pitch_degree": PITCH, "grid_x_degree": expected_grid[:17, 0].tolist(),
                "grid_y_degree": expected_grid[::17, 1].tolist(), "checkpoint_identity_checks": identities,
                "prediction_pair_checks": prediction_checks,
                "source_files": [{"path": p, "sha256_before": h, "sha256_after": digest(ROOT / p), "bytes": (ROOT / p).stat().st_size} for p, h in before.items()]}
    assert all(item["sha256_before"] == item["sha256_after"] for item in manifest["source_files"])
    manifest["all_read_source_hashes_unchanged"] = True
    (OUT / "evidence_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print("Verified: 22 final Canonical identities, 22 LN refits, 22 paired prediction artifacts, all source hashes unchanged.")


if __name__ == "__main__":
    main()
