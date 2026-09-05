from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from evaluation.mechanistic_retina.karamanlis_prediction_baselines import evaluate_retinal_model
from evaluation.mechanistic_retina.factorized_ln_split import make_inner_dev
from evaluation.mechanistic_retina.schottdorf_ln_source import LNSourcePaths, load_ln_cell
from evaluation.mechanistic_retina.schottdorf_multirecording_reporting import sha256_file
from evaluation.mechanistic_retina.schottdorf_r4_development import verify_fresh_state
from models.mechanistic_retina.contracts import ArchitectureMode, MechanisticRetinaConfig
from models.mechanistic_retina.model import build_mechanistic_retina
from training.mechanistic_retina.center_surround_ln import DevelopmentStop
from training.mechanistic_retina.losses import expected_bernoulli_nll


def main() -> None:
    output = Path(__file__).resolve().parent
    manifest = json.loads((output / "run-manifest.json").read_text())
    old_root = Path(manifest["source_results"]).parent
    ln_root = Path(manifest["ln_results"]).parent
    old = json.loads(Path(manifest["source_results"]).read_text())
    ln = json.loads(Path(manifest["ln_results"]).read_text())
    old_cells = {row["cell_id"]: row for row in old["cells"]}
    ln_cells = {row["cell_id"]: row for row in ln["cells"]}
    assert len(manifest["cell_ids"]) == 22
    assert all((output / "cells" / cid.replace("#", "_") / "results.json").exists()
               for cid in manifest["cell_ids"])
    assert sha256_file(Path(manifest["source_results"])) == manifest["source_results_sha256"]
    assert sha256_file(Path(manifest["ln_results"])) == manifest["ln_results_sha256"]
    assert all(sha256_file(Path(p)) == h for p, h in manifest["source_code_sha256"].items())
    paths = LNSourcePaths(ROOT / "data/real/schottdorf_lee_2021_repository",
                          ROOT / "data/real/schottdorf_lee_2021_macaque/1x10_256.mpg", old_root)
    references = torch.load(ln_root / "reference-predictions.pt", weights_only=True)
    torch.set_num_threads(2)
    rows, checks, hashes = [], [], {}
    figure, axes = plt.subplots(6, 4, figsize=(16, 18), constrained_layout=True)
    for axis, cid in zip(axes.flat, manifest["cell_ids"], strict=False):
        folder = output / "cells" / cid.replace("#", "_")
        result = json.loads((folder / "results.json").read_text())
        ln_result = json.loads((ln_root / "cells" / cid.replace("#", "_") / "results.json").read_text())
        assert result["inner_boundaries"] == ln_result["inner_boundaries"]
        for key in ("train_valid_bins", "validation_valid_bins", "inner_train_valid_bins", "inner_dev_valid_bins"):
            assert result[key] == ln_result[key]
        curve = result["inner_trajectory"]
        with (folder / "inner-trajectory.csv").open(newline="", encoding="utf-8") as stream:
            csv_curve = list(csv.DictReader(stream))
        assert len(csv_curve) == len(curve)
        assert all(int(a["step"]) == b["step"] and float(a["inner_dev_nll"]) == b["inner_dev_nll"]
                   for a, b in zip(csv_curve, curve, strict=True))
        assert [row["step"] for row in curve] == list(range(result["stopping_step"] + 1))
        state = DevelopmentStop(curve[0]["inner_dev_nll"], 0, curve[0]["inner_dev_nll"], 0)
        for row in curve[1:]:
            assert not state.stopped
            state = state.observe(row["inner_dev_nll"], row["step"])
        assert state.best_step == result["best_step"]
        assert state.best_nll == result["best_inner_dev_nll"]
        assert state.stopped or result["stopping_step"] == 1000
        assert result["refit_steps"] == result["best_step"]
        assert all(step == result["best_step"] for step in result["optimizer_steps"])
        assert result["gradients_finite"]
        data = load_ln_cell(paths, cid).data
        prediction = torch.load(folder / "validation-predictions.pt", weights_only=True)
        ln_prediction = torch.load(ln_root / "cells" / cid.replace("#", "_") / "validation-predictions.pt", weights_only=True)
        for other in (ln_prediction, references[cid]):
            for key in ("target", "valid_mask"):
                assert torch.equal(prediction[key], other[key])
            for key in ("source_image_ids", "trial_indices"):
                assert prediction[key] == other[key]
        checkpoint = torch.load(folder / "model-trained.pt", weights_only=True)
        raw = torch.load(folder / "model-raw.pt", weights_only=True)
        previous_raw = torch.load(old_root / "cells" / cid.replace("#", "_") / "model-raw.pt", weights_only=True)
        roundoff = verify_fresh_state(raw["model"], previous_raw["model"])
        config = dict(checkpoint["model_config"])
        config["architecture_mode"] = ArchitectureMode(config["architecture_mode"])
        model = build_mechanistic_retina(MechanisticRetinaConfig(**config), data.cone_positions_degs,
                                          data.cell_positions_degs, data.cell_types, data.polarities)
        model.load_state_dict(checkpoint["model"], strict=True)
        metrics, replay = evaluate_retinal_model(model, data.validation)
        assert torch.equal(replay, prediction["logits_trained"])
        assert metrics.population_nll == result["validation_nll_trained"]
        inner_checkpoint = torch.load(folder / "model-inner-best.pt", weights_only=True)
        model.load_state_dict(inner_checkpoint["model"], strict=True)
        inner_metrics, _ = evaluate_retinal_model(model, make_inner_dev(data.train).development)
        assert inner_metrics.population_nll == result["best_inner_dev_nll"]
        ln_nll = float(expected_bernoulli_nll(ln_prediction["logits_trained"], prediction["target"], prediction["valid_mask"]))
        old_nll = float(expected_bernoulli_nll(references[cid]["retinal_logits"], prediction["target"], prediction["valid_mask"]))
        assert ln_nll == ln_cells[cid]["ln_nll"]
        assert abs(old_nll - old_cells[cid]["validation_nll_trained"]) < 1e-6
        score = metrics.population_nll
        row = {"cell_id": cid, "group": ln_cells[cid]["group"], "best_step": result["best_step"],
               "stopping_step": result["stopping_step"], "stop_reason": "patience" if state.stopped else "max_steps",
               "best_inner_dev_nll": result["best_inner_dev_nll"], "r4_development_nll": score,
               "r4_50step_nll": old_nll, "center_surround_ln_nll": ln_nll,
               "r4_development_minus_r4_50step": score - old_nll, "r4_development_minus_ln": score - ln_nll,
               "winner_vs_50step": "R4-dev" if score < old_nll else "R4-50" if score > old_nll else "tie",
               "winner_vs_ln": "R4-dev" if score < ln_nll else "LN" if score > ln_nll else "tie",
               "validation_valid_bins": result["validation_valid_bins"], "artifact_dir": str(folder)}
        rows.append(row)
        checks.append({"cell_id": cid, "checkpoint_replay_exact": True, "split_and_target_match_ln": True,
                       "inner_best_checkpoint_nll_replay_exact": True, "csv_trajectory_matches_json": True,
                       "stop_rule_replay_passed": True, "fresh_raw_parameters_exact": True,
                       "fixed_h1_graph_max_abs_roundoff": roundoff, "refit_optimizer_steps_match_best": True})
        hashes.update(result["source_hashes"])
        axis.plot([r["step"] for r in curve], [r["inner_dev_nll"] for r in curve], linewidth=0.9)
        axis.scatter([result["best_step"]], [result["best_inner_dev_nll"]], color="red", s=14)
        axis.set_title(f"{cid} {row['group']} | best {row['best_step']} / stop {row['stopping_step']}", fontsize=9)
        axis.set_xlabel("optimizer updates"); axis.set_ylabel("inner-dev Bernoulli NLL")
        print(f"VERIFIED {cid}: {score:.9f}", flush=True)
    assert all(sha256_file(Path(p)) == h for p, h in hashes.items())
    for axis in list(axes.flat)[22:]:
        axis.set_visible(False)
    figure.savefig(output / "inner-dev-trajectories.png", dpi=150)
    plt.close(figure)
    columns = ("r4_development_nll", "r4_50step_nll", "center_surround_ln_nll")
    overall = {key: statistics.mean(row[key] for row in rows) for key in columns}
    groups = {group: {"cell_count": sum(row["group"] == group for row in rows)} |
              {key: statistics.mean(row[key] for row in rows if row["group"] == group) for key in columns}
              for group in ("MC_ON", "MC_OFF", "PC_ON", "PC_OFF")}
    summary = {"cell_count": 22, "recording_count": 37, "aggregation": "equal-cell mean validation Bernoulli NLL",
               "overall": overall, "groups": groups, "cells": rows,
               "r4_development_wins_vs_ln": sum(row["winner_vs_ln"] == "R4-dev" for row in rows),
               "r4_development_wins_vs_50step": sum(row["winner_vs_50step"] == "R4-dev" for row in rows),
               "checks": checks, "all_source_hashes_unchanged": True,
               "training_contract": manifest["training_contract"]}
    (output / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (output / "per-cell.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"overall": overall, "groups": groups}, indent=2))


if __name__ == "__main__":
    main()
