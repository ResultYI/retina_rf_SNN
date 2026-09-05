# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.6.0", "matplotlib"]
# ///
# How to run: D:/anaconda/python.exe -B replay.py (frozen existing numerical runtime).
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Final

import torch

from diagnostic_stimuli import OUT, ORIGINAL, ROOT, VARIANTS, diagnostic_bank
from baselines.center_surround_ln import CenterSurroundLN
from models.mechanistic_retina.contracts import ArchitectureMode, MechanisticRetinaConfig, PathwayClamp
from models.mechanistic_retina.model import build_mechanistic_retina
from metrics import Row, aggregate, cell_rows, mach_rows, response_metrics, write_csv

R4: Final = OUT.parent / "schottdorf_r4_development_22cell_20260830_verified"
LN: Final = OUT.parent / "schottdorf_center_surround_ln_22cell_seed61001_20260830"
CLAMPS: Final = {"normal": frozenset(), "H1_off": frozenset({PathwayClamp.H1}),
                 "AC_off": frozenset({PathwayClamp.AMACRINE_LOCAL, PathwayClamp.AMACRINE_TRANSIENT})}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    assert not (OUT / "responses.pt").exists()
    torch.set_num_threads(2)
    bank = diagnostic_bank()
    inputs = torch.load(OUT / "inputs.pt", weights_only=True)
    old = torch.load(ORIGINAL / "response-tensors.pt", weights_only=True)
    time = inputs["time_ms"]
    assert torch.equal(bank.patches, inputs["patches"])
    source_paths = tuple((ROOT / "models/mechanistic_retina").glob("*.py")) + tuple(OUT.glob("*.py"))
    source_paths += (ROOT / "baselines/center_surround_ln.py", ORIGINAL / "stimuli.py", ORIGINAL / "metrics.py",
                     ORIGINAL / "stimuli.pt", ORIGINAL / "input-tensors.pt", ORIGINAL / "response-tensors.pt",
                     ORIGINAL / "stimulus-contract.json", OUT / "protocol.json", OUT / "inputs.pt")
    hashes = {str(p): digest(p) for p in source_paths}
    all_cells, checks, rows, boundary, comparisons, support_stats = {}, [], [], [], [], []
    for index, (cid, metadata) in enumerate(old["metadata"].items()):
        paths = {"R4": R4 / "cells" / cid.replace("#", "_") / "model-trained.pt",
                 "LN": LN / "cells" / cid.replace("#", "_") / "ln-trained.pt"}
        hashes.update({str(p): digest(p) for p in paths.values()})
        rc, lc = (torch.load(paths[label], weights_only=True) for label in ("R4", "LN"))
        assert rc["cell_id"] == lc["cell_id"] == cid
        assert lc["context_bins"] == 60 and lc["history"]["dt_ms"] == rc["model_config"]["dt_ms"]
        config = dict(rc["model_config"])
        config["architecture_mode"] = ArchitectureMode(config["architecture_mode"])
        retinal = build_mechanistic_retina(MechanisticRetinaConfig(**config), rc["cone_positions_degs"],
                                           rc["cell_positions_degs"], rc["cell_types"], rc["polarities"])
        retinal.load_state_dict(rc["model"], strict=True)
        retinal.eval()
        linear = CenterSurroundLN(lc["history"]["dt_ms"], lc["history"]["tau_ms"], lc["seed"])
        linear.load_state_dict(lc["model"], strict=True)
        linear.eval()
        response = {}
        errors = []
        for mode, clamps in CLAMPS.items():
            parts = []
            for source in ("original", "diagnostic"):
                with torch.no_grad():
                    result = retinal.forward_sequence(inputs[f"{source}_drive"],
                        observed_counts=inputs[f"{source}_history"], clamps=clamps)
                if mode.startswith("H1"):
                    assert torch.count_nonzero(result.h1_surround_contribution) == 0
                if mode.startswith("AC"):
                    assert torch.count_nonzero(result.amacrine_local_current) == 0
                    assert torch.count_nonzero(result.amacrine_transient_current) == 0
                parts.append({"logit": result.logits[..., 0], "probability": result.spike_probability[..., 0],
                              "ac_local": result.amacrine_local_current[..., 0],
                              "ac_transient": result.amacrine_transient_current[..., 0]})
            for channel in ("logit", "probability"):
                error = float((parts[0][channel] - old["cells"][cid][mode][channel]).abs().max())
                errors.append(error)
                assert error == 0, (cid, mode, channel, error)
            response[f"R4_{mode}"] = {key: torch.cat([part[key] for part in parts]) for key in parts[0]}
        with torch.no_grad():
            logits = torch.cat([linear(inputs[f"{source}_drive"], inputs[f"{source}_history"])[..., 0]
                                for source in ("original", "diagnostic")])
        response["LN"] = {"logit": logits, "probability": logits.sigmoid()}
        assert all(torch.equal(v, rc["model"][k]) for k, v in retinal.state_dict().items())
        assert all(torch.equal(v, lc["model"][k]) for k, v in linear.state_dict().items())
        assert all(p.grad is None for model in (retinal, linear) for p in model.parameters())
        for label, values in response.items():
            assert all(torch.isfinite(v).all() for v in values.values())
            assert values["logit"].shape == (72, 150)
            for pair in bank.pairs:
                if pair.control:
                    assert torch.equal(values["logit"][pair.a], values["logit"][pair.b])
            mode_map = {"normal": values} if label == "LN" else {
                "normal": response["R4_normal"], label.removeprefix("R4_"): values}
            selected_rows = cell_rows(cid, metadata["group"], bank, mode_map, time)
            selected_mode = "normal" if label == "LN" else label.removeprefix("R4_")
            for row in selected_rows:
                if row["mode"] == selected_mode:
                    rows.append(row | {"model": "LN" if label == "LN" else "R4", "mode": label})
            for row in mach_rows(cid, metadata["group"], bank, {label: values}, time):
                boundary.append(row)
            for j, (family, variant, a, b, _, _) in enumerate(VARIANTS):
                for channel in ("logit", "probability"):
                    trace = values[channel][63 + 4*j] - values[channel][64 + 4*j]
                    original = values[channel][a] - values[channel][b]
                    comparisons.append({"cell_id": cid, "group": metadata["group"], "mode": label,
                        "channel": channel, "family": family, "name": variant, "kind": "variant_minus_original"}
                        | response_metrics(trace - original, time))
            for pair in bank.pairs:
                for channel in ("logit", "probability"):
                    delta = (values[channel][pair.a] - values[channel][pair.b]) - (
                        response["LN"][channel][pair.a] - response["LN"][channel][pair.b])
                    comparisons.append({"cell_id": cid, "group": metadata["group"], "mode": label,
                        "channel": channel, "family": pair.family, "name": pair.name, "kind": "model_minus_LN"}
                        | response_metrics(delta, time))
        for j, (family, variant, a, b, _, _) in enumerate(VARIANTS):
            for target, orig, diag in (("A", a, 63 + 4*j), ("B", b, 64 + 4*j)):
                for support_name in ("bc_support", "ac_support"):
                    mask = getattr(retinal.feature_bank, support_name)[0].reshape(17, 17) > 0
                    base, altered = (bank.patches[k][mask].double() + 1 for k in (orig, diag))
                    support_stats.append({"cell_id": cid, "group": metadata["group"], "family": family,
                        "variant": variant, "target": target, "support": support_name, "n_pixels": int(mask.sum()),
                        "original_mean_LM": float(base.mean()), "variant_mean_LM": float(altered.mean()),
                        "mean_delta": float(altered.mean() - base.mean()),
                        "std_delta": float(altered.std(correction=0) - base.std(correction=0)),
                        "histogram_max_error": float((base.sort().values - altered.sort().values).abs().max())})
        all_cells[cid] = response
        checks.append({"cell_id": cid, "original_R4_max_error": max(errors), "states_unchanged": True,
                       "clamps_exact_zero": True, "controls_exact_zero": True, "all_gradients_none": True})
        print(f"{index+1}/22 {cid}: original replay exact, LN/R4 frozen PASS", flush=True)
    assert all(digest(Path(p)) == value for p, value in hashes.items())
    torch.save({"cells": all_cells, "metadata": old["metadata"], "time_ms": time, "names": bank.names}, OUT / "responses.pt")
    write_csv(OUT / "per-cell-responses.csv", rows)
    write_csv(OUT / "group-responses.csv", aggregate(rows))
    write_csv(OUT / "per-cell-comparisons.csv", comparisons)
    write_csv(OUT / "group-comparisons.csv", aggregate(comparisons))
    write_csv(OUT / "mach-boundary-extrema.csv", boundary)
    write_csv(OUT / "per-cell-spatial-support-statistics.csv", support_stats)
    (OUT / "verification.json").write_text(json.dumps({"checks": checks, "sha256": hashes,
        "all_sources_unchanged": True, "training": False, "checkpoint_writes": 0}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
