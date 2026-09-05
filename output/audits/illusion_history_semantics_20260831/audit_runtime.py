# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.6.0", "numpy"]
# ///
# Run from the repository root: D:/anaconda/python.exe -B audit_runtime.py START STOP.
# Frozen inference only; nonzero histories are externally supplied binary test inputs.
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT))
from history_metrics import pair_rows
from models.mechanistic_retina.contracts import MechanisticRetinaConfig, PathwayClamp
from models.mechanistic_retina.model import build_mechanistic_retina

torch.set_num_threads(2)
torch.use_deterministic_algorithms(True)
REPLAY = ROOT / "output/real_data/schottdorf_canonical_v1_shared_bc_frozen_applications_20260830"
SOURCE = ROOT / "output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830"
input_path = REPLAY / "illusion/inputs.pt"
saved = torch.load(input_path, map_location="cpu", weights_only=True)
drive, zero, clock = saved["cone_drive"], saved["history"], saved["time_ms"]
assert drive.shape == (72, 150, 289) and zero.shape == (72, 150, 1)
assert bool((zero == 0).all()) and len(saved["pairs"]) == 35
indices = torch.arange(drive.shape[1])
histories = {"saved_zero": zero, "periodic_11": ((indices % 11) == 3).to(zero).view(1,-1,1).expand_as(zero).clone(), "dense_one": torch.ones_like(zero)}
modes = {"normal": frozenset(), "H1_off": frozenset({PathwayClamp.H1}),
         "direct_BC_off": frozenset({PathwayClamp.DIRECT_BC_SUSTAINED, PathwayClamp.DIRECT_BC_TRANSIENT}),
         "AC_off": frozenset({PathwayClamp.AMACRINE_LOCAL, PathwayClamp.AMACRINE_TRANSIENT})}
paths = sorted(SOURCE.glob("cells/*/model-trained.pt"))
assert len(paths) == 22
metadata = []
history_strengths = []
for path in paths:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    metadata.append((checkpoint["cell_id"], checkpoint["cell_types"][0] + "_" + checkpoint["polarities"][0]))
    history_strengths.append(abs(float(checkpoint["model"]["gates.history"] * checkpoint["model"]["rgc.history_gain"])))
representatives = {}
for index, (cell_id, group) in enumerate(metadata):
    if group not in representatives or history_strengths[index] > history_strengths[representatives[group]]:
        representatives[group] = index
negative_index = max(range(len(paths)), key=lambda index: history_strengths[index])
active = (clock >= 300) & (clock < 400)
upstream_fields = ("h1_state", "h1_surround_contribution", "bc_direct_presynaptic", "bc_broad_presynaptic",
                   "amacrine_local_state", "amacrine_transient_state", "total_current",
                   "rgc_divisive_state", "rgc_membrane", "rgc_adaptation")
selection = range(int(sys.argv[1]), int(sys.argv[2]))
for index in selection:
    started = time.perf_counter()
    path = paths[index]
    cp = torch.load(path, map_location="cpu", weights_only=True)
    assert cp["model_config"]["architecture_mode"] == "mechanism_identifiable"
    model = build_mechanistic_retina(MechanisticRetinaConfig(**cp["model_config"]),
        cp["cone_positions_degs"], cp["cell_positions_degs"], tuple(cp["cell_types"]), tuple(cp["polarities"]))
    model.load_state_dict(cp["model"], strict=True)
    model.eval()
    assert all(torch.equal(v, cp["model"][k]) for k,v in model.state_dict().items())
    cell_id, group = metadata[index]
    rows, checks, archives = [], [], {}
    for dtype in ([torch.float32, torch.float64] if index in representatives.values() else [torch.float32]):
        model = model.to(dtype=dtype)
        state_before = {k:v.clone() for k,v in model.state_dict().items()}
        responses, outputs = {}, {}
        for mode, clamps in modes.items():
            for name, history in histories.items():
                with torch.no_grad():
                    result = model.forward_sequence(drive.to(dtype=dtype), observed_counts=history.to(dtype=dtype), clamps=clamps)
                responses[mode,name] = {"logit":result.logits[...,0], "probability":result.spike_probability[...,0]}
                outputs[mode,name] = result
                prefix = f"{str(dtype).split('.')[-1]}__{mode}__{name}"
                archives[prefix+"__logit"] = result.logits.numpy()
                archives[prefix+"__probability"] = result.spike_probability.numpy()
            baseline = outputs[mode,"saved_zero"]
            for name in ("periodic_11","dense_one"):
                changed = outputs[mode,name]
                history_term = model.gates.history * model.rgc.history_gain * changed.rgc_history_state
                bound = 32 * torch.finfo(dtype).eps * (1 + float(baseline.logits.abs().max()) + float(changed.logits.abs().max()) + float(history_term.abs().max()) + float(model.rgc.response_bias.abs().max()))
                identity = {"cell_id":cell_id, "group":group, "dtype":str(dtype), "mode":mode, "history":name}
                current_rows = pair_rows({"pairs":saved["pairs"], "active":active, "bound":bound, "identity":identity},
                                         responses[mode,"saved_zero"], responses[mode,name])
                rows.extend(current_rows)
                logit_rows = [row for row in current_rows if row["channel"] == "logit"]
                control_zero = all(bool((changed.logits[p["a"]] - changed.logits[p["b"]] == 0).all()) for p in saved["pairs"] if p["control"])
                upstream_equal = all(torch.equal(getattr(baseline,f),getattr(changed,f)) for f in upstream_fields)
                history_matches = torch.equal(changed.rgc_history_state, outputs["normal",name].rgc_history_state)
                checked = {**identity, "roundoff_bound":bound, "upstream_bitwise_equal":upstream_equal,
                           "history_state_same_across_modes":history_matches, "controls_exact_zero":control_zero,
                           "paired_logit_max_abs":max(row["max_abs"] for row in logit_rows),
                           "resolved_sign_flips":sum(row["resolved_mean_sign_flip"] for row in logit_rows),
                           "raw_logit_change":float((changed.logits-baseline.logits).abs().max()),
                           "explicit_history_shift_residual":float((changed.logits-baseline.logits+history_term).abs().max())}
                checked["pass"] = upstream_equal and control_zero and history_matches and checked["paired_logit_max_abs"] <= bound and checked["explicit_history_shift_residual"] <= bound
                checks.append(checked)
        for mode in ("H1_off","direct_BC_off","AC_off"):
            for name in ("periodic_11","dense_one"):
                bound = next(v["roundoff_bound"] for v in checks if v["dtype"]==str(dtype) and v["mode"]==mode and v["history"]==name)
                normal_bound = next(v["roundoff_bound"] for v in checks if v["dtype"]==str(dtype) and v["mode"]=="normal" and v["history"]==name)
                max_residual = 0.0
                for p in saved["pairs"]:
                    def effect(history_name):
                        cl = responses[mode,history_name]["logit"]
                        normal = responses["normal",history_name]["logit"]
                        return (cl[p["a"]]-cl[p["b"]])-(normal[p["a"]]-normal[p["b"]])
                    residual = float((effect(name)-effect("saved_zero")).abs().max())
                    max_residual = max(max_residual, residual)
                    assert residual <= bound+normal_bound, (cell_id, mode, name, p["name"], residual, bound+normal_bound)
                checks.append({"cell_id":cell_id,"dtype":str(dtype),"mode":mode,"history":name,
                    "difference_of_differences":True,"max_abs":max_residual,"roundoff_bound":bound+normal_bound,"pass":max_residual<=bound+normal_bound})
        if index == negative_index:
            asymmetric = histories["dense_one"].to(dtype=dtype).clone()
            asymmetric[::2] = 0
            with torch.no_grad():
                negative = model.forward_sequence(drive.to(dtype=dtype), observed_counts=asymmetric)
            p = next(p for p in saved["pairs"] if p["family"]=="SBC" and not p["control"])
            baseline = outputs["normal","saved_zero"]
            observed = (negative.logits[p["a"]]-negative.logits[p["b"]])-(baseline.logits[p["a"]]-baseline.logits[p["b"]])
            shift = model.gates.history*model.rgc.history_gain*(negative.rgc_history_state[p["a"]]-negative.rgc_history_state[p["b"]])
            full_term = model.gates.history*model.rgc.history_gain*negative.rgc_history_state
            negative_bound = 32*torch.finfo(dtype).eps*(1+float(baseline.logits.abs().max())+float(negative.logits.abs().max())+float(full_term.abs().max())+float(model.rgc.response_bias.abs().max()))
            different_histories = not torch.equal(asymmetric[p["a"]],asymmetric[p["b"]])
            archives[str(dtype)+"__asymmetric_history_residual"] = observed.numpy()
            checks.append({"cell_id":cell_id,"dtype":str(dtype),"negative_control":True,
                           "max_signature_change":float(observed.abs().max()),
                           "predicted_shift_residual":float((observed+shift).abs().max()),
                           "roundoff_bound":negative_bound,"different_histories":different_histories,
                           "predicted_shift_max":float(shift.abs().max()),
                           "pass":different_histories and float(observed.abs().max())>negative_bound and float(shift.abs().max())>negative_bound and float((observed+shift).abs().max()) <= negative_bound})
        assert all(torch.equal(v,state_before[k]) for k,v in model.state_dict().items())
        assert all(p.grad is None for p in model.parameters())
        del responses, outputs
    assert all(item["pass"] for item in checks), [v for v in checks if not v["pass"]]
    stem = OUT / cell_id.replace("#","_")
    np.savez_compressed(str(stem)+"_responses.npz", **archives)
    Path(str(stem)+"_checks.json").write_text(json.dumps({"cell_id":cell_id,"group":group,"checkpoint":str(path),
        "checkpoint_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"checks":checks,"no_state_changes":True,"no_gradients":True,
        "elapsed_seconds":time.perf_counter()-started},indent=2),encoding="utf-8")
    with Path(str(stem)+"_pairs.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"index":index,"cell_id":cell_id,"pass":True,"seconds":round(time.perf_counter()-started,2)}),flush=True)
