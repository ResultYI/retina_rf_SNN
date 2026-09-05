import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
files = sorted(OUT.glob("*_checks.json"))
assert len(files) == 22
before = json.loads((OUT/"integrity_before.json").read_text())
changed = [name for name, digest in before.items() if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest]
assert not changed, changed
results = [json.loads(p.read_text()) for p in files]
pairs=torch.load(ROOT/"output/real_data/schottdorf_canonical_v1_shared_bc_frozen_applications_20260830/illusion/inputs.pt",map_location="cpu",weights_only=True)["pairs"]
primary = [r for data in results for r in data["checks"] if "paired_logit_max_abs" in r]
negative = [r for data in results for r in data["checks"] if r.get("negative_control")]
assert all(r["pass"] for r in primary+negative)
assert len(negative) == 2
for row in negative:
    assert row["different_histories"]
    assert row["max_signature_change"] > row["roundoff_bound"]
    assert row["predicted_shift_max"] > row["roundoff_bound"]
    assert row["predicted_shift_residual"] <= row["roundoff_bound"]
identities = []
dod_rows = []
for data in results:
    cp = torch.load(data["checkpoint"],map_location="cpu",weights_only=True)
    identities.append({"cell_id":data["cell_id"],"group":data["group"],"history_gate":float(cp["model"]["gates.history"]),
                       "history_gain":float(cp["model"]["rgc.history_gain"]),"history_decay":float(cp["model"]["rgc.history_decay"]),
                       "checkpoint_sha256":data["checkpoint_sha256"]})
    with np.load(OUT/(data["cell_id"].replace("#","_")+"_responses.npz")) as arrays:
        for dtype in sorted({r["dtype"] for r in data["checks"]}):
            name=dtype.split(".")[-1]
            rows=[r for r in primary if r["cell_id"]==data["cell_id"] and r["dtype"]==dtype]
            for mode in ("H1_off","direct_BC_off","AC_off"):
                z0 = arrays[f"{name}__{mode}__saved_zero__logit"][...,0]
                n0 = arrays[f"{name}__normal__saved_zero__logit"][...,0]
                for history in ("periodic_11","dense_one"):
                    zh = arrays[f"{name}__{mode}__{history}__logit"][...,0]
                    nh = arrays[f"{name}__normal__{history}__logit"][...,0]
                    limit = sum(next(r["roundoff_bound"] for r in rows if r["mode"]==m and r["history"]==history) for m in (mode,"normal"))
                    error=max(float(np.max(np.abs(((zh[p["a"]]-zh[p["b"]])-(nh[p["a"]]-nh[p["b"]]))-((z0[p["a"]]-z0[p["b"]])-(n0[p["a"]]-n0[p["b"]]))))) for p in pairs)
                    assert error<=limit, (data["cell_id"],dtype,mode,history,error,limit)
                    dod_rows.append({"cell_id":data["cell_id"],"dtype":dtype,"mode":mode,"history":history,"max_abs":error,"numerical_tolerance":limit,"pass":True})
all_pairs=[]
for path in sorted(OUT.glob("*_pairs.csv")):
    with path.open(newline="",encoding="utf-8") as stream:
        all_pairs.extend(csv.DictReader(stream))
per_dtype={}
for dtype in ("torch.float32","torch.float64"):
    rows=[r for r in primary if r["dtype"]==dtype]
    pairrows=[r for r in all_pairs if r["dtype"]==dtype]
    per_dtype[dtype]={"cells":len({r["cell_id"] for r in rows}),"comparisons":len(rows),
        "paired_logit_max_abs":max(r["paired_logit_max_abs"] for r in rows),
        "paired_logit_max_residual_to_tolerance_ratio":max(r["paired_logit_max_abs"]/r["roundoff_bound"] for r in rows),
        "raw_logit_max_change":max(r["raw_logit_change"] for r in rows),
        "explicit_shift_max_residual":max(r["explicit_history_shift_residual"] for r in rows),
        "dod_max_abs":max(r["max_abs"] for r in dod_rows if r["dtype"]==dtype),
        "upstream_all_bitwise_equal":all(r["upstream_bitwise_equal"] for r in rows),
        "controls_all_exact_zero":all(r["controls_exact_zero"] for r in rows)}
    for channel in ("logit","probability"):
        ps=[r for r in pairrows if r["channel"]==channel]
        per_dtype[dtype][channel]={"max_curve_change":max(float(r["max_abs"]) for r in ps),
            "max_mean_on_change":max(float(r["mean_on_change"]) for r in ps),
            "non_bitwise_equal_curves":sum(r["equal"]=="False" for r in ps),
            "raw_mean_sign_flips":sum(r["raw_mean_sign_flip"]=="True" for r in ps),
            "existing_threshold_category_changes":sum(r["reported_mean_sign_flip"]=="True" for r in ps),
            "resolved_mean_sign_flips":sum(r["resolved_mean_sign_flip"]=="True" for r in ps),
            "peak_index_changes":sum(r["peak_index_changed"]=="True" for r in ps)}
summary={"overall_paired_logit_shared_history":"PASS","checkpoint_count":len(results),"source_and_lineage_hashes_unchanged":len(before),
    "all_state_unchanged":all(d["no_state_changes"] for d in results),"all_gradients_absent":all(d["no_gradients"] for d in results),
    "history_gate_zero_cells":sum(r["history_gate"]==0 for r in identities),"nonzero_history_gates":[r for r in identities if r["history_gate"]!=0],
    "per_dtype":per_dtype,"negative_controls":negative,"dod_groups":len(dod_rows),"pair_comparison_rows":len(all_pairs),
    "training":False,"production_code_modified":False,"checkpoint_modified":False,"autoregressive_simulation":False}
summary["float64_representatives"] = [{"cell_id":data["cell_id"],"group":data["group"]} for data in results if any(r["dtype"]=="torch.float64" for r in data["checks"])]
pair_lookup = {pair["name"]:pair for pair in pairs}
category_changes=[]
for data in results:
    selected=[row for row in all_pairs if row["cell_id"]==data["cell_id"] and row["channel"]=="logit" and (row["reported_mean_sign_flip"]=="True" or row["raw_mean_sign_flip"]=="True")]
    if not selected:
        continue
    with np.load(OUT/(data["cell_id"].replace("#","_")+"_responses.npz")) as arrays:
        for row in selected:
            prefix=row["dtype"].split(".")[-1]+"__"+row["mode"]+"__"
            pair=pair_lookup[row["pair"]]
            means=[]
            for history in ("saved_zero",row["history"]):
                values=torch.from_numpy(arrays[prefix+history+"__logit"])[...,0]
                means.append(float((values[pair["a"]]-values[pair["b"]])[45:60].mean()))
            category_changes.append({**row,"saved_zero_mean_on":means[0],"changed_history_mean_on":means[1]})
summary["logit_category_change_max_mean_magnitude"] = max(max(abs(row["saved_zero_mean_on"]),abs(row["changed_history_mean_on"])) for row in category_changes)
(OUT/"SUMMARY.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
(OUT/"numerical_category_changes.json").write_text(json.dumps(category_changes,indent=2),encoding="utf-8")
(OUT/"checkpoint_identity.json").write_text(json.dumps(identities,indent=2),encoding="utf-8")
(OUT/"dod_recomputed.json").write_text(json.dumps(dod_rows,indent=2),encoding="utf-8")
(OUT/"integrity_after.json").write_text(json.dumps({"all_unchanged":True,"file_count":len(before),"sha256":before},indent=2),encoding="utf-8")
print(json.dumps(summary,indent=2))
