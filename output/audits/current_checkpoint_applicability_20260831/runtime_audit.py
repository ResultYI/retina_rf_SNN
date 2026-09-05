#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.6.0", "numpy", "pydantic", "scipy", "opencv-python"]
# ///
# How to run: D:/anaconda/python.exe -B runtime_audit.py from the repository root.
# Uses the existing numerical runtime. Reads inputs; JSON goes to stdout only.
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
from types import FrameType
from typing import Final
import numpy as np
import torch
from torch import nn
from data.schottdorf_lee_2021 import SchottdorfAdapterConfig, _cone_positions
from models.mechanistic_retina.contracts import MechanisticRetinaConfig, MECHANISTIC_MODEL_REVISION
from models.mechanistic_retina.model import build_mechanistic_retina
from models.mechanistic_retina.causal_contract import CANONICAL_CAUSAL_CONTRACT
from models.mechanistic_retina.spatial_contract import CANONICAL_SPATIAL_CONTRACT

ROOT: Final = Path.cwd()
BASE: Final = ROOT / "output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830"
TOL: Final = 1e-12
RADII: Final = {"midget": (0.06, 0.13), "parasol": (0.10, 0.15)}

def tensor_hash(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def file_hash(p: Path) -> str:
    with p.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()

def main() -> None:
    torch.set_num_threads(2)
    torch.manual_seed(831)
    index = json.loads((BASE / "results.json").read_text())
    manifest = json.loads((BASE / "run-manifest.json").read_text())
    comparison = json.loads((BASE / "comparison.json").read_text())
    historical_hashes = {str(Path(p).resolve()): h for p,h in comparison["source_sha256"].items()}
    expected_ids = [r["cell_id"] for r in index["cells"]]
    paths = sorted(BASE.glob("cells/*/model-trained.pt"))
    assert len(paths) == len(expected_ids) == 22
    adapter_grid = torch.from_numpy(_cone_positions(SchottdorfAdapterConfig(**index["adapter_config"])))
    rows = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        cid = payload["cell_id"]
        assert cid in expected_ids and path.parent.name == cid.replace("#", "_")
        state = payload["model"]
        config = MechanisticRetinaConfig(**payload["model_config"])
        positions, centers = payload["cone_positions_degs"], payload["cell_positions_degs"]
        cell_types, polarities = tuple(payload["cell_types"]), tuple(payload["polarities"])
        model = build_mechanistic_retina(config, positions, centers, cell_types, polarities)
        bank = model.feature_bank
        geometry_names = ("spatial_basis","path_spatial_basis","bc_support","ac_support")
        geometry_matches = {k: torch.equal(getattr(bank,k),state["feature_bank."+k]) for k in geometry_names}
        loaded = model.load_state_dict(state, strict=True)
        model.eval()
        causal = bytes(state["_causal_contract_id"].tolist()).decode()
        spatial = bytes(state["_spatial_contract_id"].tolist()).decode()
        checkpoint_hash = file_hash(path)
        row = {"cell_id":cid,"checkpoint_path":str(path.resolve()),"checkpoint_sha256":checkpoint_hash,
            "N":int(centers.shape[0]),"architecture_mode":str(config.architecture_mode),
            "schema":payload["schema"],"revision":payload["revision"],"stage":payload["stage"],
            "cell_types":list(cell_types),"polarities":list(polarities),
            "causal_contract_config":config.causal_contract,"causal_contract_state":causal,
            "spatial_contract_config":config.spatial_contract,"spatial_contract_state":spatial,
            "causal_id_sha256":tensor_hash(state["_causal_contract_id"]),
            "config_sha256":hashlib.sha256(json.dumps(payload["model_config"],sort_keys=True,separators=(",",":")).encode()).hexdigest(),
            "geometry_source":"checkpoint coordinates; adapter _cone_positions; default construction; checkpoint-loaded path_spatial_basis",
            "adapter_grid_bitwise_equal":torch.equal(positions,adapter_grid),
            "centers":centers.tolist(),"cone_count":int(positions.shape[0]),
            "cone_positions_sha256":tensor_hash(positions),"fresh_default_geometry_equals_checkpoint":geometry_matches,
            "graph_nodes":int(model.shared_subunits.cell_order.numel()),
            "graph_edges":int(model.shared_subunits.edge_index.shape[1]),
            "edge_index":model.shared_subunits.edge_index.tolist(),
            "connection_matrix":model.shared_subunits.connection_matrix().detach().tolist(),
            "checkpoint_matches_existing_comparison_hash":historical_hashes.get(str(path.resolve())) == checkpoint_hash,
            "model_config":payload["model_config"],"training_contract":payload["training_contract"],
            "state_keys_shapes":{k:list(t.shape) for k,t in state.items()},
            "strict_load_missing":list(loaded.missing_keys),"strict_load_unexpected":list(loaded.unexpected_keys)}
        identity_ok = (row["N"]==1 and row["architecture_mode"]=="mechanism_identifiable"
            and causal==config.causal_contract==CANONICAL_CAUSAL_CONTRACT
            and spatial==config.spatial_contract==CANONICAL_SPATIAL_CONTRACT
            and payload["schema"]=="schottdorf_canonical_v1_shared_bc_development"
            and payload["revision"]==MECHANISTIC_MODEL_REVISION and payload["stage"]=="trained"
            and row["checkpoint_matches_existing_comparison_hash"])
        xh1: list[torch.Tensor] = []
        kernels: list[torch.Tensor] = []
        ac_inputs: list[torch.Tensor] = []
        bc_calls: list[list[int]] = []
        call_names: list[str] = []
        modulation: list[torch.Tensor] = []
        def bank_pre(module: nn.Module, args: tuple[torch.Tensor,...]) -> None:
            xh1.append(args[0])
        def bc_pre(module: nn.Module, args: tuple[torch.Tensor,...]) -> None:
            w=module.get_parameter("raw_weights")
            bc_calls.append([id(module),id(w),w.data_ptr()])
        def ac_pre(module: nn.Module, args: tuple[torch.Tensor,...]) -> None:
            ac_inputs.append(args[0])
        def operator_post(module: nn.Module, args: tuple[torch.Tensor,...], result: torch.Tensor) -> None:
            modulation.append(result)
        def profile(frame: FrameType, event: str, result: torch.Tensor | None) -> None:
            if event=="return" and frame.f_code.co_name=="basis_kernels" and frame.f_locals.get("self") is bank:
                kernels.append(result)
        handles=[bank.register_forward_pre_hook(bank_pre),model.bipolar.register_forward_pre_hook(bc_pre),
            model.amacrine.register_forward_pre_hook(ac_pre),model.operator.register_forward_hook(operator_post)]
        for name,module in model.named_modules():
            def note(module: nn.Module, args: tuple[torch.Tensor,...], label: str=name) -> None:
                call_names.append(label)
            handles.append(module.register_forward_pre_hook(note))
        time_steps = 2*config.lag_steps
        drive = (0.2*torch.sin(torch.arange(time_steps*positions.shape[0],dtype=positions.dtype)*0.017)).reshape(1,time_steps,-1).requires_grad_(True)
        history = torch.zeros((1,time_steps,centers.shape[0]),dtype=positions.dtype)
        sys.setprofile(profile)
        try:
            output=model.forward_sequence(drive,observed_counts=history)
        finally:
            sys.setprofile(None)
            for handle in handles: handle.remove()
        assert len(xh1)==len(kernels)==len(ac_inputs)==len(modulation)==1
        kernel=kernels[0].detach()
        row["dtype"]=str(drive.dtype)
        row["production_trace"]=call_names
        row["basis_kernel_sha256"]=tensor_hash(kernel)
        row["bipolar_call_identity"]=bc_calls
        row["same_BC_parameter_used_twice"]=len(bc_calls)==2 and bc_calls[0]==bc_calls[1]
        row["AC_input_is_BC_broad"]=ac_inputs[0] is output.bc_broad_presynaptic
        row["AC_parameters"]={k:list(v.shape) for k,v in model.amacrine.named_parameters()}
        row["shared_BC_tau_shape"]=list(bank.raw_tau.shape)
        row["shared_BC_delay_shape"]=list(bank.raw_delay.shape)
        row["operator_disabled_ones"]=bool((modulation[0]==1).all()) and "operator.depthwise" not in call_names
        row["all_output_finite"]=all(bool(torch.isfinite(t).all()) for t in output.tensors())
        masks=[]
        derivatives=[]
        for n,cell_type in enumerate(cell_types):
            radii=RADII[cell_type]
            delta=positions.double().numpy()-centers[n].double().numpy()
            distance=np.hypot(delta[:,0],delta[:,1])
            supports=[]
            for label,path_indices,radius,stored,bc_output in (
                ("BC",slice(0,2),radii[0],bank.bc_support[n],output.bc_direct_presynaptic),
                ("AC",slice(2,4),radii[1],bank.ac_support[n],output.bc_broad_presynaptic)):
                expected=torch.from_numpy(distance<=radius)
                actual=(kernel[n,path_indices]!=0).any(dim=(0,1,2,3))
                path_actual=(bank.path_spatial_basis[n,path_indices]!=0).any(dim=(0,1))
                supports.append(actual)
                records=[]
                per_cone=torch.zeros(positions.shape[0],dtype=drive.dtype)
                for t in (config.lag_steps-1,time_steps-1):
                    for feature in (0,1):
                        g=torch.autograd.grad(bc_output[0,t,n,feature],xh1[0],retain_graph=True)[0]
                        v=g.detach().abs().amax(dim=(0,1))
                        per_cone=torch.maximum(per_cone,v)
                        records.append({"time_index":t,"feature":feature,
                            "outside_max":float(v[~expected].max()),"inside_max":float(v[expected].max()),
                            "finite":bool(torch.isfinite(g).all()),"jacobian_sha256":tensor_hash(g)})
                mismatch=int(torch.count_nonzero(actual!=expected))
                masks.append({"cell_index":n,"support":label,"radius_deg":radius,
                    "expected_count":int(expected.sum()),"actual_count":int(actual.sum()),
                    "stored_count":int((stored!=0).sum()),"mismatch_count":mismatch,
                    "stored_mismatch_count":int(torch.count_nonzero(stored!=expected.to(stored))),
                    "runtime_kernel_matches_path_spatial_basis":torch.equal(actual,path_actual),
                    "max_included_radius_deg":float(distance[actual.numpy()].max()),
                    "interior_holes":int(torch.count_nonzero(expected & ~actual)),
                    "outside_inclusions":int(torch.count_nonzero(actual & ~expected)),
                    "boundary_min_gap_deg":float(np.min(np.abs(distance-radius))),
                    "expected_indices":expected.nonzero().flatten().tolist(),
                    "actual_indices":actual.nonzero().flatten().tolist(),
                    "full_disk":mismatch==0})
                derivatives.append({"cell_index":n,"support":label,"outside_cone_count":int((~expected).sum()),
                    "outside_max":float(per_cone[~expected].max()),"inside_max":float(per_cone[expected].max()),
                    "inside_nonzero_cones":int(torch.count_nonzero(per_cone[expected])),
                    "per_cone_abs_max":per_cone.tolist(),"VJPs":records,
                    "passed":all(r["finite"] and r["outside_max"]<=TOL and r["inside_max"]>0 for r in records)})
            row["BC_strict_subset_AC"]=bool((supports[0]<=supports[1]).all() and (supports[1]&~supports[0]).any())
        ac_energy=output.amacrine_local_state.square().sum()+output.amacrine_transient_state.square().sum()
        ac_to_broad=torch.autograd.grad(ac_energy,output.bc_broad_presynaptic,retain_graph=True)[0]
        total_gradient=torch.autograd.grad(ac_energy,drive,retain_graph=True)[0]
        routed_gradient=torch.autograd.grad(output.bc_broad_presynaptic,drive,grad_outputs=ac_to_broad,retain_graph=True)[0]
        row["AC_to_BC_broad_max_derivative"]=float(ac_to_broad.abs().max())
        row["AC_stimulus_gradient_max"]=float(total_gradient.abs().max())
        row["AC_gradient_via_broad_residual_max"]=float((total_gradient-routed_gradient).abs().max())
        row["AC_gradient_via_broad_bitwise_equal"]=torch.equal(total_gradient,routed_gradient)
        row["independent_AC_state_keys"]=[k for k in state if k.startswith("amacrine.") and k not in
            {"amacrine.raw_tau","amacrine.raw_delay","amacrine.group_index","amacrine.tau_bounds_ms","amacrine.delay_bounds_ms"}]
        row["masks"]=masks
        row["derivatives"]=derivatives
        row["state_bitwise_unchanged"]=all(torch.equal(v,state[k]) for k,v in model.state_dict().items())
        row["parameter_grads_absent"]=all(p.grad is None for p in model.parameters())
        row["checkpoint_sha256_after"]=file_hash(path)
        row["identity_passed"]=identity_ok
        row["masks_passed"]=all(m["full_disk"] and m["stored_mismatch_count"]==0 and m["runtime_kernel_matches_path_spatial_basis"] for m in masks) and row["BC_strict_subset_AC"]
        row["derivatives_passed"]=all(d["passed"] for d in derivatives)
        row["legacy_path_passed"]=identity_ok and row["same_BC_parameter_used_twice"] and row["AC_input_is_BC_broad"] and row["operator_disabled_ones"] and not row["independent_AC_state_keys"] and set(row["AC_parameters"])=={"raw_tau","raw_delay"} and row["AC_gradient_via_broad_bitwise_equal"] and row["AC_to_BC_broad_max_derivative"]>0
        rows.append(row)
        print("AUDITED "+cid,file=sys.stderr,flush=True)
    assert set(r["cell_id"] for r in rows)==set(expected_ids)==set(manifest["cell_ids"])
    print(json.dumps({"tolerance":TOL,"torch_version":torch.__version__,"threads":torch.get_num_threads(),
        "probe":"deterministic sine drive [1,32,289], zero history; VJPs at t=15,31, both sustained/transient separately; derivative with respect to actual X_H1; no fitting",
        "adapter_config":index["adapter_config"],"cell_ids":expected_ids,"rows":rows}))
if __name__=="__main__":
    main()
